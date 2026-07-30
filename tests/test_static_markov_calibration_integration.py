"""Static Markov-to-calibration integration tests for CAL-080."""

from copy import deepcopy

import pandas as pd
import pytest

from do_attribution import MarkovAttribution
from do_attribution.calibration import AttributionCalibration, LiftTestEvidence


def _fit_markov(paths, conversions):
    model = MarkovAttribution(
        order=1,
        removal_strategy="redirect_to_null",
        return_transition_matrix=True,
        conversion_labels=["Purchase"],
    )
    fit_result = model.fit(paths, conversions)
    return model, fit_result


def _snapshot(model, fit_result):
    return {
        "attribution": dict(fit_result["attribution"]),
        "removal_effects": dict(fit_result["removal_effects"]),
        "total_conversion_probability": fit_result[
            "total_conversion_probability"
        ],
        "fit_transition_matrix": fit_result["transition_matrix"].copy(deep=True),
        "model_transition_matrix": model.transition_matrix.copy(deep=True),
        "channels": tuple(model.get_channels()),
        "states": tuple(model.states),
        "public_attribution": model.get_attribution(),
        "summary": model.summary(),
        "report": model.summary(output_format="report"),
    }


def _assert_unchanged(model, fit_result, snapshot):
    assert fit_result["attribution"] == snapshot["attribution"]
    assert fit_result["removal_effects"] == snapshot["removal_effects"]
    assert (
        fit_result["total_conversion_probability"]
        == snapshot["total_conversion_probability"]
    )
    pd.testing.assert_frame_equal(
        fit_result["transition_matrix"], snapshot["fit_transition_matrix"]
    )
    pd.testing.assert_frame_equal(
        model.transition_matrix, snapshot["model_transition_matrix"]
    )
    assert tuple(model.get_channels()) == snapshot["channels"]
    assert tuple(model.states) == snapshot["states"]
    assert model.get_attribution() == snapshot["public_attribution"]
    assert model.summary() == snapshot["summary"]
    assert model.summary(output_format="report") == snapshot["report"]


def test_approved_markov_estimates_calibrate_without_touching_model(
    example_paths,
    example_conversions,
) -> None:
    model, fit_result = _fit_markov(example_paths, example_conversions)
    snapshot = _snapshot(model, fit_result)
    approved_estimates = {
        entity: share * 1000.0
        for entity, share in fit_result["attribution"].items()
    }
    approved_before = deepcopy(approved_estimates)
    tested_entity = "Email"
    original_tested = approved_estimates[tested_entity]
    eligible = [
        entity for entity in approved_estimates if entity != tested_entity
    ]
    evidence = LiftTestEvidence(
        tested_entity=tested_entity,
        observed_lift=0.9 * original_tested,
        matched_mta_estimate=original_tested,
        test_id="email-lift",
        metadata={"source": "caller-approved Markov estimate"},
    )

    calibrated = AttributionCalibration([evidence]).apply_estimates(
        approved_estimates,
        residual_policy="proportional_untested",
        eligible_entities=eligible,
    )

    rows = {row.entity: row for row in calibrated.rows}
    required = 0.1 * original_tested
    eligible_mass = sum(approved_estimates[entity] for entity in eligible)
    assert rows[tested_entity].calibrated_estimate == pytest.approx(
        0.9 * original_tested
    )
    assert rows[tested_entity].role == "tested"
    for entity in eligible:
        assert rows[entity].calibrated_estimate == pytest.approx(
            approved_estimates[entity]
            + required * approved_estimates[entity] / eligible_mass
        )
        assert rows[entity].role == "residual_recipient"
        assert rows[entity].direct_test_adjustment == 0.0
    assert calibrated.original_total == pytest.approx(1000.0)
    assert calibrated.required_residual == pytest.approx(required)
    assert calibrated.allocated_residual == pytest.approx(required)
    assert calibrated.calibrated_total == pytest.approx(1000.0)
    exported = calibrated.to_dataframe()
    assert exported["entity"].tolist() == list(approved_estimates)
    assert exported.loc[
        exported["entity"] == tested_entity,
        "evidence_ids",
    ].iloc[0] == ("email-lift",)
    assert "calibration mechanics" in calibrated.summary()
    assert approved_estimates == approved_before
    _assert_unchanged(model, fit_result, snapshot)


def test_approved_markov_shares_require_an_explicit_calibration_policy(
    example_paths,
    example_conversions,
) -> None:
    model, fit_result = _fit_markov(example_paths, example_conversions)
    snapshot = _snapshot(model, fit_result)
    approved_shares = dict(fit_result["attribution"])
    shares_before = deepcopy(approved_shares)
    tested_entity = "Social"
    evidence = LiftTestEvidence(
        tested_entity=tested_entity,
        observed_lift=1.1 * approved_shares[tested_entity],
        matched_mta_estimate=approved_shares[tested_entity],
        test_id="social-lift",
    )
    calibrator = AttributionCalibration([evidence])

    with pytest.raises(TypeError, match="residual_policy"):
        calibrator.apply_shares(approved_shares, current_total=500.0)
    calibrated = calibrator.apply_shares(
        approved_shares,
        current_total=500.0,
        residual_policy="allow_total_change",
    )

    rows = {row.entity: row for row in calibrated.rows}
    tested_original = 500.0 * approved_shares[tested_entity]
    assert rows[tested_entity].calibrated_estimate == pytest.approx(
        1.1 * tested_original
    )
    for entity, share in approved_shares.items():
        if entity != tested_entity:
            assert rows[entity].calibrated_estimate == pytest.approx(500.0 * share)
            assert rows[entity].role == "untouched"
    assert calibrated.allocated_residual == 0.0
    assert calibrated.calibrated_total == pytest.approx(
        500.0 + 0.1 * tested_original
    )
    assert shares_before == approved_shares
    _assert_unchanged(model, fit_result, snapshot)
