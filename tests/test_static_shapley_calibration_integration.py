"""Static Shapley-to-calibration integration tests for CAL-081."""

from copy import deepcopy

import pytest

from do_attribution import ShapleyAttribution
from do_attribution.calibration import AttributionCalibration, LiftTestEvidence


def _fit_shapley(paths, conversions):
    model = ShapleyAttribution()
    fit_result = model.fit(paths, conversions)
    return model, fit_result


def _snapshot(model, fit_result):
    return {
        "attribution": dict(fit_result["attribution"]),
        "shapley_values": dict(fit_result["shapley_values"]),
        "value_function": deepcopy(fit_result["value_function"]),
        "model_value_function": deepcopy(model.value_function),
        "channels": tuple(model.get_channels()),
        "public_attribution": model.get_attribution(),
        "summary": model.summary(),
        "report": model.summary(output_format="report"),
    }


def _assert_unchanged(model, fit_result, snapshot):
    assert fit_result["attribution"] == snapshot["attribution"]
    assert fit_result["shapley_values"] == snapshot["shapley_values"]
    assert fit_result["value_function"] == snapshot["value_function"]
    assert model.value_function == snapshot["model_value_function"]
    assert tuple(model.get_channels()) == snapshot["channels"]
    assert model.get_attribution() == snapshot["public_attribution"]
    assert model.summary() == snapshot["summary"]
    assert model.summary(output_format="report") == snapshot["report"]


def test_approved_shapley_allocation_calibrates_by_declared_inputs_only(
    simple_paths,
    simple_conversions,
) -> None:
    model, fit_result = _fit_shapley(simple_paths, simple_conversions)
    snapshot = _snapshot(model, fit_result)
    approved_estimates = {
        entity: share * 1200.0
        for entity, share in fit_result["attribution"].items()
    }
    estimates_before = deepcopy(approved_estimates)
    tested_entity = "A"
    original_tested = approved_estimates[tested_entity]
    evidence = LiftTestEvidence(
        tested_entity=tested_entity,
        observed_lift=0.85 * original_tested,
        matched_mta_estimate=original_tested,
        test_id="a-experiment",
        metadata={
            "input_boundary": "caller-approved allocation; not raw Shapley lift"
        },
    )

    calibrated = AttributionCalibration([evidence]).apply_estimates(
        approved_estimates,
        residual_policy="custom_weights",
        residual_weights={"B": 0.7, "C": 0.3},
    )

    rows = {row.entity: row for row in calibrated.rows}
    required = 0.15 * original_tested
    assert rows["A"].calibrated_estimate == pytest.approx(
        0.85 * original_tested
    )
    assert rows["A"].evidence_ids == ("a-experiment",)
    assert rows["B"].residual_adjustment == pytest.approx(0.7 * required)
    assert rows["C"].residual_adjustment == pytest.approx(0.3 * required)
    assert calibrated.calibrated_total == pytest.approx(1200.0)
    assert calibrated.to_dataframe()["role"].tolist() == [
        "tested",
        "residual_recipient",
        "residual_recipient",
    ]
    summary = calibrated.summary().lower()
    assert "calibration mechanics" in summary
    assert "shapley" not in summary
    assert "incremental outcome" not in summary
    assert approved_estimates == estimates_before
    _assert_unchanged(model, fit_result, snapshot)


def test_approved_shapley_shares_can_use_an_explicit_residual_bucket(
    simple_paths,
    simple_conversions,
) -> None:
    model, fit_result = _fit_shapley(simple_paths, simple_conversions)
    snapshot = _snapshot(model, fit_result)
    approved_shares = dict(fit_result["attribution"])
    shares_before = deepcopy(approved_shares)
    tested_entity = "B"
    current_total = 500.0
    original_tested = current_total * approved_shares[tested_entity]
    evidence = LiftTestEvidence(
        tested_entity=tested_entity,
        observed_lift=1.2 * original_tested,
        matched_mta_estimate=original_tested,
        test_id="b-experiment",
    )

    calibrated = AttributionCalibration([evidence]).apply_shares(
        approved_shares,
        current_total=current_total,
        residual_policy="residual_bucket",
        residual_entity="unassigned",
        allow_signed=True,
    )

    rows = {row.entity: row for row in calibrated.rows}
    required = -0.2 * original_tested
    assert rows[tested_entity].calibrated_estimate == pytest.approx(
        1.2 * original_tested
    )
    assert rows["unassigned"].original_estimate == 0.0
    assert rows["unassigned"].residual_adjustment == pytest.approx(required)
    assert rows["unassigned"].role == "residual_bucket"
    assert calibrated.calibrated_total == pytest.approx(current_total)
    assert approved_shares == shares_before
    _assert_unchanged(model, fit_result, snapshot)
