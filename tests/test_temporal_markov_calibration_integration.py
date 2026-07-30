"""Temporal Markov-to-calibration integration tests for CAL-082.

Calibration consumes the model's authoritative epoch grid and caller-approved
per-epoch output as an external post-attribution layer.  These tests exercise
one direct (``observed_only`` + ``direct``) and one coordinate-linear
(``coordinate_linear`` + ``representative_coordinate``) reduction path and
prove that:

- the authoritative epoch keys, ordering, and boundaries are preserved;
- fitted model state, result dictionaries, summaries, and exports are
  unchanged;
- epoch recommendation and boundary detection are never invoked.
"""

from copy import deepcopy
from datetime import datetime

import pandas as pd
import pytest

import do_attribution
import do_attribution.temporal as temporal_module
import do_attribution.temporal_markov as temporal_markov_module
from do_attribution import TemporalMarkovAttribution
from do_attribution.calibration import (
    LiftTestEvidence,
    TemporalAttributionCalibration,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def temporal_markov_data():
    """The existing two-quarter Temporal Markov dataset.

    Mirrors ``small_temporal_data`` in ``tests/test_temporal_markov.py`` so the
    fitted authoritative grid is the familiar ``[2025Q1, 2025Q2]`` epoch order
    with channels ``{A, B, C}`` in Q1 and ``{B, C}`` in Q2.
    """
    q1_paths = [
        "A > B", "A > B", "A > C", "B > A", "A > B",
        "A > C", "B > A", "C > B > A", "A > B", "B > A",
    ]
    q1_dates = pd.date_range("2025-01-15", periods=10, freq="D")
    q1_convs = [1, 1, 1, 1, 0, 1, 1, 0, 1, 1]

    q2_paths = [
        "B > C", "C > B", "B > C", "B > C", "C > B",
        "B > C", "C > B", "B > C", "C > B", "B > C",
    ]
    q2_dates = pd.date_range("2025-04-15", periods=10, freq="D")
    q2_convs = [1, 1, 1, 0, 1, 1, 1, 0, 1, 1]

    paths = list(q1_paths) + list(q2_paths)
    dates = list(q1_dates) + list(q2_dates)
    convs = list(q1_convs) + list(q2_convs)
    return paths, dates, convs


@pytest.fixture
def recommendation_spies(monkeypatch):
    """Replace every epoch-recommendation and boundary-detection entry point.

    Each spy records its call and raises, so any invocation both fails the
    test immediately and leaves an audit trail.  The calls list must remain
    empty for the whole fit-snapshot-calibrate-export flow.
    """
    calls = []

    def _spy(name):
        def _record(*args, **kwargs):
            calls.append(name)
            raise AssertionError(
                f"{name} must not be called by the calibration integration path"
            )

        return _record

    monkeypatch.setattr(
        temporal_module, "recommend_epochs", _spy("temporal.recommend_epochs")
    )
    monkeypatch.setattr(
        temporal_module,
        "assign_paths_to_epochs",
        _spy("temporal.assign_paths_to_epochs"),
    )
    monkeypatch.setattr(
        temporal_module,
        "_detect_scalar_breakpoints",
        _spy("temporal._detect_scalar_breakpoints"),
    )
    monkeypatch.setattr(
        temporal_markov_module,
        "recommend_epochs",
        _spy("temporal_markov.recommend_epochs"),
    )
    monkeypatch.setattr(
        temporal_markov_module,
        "assign_paths_to_epochs",
        _spy("temporal_markov.assign_paths_to_epochs"),
    )
    monkeypatch.setattr(
        do_attribution, "recommend_epochs", _spy("package.recommend_epochs")
    )
    monkeypatch.setattr(
        do_attribution,
        "assign_paths_to_epochs",
        _spy("package.assign_paths_to_epochs"),
    )
    monkeypatch.setattr(
        TemporalMarkovAttribution,
        "fit_recommended",
        _spy("TemporalMarkovAttribution.fit_recommended"),
    )
    monkeypatch.setattr(
        TemporalMarkovAttribution,
        "detect_threshold_crossings",
        _spy("TemporalMarkovAttribution.detect_threshold_crossings"),
    )
    return calls


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _fit_temporal_markov(temporal_markov_data):
    paths, dates, convs = temporal_markov_data
    model = TemporalMarkovAttribution(
        freq="Q",
        min_samples_per_epoch=5,
        removal_strategy="redirect_to_null",
    )
    fit_result = model.fit(paths, dates, convs)
    return model, fit_result


def _authoritative_windows(model):
    """Caller-declared half-open windows built from the model's boundaries."""
    starts = [epoch.start_time.to_pydatetime() for epoch in model.epochs]
    ends = starts[1:] + [(model.epochs[-1] + 1).start_time.to_pydatetime()]
    return {
        epoch: (start, end)
        for epoch, start, end in zip(model.epochs, starts, ends)
    }


def _snapshot(model, fit_result):
    return {
        "epochs": list(model.epochs),
        "epoch_labels": [str(epoch) for epoch in model.epochs],
        "boundaries": [
            (epoch.start_time, epoch.end_time) for epoch in model.epochs
        ],
        "fit_epochs": list(fit_result["epochs"]),
        "channels": tuple(fit_result["channels"]),
        "epoch_attributions": {
            epoch: dict(result["attribution"])
            for epoch, result in model.epoch_results.items()
        },
        "epoch_removal_effects": {
            epoch: dict(result["removal_effects"])
            for epoch, result in model.epoch_results.items()
        },
        "epoch_conversion_probabilities": {
            epoch: result["total_conversion_probability"]
            for epoch, result in model.epoch_results.items()
        },
        "attribution_matrix": fit_result["attribution_matrix"].copy(deep=True),
        "removal_effect_matrix": fit_result["removal_effect_matrix"].copy(
            deep=True
        ),
        "conversion_probability_trend": fit_result[
            "conversion_probability_trend"
        ].copy(deep=True),
        "epoch_summary": model.get_epoch_summary().copy(deep=True),
        "channel_epoch_sample_sizes": model.get_channel_epoch_sample_sizes()
        .copy(deep=True),
        "volume_weighted": dict(model.get_volume_weighted_attribution()),
        "summary": model.summary(),
        "report": model.summary(output_format="report"),
        "dataframes": {
            key: frame.copy(deep=True)
            for key, frame in model.to_dataframes().items()
        },
        "epoch_recommendation": model.epoch_recommendation,
        "global_fit_done": model._global_fit_done,
        "freq": model.freq,
        "epoch_assignment": model.epoch_assignment,
    }


def _assert_unchanged(model, fit_result, snapshot):
    assert list(model.epochs) == snapshot["epochs"]
    assert [str(epoch) for epoch in model.epochs] == snapshot["epoch_labels"]
    assert [
        (epoch.start_time, epoch.end_time) for epoch in model.epochs
    ] == snapshot["boundaries"]
    assert list(fit_result["epochs"]) == snapshot["fit_epochs"]
    assert tuple(fit_result["channels"]) == snapshot["channels"]
    assert {
        epoch: dict(result["attribution"])
        for epoch, result in model.epoch_results.items()
    } == snapshot["epoch_attributions"]
    assert {
        epoch: dict(result["removal_effects"])
        for epoch, result in model.epoch_results.items()
    } == snapshot["epoch_removal_effects"]
    assert {
        epoch: result["total_conversion_probability"]
        for epoch, result in model.epoch_results.items()
    } == snapshot["epoch_conversion_probabilities"]
    pd.testing.assert_frame_equal(
        fit_result["attribution_matrix"], snapshot["attribution_matrix"]
    )
    pd.testing.assert_frame_equal(
        fit_result["removal_effect_matrix"], snapshot["removal_effect_matrix"]
    )
    pd.testing.assert_series_equal(
        fit_result["conversion_probability_trend"],
        snapshot["conversion_probability_trend"],
    )
    pd.testing.assert_frame_equal(
        model.get_epoch_summary(), snapshot["epoch_summary"]
    )
    pd.testing.assert_frame_equal(
        model.get_channel_epoch_sample_sizes(),
        snapshot["channel_epoch_sample_sizes"],
    )
    assert model.get_volume_weighted_attribution() == snapshot["volume_weighted"]
    assert model.summary() == snapshot["summary"]
    assert model.summary(output_format="report") == snapshot["report"]
    fresh_frames = model.to_dataframes()
    assert list(fresh_frames) == list(snapshot["dataframes"])
    for key, frame in snapshot["dataframes"].items():
        pd.testing.assert_frame_equal(fresh_frames[key], frame)
    assert model.epoch_recommendation is snapshot["epoch_recommendation"]
    assert model.epoch_recommendation is None
    assert model._global_fit_done is snapshot["global_fit_done"]
    assert model._global_fit_done is False
    assert model.freq == snapshot["freq"]
    assert model.epoch_assignment == snapshot["epoch_assignment"]


# --------------------------------------------------------------------------- #
# Direct reduction path                                                        #
# --------------------------------------------------------------------------- #
def test_direct_observed_only_calibration_preserves_authoritative_grid(
    temporal_markov_data,
    recommendation_spies,
) -> None:
    model, fit_result = _fit_temporal_markov(temporal_markov_data)
    snapshot = _snapshot(model, fit_result)
    q1, q2 = model.epochs

    totals = {q1: 800.0, q2: 400.0}
    approved_estimates = {
        epoch: {
            channel: share * totals[epoch]
            for channel, share in model.epoch_results[epoch][
                "attribution"
            ].items()
        }
        for epoch in model.epochs
    }
    estimates_before = deepcopy(approved_estimates)
    epoch_windows = _authoritative_windows(model)
    windows_before = deepcopy(epoch_windows)

    evidence = LiftTestEvidence(
        tested_entity="B",
        observed_lift=90.0,
        matched_mta_estimate=100.0,
        test_id="b-q1-geo",
        coordinate_name="date",
        coordinate_value=datetime(2025, 2, 1),
    )
    result = TemporalAttributionCalibration([evidence]).apply_estimates(
        approved_estimates,
        ratio_policy="observed_only",
        epoch_ratio_policy="direct",
        residual_policy="proportional_untested",
        epoch_windows=epoch_windows,
        eligible_entities_by_epoch={q1: ["A", "C"]},
    )

    # The authoritative epoch keys and ordering pass through unchanged.
    assert result.epochs == tuple(model.epochs)
    assert result.ratio_policy == "observed_only"
    assert result.epoch_ratio_policy == "direct"

    # Exactly one effective ratio exists, for the one evidence-backed pair.
    assert len(result.epoch_ratio_schedule) == 1
    ratio_row = result.epoch_ratio_schedule[0]
    assert ratio_row.tested_entity == "B"
    assert ratio_row.epoch == q1
    assert ratio_row.effective_ratio == pytest.approx(0.9)
    assert ratio_row.ratio_source == "observed"
    assert ratio_row.source_evidence_ids == ("b-q1-geo",)

    # Q1: tested entity fixed at 0.9x, residual split A:C proportionally.
    rows_q1 = {row.entity: row for row in result.rows[q1]}
    original_b = approved_estimates[q1]["B"]
    required = 0.1 * original_b
    eligible_mass = (
        approved_estimates[q1]["A"] + approved_estimates[q1]["C"]
    )
    assert rows_q1["B"].calibrated_estimate == pytest.approx(0.9 * original_b)
    assert rows_q1["B"].role == "tested"
    for channel in ("A", "C"):
        assert rows_q1[channel].calibrated_estimate == pytest.approx(
            approved_estimates[q1][channel]
            + required * approved_estimates[q1][channel] / eligible_mass
        )
        assert rows_q1[channel].role == "residual_recipient"
        assert rows_q1[channel].direct_test_adjustment == 0.0
    assert result.calibrated_totals[q1] == pytest.approx(800.0)

    # Q2 contains no evidence coordinate and stays wholly uncalibrated.
    assert all(row.role == "untouched" for row in result.rows[q2])
    assert all(row.residual_policy is None for row in result.rows[q2])
    assert result.calibrated_totals[q2] == pytest.approx(400.0)
    assert result.residual_provenance[q2]["status"] == "not_calibrated"

    # Caller inputs, fitted model state, outputs, summaries, and exports
    # are unchanged; recommendation/boundary logic was never invoked.
    assert approved_estimates == estimates_before
    assert epoch_windows == windows_before
    _assert_unchanged(model, fit_result, snapshot)
    assert recommendation_spies == []


# --------------------------------------------------------------------------- #
# Coordinate-linear reduction path                                             #
# --------------------------------------------------------------------------- #
def test_coordinate_linear_share_calibration_interpolates_on_fixed_grid(
    temporal_markov_data,
    recommendation_spies,
) -> None:
    model, fit_result = _fit_temporal_markov(temporal_markov_data)
    snapshot = _snapshot(model, fit_result)
    q1, q2 = model.epochs

    approved_shares = {
        epoch: dict(model.epoch_results[epoch]["attribution"])
        for epoch in model.epochs
    }
    totals_by_epoch = {q1: 800.0, q2: 400.0}
    shares_before = deepcopy(approved_shares)
    epoch_windows = _authoritative_windows(model)
    epoch_coordinates = {
        q1: datetime(2025, 3, 3),
        q2: datetime(2025, 4, 2),
    }

    evidence = [
        LiftTestEvidence(
            tested_entity="B",
            observed_lift=110.0,
            matched_mta_estimate=100.0,
            test_id="b-feb-geo",
            coordinate_name="date",
            coordinate_value=datetime(2025, 2, 1),
        ),
        LiftTestEvidence(
            tested_entity="B",
            observed_lift=106.0,
            matched_mta_estimate=100.0,
            test_id="b-may-geo",
            coordinate_name="date",
            coordinate_value=datetime(2025, 5, 2),
        ),
    ]
    result = TemporalAttributionCalibration(evidence).apply_shares(
        approved_shares,
        totals_by_epoch=totals_by_epoch,
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="representative_coordinate",
        residual_policy="allow_total_change",
        epoch_windows=epoch_windows,
        epoch_coordinates=epoch_coordinates,
        tested_entity_epochs={"B": [q1, q2]},
    )

    # Interpolation between the two anchors, evaluated at each declared
    # representative coordinate: 30/90 and 60/90 of the way from 1.10 to
    # 1.06.  The anchors are evidence coordinates, not epoch boundaries.
    expected_q1 = 1.10 + (30.0 / 90.0) * (1.06 - 1.10)
    expected_q2 = 1.10 + (60.0 / 90.0) * (1.06 - 1.10)

    assert result.epochs == tuple(model.epochs)
    assert [
        (row.tested_entity, row.epoch) for row in result.epoch_ratio_schedule
    ] == [("B", q1), ("B", q2)]
    by_epoch = {row.epoch: row for row in result.epoch_ratio_schedule}
    assert by_epoch[q1].effective_ratio == pytest.approx(expected_q1)
    assert by_epoch[q2].effective_ratio == pytest.approx(expected_q2)
    for row in result.epoch_ratio_schedule:
        assert row.ratio_source == "interpolated"
        assert row.left_source_evidence_id == "b-feb-geo"
        assert row.right_source_evidence_id == "b-may-geo"
        assert row.inside_observed_support is True

    # Share input times explicit totals defines the original estimates.
    original_b_q1 = approved_shares[q1]["B"] * totals_by_epoch[q1]
    original_b_q2 = approved_shares[q2]["B"] * totals_by_epoch[q2]
    rows_q1 = {row.entity: row for row in result.rows[q1]}
    rows_q2 = {row.entity: row for row in result.rows[q2]}
    assert rows_q1["B"].calibrated_estimate == pytest.approx(
        expected_q1 * original_b_q1
    )
    assert rows_q2["B"].calibrated_estimate == pytest.approx(
        expected_q2 * original_b_q2
    )
    for channel in ("A", "C"):
        assert rows_q1[channel].calibrated_estimate == pytest.approx(
            approved_shares[q1][channel] * totals_by_epoch[q1]
        )
        assert rows_q1[channel].role == "untouched"
    assert rows_q2["C"].calibrated_estimate == pytest.approx(
        approved_shares[q2]["C"] * totals_by_epoch[q2]
    )
    assert rows_q2["C"].role == "untouched"

    # allow_total_change: totals move exactly by the direct adjustments.
    assert result.original_totals[q1] == pytest.approx(800.0)
    assert result.original_totals[q2] == pytest.approx(400.0)
    assert result.calibrated_totals[q1] == pytest.approx(
        800.0 + (expected_q1 - 1.0) * original_b_q1
    )
    assert result.calibrated_totals[q2] == pytest.approx(
        400.0 + (expected_q2 - 1.0) * original_b_q2
    )

    # Caller inputs, fitted model state, outputs, summaries, and exports
    # are unchanged; recommendation/boundary logic was never invoked.
    assert approved_shares == shares_before
    _assert_unchanged(model, fit_result, snapshot)
    assert recommendation_spies == []
