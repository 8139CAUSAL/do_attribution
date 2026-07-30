"""Temporal Shapley-to-calibration integration tests for CAL-083.

Calibration consumes the model's authoritative epoch grid and caller-approved
per-epoch output as an external post-attribution layer.  These tests exercise
one explicit-ratio (``explicit_ratio`` + ``direct``) and one coordinate-based
(``coordinate_linear`` + ``coordinate_average``) reduction path and prove
that:

- the authoritative epoch keys, ordering, and boundaries are preserved;
- fitted model state, result dictionaries, summaries, and exports are
  unchanged;
- epoch recommendation and boundary detection are never invoked;
- no attribution quantity (share, raw signed Shapley value, or value
  function) is automatically converted into an incremental outcome — every
  calibrated quantity descends only from caller-approved estimates or
  shares-times-explicit-totals and declared evidence arithmetic.
"""

from copy import deepcopy
from datetime import datetime

import pandas as pd
import pytest

import do_attribution
import do_attribution.temporal as temporal_module
import do_attribution.temporal_shapley as temporal_shapley_module
from do_attribution import TemporalShapleyAttribution
from do_attribution.calibration import (
    LiftTestEvidence,
    TemporalAttributionCalibration,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def temporal_shapley_data():
    """The existing two-quarter Temporal Shapley dataset.

    Mirrors ``small_temporal_data`` in ``tests/test_temporal_shapley.py`` so
    the fitted authoritative grid is the familiar ``[2025Q1, 2025Q2]`` epoch
    order with channels ``{A, B}`` in Q1 and ``{B, C}`` in Q2.
    """
    q1_paths = [
        "A > B", "A > B", "A", "B > A", "A > B",
        "A > B", "A", "B > A", "A > B", "A",
    ]
    q1_dates = pd.date_range("2025-01-15", periods=10, freq="D")
    q1_convs = [1, 1, 0, 1, 1, 0, 1, 1, 0, 1]

    q2_paths = [
        "B > C", "C > B", "C", "B > C", "C",
        "B > C", "C > B", "B > C", "C", "C > B",
    ]
    q2_dates = pd.date_range("2025-04-15", periods=10, freq="D")
    q2_convs = [1, 1, 0, 1, 1, 1, 0, 1, 1, 0]

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
        temporal_shapley_module,
        "recommend_epochs",
        _spy("temporal_shapley.recommend_epochs"),
    )
    monkeypatch.setattr(
        temporal_shapley_module,
        "assign_paths_to_epochs",
        _spy("temporal_shapley.assign_paths_to_epochs"),
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
        TemporalShapleyAttribution,
        "fit_recommended",
        _spy("TemporalShapleyAttribution.fit_recommended"),
    )
    monkeypatch.setattr(
        TemporalShapleyAttribution,
        "detect_threshold_crossings",
        _spy("TemporalShapleyAttribution.detect_threshold_crossings"),
    )
    return calls


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _fit_temporal_shapley(temporal_shapley_data):
    paths, dates, convs = temporal_shapley_data
    model = TemporalShapleyAttribution(freq="Q", min_samples_per_epoch=5)
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
        "epoch_shapley_values": {
            epoch: dict(result["shapley_values"])
            for epoch, result in model.epoch_results.items()
        },
        "epoch_value_functions": {
            epoch: dict(result["value_function"])
            for epoch, result in model.epoch_results.items()
        },
        "attribution_matrix": fit_result["attribution_matrix"].copy(deep=True),
        "shapley_values_matrix": model.get_shapley_values_matrix().copy(
            deep=True
        ),
        "all_trends": model.get_all_trends().copy(deep=True),
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
        "global_attribution_cache": dict(model._global_attribution_cache),
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
        epoch: dict(result["shapley_values"])
        for epoch, result in model.epoch_results.items()
    } == snapshot["epoch_shapley_values"]
    assert {
        epoch: dict(result["value_function"])
        for epoch, result in model.epoch_results.items()
    } == snapshot["epoch_value_functions"]
    pd.testing.assert_frame_equal(
        fit_result["attribution_matrix"], snapshot["attribution_matrix"]
    )
    pd.testing.assert_frame_equal(
        model.get_shapley_values_matrix(), snapshot["shapley_values_matrix"]
    )
    pd.testing.assert_frame_equal(model.get_all_trends(), snapshot["all_trends"])
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
    # The lazy global static fit must never run: calibration consumes only
    # caller-approved quantities, so nothing may pull the causal-looking
    # global aggregate into existence as a side effect.
    assert model._global_fit_done is snapshot["global_fit_done"]
    assert model._global_fit_done is False
    assert dict(model._global_attribution_cache) == (
        snapshot["global_attribution_cache"]
    )
    assert dict(model._global_attribution_cache) == {}
    assert model.freq == snapshot["freq"]
    assert model.epoch_assignment == snapshot["epoch_assignment"]


# --------------------------------------------------------------------------- #
# Explicit-ratio reduction path                                                #
# --------------------------------------------------------------------------- #
def test_explicit_ratio_calibration_preserves_authoritative_grid(
    temporal_shapley_data,
    recommendation_spies,
) -> None:
    model, fit_result = _fit_temporal_shapley(temporal_shapley_data)
    snapshot = _snapshot(model, fit_result)
    q1, q2 = model.epochs

    totals = {q1: 600.0, q2: 300.0}
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

    evidence = LiftTestEvidence(
        tested_entity="B",
        observed_lift=90.0,
        matched_mta_estimate=100.0,
        test_id="b-shapley-geo",
    )
    # Declared evidence arithmetic only: 90 / 100.
    assert evidence.calibration_ratio == pytest.approx(0.9)

    ratio_schedule = {
        ("B", q1): 0.9,
        ("B", q2): 1.05,
    }
    result = TemporalAttributionCalibration([evidence]).apply_estimates(
        approved_estimates,
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy="proportional_untested",
        ratio_schedule=ratio_schedule,
        eligible_entities_by_epoch={q1: ["A"], q2: ["C"]},
    )

    # The authoritative epoch keys and ordering pass through unchanged.
    assert result.epochs == tuple(model.epochs)
    assert result.ratio_policy == "explicit_ratio"
    assert result.epoch_ratio_policy == "direct"

    # Exactly one effective ratio per declared tested-entity--epoch pair,
    # taken from the caller's explicit schedule without inference.
    assert [
        (row.tested_entity, row.epoch) for row in result.epoch_ratio_schedule
    ] == [("B", q1), ("B", q2)]
    by_epoch = {row.epoch: row for row in result.epoch_ratio_schedule}
    assert by_epoch[q1].effective_ratio == pytest.approx(0.9)
    assert by_epoch[q2].effective_ratio == pytest.approx(1.05)
    for row in result.epoch_ratio_schedule:
        assert row.ratio_policy == "explicit_ratio"
        assert row.epoch_ratio_policy == "direct"
        assert row.ratio_source == "explicit"
        assert row.source_evidence_ids == ()
        assert row.evidence_conflict_policy is None
    # The Q2 schedule ratio (1.05) deliberately differs from the declared
    # evidence ratio (0.9): the engine applied the caller's explicit
    # schedule, not a quantity inferred from evidence or attribution.
    assert by_epoch[q2].effective_ratio != evidence.calibration_ratio

    # Q1: tested entity fixed at 0.9x, residual goes wholly to A.
    rows_q1 = {row.entity: row for row in result.rows[q1]}
    original_b_q1 = approved_estimates[q1]["B"]
    assert rows_q1["B"].original_estimate == pytest.approx(original_b_q1)
    assert rows_q1["B"].calibrated_estimate == pytest.approx(0.9 * original_b_q1)
    assert rows_q1["B"].direct_test_adjustment == pytest.approx(
        -0.1 * original_b_q1
    )
    assert rows_q1["B"].residual_adjustment == 0.0
    assert rows_q1["B"].role == "tested"
    assert rows_q1["A"].direct_test_adjustment == 0.0
    assert rows_q1["A"].residual_adjustment == pytest.approx(
        0.1 * original_b_q1
    )
    assert rows_q1["A"].calibrated_estimate == pytest.approx(
        approved_estimates[q1]["A"] + 0.1 * original_b_q1
    )
    assert rows_q1["A"].role == "residual_recipient"
    assert result.calibrated_totals[q1] == pytest.approx(600.0)

    # Q2: tested entity fixed at 1.05x, negative residual goes wholly to C.
    rows_q2 = {row.entity: row for row in result.rows[q2]}
    original_b_q2 = approved_estimates[q2]["B"]
    assert rows_q2["B"].calibrated_estimate == pytest.approx(
        1.05 * original_b_q2
    )
    assert rows_q2["B"].direct_test_adjustment == pytest.approx(
        0.05 * original_b_q2
    )
    assert rows_q2["B"].role == "tested"
    assert rows_q2["C"].direct_test_adjustment == 0.0
    assert rows_q2["C"].residual_adjustment == pytest.approx(
        -0.05 * original_b_q2
    )
    assert rows_q2["C"].calibrated_estimate == pytest.approx(
        approved_estimates[q2]["C"] - 0.05 * original_b_q2
    )
    assert rows_q2["C"].role == "residual_recipient"
    assert result.calibrated_totals[q2] == pytest.approx(300.0)

    # No attribution quantity was converted into an incremental outcome:
    # every original estimate is exactly the caller's approved
    # share-times-explicit-total product, and the original totals are the
    # caller's declared totals.
    for epoch in (q1, q2):
        for row in result.rows[epoch]:
            assert row.original_estimate == pytest.approx(
                model.epoch_results[epoch]["attribution"][row.entity]
                * totals[epoch]
            )
        assert result.original_totals[epoch] == pytest.approx(totals[epoch])

    # Caller inputs, fitted model state, outputs, summaries, and exports
    # are unchanged; recommendation/boundary logic was never invoked.
    assert approved_estimates == estimates_before
    _assert_unchanged(model, fit_result, snapshot)
    assert recommendation_spies == []


# --------------------------------------------------------------------------- #
# Coordinate-average reduction path                                            #
# --------------------------------------------------------------------------- #
def test_coordinate_average_share_calibration_on_fixed_grid(
    temporal_shapley_data,
    recommendation_spies,
) -> None:
    model, fit_result = _fit_temporal_shapley(temporal_shapley_data)
    snapshot = _snapshot(model, fit_result)
    q1, q2 = model.epochs

    approved_shares = {
        epoch: dict(model.epoch_results[epoch]["attribution"])
        for epoch in model.epochs
    }
    totals_by_epoch = {q1: 600.0, q2: 300.0}
    shares_before = deepcopy(approved_shares)
    epoch_windows = _authoritative_windows(model)
    windows_before = deepcopy(epoch_windows)

    # Anchors exactly at the grid's outer boundaries so both complete epoch
    # windows lie inside observed evidence support.
    evidence = [
        LiftTestEvidence(
            tested_entity="B",
            observed_lift=110.0,
            matched_mta_estimate=100.0,
            test_id="b-jan-geo",
            coordinate_name="date",
            coordinate_value=datetime(2025, 1, 1),
        ),
        LiftTestEvidence(
            tested_entity="B",
            observed_lift=104.0,
            matched_mta_estimate=100.0,
            test_id="b-jul-geo",
            coordinate_name="date",
            coordinate_value=datetime(2025, 7, 1),
        ),
    ]
    result = TemporalAttributionCalibration(evidence).apply_shares(
        approved_shares,
        totals_by_epoch=totals_by_epoch,
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="coordinate_average",
        residual_policy="allow_total_change",
        epoch_windows=epoch_windows,
        tested_entity_epochs={"B": [q1, q2]},
    )

    # One linear segment from 1.10 (2025-01-01) to 1.04 (2025-07-01) spans
    # 181 days; the uniform average over each epoch interval equals the
    # schedule value at the interval midpoint: 45 days in for Q1
    # ([0, 90]) and 135.5 days in for Q2 ([90, 181]).
    expected_q1 = 1.10 + (45.0 / 181.0) * (1.04 - 1.10)
    expected_q2 = 1.10 + (135.5 / 181.0) * (1.04 - 1.10)

    assert result.epochs == tuple(model.epochs)
    assert [
        (row.tested_entity, row.epoch) for row in result.epoch_ratio_schedule
    ] == [("B", q1), ("B", q2)]
    by_epoch = {row.epoch: row for row in result.epoch_ratio_schedule}
    assert by_epoch[q1].effective_ratio == pytest.approx(expected_q1)
    assert by_epoch[q2].effective_ratio == pytest.approx(expected_q2)
    for row in result.epoch_ratio_schedule:
        assert row.ratio_policy == "coordinate_linear"
        assert row.epoch_ratio_policy == "coordinate_average"
        assert row.ratio_source == "coordinate_average"
        assert row.source_evidence_ids == ("b-jan-geo", "b-jul-geo")
        assert row.left_source_evidence_id == "b-jan-geo"
        assert row.right_source_evidence_id == "b-jul-geo"
        assert row.inside_observed_support is True
    assert by_epoch[q1].epoch_start == epoch_windows[q1][0]
    assert by_epoch[q1].epoch_end == epoch_windows[q1][1]
    assert by_epoch[q2].epoch_start == epoch_windows[q2][0]
    assert by_epoch[q2].epoch_end == epoch_windows[q2][1]

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
    assert rows_q1["A"].calibrated_estimate == pytest.approx(
        approved_shares[q1]["A"] * totals_by_epoch[q1]
    )
    assert rows_q1["A"].role == "untouched"
    assert rows_q2["C"].calibrated_estimate == pytest.approx(
        approved_shares[q2]["C"] * totals_by_epoch[q2]
    )
    assert rows_q2["C"].role == "untouched"

    # allow_total_change: totals move exactly by the direct adjustments.
    assert result.original_totals[q1] == pytest.approx(600.0)
    assert result.original_totals[q2] == pytest.approx(300.0)
    assert result.calibrated_totals[q1] == pytest.approx(
        600.0 + (expected_q1 - 1.0) * original_b_q1
    )
    assert result.calibrated_totals[q2] == pytest.approx(
        300.0 + (expected_q2 - 1.0) * original_b_q2
    )

    # No attribution quantity was converted into an incremental outcome:
    # original estimates equal the caller's shares times the caller's
    # explicit totals, and the observed anchor ratios equal the declared
    # lift-over-matched-estimate arithmetic (1.10 and 1.04), untouched by
    # any Shapley value or conversion probability.
    for epoch in (q1, q2):
        for row in result.rows[epoch]:
            assert row.original_estimate == pytest.approx(
                approved_shares[epoch][row.entity] * totals_by_epoch[epoch]
            )
    assert evidence[0].calibration_ratio == pytest.approx(1.10)
    assert evidence[1].calibration_ratio == pytest.approx(1.04)

    # Caller inputs, fitted model state, outputs, summaries, and exports
    # are unchanged; recommendation/boundary logic was never invoked.
    assert approved_shares == shares_before
    assert epoch_windows == windows_before
    _assert_unchanged(model, fit_result, snapshot)
    assert recommendation_spies == []
