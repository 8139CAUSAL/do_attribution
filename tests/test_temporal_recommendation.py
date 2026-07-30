"""Regression coverage for global-signal temporal epoch recommendation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from do_attribution import (
    DaySummary,
    EpochSufficiencyError,
    PinOverlapError,
    NonCalendarBoundaryError,
    SignalContinuityError,
    SignalShapeError,
    SignalSufficiencyError,
    TemporalMarkovAttribution,
    TemporalShapleyAttribution,
    StreamingEpochRecommender,
    WeightedAffineCost,
    assign_paths_to_epochs,
    recommend_epochs,
)


def _daily_paths(rates, *, n_per_day: int = 80, start: str = "2024-01-01"):
    """Deterministic daily paths whose observed rate equals ``rates``."""
    paths, conversions, anchors = [], [], []
    for day, rate in enumerate(rates):
        date = pd.Timestamp(start) + pd.Timedelta(days=day)
        converted = int(round(rate * n_per_day))
        for sample in range(n_per_day):
            paths.append("A > B > Purchase" if sample % 2 else "B > A > Purchase")
            conversions.append(int(sample < converted))
            anchors.append(date)
    return paths, conversions, anchors


def _recommend(rates, **kwargs):
    paths, conversions, anchors = _daily_paths(rates)
    params = {"min_samples": 160, "min_epoch_width": "10D", "stl_period": None}
    params.update(kwargs)
    return recommend_epochs(paths, conversions, anchors, conversion_labels=["Purchase"], **params)


def _direct_weighted_rss(times, values, weights):
    design = np.column_stack([np.ones(len(times)), times])
    beta, *_ = np.linalg.lstsq(design * np.sqrt(weights)[:, None], values * np.sqrt(weights), rcond=None)
    return float(np.square((values - design @ beta) * np.sqrt(weights)).sum())


def test_weighted_affine_cost_is_zero_for_exact_affine_segment():
    times = np.arange(12, dtype=float)
    values = 1.5 - 0.25 * times
    cost = WeightedAffineCost(times, np.ones_like(times)).fit(values)
    assert cost.error(0, len(times)) == pytest.approx(0.0, abs=1e-12)
    intercept, slope, _, n = cost.segment_parameters(0, len(times))
    assert intercept == pytest.approx(1.5)
    assert slope == pytest.approx(-0.25)
    assert n == len(times)


def test_weighted_affine_cost_matches_direct_weighted_least_squares_prefixes():
    rng = np.random.default_rng(8)
    times = np.arange(40, dtype=float)
    values = 0.5 + 0.03 * times + rng.normal(0, 0.1, len(times))
    weights = rng.integers(1, 8, len(times)).astype(float)
    cost = WeightedAffineCost(times, weights).fit(values)
    for start in range(0, 25, 3):
        for end in range(start + 3, min(40, start + 15), 4):
            assert cost.error(start, end) == pytest.approx(
                _direct_weighted_rss(times[start:end], values[start:end], weights[start:end]), abs=1e-10
            )


def test_weighted_affine_cost_rejects_singular_and_short_segments():
    cost = WeightedAffineCost([1, 1, 1, 1], [1, 1, 1, 1]).fit([1, 2, 3, 4])
    assert np.isinf(cost.error(0, 4))
    short = WeightedAffineCost([0, 1, 2], [1, 1, 0]).fit([0, 1, 2])
    assert np.isinf(short.error(0, 3))


def test_scalar_level_slope_and_flat_regimes_have_expected_knots():
    level = _recommend([0.2] * 40 + [0.7] * 40)
    slope = _recommend([0.1 + 0.005 * day for day in range(40)] + [0.3 - 0.005 * day for day in range(40)])
    flat = _recommend([0.3] * 80)
    expected = pd.Timestamp("2024-02-10")
    assert any(abs(point - expected) <= pd.Timedelta(days=2) for point in level.raw_breakpoints)
    assert any(abs(point - expected) <= pd.Timedelta(days=2) for point in slope.raw_breakpoints)
    assert len(flat.raw_breakpoints) == 0
    assert len(flat.boundaries) == 2


def test_single_linear_trend_and_equal_mean_reversal_do_not_use_mean_screening():
    trend = _recommend([0.1 + 0.005 * day for day in range(80)])
    reversal = _recommend([0.1 + 0.01 * day for day in range(30)] + [0.4 - 0.01 * day for day in range(30)])
    assert len(trend.raw_breakpoints) == 0
    assert any(abs(point - pd.Timestamp("2024-01-31")) <= pd.Timedelta(days=3) for point in reversal.raw_breakpoints)


def test_flattening_and_non_multiple_of_five_knot_are_detected_without_oversegmentation():
    flattening = _recommend([0.1 + 0.008 * day for day in range(37)] + [0.396] * 43)
    level_at_37 = _recommend([0.2] * 37 + [0.7] * 43)
    expected = pd.Timestamp("2024-02-07")
    assert any(abs(point - expected) <= pd.Timedelta(days=2) for point in flattening.raw_breakpoints)
    assert any(abs(point - expected) <= pd.Timedelta(days=2) for point in level_at_37.raw_breakpoints)
    assert len(level_at_37.raw_breakpoints) <= 2


def test_high_volume_days_receive_larger_affine_weight():
    paths, conversions, anchors = _daily_paths([0.2] * 20, n_per_day=20)
    # Add a high-volume day at a different rate; its relative precision must
    # be visible in the detector weights without creating fake path counts.
    date = pd.Timestamp("2024-01-11")
    for sample in range(180):
        paths.append("A > B")
        conversions.append(int(sample < 150))
        anchors.append(date)
    rec = recommend_epochs(paths, conversions, anchors, min_samples=80, min_epoch_width="5D", stl_period=None)
    assert rec.detector_weights.loc[date] > rec.detector_weights.median()
    assert rec.path_counts.loc[date] == 200


def test_missing_days_are_imputed_only_for_detection_and_obey_gap_limit():
    paths, conversions, anchors = _daily_paths([0.2] * 4 + [0.6] * 4)
    # Remove all paths on two interior dates.
    keep = [index for index, anchor in enumerate(anchors) if anchor not in {pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")}]
    rec = recommend_epochs(
        [paths[index] for index in keep], [conversions[index] for index in keep], [anchors[index] for index in keep],
        min_samples=80, min_epoch_width="2D", stl_period=None, max_imputation_gap="2D",
    )
    assert rec.signal.loc[pd.Timestamp("2024-01-03")] != rec.signal.loc[pd.Timestamp("2024-01-03")]
    assert 0.2 < rec.decomposition.prepared.loc[pd.Timestamp("2024-01-03")] < 0.6
    assert rec.path_counts.loc[pd.Timestamp("2024-01-03")] == 0
    assert rec.detector_weights.loc[pd.Timestamp("2024-01-03")] == 0
    with pytest.raises(SignalContinuityError, match="max_imputation_gap"):
        recommend_epochs(
            [paths[index] for index in keep], [conversions[index] for index in keep], [anchors[index] for index in keep],
            min_samples=80, min_epoch_width="2D", stl_period=None, max_imputation_gap="1D",
        )


def test_terminal_pelt_endpoint_is_not_an_internal_boundary_and_semantics_are_exclusive():
    rec = _recommend([0.3] * 20)
    assert list(rec.boundaries) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-21")]
    assert len(rec.raw_breakpoints) == 0
    with pytest.raises(NonCalendarBoundaryError):
        rec.as_freq_mapping()


def test_objective_gain_merges_deterministically_under_sample_floor():
    rec = _recommend([0.2] * 15 + [0.8] * 15 + [0.2] * 15, min_samples=1500, min_epoch_width="5D")
    # Any final merge is logged with the model-consistent gain, not means.
    assert rec.merge_log
    for action in rec.merge_log:
        assert action.reason == "min_samples"
        assert action.competing_scores


def test_anchor_assignment_is_total_disjoint_and_uses_left_closed_intervals():
    boundaries = pd.DatetimeIndex(["2024-01-01", "2024-01-05", "2024-01-10"])
    labels = assign_paths_to_epochs(pd.to_datetime(["2024-01-01", "2024-01-04", "2024-01-05", "2024-01-09"]), boundaries)
    assert labels[0] == labels[1] == pd.Interval(boundaries[0], boundaries[1], closed="left")
    assert labels[2] == labels[3] == pd.Interval(boundaries[1], boundaries[2], closed="left")
    assert len(labels) == 4


def test_fit_recommended_preserves_paths_and_endpoint_label_handling():
    paths, conversions, anchors = _daily_paths([0.2] * 30 + [0.7] * 30)
    model = TemporalShapleyAttribution(min_samples_per_epoch=160, conversion_labels=["Purchase"]).fit_recommended(
        paths, conversions, anchors, recommender_kwargs={"min_epoch_width": "10D", "stl_period": None}
    )
    assert model.epoch_recommendation is not None
    assert sum(len(value) for value in model._epoch_paths.values()) == len(paths)
    assert all("Purchase" not in result["attribution"] for result in model.epoch_results.values())
    with pytest.raises(ValueError, match="transition_drift"):
        TemporalShapleyAttribution().fit_recommended(paths, conversions, anchors, recommender_kwargs={"signal": "transition_drift"})


def test_pins_are_immutable_raw_candidates_are_unchanged_and_overlap_rejected():
    baseline = _recommend([0.2] * 30 + [0.7] * 30)
    interval = pd.Interval(pd.Timestamp("2024-01-20"), pd.Timestamp("2024-02-10"), closed="left")
    pinned = _recommend([0.2] * 30 + [0.7] * 30, pinned={interval: pd.DatetimeIndex(["2024-01-30"])})
    assert pinned.raw_breakpoints.equals(baseline.raw_breakpoints)
    assert {pd.Timestamp("2024-01-20"), pd.Timestamp("2024-01-30"), pd.Timestamp("2024-02-10")} <= set(pinned.boundaries)
    overlap = pd.Interval(pd.Timestamp("2024-02-05"), pd.Timestamp("2024-02-15"), closed="left")
    with pytest.raises(PinOverlapError):
        _recommend([0.2] * 30 + [0.7] * 30, pinned={interval: "D", overlap: "D"})


def test_pinned_thin_epoch_raises_named_sufficiency_error():
    paths, conversions, anchors = _daily_paths([0.3] * 30)
    pin = pd.Interval(pd.Timestamp("2024-01-10"), pd.Timestamp("2024-01-12"), closed="left")
    with pytest.raises(EpochSufficiencyError, match="pinned"):
        recommend_epochs(paths, conversions, anchors, min_samples=500, min_epoch_width="1D", stl_period=None, pinned={pin: pd.DatetimeIndex([])})


def _transition_paths():
    rng = np.random.default_rng(11)
    paths, conversions, anchors = [], [], []
    for week in range(17):
        for sample in range(60):
            paths.append(("A > B" if sample % 2 else "A > C") if week < 8 else ("D > B" if sample % 2 else "D > C"))
            conversions.append(int(rng.random() < 0.5))
            anchors.append(pd.Timestamp("2024-01-01") + pd.Timedelta(days=week * 7 + sample % 7))
    return paths, conversions, anchors


def test_transition_detector_uses_aligned_matrices_and_downstream_order():
    paths, conversions, anchors = _transition_paths()
    rec = recommend_epochs(paths, conversions, anchors, signal="transition_drift", min_samples=100, min_epoch_width="7D", transition_min_samples=50)
    assert isinstance(rec.detector_signal, pd.DataFrame)
    assert rec.signal.name == "adjacent_frobenius_distance"
    assert rec.metadata["detector_input"] == "aligned order-1 transition-matrix vectors"
    assert len(rec.raw_breakpoints) == 1
    model = TemporalMarkovAttribution(order=2, removal_strategy="redirect_to_null", min_samples_per_epoch=10).fit_recommended(
        paths, conversions, anchors, recommender_kwargs={"signal": "transition_drift", "min_epoch_width": "7D", "transition_min_samples": 50}
    )
    assert model.order == 2
    assert model.epoch_recommendation.detector_signal.equals(rec.detector_signal)


def test_transition_sparse_buckets_raise_instead_of_creating_zero_matrices():
    paths, conversions, anchors = _daily_paths([0.4] * 10)
    with pytest.raises(SignalSufficiencyError, match="transition"):
        recommend_epochs(paths, conversions, anchors, signal="transition_drift", min_samples=20, transition_min_samples=1000)


def test_stationary_transition_matrices_do_not_oversegment_and_keep_full_basis():
    paths, conversions, anchors = [], [], []
    for week in range(14):
        for sample in range(50):
            paths.append("A > B" if sample % 2 else "A > C")
            conversions.append(int(sample % 3 == 0))
            anchors.append(pd.Timestamp("2024-01-01") + pd.Timedelta(days=week * 7 + sample % 7))
    rec = recommend_epochs(paths, conversions, anchors, signal="transition_drift", min_samples=100, min_epoch_width="7D", transition_min_samples=40)
    assert len(rec.raw_breakpoints) == 0
    assert {"A", "B", "C", "START", "CONVERSION", "NULL"} <= set(rec.metadata["state_basis"])
    assert rec.signal.iloc[1:].eq(0.0).all()


def test_callback_validation_rejects_multivariate_and_irregular_outputs():
    paths, conversions, anchors = _daily_paths([0.3] * 20)
    with pytest.raises(SignalShapeError):
        recommend_epochs(paths, conversions, anchors, signal=lambda *_: pd.DataFrame({"x": [1, 2]}), min_samples=20)
    with pytest.raises(SignalShapeError, match="regular"):
        recommend_epochs(
            paths, conversions, anchors,
            signal=lambda *_: pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04"])),
            min_samples=20,
        )


def test_boundary_plot_exposes_disconnected_segments_and_counts():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from do_attribution.visualization import plot_epoch_recommendation

    rec = _recommend([0.2] * 30 + [0.7] * 30)
    fig = plot_epoch_recommendation(rec, show_prepared_signal=True)
    try:
        assert len(fig.axes) == 2
        axis = fig.axes[0]
        fitted = [line for line in axis.lines if line.get_label() == "Fitted affine epoch"]
        assert fitted
        # Individual segment artists are the no-bridge guarantee.
        black_segments = [line for line in axis.lines if line.get_color() == "black"]
        assert len(black_segments) == len(rec.segment_fits)
        final_internal = set(rec.boundaries[1:-1])
        verticals = {pd.Timestamp(line.get_xdata()[0]).normalize() for line in axis.lines if len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1]}
        assert final_internal <= verticals
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_boundary_plot_distinguishes_rejected_candidates_from_pins():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from do_attribution.visualization import plot_epoch_recommendation

    interval = pd.Interval(pd.Timestamp("2024-01-20"), pd.Timestamp("2024-02-10"), closed="left")
    rec = _recommend([0.2] * 30 + [0.7] * 30, pinned={interval: pd.DatetimeIndex(["2024-01-30"])})
    fig = plot_epoch_recommendation(rec)
    try:
        labels = fig.axes[0].get_legend_handles_labels()[1]
        assert "Pinned boundary" in labels
        assert "Rejected raw candidate" in labels
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_transition_plot_states_its_detector_input():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from do_attribution.visualization import plot_epoch_recommendation

    paths, conversions, anchors = _transition_paths()
    rec = recommend_epochs(paths, conversions, anchors, signal="transition_drift", min_samples=100, min_epoch_width="7D", transition_min_samples=50)
    fig = plot_epoch_recommendation(rec)
    try:
        assert "aligned order-1 transition-matrix vectors" in fig.axes[0].get_title()
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_streaming_snapshot_restore_is_json_ready_and_deterministic():
    import json

    original = StreamingEpochRecommender(confirmation_window=3, max_run_length=40)
    for day in range(30):
        converted = 2 if day < 20 else 8
        original.update(DaySummary(pd.Timestamp("2024-01-01") + pd.Timedelta(days=day), ["A > B"] * 10, [int(index < converted) for index in range(10)]))
    snapshot = original.snapshot()
    # The persistence contract is JSON primitives only.
    json.dumps(snapshot)
    restored = StreamingEpochRecommender.restore(snapshot)
    next_summary = DaySummary(pd.Timestamp("2024-01-31"), ["A > B"] * 10, [1] * 8 + [0] * 2)
    assert original.update(next_summary) == restored.update(next_summary)
    recommendation = original.get_current_recommendation()
    assert recommendation.streaming is True
    assert recommendation.boundaries[0] == pd.Timestamp("2024-01-01")
    assert recommendation.boundaries[-1] == pd.Timestamp("2024-02-01")
