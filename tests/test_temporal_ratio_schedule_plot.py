"""Coordinate schedule and final-ratio plotting tests for CAL-074."""

import pytest

matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")

from matplotlib.figure import Figure  # noqa: E402

from do_attribution.calibration import (  # noqa: E402
    LiftTestEvidence,
    TemporalAttributionCalibration,
)


def _evidence(entity: str, evidence_id: str, ratio: float, coordinate: float):
    return LiftTestEvidence(
        tested_entity=entity,
        observed_lift=100.0 * ratio,
        matched_mta_estimate=100.0,
        test_id=evidence_id,
        coordinate_name="time",
        coordinate_value=coordinate,
    )


def _multi_entity_result():
    evidence = [
        _evidence("A", "a-left", 0.8, 0.0),
        _evidence("A", "a-right", 1.2, 10.0),
        _evidence("B", "b-left", 1.1, 0.0),
        _evidence("B", "b-right", 0.9, 10.0),
    ]
    estimates = {
        epoch: {"A": 100.0, "B": 100.0, "C": 100.0}
        for epoch in ("E1", "E2", "E3")
    }
    return TemporalAttributionCalibration(evidence).apply_estimates(
        estimates,
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="representative_coordinate",
        residual_policy="allow_total_change",
        epoch_windows={"E1": (0.0, 5.0), "E2": (5.0, 10.0), "E3": (10.0, 15.0)},
        epoch_coordinates={"E1": 2.5, "E2": 7.5, "E3": 12.0},
        tested_entity_epochs={
            "A": ["E1", "E2", "E3"],
            "B": ["E1", "E2", "E3"],
        },
        ratio_bounds=(0.5, 1.5),
    )


def test_plot_separates_schedule_layers_final_ratios_coverage_and_bounds() -> None:
    result = _multi_entity_result()

    figure = result.plot_ratio_schedule()
    try:
        assert isinstance(figure, Figure)
        assert len(figure.axes) == 2
        schedule_ax, epoch_ax = figure.axes
        assert schedule_ax.get_title() == (
            "Coordinate Schedule: Observed vs Constructed Values"
        )
        assert epoch_ax.get_title() == "Final Epoch-Level Ratios and Coverage"
        assert schedule_ax.get_ylabel() == "Schedule ratio"
        assert epoch_ax.get_ylabel() == "Effective ratio"
        assert epoch_ax.get_xlabel() == "Authoritative attribution epoch"

        schedule_labels = schedule_ax.get_legend_handles_labels()[1]
        for entity in ("A", "B"):
            assert f"Schedule path — {entity}" in schedule_labels
            assert f"Observed anchor — {entity}" in schedule_labels
            assert f"Constructed schedule value — {entity}" in schedule_labels
            assert f"Outside observed support — {entity}" in schedule_labels
        schedule_ticks = [tick.get_text() for tick in schedule_ax.get_xticklabels()]
        assert schedule_ticks == [
            "time=0.0",
            "time=2.5",
            "time=7.5",
            "time=10.0",
            "time=12.0",
        ]
        assert {
            text.get_text() for text in schedule_ax.texts
        } == {"not calibrated / no extrapolation"}

        epoch_labels = epoch_ax.get_legend_handles_labels()[1]
        for entity in ("A", "B"):
            assert f"Final epoch ratio — {entity}" in epoch_labels
            assert f"Not calibrated — {entity}" in epoch_labels
        assert "Lower ratio bound (0.5000)" in epoch_labels
        assert "Upper ratio bound (1.5000)" in epoch_labels
        assert [tick.get_text() for tick in epoch_ax.get_xticklabels()] == [
            "E1",
            "E2",
            "E3",
        ]
    finally:
        plt.close(figure)


def test_final_ratio_artists_contain_only_calibrated_epoch_values() -> None:
    result = _multi_entity_result()
    expected = {
        entity: [
            row.effective_ratio
            for row in result.epoch_ratio_schedule
            if row.tested_entity == entity
        ]
        for entity in ("A", "B")
    }

    figure = result.plot_ratio_schedule()
    try:
        epoch_ax = figure.axes[1]
        collections = {
            collection.get_label(): collection
            for collection in epoch_ax.collections
        }
        for entity in ("A", "B"):
            offsets = collections[f"Final epoch ratio — {entity}"].get_offsets()
            assert offsets[:, 0].tolist() == [0.0, 1.0]
            assert offsets[:, 1].tolist() == pytest.approx(expected[entity])
            uncovered = collections[f"Not calibrated — {entity}"].get_offsets()
            assert uncovered[:, 0].tolist() == [2.0]
    finally:
        plt.close(figure)


def test_direct_explicit_policy_reports_no_coordinate_schedule_without_fabrication() -> None:
    result = TemporalAttributionCalibration(
        [
            LiftTestEvidence(
                tested_entity="A",
                observed_lift=90.0,
                matched_mta_estimate=100.0,
                test_id="a-source",
            )
        ]
    ).apply_estimates(
        {"E1": {"A": 100.0}, "E2": {"A": 100.0}},
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        ratio_schedule={("A", "E1"): 0.9},
    )
    snapshot = (
        result.coordinate_ratio_schedule,
        result.epoch_ratio_schedule,
        result.epoch_reduction_provenance,
    )

    figure = result.plot_ratio_schedule()
    try:
        schedule_ax, epoch_ax = figure.axes
        assert schedule_ax.get_legend() is None
        assert [text.get_text() for text in schedule_ax.texts] == [
            "No coordinate schedule: explicit_ratio supplies epoch-level values directly."
        ]
        labels = epoch_ax.get_legend_handles_labels()[1]
        assert "Final epoch ratio — A" in labels
        assert "Not calibrated — A" in labels
        assert (
            result.coordinate_ratio_schedule,
            result.epoch_ratio_schedule,
            result.epoch_reduction_provenance,
        ) == snapshot
    finally:
        plt.close(figure)


def _reported_uncertainty_result(standard_errors):
    """Two entities; B's first record has a negative matched estimate."""
    records = (
        ("A", "a-left", 80.0, 100.0, 0.0),
        ("A", "a-right", 120.0, 100.0, 10.0),
        ("B", "b-left", -55.0, -50.0, 0.0),
        ("B", "b-right", 90.0, 100.0, 10.0),
    )
    evidence = [
        LiftTestEvidence(
            tested_entity=entity,
            observed_lift=lift,
            matched_mta_estimate=matched,
            test_id=evidence_id,
            coordinate_name="time",
            coordinate_value=coordinate,
            standard_error=standard_error,
        )
        for (entity, evidence_id, lift, matched, coordinate), standard_error in zip(
            records, standard_errors
        )
    ]
    estimates = {
        epoch: {"A": 100.0, "B": 100.0, "C": 100.0}
        for epoch in ("E1", "E2", "E3")
    }
    return TemporalAttributionCalibration(evidence).apply_estimates(
        estimates,
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="representative_coordinate",
        residual_policy="allow_total_change",
        epoch_windows={"E1": (0.0, 5.0), "E2": (5.0, 10.0), "E3": (10.0, 15.0)},
        epoch_coordinates={"E1": 2.5, "E2": 7.5, "E3": 12.0},
        tested_entity_epochs={
            "A": ["E1", "E2", "E3"],
            "B": ["E1", "E2", "E3"],
        },
    )


def _error_bars(axis):
    """Return {label: [(x, low, high), ...]} for every error-bar container."""
    from matplotlib.container import ErrorbarContainer

    bars = {}
    for container in axis.containers:
        if not isinstance(container, ErrorbarContainer):
            continue
        segments = [
            segment
            for collection in container.lines[2]
            for segment in collection.get_segments()
        ]
        bars[container.get_label()] = [
            (float(segment[0][0]), float(segment[0][1]), float(segment[1][1]))
            for segment in segments
        ]
    return bars


def test_observed_anchors_carry_the_experiment_reported_standard_error() -> None:
    result = _reported_uncertainty_result((5.0, None, 4.0, 0.0))

    figure = result.plot_ratio_schedule()
    try:
        schedule_ax, epoch_ax = figure.axes
        bars = _error_bars(schedule_ax)
        label_a = "Experiment-reported ±1 SE of observed lift (ratio scale) — A"
        label_b = "Experiment-reported ±1 SE of observed lift (ratio scale) — B"
        assert set(bars) == {label_a, label_b}

        # A: only a-left reported an error; 5 lift units on a matched
        # estimate of 100 is 0.05 on the ratio axis, around the 0.8 anchor.
        assert len(bars[label_a]) == 1
        x, low, high = bars[label_a][0]
        assert x == 0.0
        assert (low, high) == pytest.approx((0.75, 0.85))

        # B: the absolute matched estimate scales the error (4 / 50), and a
        # reported error of zero is still a reported error.
        assert len(bars[label_b]) == 2
        (x_left, low_left, high_left), (x_right, low_right, high_right) = bars[label_b]
        assert (low_left, high_left) == pytest.approx((1.02, 1.18))
        assert (low_right, high_right) == pytest.approx((0.9, 0.9))
        assert x_left == 0.0 and x_right > x_left

        legend = schedule_ax.get_legend_handles_labels()[1]
        assert label_a in legend and label_b in legend
        assert "Observed anchor — A" in legend
        # Constructed values and final epoch ratios never carry the bars.
        assert _error_bars(epoch_ax) == {}
        for text in legend + epoch_ax.get_legend_handles_labels()[1]:
            assert "confidence interval" not in text.lower()
    finally:
        plt.close(figure)


def test_no_error_bars_are_drawn_when_no_uncertainty_was_supplied() -> None:
    result = _reported_uncertainty_result((None, None, None, None))

    figure = result.plot_ratio_schedule()
    try:
        schedule_ax, epoch_ax = figure.axes
        assert _error_bars(schedule_ax) == {}
        assert _error_bars(epoch_ax) == {}
        assert not any(
            "SE" in label for label in schedule_ax.get_legend_handles_labels()[1]
        )
    finally:
        plt.close(figure)


def test_standard_error_changes_only_the_error_bars_of_the_schedule_plot() -> None:
    def artists(figure):
        return [
            (
                [collection.get_label() for collection in axis.collections
                 if not collection.get_label().startswith("_")],
                [
                    [tuple(point) for point in collection.get_offsets()]
                    for collection in axis.collections
                    if not collection.get_label().startswith("_")
                ],
                [
                    (line.get_label(), [tuple(point) for point in line.get_xydata()])
                    for line in axis.get_lines()
                    if not line.get_label().startswith("_")
                ],
                [text.get_text() for text in axis.texts],
                [tick.get_text() for tick in axis.get_xticklabels()],
                axis.get_title(),
            )
            for axis in figure.axes
        ]

    without = _reported_uncertainty_result((None, None, None, None))
    low = _reported_uncertainty_result((1.0, 1.0, 1.0, 1.0))
    high = _reported_uncertainty_result((30.0, 30.0, 30.0, 30.0))
    figures = [
        result.plot_ratio_schedule() for result in (without, low, high)
    ]
    try:
        reference = artists(figures[0])
        assert artists(figures[1]) == reference
        assert artists(figures[2]) == reference
        low_bars = _error_bars(figures[1].axes[0])
        high_bars = _error_bars(figures[2].axes[0])
        assert set(low_bars) == set(high_bars) and low_bars != high_bars
    finally:
        for figure in figures:
            plt.close(figure)
