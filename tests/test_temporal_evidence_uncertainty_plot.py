"""Evidence-uncertainty plotting tests.

``plot_evidence_uncertainty`` shows each stored evidence record's observed
lift on a lift axis, with the experiment's reported standard error as a
horizontal bar.  The bars describe the experiment's own estimate; they are
never an interval on calibrated output.
"""

from dataclasses import replace
from datetime import date

import pytest

matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")

from matplotlib.collections import PathCollection  # noqa: E402
from matplotlib.container import ErrorbarContainer  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from do_attribution.calibration import (  # noqa: E402
    LiftTestEvidence,
    TemporalAttributionCalibration,
)


# (tested entity, evidence id, observed lift, matched estimate, coordinate).
# b-left is a negative lift on a negative matched estimate.
RECORDS = (
    ("A", "a-left", 80.0, 100.0, 0.0),
    ("A", "a-right", 120.0, 100.0, 10.0),
    ("B", "b-left", -55.0, -50.0, 0.0),
    ("B", "b-right", 90.0, 100.0, 10.0),
)


def _result(standard_errors):
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
            RECORDS, standard_errors
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


def _points(axis):
    """Return {label: [(lift, row), ...]} for the observed-lift markers."""
    return {
        collection.get_label(): [
            (float(x), float(y)) for x, y in collection.get_offsets()
        ]
        for collection in axis.collections
        if isinstance(collection, PathCollection)
    }


def _error_bars(axis):
    """Return {label: [(low, high, row), ...]} for every error-bar container."""
    bars = {}
    for container in axis.containers:
        if not isinstance(container, ErrorbarContainer):
            continue
        assert container.has_xerr and not container.has_yerr
        segments = [
            segment
            for collection in container.lines[2]
            for segment in collection.get_segments()
        ]
        bars[container.get_label()] = [
            (float(segment[0][0]), float(segment[1][0]), float(segment[0][1]))
            for segment in segments
        ]
    return bars


def _annotations(axis):
    return [text.get_text() for text in axis.texts]


def test_plot_shows_lift_points_with_reported_standard_error_bars() -> None:
    result = _result((5.0, None, 4.0, 0.0))

    figure = result.plot_evidence_uncertainty()
    try:
        assert isinstance(figure, Figure)
        assert len(figure.axes) == 1
        axis = figure.axes[0]
        assert axis.get_title() == (
            "Experimental Evidence: Observed Lift ± Experiment-Reported "
            "Standard Error"
        )
        assert axis.get_ylabel() == "Evidence record"

        # One row per stored evidence record, in evidence order.
        assert [tick.get_text() for tick in axis.get_yticklabels()] == [
            "a-left (A)",
            "a-right (A)",
            "b-left (B)",
            "b-right (B)",
        ]
        assert _points(axis) == {
            "Observed experimental lift — A": [(80.0, 0.0), (120.0, 1.0)],
            "Observed experimental lift — B": [(-55.0, 2.0), (90.0, 3.0)],
        }

        # Horizontal bars of exactly +/- one reported standard error, in the
        # lift units of the axis, only where an error was reported.
        bars = _error_bars(axis)
        assert set(bars) == {
            "Experiment-reported ±1 standard error — A",
            "Experiment-reported ±1 standard error — B",
        }
        assert bars["Experiment-reported ±1 standard error — A"] == [
            pytest.approx((75.0, 85.0, 0.0))
        ]
        assert bars["Experiment-reported ±1 standard error — B"] == [
            pytest.approx((-59.0, -51.0, 2.0)),
            pytest.approx((90.0, 90.0, 3.0)),
        ]

        assert _annotations(axis) == [
            "80.00 ± 5.00",
            "120.00 (no uncertainty supplied)",
            "-55.00 ± 4.00",
            "90.00 ± 0.00",
        ]
    finally:
        plt.close(figure)


def test_bars_are_labelled_as_the_experiments_reported_uncertainty() -> None:
    figure = _result((5.0, None, 4.0, 0.0)).plot_evidence_uncertainty()
    try:
        axis = figure.axes[0]
        legend = axis.get_legend_handles_labels()[1]
        assert set(legend) == {
            "Observed experimental lift — A",
            "Observed experimental lift — B",
            "Experiment-reported ±1 standard error — A",
            "Experiment-reported ±1 standard error — B",
        }
        label = " ".join(axis.get_xlabel().split())
        assert label.startswith("Observed experimental lift")
        assert (
            "±1 standard error reported by the experiment for its observed lift"
            in label
        )
        assert (
            "not a confidence interval on calibrated output or attribution "
            "estimates"
        ) in label
        for text in [axis.get_title(), *legend, *_annotations(axis)]:
            lowered = text.lower()
            assert "confidence" not in lowered
            assert "calibrated" not in lowered
    finally:
        plt.close(figure)


def test_records_without_a_standard_error_are_points_marked_no_uncertainty_supplied() -> None:
    figure = _result((None, None, None, None)).plot_evidence_uncertainty()
    try:
        axis = figure.axes[0]
        assert _error_bars(axis) == {}
        assert sum(len(points) for points in _points(axis).values()) == 4
        assert _annotations(axis) == [
            "80.00 (no uncertainty supplied)",
            "120.00 (no uncertainty supplied)",
            "-55.00 (no uncertainty supplied)",
            "90.00 (no uncertainty supplied)",
        ]
        assert axis.get_legend_handles_labels()[1] == [
            "Observed experimental lift — A",
            "Observed experimental lift — B",
        ]
    finally:
        plt.close(figure)


def test_changing_the_standard_error_moves_only_the_bars_and_their_annotations() -> None:
    low = _result((1.0, 1.0, 1.0, 1.0))
    high = _result((30.0, 30.0, 30.0, 30.0))
    figures = [low.plot_evidence_uncertainty(), high.plot_evidence_uncertainty()]
    try:
        low_axis, high_axis = (figure.axes[0] for figure in figures)
        assert _points(low_axis) == _points(high_axis)
        assert [tick.get_text() for tick in low_axis.get_yticklabels()] == [
            tick.get_text() for tick in high_axis.get_yticklabels()
        ]
        assert low_axis.get_legend_handles_labels()[1] == (
            high_axis.get_legend_handles_labels()[1]
        )
        assert _error_bars(low_axis)["Experiment-reported ±1 standard error — A"] == [
            pytest.approx((79.0, 81.0, 0.0)),
            pytest.approx((119.0, 121.0, 1.0)),
        ]
        assert _error_bars(high_axis)["Experiment-reported ±1 standard error — A"] == [
            pytest.approx((50.0, 110.0, 0.0)),
            pytest.approx((90.0, 150.0, 1.0)),
        ]
        assert _annotations(low_axis)[0] == "80.00 ± 1.00"
        assert _annotations(high_axis)[0] == "80.00 ± 30.00"
    finally:
        for figure in figures:
            plt.close(figure)

    # The calibration itself is identical; only the evidence metadata differs.
    assert low.rows == high.rows
    assert low.epoch_ratio_schedule == high.epoch_ratio_schedule
    assert low.coordinate_ratio_schedule == high.coordinate_ratio_schedule
    assert low.calibrated_totals == high.calibrated_totals
    assert [replace(record, standard_error=None) for record in low.evidence] == [
        replace(record, standard_error=None) for record in high.evidence
    ]


def test_plot_reads_stored_evidence_without_mutating_the_result() -> None:
    result = _result((5.0, None, 4.0, 0.0))
    pristine = _result((5.0, None, 4.0, 0.0))

    figure = result.plot_evidence_uncertainty()
    plt.close(figure)

    assert result == pristine


def test_plot_does_not_need_coordinates_or_epoch_windows() -> None:
    result = TemporalAttributionCalibration(
        [
            LiftTestEvidence(
                tested_entity="A",
                observed_lift=90.0,
                matched_mta_estimate=100.0,
                test_id="a-no-coordinate",
                standard_error=3.0,
            ),
            LiftTestEvidence(
                tested_entity="A",
                observed_lift=95.0,
                matched_mta_estimate=100.0,
                test_id="a-dated",
                coordinate_name="date",
                coordinate_value=date(2024, 1, 2),
            ),
        ]
    ).apply_estimates(
        {"E1": {"A": 100.0}, "E2": {"A": 100.0}},
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        ratio_schedule={("A", "E1"): 0.9},
    )

    figure = result.plot_evidence_uncertainty()
    try:
        axis = figure.axes[0]
        assert [tick.get_text() for tick in axis.get_yticklabels()] == [
            "a-no-coordinate (A)",
            "a-dated (A)",
        ]
        assert _error_bars(axis) == {
            "Experiment-reported ±1 standard error — A": [
                pytest.approx((87.0, 93.0, 0.0))
            ]
        }
        assert _annotations(axis) == [
            "90.00 ± 3.00",
            "95.00 (no uncertainty supplied)",
        ]
    finally:
        plt.close(figure)
