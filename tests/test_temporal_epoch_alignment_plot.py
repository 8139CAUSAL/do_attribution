"""Authoritative epoch-alignment plotting tests for CAL-073."""

from datetime import date

import pytest

matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")

from matplotlib.collections import PathCollection  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from do_attribution.calibration import (  # noqa: E402
    LiftTestEvidence,
    TemporalAttributionCalibration,
)


def _dated_evidence(entity: str, evidence_id: str, coordinate: date, ratio: float):
    return LiftTestEvidence(
        tested_entity=entity,
        observed_lift=100.0 * ratio,
        matched_mta_estimate=100.0,
        test_id=evidence_id,
        coordinate_name="date",
        coordinate_value=coordinate,
    )


def _alignment_result():
    evidence = [
        _dated_evidence("A", "a-one", date(2024, 1, 2), 0.8),
        _dated_evidence("A", "a-two", date(2024, 1, 8), 1.0),
        _dated_evidence("B", "b-unmatched", date(2024, 2, 1), 1.1),
    ]
    return TemporalAttributionCalibration(evidence).apply_estimates(
        {
            "E1": {"A": 100.0, "B": 100.0},
            "E2": {"A": 100.0, "B": 100.0},
        },
        ratio_policy="observed_only",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        epoch_windows={
            "E1": (date(2024, 1, 1), date(2024, 1, 10)),
            "E2": (date(2024, 1, 10), date(2024, 1, 20)),
        },
        epoch_support={"E1": 125.0, "E2": 80.0},
        evidence_conflict_policy="select_evidence",
        selected_evidence={("A", "E1"): "a-one"},
    )


def test_plot_shows_boundaries_epoch_labels_all_evidence_and_support() -> None:
    result = _alignment_result()

    figure = result.plot_epoch_alignment()
    try:
        assert isinstance(figure, Figure)
        assert len(figure.axes) == 2
        evidence_ax, support_ax = figure.axes
        assert evidence_ax.get_title() == "Authoritative Attribution Epoch Alignment"
        assert evidence_ax.get_ylabel() == "Tested entity evidence"
        assert support_ax.get_ylabel() == "Epoch support"
        assert [tick.get_text() for tick in evidence_ax.get_yticklabels()] == [
            "A",
            "B",
        ]

        boundary_lines = [
            line
            for line in evidence_ax.lines
            if line.get_linestyle() == "--"
        ]
        assert len(boundary_lines) == 3
        text = {artist.get_text() for artist in evidence_ax.texts}
        assert {"Epoch E1", "Epoch E2", "a-one", "a-two", "b-unmatched"} <= text

        point_count = sum(
            len(collection.get_offsets())
            for collection in evidence_ax.collections
            if isinstance(collection, PathCollection)
        )
        assert point_count == 3
        evidence_legend = evidence_ax.get_legend_handles_labels()[1]
        assert "Authoritative epoch boundary" in evidence_legend
        assert "Evidence: A (matched)" in evidence_legend
        assert "Evidence: B (unmatched / unpositioned)" in evidence_legend
        assert support_ax.get_legend_handles_labels()[1] == [
            "Stored epoch support (diagnostic)"
        ]
        assert len(support_ax.patches) == 2
    finally:
        plt.close(figure)


def test_changing_evidence_coordinates_does_not_change_boundary_artists() -> None:
    def result_at(coordinate: float):
        evidence = LiftTestEvidence(
            tested_entity="A",
            observed_lift=90.0,
            matched_mta_estimate=100.0,
            test_id=f"a-{coordinate}",
            coordinate_name="time",
            coordinate_value=coordinate,
        )
        return TemporalAttributionCalibration([evidence]).apply_estimates(
            {"E1": {"A": 100.0}},
            ratio_policy="observed_only",
            epoch_ratio_policy="direct",
            residual_policy="allow_total_change",
            epoch_windows={"E1": (0.0, 10.0)},
        )

    figures = [
        result_at(2.0).plot_epoch_alignment(),
        result_at(7.0).plot_epoch_alignment(),
    ]
    try:
        boundaries = []
        for figure in figures:
            boundaries.append(
                [
                    tuple(line.get_xdata())
                    for line in figure.axes[0].lines
                    if line.get_linestyle() == "--"
                ]
            )
        assert boundaries[0] == boundaries[1] == [(0.0, 0.0), (10.0, 10.0)]
    finally:
        for figure in figures:
            plt.close(figure)


def test_plot_without_windows_uses_authoritative_epoch_categories() -> None:
    result = TemporalAttributionCalibration(
        [
            LiftTestEvidence(
                tested_entity="A",
                observed_lift=90.0,
                matched_mta_estimate=100.0,
                test_id="a-no-coordinate",
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
        result.epochs,
        result.epoch_windows,
        result.evidence_points,
        result.rows,
    )

    figure = result.plot_epoch_alignment()
    try:
        ax = figure.axes[0]
        figure.canvas.draw()
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["E1", "E2"]
        assert "Evidence: A (unmatched / unpositioned)" in ax.get_legend_handles_labels()[1]
        assert len(ax.collections[0].get_offsets()) == 1
        assert (
            result.epochs,
            result.epoch_windows,
            result.evidence_points,
            result.rows,
        ) == snapshot
    finally:
        plt.close(figure)
