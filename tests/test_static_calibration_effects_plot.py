"""Static calibration-effects plotting contract."""

from __future__ import annotations

import pytest

matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")

from matplotlib.figure import Figure  # noqa: E402

from do_attribution.calibration import (  # noqa: E402
    AttributionCalibration,
    LiftTestEvidence,
)


def _result():
    return AttributionCalibration(
        evidence=[
            LiftTestEvidence(
                tested_entity="A",
                observed_lift=360.0,
                matched_mta_estimate=400.0,
                test_id="a-test",
            )
        ]
    ).apply_estimates(
        {"A": 400.0, "B": 350.0, "C": 250.0},
        residual_policy="proportional_untested",
        eligible_entities=["B", "C"],
    )


def _artist_values(figure: Figure):
    axis = figure.axes[0]
    containers = {container.get_label(): container for container in axis.containers}
    calibrated = next(
        collection
        for collection in axis.collections
        if collection.get_label() == "Calibrated estimate"
    )
    return (
        {
            label: [patch.get_width() for patch in container.patches]
            for label, container in containers.items()
        },
        calibrated.get_offsets()[:, 0].tolist(),
    )


def test_static_plot_exposes_the_complete_additive_identity_by_entity() -> None:
    result = _result()
    figure = result.plot_calibration_effects()
    try:
        assert isinstance(figure, Figure)
        assert len(figure.axes) == 1
        axis = figure.axes[0]
        widths, calibrated = _artist_values(figure)
        assert [label.get_text() for label in axis.get_yticklabels()] == [
            "A",
            "B",
            "C",
        ]
        assert widths["Original estimate"] == pytest.approx(
            [400.0, 350.0, 250.0]
        )
        assert widths["Direct test adjustment"] == pytest.approx(
            [-40.0, 0.0, 0.0]
        )
        assert widths["Residual adjustment"] == pytest.approx(
            [0.0, 70.0 / 3.0, 50.0 / 3.0]
        )
        assert calibrated == pytest.approx(
            [360.0, 1120.0 / 3.0, 800.0 / 3.0]
        )
        for index, row in enumerate(result.rows):
            assert (
                widths["Original estimate"][index]
                + widths["Direct test adjustment"][index]
                + widths["Residual adjustment"][index]
                == pytest.approx(calibrated[index])
                == pytest.approx(row.calibrated_estimate)
            )
        assert "Original + Direct + Residual = Calibrated" in axis.get_title()
    finally:
        plt.close(figure)


def test_direct_and_residual_effects_have_distinct_signed_artists() -> None:
    figure = _result().plot_calibration_effects()
    try:
        axis = figure.axes[0]
        containers = {container.get_label(): container for container in axis.containers}
        direct = containers["Direct test adjustment"]
        residual = containers["Residual adjustment"]
        assert direct.patches[0].get_width() == pytest.approx(-40.0)
        assert residual.patches[1].get_width() > 0.0
        assert direct.patches[0].get_facecolor() != residual.patches[0].get_facecolor()
        labels = axis.get_legend_handles_labels()[1]
        assert set(labels) == {
            "Original estimate",
            "Direct test adjustment",
            "Residual adjustment",
            "Calibrated estimate",
        }
    finally:
        plt.close(figure)


def test_static_plot_is_deterministic_and_does_not_mutate_the_result() -> None:
    result = _result()
    snapshot = (
        result.rows,
        result.evidence,
        result.original_total,
        result.raw_post_test_total,
        result.required_residual,
        result.allocated_residual,
        result.calibrated_total,
    )
    first = result.plot_calibration_effects()
    second = result.plot_calibration_effects()
    try:
        assert _artist_values(first) == _artist_values(second)
        assert (
            result.rows,
            result.evidence,
            result.original_total,
            result.raw_post_test_total,
            result.required_residual,
            result.allocated_residual,
            result.calibrated_total,
        ) == snapshot
    finally:
        plt.close(first)
        plt.close(second)
