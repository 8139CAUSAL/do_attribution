"""Original-versus-calibrated entity-mix plotting tests for CAL-077."""

import pytest

matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")

from matplotlib.figure import Figure  # noqa: E402

from do_attribution.calibration import (  # noqa: E402
    LiftTestEvidence,
    TemporalAttributionCalibration,
)


def _evidence(entity: str, evidence_id: str):
    return LiftTestEvidence(
        tested_entity=entity,
        observed_lift=80.0,
        matched_mta_estimate=100.0,
        test_id=evidence_id,
    )


def _container_values(ax):
    return {
        container.get_label(): list(container.datavalues)
        for container in ax.containers
    }


def _changing_mix_result():
    return TemporalAttributionCalibration(
        [_evidence("A", "a-test"), _evidence("D", "d-test")]
    ).apply_estimates(
        {
            "E1": {"A": 100.0, "D": 50.0, "B": 50.0},
            "E2": {"A": 100.0, "D": 50.0, "C": 150.0},
            "E3": {"A": 100.0, "D": 50.0, "B": 50.0, "C": 100.0},
        },
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        ratio_schedule={
            (entity, epoch): ratio
            for epoch in ("E1", "E2", "E3")
            for entity, ratio in (("A", 0.8), ("D", 1.2))
        },
    )


def test_original_and_calibrated_shares_show_changing_multi_tested_mix() -> None:
    result = _changing_mix_result()

    figure = result.plot_entity_mix()
    try:
        assert isinstance(figure, Figure)
        assert len(figure.axes) == 2
        original_ax, calibrated_ax = figure.axes
        assert original_ax.get_title() == (
            "Original Entity Mix by Authoritative Epoch"
        )
        assert calibrated_ax.get_title() == (
            "Calibrated Entity Mix by Authoritative Epoch"
        )
        original = _container_values(original_ax)
        assert original["Tested entity — A"] == pytest.approx(
            [0.5, 1.0 / 3.0, 1.0 / 3.0]
        )
        assert original["Tested entity — D"] == pytest.approx(
            [0.25, 1.0 / 6.0, 1.0 / 6.0]
        )
        assert original["Entity — B"] == pytest.approx([0.25, 0.0, 1.0 / 6.0])
        assert original["Entity — C"] == pytest.approx([0.0, 0.5, 1.0 / 3.0])

        calibrated = _container_values(calibrated_ax)
        assert calibrated["Tested entity — A"] == pytest.approx(
            [80.0 / 190.0, 80.0 / 290.0, 80.0 / 290.0]
        )
        assert calibrated["Tested entity — D"] == pytest.approx(
            [60.0 / 190.0, 60.0 / 290.0, 60.0 / 290.0]
        )
        assert calibrated["Entity — B"] == pytest.approx(
            [50.0 / 190.0, 0.0, 50.0 / 290.0]
        )
        assert calibrated["Entity — C"] == pytest.approx(
            [0.0, 150.0 / 290.0, 100.0 / 290.0]
        )
        assert [tick.get_text() for tick in calibrated_ax.get_xticklabels()] == [
            "E1",
            "E2",
            "E3",
        ]
    finally:
        plt.close(figure)


def test_one_legend_sits_outside_both_plot_areas() -> None:
    """The stacked bars fill each panel, so a legend inside an axes hides shares."""
    result = _changing_mix_result()

    figure = result.plot_entity_mix()
    try:
        original_ax, calibrated_ax = figure.axes
        legend = original_ax.get_legend()
        assert legend is not None
        assert calibrated_ax.get_legend() is None
        assert [text.get_text() for text in legend.get_texts()] == [
            "Tested entity — A",
            "Tested entity — D",
            "Entity — B",
            "Entity — C",
        ]

        figure.canvas.draw()
        legend_box = legend.get_window_extent()
        for ax in (original_ax, calibrated_ax):
            assert not legend_box.overlaps(ax.get_window_extent())
        # The layout makes room for the key instead of pushing it off-canvas.
        assert figure.bbox.x0 <= legend_box.x0 < legend_box.x1 <= figure.bbox.x1
        assert figure.bbox.y0 <= legend_box.y0 < legend_box.y1 <= figure.bbox.y1
    finally:
        plt.close(figure)


def test_residual_bucket_is_preserved_as_a_distinct_mix_entity() -> None:
    result = TemporalAttributionCalibration([_evidence("A", "a-test")]).apply_estimates(
        {"E1": {"A": 100.0, "B": 100.0}},
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy="residual_bucket",
        ratio_schedule={("A", "E1"): 0.8},
        residual_entity="baseline",
    )
    snapshot = (result.rows, result.original_totals, result.calibrated_totals)

    figure = result.plot_entity_mix()
    try:
        original_ax, calibrated_ax = figure.axes
        assert _container_values(original_ax)["Residual bucket — baseline"] == [0.0]
        assert _container_values(calibrated_ax)[
            "Residual bucket — baseline"
        ] == pytest.approx([0.1])
        assert _container_values(calibrated_ax)[
            "Tested entity — A"
        ] == pytest.approx([0.4])
        assert _container_values(calibrated_ax)["Entity — B"] == pytest.approx(
            [0.5]
        )
        assert (result.rows, result.original_totals, result.calibrated_totals) == snapshot
    finally:
        plt.close(figure)


def test_zero_total_epochs_mark_undefined_shares_instead_of_fabricating_mix() -> None:
    result = TemporalAttributionCalibration([_evidence("A", "a-test")]).apply_estimates(
        {"E0": {"A": 0.0, "B": 0.0}},
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        ratio_schedule={("A", "E0"): 0.8},
    )

    figure = result.plot_entity_mix()
    try:
        for ax in figure.axes:
            values = _container_values(ax)
            assert values["Tested entity — A"] == [0.0]
            assert values["Entity — B"] == [0.0]
            collections = {
                collection.get_label(): collection for collection in ax.collections
            }
            marker = collections["Share undefined (epoch total is zero)"]
            assert marker.get_offsets().tolist() == [[0.0, 0.0]]
    finally:
        plt.close(figure)
