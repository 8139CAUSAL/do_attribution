"""Direct tested-entity calibration-effects plotting tests for CAL-075."""

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
        observed_lift=90.0,
        matched_mta_estimate=100.0,
        test_id=evidence_id,
    )


def _multiple_tested_result():
    return TemporalAttributionCalibration(
        [_evidence("A", "a-test"), _evidence("D", "d-test")]
    ).apply_estimates(
        {
            "E1": {"A": 100.0, "D": 50.0, "B": 50.0},
            "E2": {"A": 120.0, "D": 40.0, "B": 40.0},
            "E3": {"A": 80.0, "D": 60.0, "B": 60.0},
        },
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        ratio_schedule={
            ("A", "E1"): 0.8,
            ("D", "E1"): 1.2,
            ("A", "E2"): 1.1,
        },
    )


def _container_values(ax):
    return {
        container.get_label(): list(container.datavalues)
        for container in ax.containers
    }


def test_plot_has_original_direct_and_adjustment_values_for_each_tested_entity() -> None:
    result = _multiple_tested_result()

    figure = result.plot_calibration_effects()
    try:
        assert isinstance(figure, Figure)
        assert len(figure.axes) == 2
        value_ax, adjustment_ax = figure.axes
        assert value_ax.get_title() == (
            "Tested-Entity Original vs Evidence-Backed Direct Values"
        )
        assert adjustment_ax.get_title() == (
            "Direct Test Adjustments (Residual Adjustments Excluded)"
        )
        values = _container_values(value_ax)
        assert values["Original — A"] == [100.0, 120.0, 80.0]
        assert values["Direct calibrated — A"] == [80.0, 132.0, 80.0]
        assert values["Original — D"] == [50.0, 40.0, 60.0]
        assert values["Direct calibrated — D"] == [60.0, 40.0, 60.0]
        adjustments = _container_values(adjustment_ax)
        assert adjustments["Direct test adjustment — A"] == [-20.0, 12.0, 0.0]
        assert adjustments["Direct test adjustment — D"] == [10.0, 0.0, 0.0]
        assert [tick.get_text() for tick in adjustment_ax.get_xticklabels()] == [
            "E1",
            "E2",
            "E3",
        ]
    finally:
        plt.close(figure)


def test_uncalibrated_entity_epochs_are_explicitly_marked_at_original_values() -> None:
    figure = _multiple_tested_result().plot_calibration_effects()
    try:
        value_ax, adjustment_ax = figure.axes
        value_collections = {
            collection.get_label(): collection
            for collection in value_ax.collections
        }
        adjustment_collections = {
            collection.get_label(): collection
            for collection in adjustment_ax.collections
        }
        assert (
            value_collections["Uncalibrated; original retained — A"]
            .get_offsets()[:, 1]
            .tolist()
            == [80.0]
        )
        assert (
            value_collections["Uncalibrated; original retained — D"]
            .get_offsets()[:, 1]
            .tolist()
            == [40.0, 60.0]
        )
        assert (
            adjustment_collections["No direct adjustment — A"]
            .get_offsets()[:, 1]
            .tolist()
            == [0.0]
        )
        assert (
            adjustment_collections["No direct adjustment — D"]
            .get_offsets()[:, 1]
            .tolist()
            == [0.0, 0.0]
        )
    finally:
        plt.close(figure)


def test_residual_recipient_values_never_enter_direct_effect_artists() -> None:
    result = TemporalAttributionCalibration([_evidence("A", "a-test")]).apply_estimates(
        {"E1": {"A": 100.0, "B": 100.0}},
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy="proportional_untested",
        ratio_schedule={("A", "E1"): 0.8},
        eligible_entities=["B"],
    )
    snapshot = (result.rows, result.epoch_ratio_schedule)

    figure = result.plot_calibration_effects()
    try:
        value_ax, adjustment_ax = figure.axes
        labels = (
            value_ax.get_legend_handles_labels()[1]
            + adjustment_ax.get_legend_handles_labels()[1]
        )
        assert all("B" not in label for label in labels)
        values = _container_values(value_ax)
        assert values["Original — A"] == [100.0]
        assert values["Direct calibrated — A"] == [80.0]
        assert _container_values(adjustment_ax)["Direct test adjustment — A"] == [
            -20.0
        ]
        assert result.rows["E1"][1].residual_adjustment == 20.0
        assert (result.rows, result.epoch_ratio_schedule) == snapshot
    finally:
        plt.close(figure)
