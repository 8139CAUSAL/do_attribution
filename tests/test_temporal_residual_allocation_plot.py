"""Residual-allocation diagnostic plotting tests for CAL-076."""

import pytest

matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")

from matplotlib.figure import Figure  # noqa: E402

from do_attribution.calibration import (  # noqa: E402
    LiftTestEvidence,
    TemporalAttributionCalibration,
)


def _calibrator():
    return TemporalAttributionCalibration(
        [
            LiftTestEvidence(
                tested_entity="A",
                observed_lift=80.0,
                matched_mta_estimate=100.0,
                test_id="a-test",
            )
        ]
    )


def _mixed_policy_result():
    epochs = ("E0", "E1", "E2", "E3", "E4", "E5")
    return _calibrator().apply_estimates(
        {epoch: {"A": 100.0, "B": 60.0, "C": 40.0} for epoch in epochs},
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy={
            "E1": "allow_total_change",
            "E2": "proportional_untested",
            "E3": "custom_weights",
            "E4": "residual_bucket",
            "E5": "explicit_adjustments",
        },
        tested_entity_epochs={"A": ["E1", "E2", "E3", "E4", "E5"]},
        ratio_schedule={("A", epoch): 0.8 for epoch in epochs[1:]},
        eligible_entities_by_epoch={"E2": ["B", "C"]},
        residual_weights_by_epoch={"E3": {"B": 0.25, "C": 0.75}},
        residual_entity_by_epoch={"E4": "baseline"},
        residual_adjustments_by_epoch={"E5": {"B": 7.0, "C": 13.0}},
    )


def _container_values(ax):
    return {
        container.get_label(): list(container.datavalues)
        for container in ax.containers
    }


def test_required_allocated_and_every_residual_role_are_auditable() -> None:
    result = _mixed_policy_result()

    figure = result.plot_residual_allocation()
    try:
        assert isinstance(figure, Figure)
        assert len(figure.axes) == 2
        total_ax, recipient_ax = figure.axes
        assert total_ax.get_title() == "Required vs Allocated Residual by Epoch"
        assert recipient_ax.get_title() == (
            "Residual Recipients and Buckets (Direct Adjustments Excluded)"
        )
        totals = _container_values(total_ax)
        assert totals["Required residual to preserve total"] == [
            0.0,
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
        ]
        assert totals["Allocated residual"] == [0.0, 0.0, 20.0, 20.0, 20.0, 20.0]
        recipients = _container_values(recipient_ax)
        assert recipients["Residual recipient — B"] == [0.0, 0.0, 12.0, 5.0, 0.0, 7.0]
        assert recipients["Residual recipient — C"] == [0.0, 0.0, 8.0, 15.0, 0.0, 13.0]
        assert recipients["Residual bucket — baseline"] == [0.0, 0.0, 0.0, 0.0, 20.0, 0.0]
        assert all("A" not in label for label in recipients)
        assert [tick.get_text() for tick in recipient_ax.get_xticklabels()] == list(
            result.epochs
        )
    finally:
        plt.close(figure)


def test_open_closed_and_uncalibrated_epoch_statuses_are_explicit() -> None:
    figure = _mixed_policy_result().plot_residual_allocation()
    try:
        total_ax = figure.axes[0]
        annotations = {text.get_text() for text in total_ax.texts}
        assert annotations == {
            "not calibrated",
            "allow_total_change (total may change)",
            "closed: proportional_untested",
            "closed: custom_weights",
            "closed: residual_bucket",
            "closed: explicit_adjustments",
        }
        collection_labels = {
            collection.get_label() for collection in total_ax.collections
        }
        assert "Open allocation / total may change" in collection_labels
        assert "Not calibrated / no residual policy" in collection_labels
    finally:
        plt.close(figure)


def test_reverse_residuals_remain_signed_in_all_plot_layers() -> None:
    result = _calibrator().apply_estimates(
        {"E1": {"A": 100.0, "B": 60.0, "C": 40.0}},
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy="custom_weights",
        ratio_schedule={("A", "E1"): 1.2},
        residual_weights={"B": 0.5, "C": 0.5},
    )

    figure = result.plot_residual_allocation()
    try:
        total_ax, recipient_ax = figure.axes
        assert _container_values(total_ax)["Required residual to preserve total"] == [
            -20.0
        ]
        assert _container_values(total_ax)["Allocated residual"] == [-20.0]
        recipients = _container_values(recipient_ax)
        assert recipients["Residual recipient — B"] == [-10.0]
        assert recipients["Residual recipient — C"] == [-10.0]
    finally:
        plt.close(figure)


def test_allow_total_change_without_recipients_is_explicit_and_nonmutating() -> None:
    result = _calibrator().apply_estimates(
        {"E1": {"A": 100.0, "B": 100.0}},
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        ratio_schedule={("A", "E1"): 0.8},
    )
    snapshot = (
        result.rows,
        result.required_residuals,
        result.calibrated_totals,
        result.residual_provenance,
    )

    figure = result.plot_residual_allocation()
    try:
        total_ax, recipient_ax = figure.axes
        totals = _container_values(total_ax)
        assert totals["Required residual to preserve total"] == [20.0]
        assert totals["Allocated residual"] == [0.0]
        assert [text.get_text() for text in recipient_ax.texts] == [
            "No recipient or residual-bucket adjustments were allocated."
        ]
        assert recipient_ax.get_legend() is None
        assert (
            result.rows,
            result.required_residuals,
            result.calibrated_totals,
            result.residual_provenance,
        ) == snapshot
    finally:
        plt.close(figure)
