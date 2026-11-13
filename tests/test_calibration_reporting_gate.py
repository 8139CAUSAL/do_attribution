"""Cross-surface reporting, provenance, and no-mutation gate for CAL-079."""

import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")

from do_attribution.calibration import (  # noqa: E402
    AttributionCalibration,
    LiftTestEvidence,
    TemporalAttributionCalibration,
)


def _result():
    return TemporalAttributionCalibration(
        [
            LiftTestEvidence(
                tested_entity="A",
                observed_lift=80.0,
                matched_mta_estimate=100.0,
                test_id="a-source",
                metadata={"audit": {"source": ["experiment"]}},
            )
        ]
    ).apply_estimates(
        {
            "E1": {"A": 100.0, "B": 60.0, "C": 40.0},
            "E2": {"A": 100.0, "B": 40.0, "C": 60.0},
        },
        ratio_policy="explicit_strength",
        epoch_ratio_policy="direct",
        residual_policy="custom_weights",
        source_evidence={"A": "a-source"},
        evidence_strength={("A", "E1"): 0.5, ("A", "E2"): 1.0},
        residual_weights={"B": 0.4, "C": 0.6},
        epoch_support={"E1": 10.0, "E2": 20.0},
    )


def _static_result():
    return AttributionCalibration(
        [
            LiftTestEvidence(
                tested_entity="A",
                observed_lift=80.0,
                matched_mta_estimate=100.0,
                test_id="a-static-source",
            )
        ]
    ).apply_estimates(
        {"A": 100.0, "B": 60.0, "C": 40.0},
        residual_policy="custom_weights",
        residual_weights={"B": 0.4, "C": 0.6},
    )


def test_static_plot_reads_without_mutating_result_state() -> None:
    result = _static_result()
    snapshot = (
        result.rows,
        result.evidence,
        result.original_total,
        result.raw_post_test_total,
        result.required_residual,
        result.allocated_residual,
        result.calibrated_total,
    )
    figure = result.plot_calibration_effects()
    try:
        assert figure.axes
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
        plt.close(figure)


def test_every_plot_and_reporting_projection_reads_without_mutating_state() -> None:
    result = _result()
    snapshot = (
        result.rows,
        result.evidence,
        result.epoch_ratio_schedule,
        result.ratio_provenance,
        result.epoch_reduction_provenance,
        result.residual_provenance,
        result.original_totals,
        result.calibrated_totals,
    )
    first_summary = result.summary()
    first_tables = result.to_dataframes()

    figures = [
        result.plot_epoch_alignment(),
        result.plot_ratio_schedule(),
        result.plot_evidence_uncertainty(),
        result.plot_calibration_effects(),
        result.plot_residual_allocation(),
        result.plot_entity_mix(),
        result.plot_evidence_strength(),
    ]
    try:
        assert all(figure.axes for figure in figures)
        assert result.summary() == first_summary
        second_tables = result.to_dataframes()
        assert list(second_tables) == list(first_tables)
        for key in first_tables:
            pd.testing.assert_frame_equal(first_tables[key], second_tables[key])
        assert (
            result.rows,
            result.evidence,
            result.epoch_ratio_schedule,
            result.ratio_provenance,
            result.epoch_reduction_provenance,
            result.residual_provenance,
            result.original_totals,
            result.calibrated_totals,
        ) == snapshot
    finally:
        for figure in figures:
            plt.close(figure)


def test_every_effective_ratio_and_residual_adjustment_is_traceable() -> None:
    result = _result()
    known_evidence_ids = {item.test_id for item in result.evidence}
    candidate_pairs = set(result.ratio_provenance["candidate_pairs"])
    schedule_status = result.ratio_provenance["schedule_status_by_pair"]
    schedule_provenance = result.ratio_provenance[
        "schedule_provenance_by_pair"
    ]
    reduction_provenance = result.epoch_reduction_provenance["by_pair"]

    for ratio_row in result.epoch_ratio_schedule:
        pair = (ratio_row.tested_entity, ratio_row.epoch)
        assert pair in candidate_pairs
        assert schedule_status[pair] == "strength_derived"
        assert schedule_provenance[pair]["source_evidence_id"] in known_evidence_ids
        assert schedule_provenance[pair]["evidence_strength"] in {0.5, 1.0}
        assert schedule_provenance[pair]["effective_ratio"] == pytest.approx(
            ratio_row.effective_ratio
        )
        assert reduction_provenance[pair]["effective_ratio"] == pytest.approx(
            ratio_row.effective_ratio
        )
        assert set(ratio_row.source_evidence_ids) <= known_evidence_ids

    for epoch in result.epochs:
        row_adjustments = tuple(
            (row.entity, row.residual_adjustment)
            for row in result.rows[epoch]
            if row.residual_adjustment != 0.0
        )
        provenance = result.residual_provenance[epoch]
        assert provenance["residual_policy"] == "custom_weights"
        assert provenance["adjustments"] == row_adjustments
        assert sum(value for _, value in row_adjustments) == pytest.approx(
            result.required_residuals[epoch]
        )

    tables = result.to_dataframes()
    ratio_export = tables["epoch_ratios"].set_index(
        ["tested_entity", "epoch"]
    )
    residual_export = tables["residual_provenance"].set_index("epoch")
    for ratio_row in result.epoch_ratio_schedule:
        pair = (ratio_row.tested_entity, ratio_row.epoch)
        assert ratio_export.loc[pair, "effective_ratio"] == pytest.approx(
            ratio_row.effective_ratio
        )
    for epoch in result.epochs:
        assert residual_export.loc[epoch, "provenance"]["adjustments"] == (
            result.residual_provenance[epoch]["adjustments"]
        )
