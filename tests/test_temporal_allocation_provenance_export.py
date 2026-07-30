"""Temporal allocation, totals, and provenance export tests for CAL-071."""

from dataclasses import replace

import pandas as pd

from do_attribution.calibration import (
    LiftTestEvidence,
    TemporalAttributionCalibration,
)


ALLOCATION_COLUMNS = [
    "epoch",
    "entity",
    "original_estimate",
    "direct_test_adjustment",
    "residual_adjustment",
    "net_adjustment",
    "calibrated_estimate",
    "original_share",
    "calibrated_share",
    "role",
]
TOTAL_COLUMNS = [
    "epoch",
    "original_total",
    "raw_post_test_total",
    "required_residual",
    "allocated_residual",
    "calibrated_total",
    "residual_policy",
]
RATIO_PROVENANCE_COLUMNS = [
    "tested_entity",
    "epoch",
    "ratio_policy",
    "candidate",
    "schedule_status",
    "schedule_provenance",
    "evidence_conflict_policy",
    "conflict_resolution",
    "uncalibrated",
]
REDUCTION_PROVENANCE_COLUMNS = [
    "tested_entity",
    "epoch",
    "epoch_ratio_policy",
    "calibrated",
    "provenance",
]
RESIDUAL_PROVENANCE_COLUMNS = [
    "epoch",
    "residual_policy",
    "allocated_residual",
    "provenance",
]
ALL_TABLE_KEYS = [
    "evidence",
    "coordinate_schedule",
    "epoch_ratios",
    "allocations",
    "totals",
    "ratio_provenance",
    "epoch_reduction_provenance",
    "residual_provenance",
    "warnings",
]


def _mixed_result():
    epochs = ("E0", "E1", "E2", "E3", "E4", "E5")
    estimates = {
        epoch: {"A": 100.0, "B": 60.0, "C": 40.0}
        for epoch in epochs
    }
    calibrator = TemporalAttributionCalibration(
        [
            LiftTestEvidence(
                tested_entity="A",
                observed_lift=80.0,
                matched_mta_estimate=100.0,
                test_id="a-test",
            )
        ]
    )
    return calibrator.apply_estimates(
        estimates,
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy={
            "E1": "allow_total_change",
            "E2": "proportional_untested",
            "E3": "custom_weights",
            "E4": "residual_bucket",
            "E5": "explicit_adjustments",
        },
        ratio_schedule={
            ("A", epoch): 0.8 for epoch in epochs if epoch != "E0"
        },
        eligible_entities_by_epoch={"E2": ["B", "C"]},
        residual_weights_by_epoch={"E3": {"B": 0.25, "C": 0.75}},
        residual_entity_by_epoch={"E4": "baseline"},
        residual_adjustments_by_epoch={"E5": {"B": 7.0, "C": 13.0}},
    )


def test_complete_return_contract_has_exact_table_and_column_order() -> None:
    tables = _mixed_result().to_dataframes()

    assert list(tables) == ALL_TABLE_KEYS
    assert list(tables["allocations"].columns) == ALLOCATION_COLUMNS
    assert list(tables["totals"].columns) == TOTAL_COLUMNS
    assert list(tables["ratio_provenance"].columns) == RATIO_PROVENANCE_COLUMNS
    assert (
        list(tables["epoch_reduction_provenance"].columns)
        == REDUCTION_PROVENANCE_COLUMNS
    )
    assert (
        list(tables["residual_provenance"].columns)
        == RESIDUAL_PROVENANCE_COLUMNS
    )
    assert list(tables["warnings"].columns) == ["warning"]


def test_allocations_preserve_epoch_entity_and_appended_bucket_order() -> None:
    allocations = _mixed_result().to_dataframes()["allocations"]

    assert list(zip(allocations["epoch"], allocations["entity"])) == [
        ("E0", "A"), ("E0", "B"), ("E0", "C"),
        ("E1", "A"), ("E1", "B"), ("E1", "C"),
        ("E2", "A"), ("E2", "B"), ("E2", "C"),
        ("E3", "A"), ("E3", "B"), ("E3", "C"),
        ("E4", "A"), ("E4", "B"), ("E4", "C"), ("E4", "baseline"),
        ("E5", "A"), ("E5", "B"), ("E5", "C"),
    ]
    tested = allocations[(allocations["epoch"] == "E2") & (allocations["entity"] == "A")].iloc[0]
    recipient = allocations[(allocations["epoch"] == "E2") & (allocations["entity"] == "B")].iloc[0]
    bucket = allocations[allocations["entity"] == "baseline"].iloc[0]
    assert tested["direct_test_adjustment"] == -20.0
    assert tested["residual_adjustment"] == 0.0
    assert tested["net_adjustment"] == -20.0
    assert tested["role"] == "tested"
    assert recipient["direct_test_adjustment"] == 0.0
    assert recipient["residual_adjustment"] == 12.0
    assert recipient["net_adjustment"] == 12.0
    assert recipient["role"] == "residual_recipient"
    assert bucket["original_estimate"] == 0.0
    assert bucket["residual_adjustment"] == 20.0
    assert bucket["role"] == "residual_bucket"


def test_totals_reproduce_every_epoch_identity_and_uncovered_state() -> None:
    totals = _mixed_result().to_dataframes()["totals"].set_index("epoch")

    assert totals.index.tolist() == ["E0", "E1", "E2", "E3", "E4", "E5"]
    assert totals.loc["E0", "original_total"] == 200.0
    assert totals.loc["E0", "raw_post_test_total"] == 200.0
    assert totals.loc["E0", "required_residual"] == 0.0
    assert totals.loc["E0", "allocated_residual"] == 0.0
    assert totals.loc["E0", "calibrated_total"] == 200.0
    assert pd.isna(totals.loc["E0", "residual_policy"])
    assert totals.loc["E1", "raw_post_test_total"] == 180.0
    assert totals.loc["E1", "required_residual"] == 20.0
    assert totals.loc["E1", "allocated_residual"] == 0.0
    assert totals.loc["E1", "calibrated_total"] == 180.0
    assert totals.loc["E1", "residual_policy"] == "allow_total_change"
    for epoch in ("E2", "E3", "E4", "E5"):
        assert totals.loc[epoch, "raw_post_test_total"] == 180.0
        assert totals.loc[epoch, "required_residual"] == 20.0
        assert totals.loc[epoch, "allocated_residual"] == 20.0
        assert totals.loc[epoch, "calibrated_total"] == 200.0


def test_provenance_tables_cover_calibrated_and_uncalibrated_pairs_and_epochs() -> None:
    tables = _mixed_result().to_dataframes()
    ratio = tables["ratio_provenance"].set_index(["tested_entity", "epoch"])
    reduction = tables["epoch_reduction_provenance"].set_index(
        ["tested_entity", "epoch"]
    )
    residual = tables["residual_provenance"].set_index("epoch")

    assert ratio.index.tolist() == [("A", epoch) for epoch in ("E0", "E1", "E2", "E3", "E4", "E5")]
    assert ratio.loc[("A", "E0"), "candidate"] == False
    assert ratio.loc[("A", "E0"), "schedule_status"] == "not_calibrated"
    assert ratio.loc[("A", "E0"), "uncalibrated"] == True
    assert ratio.loc[("A", "E2"), "candidate"] == True
    assert ratio.loc[("A", "E2"), "schedule_status"] == "explicit"
    assert ratio.loc[("A", "E2"), "uncalibrated"] == False
    assert reduction.loc[("A", "E0"), "calibrated"] == False
    assert reduction.loc[("A", "E1"), "calibrated"] == True
    assert residual.index.tolist() == ["E0", "E1", "E2", "E3", "E4", "E5"]
    assert pd.isna(residual.loc["E0", "residual_policy"])
    assert residual.loc["E0", "provenance"]["status"] == "not_calibrated"
    assert residual.loc["E4", "provenance"]["residual_policy"] == "residual_bucket"
    assert residual.loc["E4", "allocated_residual"] == 20.0


def test_every_dataframe_and_nested_provenance_cell_is_a_fresh_snapshot() -> None:
    result = _mixed_result()
    first = result.to_dataframes()
    first["allocations"].at[0, "original_estimate"] = -1.0
    first["totals"].drop(index=0, inplace=True)
    first["ratio_provenance"].at[0, "schedule_provenance"]["changed"] = True
    first["epoch_reduction_provenance"].at[1, "provenance"]["changed"] = True
    first["residual_provenance"].at[0, "provenance"]["changed"] = True

    second = result.to_dataframes()

    assert second["allocations"].loc[0, "original_estimate"] == 100.0
    assert second["totals"].index.equals(pd.RangeIndex(6))
    assert "changed" not in second["ratio_provenance"].loc[0, "schedule_provenance"]
    assert "changed" not in second["epoch_reduction_provenance"].loc[1, "provenance"]
    assert "changed" not in second["residual_provenance"].loc[0, "provenance"]
    assert "changed" not in result.ratio_provenance["schedule_provenance_by_pair"][("A", "E0")]


def test_warnings_are_exported_in_stored_order_without_mutable_aliasing() -> None:
    result = replace(
        _mixed_result(),
        warnings=("First warning.", "Second warning."),
    )

    first = result.to_dataframes()["warnings"]
    first.at[0, "warning"] = "changed"
    second = result.to_dataframes()["warnings"]

    assert second["warning"].tolist() == ["First warning.", "Second warning."]
    assert result.warnings == ("First warning.", "Second warning.")
