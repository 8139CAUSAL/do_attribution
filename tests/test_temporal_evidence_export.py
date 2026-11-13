"""Temporal normalized-evidence DataFrame export tests for CAL-068."""

from datetime import date

import pandas as pd

from do_attribution.calibration import (
    LiftTestEvidence,
    TemporalAttributionCalibration,
)


EVIDENCE_COLUMNS = [
    "tested_entity",
    "evidence_id",
    "observed_ratio",
    "coordinate_name",
    "coordinate_value",
    "window_start",
    "window_end",
    "matched_epoch",
    "conflict_group",
    "observed_lift",
    "standard_error",
    "uncertainty_status",
]


def _dated_evidence(
    evidence_id: str,
    coordinate: date,
    ratio: float,
    standard_error: float | None = None,
) -> LiftTestEvidence:
    return LiftTestEvidence(
        tested_entity="A",
        observed_lift=100.0 * ratio,
        matched_mta_estimate=100.0,
        test_id=evidence_id,
        coordinate_name="date",
        coordinate_value=coordinate,
        window_start=f"{evidence_id}-window-start",
        window_end=f"{evidence_id}-window-end",
        standard_error=standard_error,
    )


def _result_with_conflict_and_unmatched_evidence():
    evidence = [
        _dated_evidence("a-early", date(2024, 1, 2), 0.8),
        _dated_evidence("a-late", date(2024, 1, 8), 1.0),
        _dated_evidence("a-unmatched", date(2024, 2, 1), 1.2),
    ]
    return TemporalAttributionCalibration(evidence).apply_estimates(
        {
            "E1": {"A": 100.0, "B": 100.0},
            "E2": {"A": 100.0, "B": 100.0},
        },
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="representative_coordinate",
        residual_policy="allow_total_change",
        epoch_windows={
            "E1": (date(2024, 1, 1), date(2024, 1, 10)),
            "E2": (date(2024, 1, 10), date(2024, 1, 20)),
        },
        epoch_coordinates={
            "E1": date(2024, 1, 5),
            "E2": date(2024, 1, 15),
        },
        tested_entity_epochs={"A": ["E1", "E2"]},
    )


def test_evidence_table_uses_the_frozen_schema_and_preserves_record_order() -> None:
    result = _result_with_conflict_and_unmatched_evidence()

    tables = result.to_dataframes()
    evidence = tables["evidence"]

    assert list(tables) == [
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
    assert isinstance(evidence, pd.DataFrame)
    assert list(evidence.columns) == EVIDENCE_COLUMNS
    assert evidence.index.equals(pd.RangeIndex(3))
    assert evidence["evidence_id"].tolist() == [
        "a-early",
        "a-late",
        "a-unmatched",
    ]
    assert evidence["observed_ratio"].tolist() == [0.8, 1.0, 1.2]
    assert evidence["coordinate_value"].tolist() == [
        date(2024, 1, 2),
        date(2024, 1, 8),
        date(2024, 2, 1),
    ]
    assert evidence["window_start"].tolist() == [
        "a-early-window-start",
        "a-late-window-start",
        "a-unmatched-window-start",
    ]
    assert evidence["observed_lift"].tolist() == [80.0, 100.0, 120.0]
    assert evidence["standard_error"].isna().all()
    assert evidence["uncertainty_status"].tolist() == [
        "no uncertainty supplied"
    ] * 3


def _result_with_standard_errors(standard_errors):
    evidence = [
        _dated_evidence("a-early", date(2024, 1, 2), 0.8, standard_errors[0]),
        _dated_evidence("a-late", date(2024, 1, 8), 1.0, standard_errors[1]),
        _dated_evidence("a-unmatched", date(2024, 2, 1), 1.2, standard_errors[2]),
    ]
    return TemporalAttributionCalibration(evidence).apply_estimates(
        {
            "E1": {"A": 100.0, "B": 100.0},
            "E2": {"A": 100.0, "B": 100.0},
        },
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="representative_coordinate",
        residual_policy="allow_total_change",
        epoch_windows={
            "E1": (date(2024, 1, 1), date(2024, 1, 10)),
            "E2": (date(2024, 1, 10), date(2024, 1, 20)),
        },
        epoch_coordinates={
            "E1": date(2024, 1, 5),
            "E2": date(2024, 1, 15),
        },
        tested_entity_epochs={"A": ["E1", "E2"]},
    )


def test_evidence_table_carries_lift_and_reported_standard_error_per_record() -> None:
    result = _result_with_standard_errors((5.0, None, 0.0))

    evidence = result.to_dataframes()["evidence"]

    assert evidence["evidence_id"].tolist() == [
        record.test_id for record in result.evidence
    ]
    assert evidence["observed_lift"].tolist() == [
        record.observed_lift for record in result.evidence
    ]
    assert evidence.loc[0, "standard_error"] == 5.0
    assert pd.isna(evidence.loc[1, "standard_error"])
    # A reported standard error of zero is a value, not a missing one.
    assert evidence.loc[2, "standard_error"] == 0.0
    assert evidence["uncertainty_status"].tolist() == [
        "standard error reported by the experiment",
        "no uncertainty supplied",
        "standard error reported by the experiment",
    ]
    # The unmatched observation keeps its uncertainty annotation too.
    assert pd.isna(evidence.loc[2, "matched_epoch"])


def test_standard_error_changes_only_the_evidence_annotation_columns() -> None:
    without = _result_with_standard_errors((None, None, None)).to_dataframes()
    low = _result_with_standard_errors((1.0, 2.0, 3.0)).to_dataframes()
    high = _result_with_standard_errors((1000.0, 2000.0, 3000.0)).to_dataframes()

    annotation_columns = ["standard_error", "uncertainty_status"]
    for tables in (low, high):
        assert list(tables) == list(without)
        for key in without:
            if key == "evidence":
                continue
            pd.testing.assert_frame_equal(tables[key], without[key])
        pd.testing.assert_frame_equal(
            tables["evidence"].drop(columns=annotation_columns),
            without["evidence"].drop(columns=annotation_columns),
        )
    assert low["evidence"]["standard_error"].tolist() == [1.0, 2.0, 3.0]
    assert high["evidence"]["standard_error"].tolist() == [1000.0, 2000.0, 3000.0]
    pd.testing.assert_series_equal(
        low["evidence"]["uncertainty_status"],
        high["evidence"]["uncertainty_status"],
    )


def test_matches_and_conflict_groups_are_derived_without_hiding_unmatched_points() -> None:
    evidence = _result_with_conflict_and_unmatched_evidence().to_dataframes()[
        "evidence"
    ]

    assert evidence["matched_epoch"].iloc[:2].tolist() == ["E1", "E1"]
    assert pd.isna(evidence.loc[2, "matched_epoch"])
    expected_group = ("a-early", "a-late")
    assert evidence["conflict_group"].iloc[:2].tolist() == [
        expected_group,
        expected_group,
    ]
    assert pd.isna(evidence.loc[2, "conflict_group"])
    assert evidence.loc[2, "coordinate_value"] == date(2024, 2, 1)


class _MutableHashableEntity:
    def __init__(self, label: str) -> None:
        self.label = label

    def __hash__(self) -> int:
        return hash(self.label)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _MutableHashableEntity)
            and self.label == other.label
        )


def test_evidence_exports_are_fresh_deep_snapshots() -> None:
    tested = _MutableHashableEntity("A")
    result = TemporalAttributionCalibration(
        [
            LiftTestEvidence(
                tested_entity=tested,
                observed_lift=90.0,
                matched_mta_estimate=100.0,
                test_id="mutable-a",
                coordinate_name="time",
                coordinate_value=1.0,
            )
        ]
    ).apply_estimates(
        {"E1": [(tested, 100.0), ("B", 100.0)]},
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        ratio_schedule={(tested, "E1"): 0.9},
        tested_entity_epochs={tested: ["E1"]},
    )

    first = result.to_dataframes()["evidence"]
    first.at[0, "tested_entity"].label = "mutated"
    first.at[0, "evidence_id"] = "changed"
    first.at[0, "conflict_group"] = ("changed", "also-changed")
    first.drop(index=0, inplace=True)

    second = result.to_dataframes()["evidence"]

    assert result.evidence_points[0].tested_entity.label == "A"
    assert result.evidence_points[0].evidence_id == "mutable-a"
    assert second.loc[0, "tested_entity"].label == "A"
    assert second.loc[0, "evidence_id"] == "mutable-a"
    assert pd.isna(second.loc[0, "conflict_group"])
    assert second.index.equals(pd.RangeIndex(1))
