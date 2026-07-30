"""Temporal coordinate-schedule DataFrame export tests for CAL-069."""

import pandas as pd

from do_attribution.calibration import (
    LiftTestEvidence,
    TemporalAttributionCalibration,
)


SCHEDULE_COLUMNS = [
    "tested_entity",
    "coordinate_name",
    "coordinate_value",
    "schedule_ratio",
    "schedule_source",
    "source_evidence_ids",
    "left_source_evidence_id",
    "right_source_evidence_id",
    "inside_observed_support",
]


def _evidence(entity: object, evidence_id: str, ratio: float, coordinate: float):
    return LiftTestEvidence(
        tested_entity=entity,
        observed_lift=100.0 * ratio,
        matched_mta_estimate=100.0,
        test_id=evidence_id,
        coordinate_name="time",
        coordinate_value=coordinate,
    )


def test_coordinate_linear_export_includes_compact_requested_evaluations() -> None:
    result = TemporalAttributionCalibration(
        [
            _evidence("A", "a-left", 0.8, 2.0),
            _evidence("A", "a-right", 1.2, 8.0),
        ]
    ).apply_estimates(
        {
            "E1": {"A": 100.0, "B": 100.0},
            "E2": {"A": 100.0, "B": 100.0},
            "E3": {"A": 100.0, "B": 100.0},
        },
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="representative_coordinate",
        residual_policy="allow_total_change",
        epoch_windows={"E1": (0.0, 5.0), "E2": (5.0, 10.0), "E3": (10.0, 15.0)},
        epoch_coordinates={"E1": 2.0, "E2": 7.0, "E3": 12.0},
        tested_entity_epochs={"A": ["E1", "E2", "E3"]},
    )

    tables = result.to_dataframes()
    schedule = tables["coordinate_schedule"]

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
    assert list(schedule.columns) == SCHEDULE_COLUMNS
    assert schedule.index.equals(pd.RangeIndex(4))
    assert schedule["coordinate_value"].tolist() == [2.0, 7.0, 8.0, 12.0]
    assert schedule["schedule_source"].tolist() == [
        "observed",
        "interpolated",
        "observed",
        "unsupported",
    ]
    assert schedule["schedule_ratio"].iloc[:3].tolist() == [0.8, 1.1333333333333333, 1.2]
    assert pd.isna(schedule.loc[3, "schedule_ratio"])
    assert schedule.loc[1, "source_evidence_ids"] == ("a-left", "a-right")
    assert schedule.loc[1, "left_source_evidence_id"] == "a-left"
    assert schedule.loc[1, "right_source_evidence_id"] == "a-right"
    assert schedule["inside_observed_support"].tolist() == [True, True, True, False]
    assert tuple(result.coordinate_ratio_schedule)[1].schedule_source == "interpolated"
    assert tuple(result.coordinate_ratio_schedule)[-1].schedule_source == "unsupported"


def test_schedule_rows_keep_first_seen_entity_then_coordinate_order() -> None:
    result = TemporalAttributionCalibration(
        [
            _evidence("B", "b-left", 0.9, 0.0),
            _evidence("B", "b-right", 1.1, 10.0),
            _evidence("A", "a-left", 0.8, 0.0),
            _evidence("A", "a-right", 1.2, 10.0),
        ]
    ).apply_estimates(
        {"E1": {"A": 100.0, "B": 100.0}},
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="representative_coordinate",
        residual_policy="allow_total_change",
        epoch_windows={"E1": (2.0, 8.0)},
        epoch_coordinates={"E1": 5.0},
        tested_entity_epochs={"B": ["E1"], "A": ["E1"]},
    )

    schedule = result.to_dataframes()["coordinate_schedule"]

    assert list(zip(schedule["tested_entity"], schedule["coordinate_value"])) == [
        ("B", 0.0),
        ("B", 5.0),
        ("B", 10.0),
        ("A", 0.0),
        ("A", 5.0),
        ("A", 10.0),
    ]


def test_observed_and_constant_schedules_export_their_stored_sources() -> None:
    evidence = [_evidence("A", "a-source", 0.8, 1.0)]
    observed = TemporalAttributionCalibration(evidence).apply_estimates(
        {"E1": {"A": 100.0, "B": 100.0}},
        ratio_policy="observed_only",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        epoch_windows={"E1": (0.0, 2.0)},
    ).to_dataframes()["coordinate_schedule"]
    constant = TemporalAttributionCalibration(evidence).apply_estimates(
        {"E1": {"A": 100.0, "B": 100.0}},
        ratio_policy="constant",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        source_evidence={"A": "a-source"},
        tested_entity_epochs={"A": ["E1"]},
    ).to_dataframes()["coordinate_schedule"]

    assert observed.loc[0, "schedule_source"] == "observed"
    assert observed.loc[0, "coordinate_value"] == 1.0
    assert observed.loc[0, "source_evidence_ids"] == ("a-source",)
    assert constant.loc[0, "schedule_source"] == "constant"
    assert pd.isna(constant.loc[0, "coordinate_name"])
    assert pd.isna(constant.loc[0, "coordinate_value"])
    assert constant.loc[0, "schedule_ratio"] == 0.8


def test_direct_explicit_policies_return_an_empty_table_with_exact_schema() -> None:
    evidence = [_evidence("A", "a-source", 0.8, 1.0)]
    explicit = TemporalAttributionCalibration(evidence).apply_estimates(
        {"E1": {"A": 100.0}},
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        ratio_schedule={("A", "E1"): 0.9},
    )
    strength = TemporalAttributionCalibration(evidence).apply_estimates(
        {"E1": {"A": 100.0}},
        ratio_policy="explicit_strength",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        source_evidence={"A": "a-source"},
        evidence_strength={("A", "E1"): 0.5},
        tested_entity_epochs={"A": ["E1"]},
    )

    for result in (explicit, strength):
        schedule = result.to_dataframes()["coordinate_schedule"]
        assert schedule.empty
        assert list(schedule.columns) == SCHEDULE_COLUMNS


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


def test_schedule_exports_are_fresh_deep_snapshots() -> None:
    tested = _MutableHashableEntity("A")
    result = TemporalAttributionCalibration(
        [_evidence(tested, "mutable-a", 0.8, 1.0)]
    ).apply_estimates(
        {"E1": [(tested, 100.0), ("B", 100.0)]},
        ratio_policy="constant",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        source_evidence={tested: "mutable-a"},
        tested_entity_epochs={tested: ["E1"]},
    )

    first = result.to_dataframes()["coordinate_schedule"]
    first.at[0, "tested_entity"].label = "mutated"
    first.at[0, "source_evidence_ids"] = ("changed",)
    first.drop(index=0, inplace=True)

    second = result.to_dataframes()["coordinate_schedule"]

    assert result.coordinate_ratio_schedule[0].tested_entity.label == "A"
    assert second.loc[0, "tested_entity"].label == "A"
    assert second.loc[0, "source_evidence_ids"] == ("mutable-a",)
    assert second.index.equals(pd.RangeIndex(1))
