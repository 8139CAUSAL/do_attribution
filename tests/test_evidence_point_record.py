"""Normalized temporal evidence-record tests for CAL-013."""

from dataclasses import FrozenInstanceError, fields
from datetime import date

import pytest

from do_attribution.calibration import (
    EvidencePointRecord,
    LiftTestEvidence,
    TemporalAttributionCalibration,
    _conflict_groups_by_epoch,
    _normalize_evidence_points,
)


EXPECTED_FIELDS = [
    "tested_entity",
    "evidence_id",
    "observed_ratio",
    "coordinate_name",
    "coordinate_value",
    "window_start",
    "window_end",
    "matched_epoch",
    "conflict_group",
    "metadata",
]


def _evidence(
    test_id: str,
    coordinate: date | None,
    *,
    entity: str = "A",
    ratio: float = 1.0,
    window_start: object | None = None,
    window_end: object | None = None,
    metadata: dict | None = None,
) -> LiftTestEvidence:
    return LiftTestEvidence(
        tested_entity=entity,
        observed_lift=ratio * 100.0,
        matched_mta_estimate=100.0,
        test_id=test_id,
        coordinate_name=None if coordinate is None else "date",
        coordinate_value=coordinate,
        window_start=window_start,
        window_end=window_end,
        metadata={} if metadata is None else metadata,
    )


def _windows() -> dict[str, tuple[date, date]]:
    return {
        "E1": (date(2026, 1, 1), date(2026, 1, 10)),
        "E2": (date(2026, 1, 10), date(2026, 1, 20)),
        "E3": (date(2026, 1, 20), date(2026, 1, 30)),
    }


def test_fields_match_plan_order_and_record_is_frozen() -> None:
    record = EvidencePointRecord(
        tested_entity="A",
        evidence_id="a-1",
        observed_ratio=0.9,
    )
    assert [item.name for item in fields(record)] == EXPECTED_FIELDS
    assert not hasattr(record, "epoch_owner")
    assert not hasattr(record, "entity_epoch")
    with pytest.raises(FrozenInstanceError):
        record.matched_epoch = "E1"


@pytest.mark.parametrize(
    ("coordinate", "expected_epoch"),
    [
        (date(2026, 1, 1), "E1"),
        (date(2026, 1, 9), "E1"),
        (date(2026, 1, 10), "E2"),
        (date(2026, 1, 20), "E3"),
        (date(2026, 1, 30), "E3"),
        (date(2025, 12, 31), None),
        (date(2026, 1, 31), None),
    ],
)
def test_authoritative_window_boundary_alignment_is_deterministic(
    coordinate: date,
    expected_epoch: str | None,
) -> None:
    records = _normalize_evidence_points(
        [_evidence("a-1", coordinate)], epoch_windows=_windows()
    )
    assert records[0].coordinate_value == coordinate
    assert records[0].matched_epoch == expected_epoch


@pytest.mark.parametrize(
    ("coordinate", "expected_epoch"),
    [
        (-0.1, None),
        (0.0, "E1"),
        (0.5, "E1"),
        (1.0, None),
        (1.5, None),
        (2.0, "E2"),
        (3.0, "E2"),
        (3.1, None),
    ],
)
def test_membership_keeps_gaps_unmatched_and_closes_only_final_endpoint(
    coordinate: float,
    expected_epoch: str | None,
) -> None:
    evidence = LiftTestEvidence(
        tested_entity="A",
        observed_lift=90.0,
        matched_mta_estimate=100.0,
        test_id="a-gap",
        coordinate_name="index",
        coordinate_value=coordinate,
    )
    point = _normalize_evidence_points(
        [evidence],
        epoch_windows={"E1": (0.0, 1.0), "E2": (2.0, 3.0)},
    )[0]
    assert point.matched_epoch == expected_epoch


def test_no_windows_or_coordinate_means_unmatched_without_midpoint_inference() -> None:
    coordinate_without_windows = _normalize_evidence_points(
        [_evidence("a-1", date(2026, 1, 5))]
    )[0]
    provenance_window_without_coordinate = _normalize_evidence_points(
        [
            _evidence(
                "a-2",
                None,
                window_start=date(2026, 1, 1),
                window_end=date(2026, 1, 9),
            )
        ],
        epoch_windows=_windows(),
    )[0]
    assert coordinate_without_windows.matched_epoch is None
    assert provenance_window_without_coordinate.matched_epoch is None
    assert provenance_window_without_coordinate.coordinate_value is None


def test_temporal_constructor_stores_normalized_points_without_epoch_claims() -> None:
    calibrator = TemporalAttributionCalibration(
        evidence=[
            _evidence("a-1", date(2026, 1, 5)),
            _evidence("b-1", None, entity="B"),
        ]
    )
    assert tuple(point.evidence_id for point in calibrator._evidence_points) == (
        "a-1",
        "b-1",
    )
    assert all(point.matched_epoch is None for point in calibrator._evidence_points)


def test_unmatched_and_conflicting_evidence_remain_without_reduction() -> None:
    evidence = [
        _evidence("a-1", date(2026, 1, 5), ratio=0.8),
        _evidence("a-2", date(2026, 1, 7), ratio=1.2),
        _evidence("b-1", date(2026, 1, 8), entity="B", ratio=1.1),
        _evidence("a-outside", date(2026, 2, 1), ratio=0.7),
    ]
    records = _normalize_evidence_points(evidence, epoch_windows=_windows())

    assert [record.evidence_id for record in records] == [
        "a-1",
        "a-2",
        "b-1",
        "a-outside",
    ]
    assert [record.observed_ratio for record in records] == [0.8, 1.2, 1.1, 0.7]
    assert records[0].conflict_group == ("a-1", "a-2")
    assert records[1].conflict_group == ("a-1", "a-2")
    assert records[2].conflict_group is None
    assert records[3].matched_epoch is None
    assert records[3].conflict_group is None

    groups = _conflict_groups_by_epoch(records)
    assert list(groups) == [("A", "E1")]
    assert tuple(item.evidence_id for item in groups[("A", "E1")]) == (
        "a-1",
        "a-2",
    )


def test_nested_metadata_conflict_ids_and_windows_are_defensively_copied() -> None:
    metadata = {"source": {"ids": ["raw-a"]}}
    conflict_ids = ["a-1", "a-2"]
    record = EvidencePointRecord(
        tested_entity="A",
        evidence_id="a-1",
        observed_ratio=0.9,
        matched_epoch="E1",
        conflict_group=conflict_ids,
        metadata=metadata,
    )
    conflict_ids.append("a-3")
    metadata["source"]["ids"].append("mutated")
    assert record.conflict_group == ("a-1", "a-2")
    assert record.metadata["source"]["ids"] == ("raw-a",)
    with pytest.raises(TypeError):
        record.metadata["source"]["new"] = "value"

    windows = _windows()
    normalized = _normalize_evidence_points(
        [_evidence("a-window", date(2026, 1, 5))], epoch_windows=windows
    )
    windows["E1"] = (date(2030, 1, 1), date(2030, 1, 2))
    assert normalized[0].matched_epoch == "E1"


def test_invalid_or_incompatible_authoritative_windows_raise() -> None:
    evidence = [_evidence("a-1", date(2026, 1, 5))]
    with pytest.raises(ValueError, match="ordered and nonoverlapping"):
        _normalize_evidence_points(
            evidence,
            epoch_windows={
                "E1": (date(2026, 1, 1), date(2026, 1, 10)),
                "E2": (date(2026, 1, 9), date(2026, 1, 20)),
            },
        )
    with pytest.raises(TypeError, match="compatible coordinate type"):
        _normalize_evidence_points(
            evidence,
            epoch_windows={"E1": (0.0, 1.0)},
        )
