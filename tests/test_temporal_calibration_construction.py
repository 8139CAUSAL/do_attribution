"""Temporal calibrator construction and evidence indexes for CAL-033."""

from datetime import date

import pytest

from do_attribution.calibration import (
    LiftTestEvidence,
    TemporalAttributionCalibration,
)


def _evidence(
    entity: str,
    test_id: str,
    *,
    coordinate_name: str | None,
    coordinate_value: float | date | None,
) -> LiftTestEvidence:
    return LiftTestEvidence(
        tested_entity=entity,
        observed_lift=90.0,
        matched_mta_estimate=100.0,
        test_id=test_id,
        coordinate_name=coordinate_name,
        coordinate_value=coordinate_value,
        metadata={"source": {"ids": [test_id]}},
    )


def test_constructor_builds_ordered_immutable_evidence_indexes() -> None:
    may = date(2026, 5, 15)
    records = [
        _evidence("B", "b-may-1", coordinate_name="date", coordinate_value=may),
        _evidence("A", "a-index", coordinate_name="index", coordinate_value=2.0),
        _evidence("B", "b-may-2", coordinate_name="date", coordinate_value=may),
        _evidence("C", "c-static", coordinate_name=None, coordinate_value=None),
    ]

    calibrator = TemporalAttributionCalibration(evidence=records)
    records.reverse()

    assert [item.test_id for item in calibrator._evidence] == [
        "b-may-1",
        "a-index",
        "b-may-2",
        "c-static",
    ]
    assert list(calibrator._evidence_by_entity) == ["B", "A", "C"]
    assert [
        item.test_id for item in calibrator._evidence_by_entity["B"]
    ] == ["b-may-1", "b-may-2"]
    assert list(calibrator._evidence_by_id) == [
        "b-may-1",
        "a-index",
        "b-may-2",
        "c-static",
    ]
    assert list(calibrator._evidence_by_coordinate_name) == ["date", "index", None]
    assert [
        item.test_id for item in calibrator._evidence_by_coordinate_name["date"]
    ] == ["b-may-1", "b-may-2"]
    assert list(calibrator._evidence_by_coordinate_value) == [may, 2.0, None]
    assert [
        item.test_id for item in calibrator._evidence_by_coordinate_value[may]
    ] == ["b-may-1", "b-may-2"]

    for index in (
        calibrator._evidence_by_entity,
        calibrator._evidence_by_id,
        calibrator._evidence_by_coordinate_name,
        calibrator._evidence_by_coordinate_value,
    ):
        with pytest.raises(TypeError):
            index["new"] = ()
    with pytest.raises(TypeError):
        calibrator._evidence_by_coordinate_name["date"][0].metadata["source"] = {}


def test_constructor_does_not_align_epochs_or_select_policies() -> None:
    calibrator = TemporalAttributionCalibration(
        evidence=[
            _evidence(
                "A",
                "a-date",
                coordinate_name="date",
                coordinate_value=date(2026, 1, 5),
            ),
            _evidence(
                "A",
                "a-index",
                coordinate_name="index",
                coordinate_value=7.0,
            ),
            _evidence("B", "b-static", coordinate_name=None, coordinate_value=None),
        ]
    )

    assert [point.evidence_id for point in calibrator._evidence_points] == [
        "a-date",
        "a-index",
        "b-static",
    ]
    assert all(point.matched_epoch is None for point in calibrator._evidence_points)
    assert all(point.conflict_group is None for point in calibrator._evidence_points)
    assert not any("policy" in name for name in vars(calibrator))
    assert not any("epoch" in name for name in vars(calibrator))
