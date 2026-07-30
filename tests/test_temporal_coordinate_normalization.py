"""Coordinate-linear anchor normalization tests for CAL-045."""

from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta, timezone

import pytest

from do_attribution.calibration import (
    LiftTestEvidence,
    _normalize_coordinate_linear_anchors,
)


def _evidence(entity, test_id, name, value, ratio=1.0):
    return LiftTestEvidence(
        tested_entity=entity,
        observed_lift=ratio * 100.0,
        matched_mta_estimate=100.0,
        test_id=test_id,
        coordinate_name=name,
        coordinate_value=value,
    )


def test_numeric_coordinates_preserve_originals_and_stable_unsorted_order() -> None:
    anchors = _normalize_coordinate_linear_anchors(
        [
            _evidence("A", "a-right", "index", 3, 1.2),
            _evidence("A", "a-left", "index", 1.5, 0.8),
        ]
    )
    assert [anchor.evidence_id for anchor in anchors.anchors] == ["a-right", "a-left"]
    assert [anchor.coordinate_value for anchor in anchors.anchors] == [3, 1.5]
    assert [anchor.normalized_coordinate for anchor in anchors.anchors] == [3.0, 1.5]
    assert all(anchor.coordinate_kind == "numeric" for anchor in anchors.anchors)
    with pytest.raises(FrozenInstanceError):
        anchors.anchors[0].normalized_coordinate = 4.0
    with pytest.raises(TypeError):
        anchors.by_entity["A"] = ()


def test_date_coordinates_use_ordinal_day_arithmetic_and_round_trip() -> None:
    left = date(2026, 1, 1)
    right = date(2026, 1, 11)
    anchors = _normalize_coordinate_linear_anchors(
        [
            _evidence("A", "a-left", "date", left),
            _evidence("A", "a-right", "date", right),
        ]
    ).by_entity["A"]
    assert anchors[0].coordinate_value == left
    assert anchors[1].coordinate_value == right
    assert anchors[1].normalized_coordinate - anchors[0].normalized_coordinate == 10.0
    assert all(anchor.coordinate_kind == "date" for anchor in anchors)


def test_naive_and_aware_datetimes_have_deterministic_second_arithmetic() -> None:
    naive = _normalize_coordinate_linear_anchors(
        [
            _evidence("A", "a-left", "timestamp", datetime(2026, 1, 1, 0, 0)),
            _evidence("A", "a-right", "timestamp", datetime(2026, 1, 1, 0, 1)),
        ]
    ).by_entity["A"]
    assert naive[1].normalized_coordinate - naive[0].normalized_coordinate == 60.0

    eastern = timezone(timedelta(hours=-5))
    aware = _normalize_coordinate_linear_anchors(
        [
            _evidence(
                "A",
                "a-left",
                "timestamp",
                datetime(2026, 1, 1, 0, 0, tzinfo=eastern),
            ),
            _evidence(
                "A",
                "a-right",
                "timestamp",
                datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc),
            ),
        ]
    ).by_entity["A"]
    assert aware[1].normalized_coordinate - aware[0].normalized_coordinate == 3600.0
    assert aware[0].coordinate_value.tzinfo == eastern


def test_each_entity_has_an_independent_compatible_coordinate_axis() -> None:
    anchors = _normalize_coordinate_linear_anchors(
        [
            _evidence("B", "b-1", "date", date(2026, 1, 1)),
            _evidence("A", "a-1", "index", 1.0),
            _evidence("B", "b-2", "date", date(2026, 1, 2)),
            _evidence("A", "a-2", "index", 2.0),
        ]
    )
    assert tuple(anchors.by_entity) == ("B", "A")
    assert [anchor.coordinate_kind for anchor in anchors.by_entity["B"]] == ["date", "date"]
    assert [anchor.coordinate_kind for anchor in anchors.by_entity["A"]] == ["numeric", "numeric"]


def test_selected_entities_may_restrict_anchor_preparation_explicitly() -> None:
    evidence = [
        _evidence("A", "a-1", "index", 1.0),
        _evidence("A", "a-2", "index", 2.0),
        _evidence("B", "b-only", "index", 1.0),
    ]
    anchors = _normalize_coordinate_linear_anchors(evidence, selected_entities=["A"])
    assert tuple(anchors.by_entity) == ("A",)
    with pytest.raises(ValueError, match="without evidence"):
        _normalize_coordinate_linear_anchors(evidence, selected_entities=["C"])


@pytest.mark.parametrize(
    ("evidence", "error", "message"),
    [
        (
            [_evidence("A", "a-only", "index", 1.0)],
            ValueError,
            "at least two",
        ),
        (
            [
                _evidence("A", "a-1", "index", 1.0),
                _evidence("A", "a-2", "spend", 2.0),
            ],
            ValueError,
            "common coordinate_name",
        ),
        (
            [
                _evidence("A", "a-1", "time", date(2026, 1, 1)),
                _evidence("A", "a-2", "time", datetime(2026, 1, 2)),
            ],
            TypeError,
            "compatible coordinate type",
        ),
        (
            [
                _evidence("A", "a-1", "time", datetime(2026, 1, 1)),
                _evidence(
                    "A",
                    "a-2",
                    "time",
                    datetime(2026, 1, 2, tzinfo=timezone.utc),
                ),
            ],
            TypeError,
            "timezone-aware and timezone-naive",
        ),
        (
            [
                _evidence("A", "a-1", "index", 1.0),
                _evidence("A", "a-2", "index", 1),
            ],
            ValueError,
            "duplicate coordinate",
        ),
    ],
)
def test_incompatible_or_duplicate_anchor_sets_raise(evidence, error, message) -> None:
    with pytest.raises(error, match=message):
        _normalize_coordinate_linear_anchors(evidence)


@pytest.mark.parametrize("value", [True, "Q1"])
def test_bool_and_string_epoch_labels_are_never_coordinates(value) -> None:
    with pytest.raises((TypeError, ValueError), match="coordinate_value"):
        _evidence("A", "a-invalid", "index", value)
