"""Piecewise-linear schedule construction and evaluation for CAL-046."""

from datetime import date, datetime, timedelta
import inspect

import pytest

from do_attribution.calibration import (
    LiftTestEvidence,
    _build_coordinate_linear_schedule,
    _evaluate_coordinate_linear_schedule,
)


def _evidence(entity, test_id, name, value, ratio):
    return LiftTestEvidence(
        tested_entity=entity,
        observed_lift=ratio * 100.0,
        matched_mta_estimate=100.0,
        test_id=test_id,
        coordinate_name=name,
        coordinate_value=value,
    )


def test_builder_sorts_anchors_and_reproduces_them_exactly() -> None:
    schedule = _build_coordinate_linear_schedule(
        [
            _evidence("A", "a-right", "index", 10.0, 1.2),
            _evidence("A", "a-left", "index", 0.0, 0.8),
        ]
    )
    assert [row.source_evidence_ids for row in schedule.anchor_rows] == [
        ("a-left",),
        ("a-right",),
    ]
    assert [row.coordinate_value for row in schedule.anchor_rows] == [0.0, 10.0]
    assert [row.schedule_ratio for row in schedule.anchor_rows] == [0.8, 1.2]

    left = _evaluate_coordinate_linear_schedule(
        schedule, tested_entity="A", coordinate_value=0.0
    )
    right = _evaluate_coordinate_linear_schedule(
        schedule, tested_entity="A", coordinate_value=10.0
    )
    assert left.schedule_source == right.schedule_source == "observed"
    assert left.schedule_ratio == 0.8
    assert right.schedule_ratio == 1.2
    assert left.source_evidence_ids == ("a-left",)
    assert right.source_evidence_ids == ("a-right",)


@pytest.mark.parametrize(
    ("left_ratio", "right_ratio", "expected"),
    [(0.8, 1.2, 1.0), (1.2, 0.8, 1.0)],
)
def test_interpolation_uses_nearest_brackets_and_endpoint_containment(
    left_ratio, right_ratio, expected
) -> None:
    schedule = _build_coordinate_linear_schedule(
        [
            _evidence("A", "a-left", "index", 0.0, left_ratio),
            _evidence("A", "a-mid", "index", 10.0, right_ratio),
            _evidence("A", "a-right", "index", 20.0, 1.5),
        ]
    )
    row = _evaluate_coordinate_linear_schedule(
        schedule, tested_entity="A", coordinate_value=5.0
    )
    assert row.schedule_source == "interpolated"
    assert row.schedule_ratio == pytest.approx(expected)
    assert row.source_evidence_ids == ("a-left", "a-mid")
    assert row.left_source_evidence_id == "a-left"
    assert row.right_source_evidence_id == "a-mid"
    assert min(left_ratio, right_ratio) <= row.schedule_ratio <= max(left_ratio, right_ratio)


def test_no_extrapolation_emits_explicit_unsupported_rows() -> None:
    schedule = _build_coordinate_linear_schedule(
        [
            _evidence("A", "a-left", "index", 0.0, 0.8),
            _evidence("A", "a-right", "index", 10.0, 1.2),
        ]
    )
    for coordinate in (-1.0, 11.0):
        row = _evaluate_coordinate_linear_schedule(
            schedule, tested_entity="A", coordinate_value=coordinate
        )
        assert row.schedule_source == "unsupported"
        assert row.schedule_ratio is None
        assert row.source_evidence_ids == ()
        assert not row.inside_observed_support


def test_date_and_datetime_interpolation_use_normalized_arithmetic() -> None:
    date_schedule = _build_coordinate_linear_schedule(
        [
            _evidence("A", "a-left", "date", date(2026, 1, 1), 1.0),
            _evidence("A", "a-right", "date", date(2026, 1, 11), 2.0),
        ]
    )
    date_row = _evaluate_coordinate_linear_schedule(
        date_schedule,
        tested_entity="A",
        coordinate_value=date(2026, 1, 6),
    )
    assert date_row.schedule_ratio == pytest.approx(1.5)
    assert date_row.coordinate_value == date(2026, 1, 6)

    start = datetime(2026, 1, 1, 0, 0)
    datetime_schedule = _build_coordinate_linear_schedule(
        [
            _evidence("A", "a-left", "timestamp", start, 0.5),
            _evidence("A", "a-right", "timestamp", start + timedelta(hours=2), 1.5),
        ]
    )
    datetime_row = _evaluate_coordinate_linear_schedule(
        datetime_schedule,
        tested_entity="A",
        coordinate_value=start + timedelta(hours=1),
    )
    assert datetime_row.schedule_ratio == pytest.approx(1.0)


def test_entities_have_independent_schedules_and_axis_validation() -> None:
    schedule = _build_coordinate_linear_schedule(
        [
            _evidence("B", "b-left", "date", date(2026, 1, 1), 1.0),
            _evidence("A", "a-left", "index", 0.0, 0.5),
            _evidence("B", "b-right", "date", date(2026, 1, 3), 2.0),
            _evidence("A", "a-right", "index", 10.0, 1.5),
        ]
    )
    assert tuple(schedule.normalized_anchors_by_entity) == ("B", "A")
    assert _evaluate_coordinate_linear_schedule(
        schedule, tested_entity="A", coordinate_value=5.0
    ).schedule_ratio == pytest.approx(1.0)
    assert _evaluate_coordinate_linear_schedule(
        schedule, tested_entity="B", coordinate_value=date(2026, 1, 2)
    ).schedule_ratio == pytest.approx(1.5)
    with pytest.raises(TypeError, match="coordinate type"):
        _evaluate_coordinate_linear_schedule(
            schedule, tested_entity="A", coordinate_value=date(2026, 1, 2)
        )
    with pytest.raises(ValueError, match="no tested entity"):
        _evaluate_coordinate_linear_schedule(
            schedule, tested_entity="C", coordinate_value=5.0
        )


def test_schedule_is_immutable_compact_and_independent_of_epoch_design() -> None:
    evidence = [
        _evidence("A", "a-left", "index", 0.0, 0.8),
        _evidence("A", "a-right", "index", 10.0, 1.2),
    ]
    schedule = _build_coordinate_linear_schedule(evidence)
    evidence.reverse()

    assert len(schedule.anchor_rows) == 2
    assert "epoch" not in inspect.signature(_build_coordinate_linear_schedule).parameters
    with pytest.raises(TypeError):
        schedule.normalized_anchors_by_entity["A"] = ()


def test_signed_anchor_ratios_require_explicit_permission() -> None:
    evidence = [
        _evidence("A", "a-left", "index", 0.0, -0.5),
        _evidence("A", "a-right", "index", 10.0, 0.5),
    ]
    with pytest.raises(ValueError, match="allow_signed=True"):
        _build_coordinate_linear_schedule(evidence)
    schedule = _build_coordinate_linear_schedule(evidence, allow_signed=True)
    assert _evaluate_coordinate_linear_schedule(
        schedule, tested_entity="A", coordinate_value=5.0
    ).schedule_ratio == pytest.approx(0.0)
