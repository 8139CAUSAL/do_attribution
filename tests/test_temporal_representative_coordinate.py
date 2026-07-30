"""Representative-coordinate epoch reduction for CAL-055."""

from datetime import date, datetime, timezone

import pytest

from do_attribution.calibration import (
    LiftTestEvidence,
    _build_coordinate_linear_schedule,
    _reduce_representative_coordinate,
    _validate_epoch_ratio_reduction_request,
)


def _evidence(entity, test_id, coordinate, ratio, coordinate_name="index"):
    return LiftTestEvidence(
        tested_entity=entity,
        observed_lift=ratio * 100.0,
        matched_mta_estimate=100.0,
        test_id=test_id,
        coordinate_name=coordinate_name,
        coordinate_value=coordinate,
    )


def _numeric_schedule():
    return _build_coordinate_linear_schedule(
        [
            _evidence("A", "a-left", 2.0, 0.8),
            _evidence("B", "b-left", 2.0, 1.0),
            _evidence("A", "a-right", 8.0, 1.2),
            _evidence("B", "b-right", 8.0, 0.5),
        ]
    )


def _numeric_request(pairs, coordinates, *, support=None):
    return _validate_epoch_ratio_reduction_request(
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="representative_coordinate",
        epochs=("E1", "E2"),
        calibrated_pairs=pairs,
        epoch_windows={"E1": (0.0, 5.0), "E2": (5.0, 10.0)},
        epoch_coordinates=coordinates,
        epoch_support=support,
    )


def test_exact_and_interpolated_representative_coordinates_emit_exact_rows() -> None:
    request = _numeric_request(
        (("A", "E1"), ("A", "E2")),
        {"E1": 2.0, "E2": 7.0},
    )
    result = _reduce_representative_coordinate(request, _numeric_schedule())
    exact, interpolated = result.rows
    assert exact.effective_ratio == 0.8
    assert exact.ratio_source == "observed"
    assert exact.source_evidence_ids == ("a-left",)
    assert exact.epoch_coordinate == 2.0
    assert interpolated.effective_ratio == pytest.approx(1.1333333333333333)
    assert interpolated.ratio_source == "interpolated"
    assert interpolated.source_evidence_ids == ("a-left", "a-right")
    assert interpolated.left_source_evidence_id == "a-left"
    assert interpolated.right_source_evidence_id == "a-right"


def test_outside_support_coordinate_emits_no_row_and_never_extrapolates() -> None:
    request = _numeric_request(
        (("A", "E1"), ("A", "E2")),
        {"E1": 1.0, "E2": 7.0},
    )
    result = _reduce_representative_coordinate(request, _numeric_schedule())
    assert [(row.tested_entity, row.epoch) for row in result.rows] == [
        ("A", "E2")
    ]
    assert result.uncalibrated_pairs == (("A", "E1"),)
    unsupported = result.provenance_by_pair[("A", "E1")]
    assert unsupported["ratio_source"] == "unsupported"
    assert unsupported["effective_ratio"] is None
    assert unsupported["source_evidence_ids"] == ()
    assert unsupported["inside_observed_support"] is False


def test_entities_use_independent_schedules_at_the_same_epoch_coordinate() -> None:
    request = _numeric_request(
        (("B", "E2"), ("A", "E2")),
        {"E2": 7.0},
    )
    result = _reduce_representative_coordinate(request, _numeric_schedule())
    assert [(row.tested_entity, row.epoch) for row in result.rows] == [
        ("A", "E2"),
        ("B", "E2"),
    ]
    assert result.rows[0].effective_ratio == pytest.approx(1.1333333333333333)
    assert result.rows[1].effective_ratio == pytest.approx(0.5833333333333334)


def test_representative_provenance_records_declared_coordinate_rule_and_support() -> None:
    request = _numeric_request(
        (("A", "E2"),),
        {"E2": 7.0},
        support={"E2": 42.0},
    )
    result = _reduce_representative_coordinate(request, _numeric_schedule())
    row = result.rows[0]
    provenance = result.provenance_by_pair[("A", "E2")]
    assert row.epoch_start == 5.0
    assert row.epoch_end == 10.0
    assert row.epoch_support == 42.0
    assert provenance["evaluation_rule"] == (
        "schedule_at_declared_epoch_coordinate"
    )
    assert provenance["epoch_coordinate"] == 7.0
    assert provenance["left_source_evidence_id"] == "a-left"
    assert provenance["right_source_evidence_id"] == "a-right"


def test_date_coordinate_identity_uses_declared_dates() -> None:
    schedule = _build_coordinate_linear_schedule(
        [
            _evidence("A", "a-left", date(2026, 1, 2), 0.8, "date"),
            _evidence("A", "a-right", date(2026, 1, 10), 1.2, "date"),
        ]
    )
    request = _validate_epoch_ratio_reduction_request(
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="representative_coordinate",
        epochs=("E1", "E2"),
        calibrated_pairs=(("A", "E1"), ("A", "E2")),
        epoch_windows={
            "E1": (date(2026, 1, 1), date(2026, 1, 6)),
            "E2": (date(2026, 1, 6), date(2026, 1, 12)),
        },
        epoch_coordinates={
            "E1": date(2026, 1, 4),
            "E2": date(2026, 1, 8),
        },
    )
    result = _reduce_representative_coordinate(request, schedule)
    assert [row.effective_ratio for row in result.rows] == pytest.approx([0.9, 1.1])
    assert result.rows[0].epoch_coordinate == date(2026, 1, 4)


def test_timezone_aware_datetime_coordinate_identity() -> None:
    left = datetime(2026, 1, 1, tzinfo=timezone.utc)
    middle = datetime(2026, 1, 2, tzinfo=timezone.utc)
    right = datetime(2026, 1, 3, tzinfo=timezone.utc)
    schedule = _build_coordinate_linear_schedule(
        [
            _evidence("A", "a-left", left, 0.5, "datetime"),
            _evidence("A", "a-right", right, 1.5, "datetime"),
        ]
    )
    request = _validate_epoch_ratio_reduction_request(
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="representative_coordinate",
        epochs=("E1",),
        calibrated_pairs=(("A", "E1"),),
        epoch_windows={
            "E1": (
                datetime(2025, 12, 31, tzinfo=timezone.utc),
                datetime(2026, 1, 4, tzinfo=timezone.utc),
            )
        },
        epoch_coordinates={"E1": middle},
    )
    result = _reduce_representative_coordinate(request, schedule)
    assert result.rows[0].effective_ratio == 1.0
    assert result.rows[0].epoch_coordinate == middle


def test_unknown_schedule_entity_and_wrong_policy_request_raise() -> None:
    request = _numeric_request((("C", "E2"),), {"E2": 7.0})
    with pytest.raises(ValueError, match="without a coordinate schedule"):
        _reduce_representative_coordinate(request, _numeric_schedule())

    direct_request = _validate_epoch_ratio_reduction_request(
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        epochs=("E1", "E2"),
        calibrated_pairs=(),
    )
    with pytest.raises(ValueError, match="representative reduction requires"):
        _reduce_representative_coordinate(direct_request, _numeric_schedule())


def test_result_is_deterministic_deeply_read_only_and_does_not_change_schedule() -> None:
    schedule = _numeric_schedule()
    anchors_before = schedule.anchor_rows
    request = _numeric_request((("A", "E2"),), {"E2": 7.0})
    first = _reduce_representative_coordinate(request, schedule)
    second = _reduce_representative_coordinate(request, schedule)
    assert first == second
    assert schedule.anchor_rows == anchors_before
    with pytest.raises(TypeError):
        first.provenance_by_pair[("A", "E2")]["epoch_coordinate"] = 9.0
