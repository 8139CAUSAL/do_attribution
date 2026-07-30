"""Uniform piecewise-linear epoch interval averages for CAL-056."""

from datetime import date, datetime, timezone

import pytest

from do_attribution.calibration import (
    LiftTestEvidence,
    _build_coordinate_linear_schedule,
    _reduce_coordinate_average,
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


def _schedule():
    return _build_coordinate_linear_schedule(
        [
            _evidence("A", "a-left", 0.0, 1.0),
            _evidence("B", "b-left", 0.0, 0.5),
            _evidence("A", "a-middle", 5.0, 3.0),
            _evidence("B", "b-middle", 5.0, 1.0),
            _evidence("A", "a-right", 10.0, 1.0),
            _evidence("B", "b-right", 10.0, 1.5),
        ]
    )


def _request(pairs, windows, *, support=None):
    return _validate_epoch_ratio_reduction_request(
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="coordinate_average",
        epochs=tuple(windows),
        calibrated_pairs=pairs,
        epoch_windows=windows,
        epoch_support=support,
    )


def test_single_segment_uniform_average_matches_endpoint_mean() -> None:
    schedule = _build_coordinate_linear_schedule(
        [
            _evidence("A", "a-left", 0.0, 0.8),
            _evidence("A", "a-right", 10.0, 1.2),
        ]
    )
    request = _request((("A", "E1"),), {"E1": (2.0, 6.0)})
    result = _reduce_coordinate_average(request, schedule)
    assert result.rows[0].effective_ratio == pytest.approx(0.96)
    assert result.rows[0].source_evidence_ids == ("a-left", "a-right")
    assert len(
        result.provenance_by_pair[("A", "E1")]["contributing_segments"]
    ) == 1


def test_multi_segment_hand_calculation_integrates_every_crossed_segment() -> None:
    request = _request((("A", "E1"),), {"E1": (2.0, 8.0)})
    result = _reduce_coordinate_average(request, _schedule())
    row = result.rows[0]
    assert row.effective_ratio == pytest.approx(2.4)
    assert row.source_evidence_ids == ("a-left", "a-middle", "a-right")
    assert row.left_source_evidence_id == "a-left"
    assert row.right_source_evidence_id == "a-right"
    segments = result.provenance_by_pair[("A", "E1")][
        "contributing_segments"
    ]
    assert len(segments) == 2
    assert [segment["normalized_area"] for segment in segments] == pytest.approx(
        [7.2, 7.2]
    )
    assert sum(segment["normalized_area"] for segment in segments) / 6.0 == (
        pytest.approx(2.4)
    )


def test_complete_support_average_uses_the_full_authoritative_interval() -> None:
    request = _request((("A", "E1"),), {"E1": (0.0, 10.0)})
    result = _reduce_coordinate_average(request, _schedule())
    assert result.rows[0].effective_ratio == 2.0
    provenance = result.provenance_by_pair[("A", "E1")]
    assert provenance["interval_start"] == 0.0
    assert provenance["interval_end"] == 10.0
    assert provenance["normalized_interval_width"] == 10.0
    assert provenance["evaluation_rule"] == (
        "uniform_piecewise_linear_interval_average"
    )


@pytest.mark.parametrize("window", [(-1.0, 8.0), (2.0, 11.0)])
def test_partial_observed_support_raises_without_silent_partial_average(window) -> None:
    request = _request((("A", "E1"),), {"E1": window})
    with pytest.raises(ValueError, match="partial support is not averaged"):
        _reduce_coordinate_average(request, _schedule())


def test_entities_average_independently_on_the_same_authoritative_window() -> None:
    request = _request(
        (("B", "E1"), ("A", "E1")),
        {"E1": (2.0, 8.0)},
    )
    result = _reduce_coordinate_average(request, _schedule())
    assert [(row.tested_entity, row.epoch) for row in result.rows] == [
        ("A", "E1"),
        ("B", "E1"),
    ]
    assert result.rows[0].effective_ratio == pytest.approx(2.4)
    assert result.rows[1].effective_ratio == pytest.approx(1.0)


def test_date_multi_segment_average_uses_day_axis() -> None:
    schedule = _build_coordinate_linear_schedule(
        [
            _evidence("A", "a-left", date(2026, 1, 1), 1.0, "date"),
            _evidence("A", "a-middle", date(2026, 1, 6), 3.0, "date"),
            _evidence("A", "a-right", date(2026, 1, 11), 1.0, "date"),
        ]
    )
    request = _request(
        (("A", "E1"),),
        {"E1": (date(2026, 1, 3), date(2026, 1, 9))},
    )
    result = _reduce_coordinate_average(request, schedule)
    assert result.rows[0].effective_ratio == pytest.approx(2.4)
    assert result.provenance_by_pair[("A", "E1")]["coordinate_kind"] == "date"


def test_timezone_aware_datetime_average_uses_elapsed_seconds() -> None:
    utc = timezone.utc
    schedule = _build_coordinate_linear_schedule(
        [
            _evidence("A", "a-left", datetime(2026, 1, 1, tzinfo=utc), 1.0, "datetime"),
            _evidence("A", "a-middle", datetime(2026, 1, 2, tzinfo=utc), 3.0, "datetime"),
            _evidence("A", "a-right", datetime(2026, 1, 3, tzinfo=utc), 1.0, "datetime"),
        ]
    )
    request = _request(
        (("A", "E1"),),
        {
            "E1": (
                datetime(2026, 1, 1, 12, tzinfo=utc),
                datetime(2026, 1, 2, 12, tzinfo=utc),
            )
        },
    )
    result = _reduce_coordinate_average(request, schedule)
    assert result.rows[0].effective_ratio == pytest.approx(2.5)
    assert result.provenance_by_pair[("A", "E1")]["coordinate_kind"] == (
        "datetime"
    )


def test_average_preserves_optional_scalar_support_as_diagnostic_only() -> None:
    request = _request(
        (("A", "E1"),),
        {"E1": (2.0, 8.0)},
        support={"E1": 123.0},
    )
    result = _reduce_coordinate_average(request, _schedule())
    assert result.rows[0].epoch_support == 123.0
    assert result.rows[0].epoch_coordinate is None
    assert result.provenance_by_pair[("A", "E1")]["epoch_support"] == 123.0


def test_unknown_entity_and_wrong_policy_request_raise() -> None:
    unknown_request = _request((("C", "E1"),), {"E1": (2.0, 8.0)})
    with pytest.raises(ValueError, match="without a coordinate schedule"):
        _reduce_coordinate_average(unknown_request, _schedule())

    representative_request = _validate_epoch_ratio_reduction_request(
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="representative_coordinate",
        epochs=("E1",),
        calibrated_pairs=(("A", "E1"),),
        epoch_windows={"E1": (2.0, 8.0)},
        epoch_coordinates={"E1": 5.0},
    )
    with pytest.raises(ValueError, match="coordinate-average reduction requires"):
        _reduce_coordinate_average(representative_request, _schedule())


def test_average_result_is_deterministic_deeply_read_only_and_does_not_mutate() -> None:
    schedule = _schedule()
    before = schedule.anchor_rows
    request = _request((("A", "E1"),), {"E1": (2.0, 8.0)})
    first = _reduce_coordinate_average(request, schedule)
    second = _reduce_coordinate_average(request, schedule)
    assert first == second
    assert schedule.anchor_rows == before
    assert first.uncalibrated_pairs == ()
    with pytest.raises(TypeError):
        first.provenance_by_pair[("A", "E1")]["interval_start"] = 3.0
