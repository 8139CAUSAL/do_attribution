"""Explicit within-epoch support-weighted reductions for CAL-057."""

from datetime import date, datetime, timezone

import pytest

from do_attribution.calibration import (
    LiftTestEvidence,
    _build_coordinate_linear_schedule,
    _reduce_support_weighted_average,
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
            _evidence("B", "b-left", 0.0, 2.0),
            _evidence("A", "a-right", 10.0, 3.0),
            _evidence("B", "b-right", 10.0, 1.0),
        ]
    )


def _request(pairs, coordinate_weights, *, support=None):
    return _validate_epoch_ratio_reduction_request(
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="support_weighted_average",
        epochs=("E1",),
        calibrated_pairs=pairs,
        epoch_windows={"E1": (0.0, 10.0)},
        epoch_coordinate_weights={"E1": coordinate_weights},
        epoch_support=support,
    )


def test_exact_weighted_identity_uses_unnormalized_coordinate_weights() -> None:
    request = _request(
        (("A", "E1"),),
        [(0.0, 1.0), (5.0, 2.0), (10.0, 1.0)],
    )
    result = _reduce_support_weighted_average(request, _schedule())
    row = result.rows[0]
    assert row.effective_ratio == 2.0
    assert row.ratio_source == "support_weighted_average"
    assert row.source_evidence_ids == ("a-left", "a-right")
    provenance = result.provenance_by_pair[("A", "E1")]
    assert provenance["total_coordinate_weight"] == 4.0
    assert provenance["weighted_ratio_sum"] == 8.0
    assert [
        point["evaluated_ratio"]
        for point in provenance["coordinate_evaluations"]
    ] == [1.0, 2.0, 3.0]
    assert [
        point["weighted_term"]
        for point in provenance["coordinate_evaluations"]
    ] == [1.0, 4.0, 3.0]


@pytest.mark.parametrize(
    "weights, expected",
    [
        ([(0.0, 1.0), (10.0, 0.0)], 1.0),
        ([(0.0, 0.0), (10.0, 5.0)], 3.0),
        ([(2.5, 3.0), (7.5, 1.0)], 1.75),
        ([(0.0, 1e100), (10.0, 1e100)], 2.0),
    ],
)
def test_weighted_property_cases_follow_declared_identity(weights, expected) -> None:
    result = _reduce_support_weighted_average(
        _request((("A", "E1"),), weights),
        _schedule(),
    )
    assert result.rows[0].effective_ratio == pytest.approx(expected)


def test_entities_use_same_epoch_weights_but_independent_schedules() -> None:
    request = _request(
        (("B", "E1"), ("A", "E1")),
        [(0.0, 1.0), (10.0, 3.0)],
    )
    result = _reduce_support_weighted_average(request, _schedule())
    assert [(row.tested_entity, row.epoch) for row in result.rows] == [
        ("A", "E1"),
        ("B", "E1"),
    ]
    assert result.rows[0].effective_ratio == 2.5
    assert result.rows[1].effective_ratio == 1.25


def test_every_coordinate_must_be_inside_observed_support_even_at_zero_weight() -> None:
    schedule = _build_coordinate_linear_schedule(
        [
            _evidence("A", "a-left", 2.0, 1.0),
            _evidence("A", "a-right", 8.0, 3.0),
        ]
    )
    request = _request(
        (("A", "E1"),),
        [(1.0, 0.0), (5.0, 1.0)],
    )
    with pytest.raises(ValueError, match="every coordinate inside observed"):
        _reduce_support_weighted_average(request, schedule)


def test_single_exact_coordinate_retains_one_anchor_without_fake_bracket() -> None:
    result = _reduce_support_weighted_average(
        _request((("A", "E1"),), [(0.0, 5.0)]),
        _schedule(),
    )
    row = result.rows[0]
    assert row.effective_ratio == 1.0
    assert row.source_evidence_ids == ("a-left",)
    assert row.left_source_evidence_id is None
    assert row.right_source_evidence_id is None


def test_date_coordinate_weights_use_declared_points() -> None:
    schedule = _build_coordinate_linear_schedule(
        [
            _evidence("A", "a-left", date(2026, 1, 1), 1.0, "date"),
            _evidence("A", "a-right", date(2026, 1, 11), 3.0, "date"),
        ]
    )
    request = _validate_epoch_ratio_reduction_request(
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="support_weighted_average",
        epochs=("E1",),
        calibrated_pairs=(("A", "E1"),),
        epoch_windows={"E1": (date(2026, 1, 1), date(2026, 1, 11))},
        epoch_coordinate_weights={
            "E1": [(date(2026, 1, 3), 1.0), (date(2026, 1, 9), 3.0)]
        },
    )
    result = _reduce_support_weighted_average(request, schedule)
    assert result.rows[0].effective_ratio == pytest.approx(2.3)


def test_timezone_aware_datetime_weights_use_elapsed_schedule_values() -> None:
    utc = timezone.utc
    schedule = _build_coordinate_linear_schedule(
        [
            _evidence("A", "a-left", datetime(2026, 1, 1, tzinfo=utc), 0.5, "datetime"),
            _evidence("A", "a-right", datetime(2026, 1, 3, tzinfo=utc), 1.5, "datetime"),
        ]
    )
    request = _validate_epoch_ratio_reduction_request(
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="support_weighted_average",
        epochs=("E1",),
        calibrated_pairs=(("A", "E1"),),
        epoch_windows={
            "E1": (
                datetime(2026, 1, 1, tzinfo=utc),
                datetime(2026, 1, 3, tzinfo=utc),
            )
        },
        epoch_coordinate_weights={
            "E1": [(datetime(2026, 1, 2, tzinfo=utc), 7.0)]
        },
    )
    result = _reduce_support_weighted_average(request, schedule)
    assert result.rows[0].effective_ratio == 1.0


@pytest.mark.parametrize(
    "weights, expected",
    [
        ([(5.0, -1.0)], "nonnegative"),
        ([(2.0, 0.0), (8.0, 0.0)], "positive total"),
        ([(11.0, 1.0)], "inside that epoch"),
    ],
)
def test_invalid_coordinate_weights_raise_before_reduction(weights, expected) -> None:
    with pytest.raises(ValueError, match=expected):
        _request((("A", "E1"),), weights)


def test_scalar_epoch_support_cannot_substitute_for_coordinate_weights() -> None:
    with pytest.raises(ValueError, match="scalar epoch_support is not a substitute"):
        _validate_epoch_ratio_reduction_request(
            ratio_policy="coordinate_linear",
            epoch_ratio_policy="support_weighted_average",
            epochs=("E1",),
            calibrated_pairs=(("A", "E1"),),
            epoch_windows={"E1": (0.0, 10.0)},
            epoch_support={"E1": 100.0},
        )


def test_scalar_support_is_preserved_but_not_used_in_weighted_identity() -> None:
    result = _reduce_support_weighted_average(
        _request(
            (("A", "E1"),),
            [(0.0, 1.0), (10.0, 1.0)],
            support={"E1": 999.0},
        ),
        _schedule(),
    )
    assert result.rows[0].effective_ratio == 2.0
    assert result.rows[0].epoch_support == 999.0
    assert result.provenance_by_pair[("A", "E1")]["epoch_support"] == 999.0


def test_unknown_entity_and_wrong_policy_request_raise() -> None:
    unknown = _request((("C", "E1"),), [(5.0, 1.0)])
    with pytest.raises(ValueError, match="without a coordinate schedule"):
        _reduce_support_weighted_average(unknown, _schedule())

    average_request = _validate_epoch_ratio_reduction_request(
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="coordinate_average",
        epochs=("E1",),
        calibrated_pairs=(("A", "E1"),),
        epoch_windows={"E1": (0.0, 10.0)},
    )
    with pytest.raises(ValueError, match="support-weighted reduction requires"):
        _reduce_support_weighted_average(average_request, _schedule())


def test_result_is_deterministic_deeply_read_only_and_input_isolated() -> None:
    weights = [(0.0, 1.0), (10.0, 3.0)]
    request = _request((("A", "E1"),), weights)
    schedule = _schedule()
    before = schedule.anchor_rows
    first = _reduce_support_weighted_average(request, schedule)
    weights[0] = (5.0, 99.0)
    second = _reduce_support_weighted_average(request, schedule)
    assert first == second
    assert schedule.anchor_rows == before
    assert first.uncalibrated_pairs == ()
    with pytest.raises(TypeError):
        first.provenance_by_pair[("A", "E1")]["coordinate_evaluations"][0][
            "weight"
        ] = 9.0
