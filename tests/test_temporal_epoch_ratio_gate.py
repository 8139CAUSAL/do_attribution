"""Final bounds and cross-policy epoch-ratio gate for CAL-058."""

from math import inf

import pytest

from do_attribution.calibration import (
    LiftTestEvidence,
    _apply_epoch_ratio_bounds,
    _build_coordinate_linear_schedule,
    _build_explicit_ratio_schedule,
    _reduce_coordinate_average,
    _reduce_direct_epoch_ratios,
    _reduce_representative_coordinate,
    _reduce_support_weighted_average,
    _validate_epoch_ratio_reduction_request,
)


def _evidence(test_id, coordinate, ratio):
    return LiftTestEvidence(
        tested_entity="A",
        observed_lift=ratio * 100.0,
        matched_mta_estimate=100.0,
        test_id=test_id,
        coordinate_name="index",
        coordinate_value=coordinate,
    )


def _coordinate_schedule():
    return _build_coordinate_linear_schedule(
        [_evidence("a-left", 0.0, 0.5), _evidence("a-right", 10.0, 1.5)]
    )


def _direct_result(ratio=1.0):
    evidence = [_evidence("a-left", 0.0, 0.5), _evidence("a-right", 10.0, 1.5)]
    schedule = _build_explicit_ratio_schedule(
        evidence,
        epochs=("E1",),
        ratio_schedule={("A", "E1"): ratio},
        allow_signed=True,
    )
    request = _validate_epoch_ratio_reduction_request(
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        epochs=("E1",),
        calibrated_pairs=(("A", "E1"),),
        epoch_windows={"E1": (0.0, 10.0)},
    )
    return _reduce_direct_epoch_ratios(request, schedule)


def _representative_result(coordinate=5.0):
    request = _validate_epoch_ratio_reduction_request(
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="representative_coordinate",
        epochs=("E1",),
        calibrated_pairs=(("A", "E1"),),
        epoch_windows={"E1": (0.0, 10.0)},
        epoch_coordinates={"E1": coordinate},
    )
    return _reduce_representative_coordinate(request, _coordinate_schedule())


def _average_result(window=(0.0, 10.0)):
    request = _validate_epoch_ratio_reduction_request(
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="coordinate_average",
        epochs=("E1",),
        calibrated_pairs=(("A", "E1"),),
        epoch_windows={"E1": window},
    )
    return _reduce_coordinate_average(request, _coordinate_schedule())


def _weighted_result(weights=((0.0, 1.0), (10.0, 1.0))):
    request = _validate_epoch_ratio_reduction_request(
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="support_weighted_average",
        epochs=("E1",),
        calibrated_pairs=(("A", "E1"),),
        epoch_windows={"E1": (0.0, 10.0)},
        epoch_coordinate_weights={"E1": weights},
    )
    return _reduce_support_weighted_average(request, _coordinate_schedule())


@pytest.mark.parametrize(
    "factory, expected_policy",
    [
        (_direct_result, "direct"),
        (_representative_result, "representative_coordinate"),
        (_average_result, "coordinate_average"),
        (_weighted_result, "support_weighted_average"),
    ],
)
def test_all_four_reductions_accept_bounds_after_final_resolution(
    factory, expected_policy
) -> None:
    original = factory()
    bounded = _apply_epoch_ratio_bounds(original, ratio_bounds=(0.8, 1.2))
    row = bounded.rows[0]
    assert row.effective_ratio == 1.0
    assert row.epoch_ratio_policy == expected_policy
    assert row.ratio_bound_lower == 0.8
    assert row.ratio_bound_upper == 1.2
    assert bounded.provenance_by_pair[("A", "E1")]["bounds_action"] == (
        "validated_without_clipping"
    )
    assert original.rows[0].ratio_bound_lower is None
    assert original.rows[0].ratio_bound_upper is None


@pytest.mark.parametrize(
    "result",
    [
        pytest.param(_direct_result(1.3), id="direct"),
        pytest.param(_representative_result(9.0), id="representative"),
        pytest.param(_average_result((5.0, 10.0)), id="coordinate-average"),
        pytest.param(
            _weighted_result(((0.0, 1.0), (10.0, 3.0))),
            id="support-weighted",
        ),
    ],
)
def test_out_of_bound_final_ratios_raise_without_clipping(result) -> None:
    before = result.rows[0].effective_ratio
    with pytest.raises(ValueError, match="outside declared raise-only bounds"):
        _apply_epoch_ratio_bounds(result, ratio_bounds=(0.8, 1.2))
    assert result.rows[0].effective_ratio == before
    assert result.rows[0].ratio_bound_lower is None


def test_bound_error_identifies_pair_policies_source_ids_and_limits() -> None:
    with pytest.raises(ValueError) as exc_info:
        _apply_epoch_ratio_bounds(_direct_result(1.3), ratio_bounds=(0.8, 1.2))
    message = str(exc_info.value)
    for fragment in (
        "tested_entity='A'",
        "epoch='E1'",
        "effective_ratio=1.3",
        "lower=0.8",
        "upper=1.2",
        "ratio_policy='explicit_ratio'",
        "epoch_ratio_policy='direct'",
        "ratio_source='explicit'",
        "source_evidence_ids=()",
    ):
        assert fragment in message


@pytest.mark.parametrize(
    "bounds, expected",
    [
        ([0.0, 1.0], "two-item tuple"),
        ((2.0, 1.0), "lower <= upper"),
        ((0.0, inf), "finite"),
        ((False, 1.0), "real number"),
    ],
)
def test_invalid_bounds_raise(bounds, expected) -> None:
    with pytest.raises((TypeError, ValueError), match=expected):
        _apply_epoch_ratio_bounds(_direct_result(), ratio_bounds=bounds)


@pytest.mark.parametrize(
    "ratio, bounds",
    [
        (0.8, (0.8, 1.2)),
        (1.2, (0.8, 1.2)),
        (0.0, (0.0, 0.0)),
        (-1.0, (-1.0, 2.0)),
    ],
)
def test_inclusive_bound_property_cases_preserve_exact_ratio(ratio, bounds) -> None:
    result = _direct_result(ratio)
    bounded = _apply_epoch_ratio_bounds(result, ratio_bounds=bounds)
    assert bounded.rows[0].effective_ratio == ratio


def test_none_bounds_return_same_immutable_result_without_reconstruction() -> None:
    result = _representative_result()
    assert _apply_epoch_ratio_bounds(result, ratio_bounds=None) is result


def test_unsupported_pairs_remain_unbounded_and_authoritative_grid_is_preserved() -> None:
    request = _validate_epoch_ratio_reduction_request(
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="representative_coordinate",
        epochs=("E1", "E2"),
        calibrated_pairs=(("A", "E1"), ("A", "E2")),
        epoch_windows={"E1": (-2.0, 0.0), "E2": (0.0, 10.0)},
        epoch_coordinates={"E1": -1.0, "E2": 5.0},
    )
    original = _reduce_representative_coordinate(request, _coordinate_schedule())
    bounded = _apply_epoch_ratio_bounds(original, ratio_bounds=(0.0, 2.0))
    assert bounded.uncalibrated_pairs == (("A", "E1"),)
    assert bounded.provenance_by_pair[("A", "E1")]["effective_ratio"] is None
    assert "ratio_bound_lower" not in bounded.provenance_by_pair[("A", "E1")]
    assert bounded.rows[0].epoch_start == 0.0
    assert bounded.rows[0].epoch_end == 10.0
    assert bounded.rows[0].epoch == "E2"


def test_bounded_result_keeps_exact_one_row_per_pair_and_complete_provenance() -> None:
    bounded = _apply_epoch_ratio_bounds(
        _weighted_result(((0.0, 1.0), (10.0, 1.0))),
        ratio_bounds=(0.5, 1.5),
    )
    assert len(bounded.rows) == 1
    assert (bounded.rows[0].tested_entity, bounded.rows[0].epoch) == ("A", "E1")
    provenance = bounded.provenance_by_pair[("A", "E1")]
    assert provenance["coordinate_evaluations"]
    assert provenance["ratio_bound_lower"] == 0.5
    assert provenance["ratio_bound_upper"] == 1.5


def test_bounds_result_is_deterministic_and_deeply_read_only() -> None:
    original = _average_result()
    first = _apply_epoch_ratio_bounds(original, ratio_bounds=(0.5, 1.5))
    second = _apply_epoch_ratio_bounds(original, ratio_bounds=(0.5, 1.5))
    assert first == second
    with pytest.raises(TypeError):
        first.provenance_by_pair[("A", "E1")]["ratio_bound_lower"] = 0.0
