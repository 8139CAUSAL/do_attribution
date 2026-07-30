"""Epoch-ratio dispatch and exact output coverage for CAL-053."""

import pytest

from do_attribution.calibration import (
    EpochRatioRow,
    _validate_epoch_ratio_reduction_output,
    _validate_epoch_ratio_reduction_request,
)


EPOCHS = ("E1", "E2")
WINDOWS = {"E1": (0.0, 10.0), "E2": (10.0, 20.0)}
PAIRS = (("A", "E1"), ("B", "E2"))


def _request(
    ratio_policy="explicit_ratio",
    epoch_ratio_policy="direct",
    **overrides,
):
    values = {
        "ratio_policy": ratio_policy,
        "epoch_ratio_policy": epoch_ratio_policy,
        "epochs": EPOCHS,
        "calibrated_pairs": PAIRS,
        "epoch_windows": WINDOWS,
    }
    values.update(overrides)
    return _validate_epoch_ratio_reduction_request(**values)


def _row(
    entity,
    epoch,
    *,
    ratio_policy="explicit_ratio",
    epoch_ratio_policy="direct",
):
    start, end = WINDOWS[epoch]
    return EpochRatioRow(
        tested_entity=entity,
        epoch=epoch,
        epoch_start=start,
        epoch_end=end,
        epoch_coordinate=None,
        epoch_support=None,
        effective_ratio=0.9,
        ratio_policy=ratio_policy,
        epoch_ratio_policy=epoch_ratio_policy,
        ratio_source="explicit",
        evidence_conflict_policy=None,
        source_evidence_ids=(),
        evidence_weights=None,
        left_source_evidence_id=None,
        right_source_evidence_id=None,
        inside_observed_support=True,
        ratio_bound_lower=None,
        ratio_bound_upper=None,
    )


@pytest.mark.parametrize(
    "ratio_policy",
    ["observed_only", "constant", "explicit_ratio", "explicit_strength"],
)
def test_all_direct_ratio_policies_accept_only_direct_reduction(ratio_policy) -> None:
    request = _request(ratio_policy=ratio_policy)
    assert request.ratio_policy == ratio_policy
    assert request.epoch_ratio_policy == "direct"
    with pytest.raises(ValueError, match="requires epoch_ratio_policy='direct'"):
        _request(
            ratio_policy=ratio_policy,
            epoch_ratio_policy="representative_coordinate",
            epoch_coordinates={"E1": 5.0, "E2": 15.0},
        )


@pytest.mark.parametrize(
    "epoch_ratio_policy, metadata",
    [
        (
            "representative_coordinate",
            {"epoch_coordinates": {"E1": 5.0, "E2": 15.0}},
        ),
        ("coordinate_average", {}),
        (
            "support_weighted_average",
            {
                "epoch_coordinate_weights": {
                    "E1": [(2.0, 1.0)],
                    "E2": [(12.0, 1.0)],
                }
            },
        ),
    ],
)
def test_coordinate_linear_accepts_exactly_three_coordinate_reductions(
    epoch_ratio_policy, metadata
) -> None:
    request = _request(
        ratio_policy="coordinate_linear",
        epoch_ratio_policy=epoch_ratio_policy,
        **metadata,
    )
    assert request.epoch_ratio_policy == epoch_ratio_policy
    with pytest.raises(ValueError, match="incompatible"):
        _request(ratio_policy="coordinate_linear", epoch_ratio_policy="direct")


@pytest.mark.parametrize(
    "name, value",
    [
        ("ratio_policy", "spline"),
        ("epoch_ratio_policy", "median"),
        ("epoch_ratio_policy", None),
    ],
)
def test_unknown_or_omitted_reduction_policy_raises(name, value) -> None:
    values = {name: value}
    with pytest.raises(ValueError, match=name):
        _request(**values)


def test_representative_coordinate_requires_complete_affected_metadata() -> None:
    with pytest.raises(ValueError, match="requires epoch_coordinates"):
        _request(
            ratio_policy="coordinate_linear",
            epoch_ratio_policy="representative_coordinate",
        )
    with pytest.raises(ValueError, match="missing"):
        _request(
            ratio_policy="coordinate_linear",
            epoch_ratio_policy="representative_coordinate",
            epoch_coordinates={"E1": 5.0},
        )
    with pytest.raises(ValueError, match="incompatible"):
        _request(
            ratio_policy="coordinate_linear",
            epoch_ratio_policy="representative_coordinate",
            epoch_coordinates={"E1": 5.0, "E2": 15.0},
            epoch_coordinate_weights={
                "E1": [(2.0, 1.0)],
                "E2": [(12.0, 1.0)],
            },
        )


def test_average_and_weighted_metadata_are_policy_specific() -> None:
    with pytest.raises(ValueError, match="does not accept"):
        _request(
            ratio_policy="coordinate_linear",
            epoch_ratio_policy="coordinate_average",
            epoch_coordinates={"E1": 5.0, "E2": 15.0},
        )
    with pytest.raises(ValueError, match="requires epoch_coordinate_weights"):
        _request(
            ratio_policy="coordinate_linear",
            epoch_ratio_policy="support_weighted_average",
            epoch_support={"E1": 100.0, "E2": 200.0},
        )
    with pytest.raises(ValueError, match="incompatible"):
        _request(
            ratio_policy="coordinate_linear",
            epoch_ratio_policy="support_weighted_average",
            epoch_coordinates={"E1": 5.0, "E2": 15.0},
            epoch_coordinate_weights={
                "E1": [(2.0, 1.0)],
                "E2": [(12.0, 1.0)],
            },
        )


def test_direct_rejects_coordinate_reduction_metadata() -> None:
    with pytest.raises(ValueError, match="does not accept"):
        _request(epoch_coordinates={"E1": 5.0, "E2": 15.0})
    with pytest.raises(ValueError, match="does not accept"):
        _request(
            epoch_coordinate_weights={
                "E1": [(2.0, 1.0)],
                "E2": [(12.0, 1.0)],
            }
        )


@pytest.mark.parametrize(
    "pairs, expected",
    [
        (("A", "E1"), "two-item"),
        ((("A", "E3"),), "unknown epoch"),
        ((("A", "E1"), ("A", "E1")), "duplicate pair"),
    ],
)
def test_calibrated_pairs_must_be_unique_valid_pairs(pairs, expected) -> None:
    with pytest.raises((TypeError, ValueError), match=expected):
        _request(calibrated_pairs=pairs)


def test_dispatch_normalizes_authoritative_metadata_immutably() -> None:
    windows = dict(WINDOWS)
    coordinates = {"E1": 5.0, "E2": 15.0}
    support = {"E1": 11.0, "E2": 22.0}
    request = _request(
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="representative_coordinate",
        epoch_windows=windows,
        epoch_coordinates=coordinates,
        epoch_support=support,
    )
    windows["E1"] = (100.0, 200.0)
    coordinates["E1"] = 9.0
    support["E1"] = 99.0
    assert request.epoch_windows["E1"] == (0.0, 10.0)
    assert request.epoch_coordinates["E1"] == 5.0
    assert request.epoch_support["E1"] == 11.0
    with pytest.raises(TypeError):
        request.epoch_coordinates["E1"] = 6.0


def test_output_requires_one_matching_row_per_calibrated_pair() -> None:
    request = _request()
    rows = (_row("A", "E1"), _row("B", "E2"))
    assert _validate_epoch_ratio_reduction_output(request, rows) == rows

    with pytest.raises(ValueError, match="multiple rows"):
        _validate_epoch_ratio_reduction_output(
            request,
            (rows[0], rows[0], rows[1]),
        )
    with pytest.raises(ValueError, match="missing"):
        _validate_epoch_ratio_reduction_output(request, (rows[0],))
    with pytest.raises(ValueError, match="extra"):
        _validate_epoch_ratio_reduction_output(
            request,
            rows + (_row("C", "E1"),),
        )


def test_output_policy_fields_must_match_dispatch_request() -> None:
    request = _request()
    with pytest.raises(ValueError, match="ratio_policy must match"):
        _validate_epoch_ratio_reduction_output(
            request,
            (
                _row("A", "E1", ratio_policy="constant"),
                _row("B", "E2"),
            ),
        )


def test_empty_calibrated_pair_set_accepts_only_empty_output() -> None:
    request = _request(calibrated_pairs=())
    assert _validate_epoch_ratio_reduction_output(request, ()) == ()
    with pytest.raises(ValueError, match="extra"):
        _validate_epoch_ratio_reduction_output(
            request,
            (_row("A", "E1"),),
        )
