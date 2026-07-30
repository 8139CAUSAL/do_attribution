"""Exact epoch-local temporal custom-weight allocation for CAL-062."""

import math

import pytest

import do_attribution.calibration as calibration
from do_attribution.calibration import EpochRatioRow, LiftTestEvidence


EPOCHS = ("E1", "E2", "E3")
ESTIMATES = {
    "E1": {"A": 100.0, "B": 30.0, "C": 70.0},
    "E2": {"A": 100.0, "B": 80.0, "C": 20.0},
    "E3": {"A": 100.0, "B": 40.0, "C": 60.0},
}


def _evidence():
    item = LiftTestEvidence(
        tested_entity="A",
        observed_lift=80.0,
        matched_mta_estimate=100.0,
        test_id="a-test",
    )
    return {"A": (item,)}


def _ratio(epoch, ratio=0.8):
    return EpochRatioRow(
        tested_entity="A",
        epoch=epoch,
        epoch_start=None,
        epoch_end=None,
        epoch_coordinate=None,
        epoch_support=None,
        effective_ratio=ratio,
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
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


def _direct(*, rows=None):
    return calibration._temporal_direct_state(
        ESTIMATES,
        epoch_ratio_rows=(
            (_ratio("E2"), _ratio("E1")) if rows is None else rows
        ),
        evidence_by_entity=_evidence(),
        allow_signed=False,
        tolerance=1e-9,
    )


def _request(**overrides):
    values = {
        "epochs": EPOCHS,
        "calibrated_epochs": ("E2", "E1"),
        "residual_policy": "custom_weights",
        "residual_weights_by_epoch": {
            "E1": {"B": 0.25, "C": 0.75},
            "E2": {"C": 0.2, "B": 0.8},
        },
    }
    values.update(overrides)
    return calibration._normalize_temporal_residual_request(**values)


def _result(*, state=None, request=None):
    state = _direct() if state is None else state
    request = _request() if request is None else request
    allocations = calibration._allocate_temporal_custom_weights(
        state,
        request,
        allow_signed=False,
        tolerance=1e-9,
    )
    return calibration._finalize_temporal_residual_allocations(
        state,
        request,
        allocations,
        allow_signed=False,
        tolerance=1e-9,
    )


def test_each_epoch_uses_only_its_exact_declared_weights_and_order() -> None:
    result = _result()

    assert result.allocations_by_epoch["E1"].adjustments == (
        ("B", 5.0),
        ("C", 15.0),
    )
    assert result.allocations_by_epoch["E2"].adjustments == (
        ("C", 4.0),
        ("B", 16.0),
    )
    assert result.calibrated_values_by_epoch["E1"] == (
        ("A", 80.0),
        ("B", 35.0),
        ("C", 85.0),
    )
    assert result.calibrated_values_by_epoch["E2"] == (
        ("A", 80.0),
        ("B", 96.0),
        ("C", 24.0),
    )


def test_each_custom_allocation_sums_to_its_epoch_residual_and_closes_total() -> None:
    result = _result()

    for epoch in ("E1", "E2"):
        allocation = result.allocations_by_epoch[epoch]
        assert sum(value for _, value in allocation.adjustments) == pytest.approx(
            result.required_residuals[epoch]
        )
        assert result.allocated_residuals[epoch] == pytest.approx(20.0)
        assert result.calibrated_totals[epoch] == result.original_totals[epoch]


def test_global_weights_repeat_only_when_custom_policy_is_uniform() -> None:
    request = _request(
        residual_weights_by_epoch=None,
        residual_weights={"B": 0.25, "C": 0.75},
    )
    result = _result(request=request)

    assert result.allocations_by_epoch["E1"].adjustments == (
        ("B", 5.0),
        ("C", 15.0),
    )
    assert result.allocations_by_epoch["E2"].adjustments == (
        ("B", 5.0),
        ("C", 15.0),
    )


@pytest.mark.parametrize(
    "weights, expected",
    [
        ({"E1": {"B": 1.0}}, "missing"),
        (
            {
                "E1": {"B": 1.0},
                "E2": {"B": 1.0},
                "E3": {"B": 1.0},
            },
            "extra",
        ),
    ],
)
def test_weight_mapping_requires_exact_custom_policy_epoch_coverage(
    weights, expected
) -> None:
    with pytest.raises(ValueError, match=expected):
        _request(residual_weights_by_epoch=weights)


@pytest.mark.parametrize(
    "bad_weights, expected",
    [
        ({"B": 0.4, "C": 0.5}, "sum to one"),
        ({"B": -0.1, "C": 1.1}, "nonnegative"),
        ({"B": math.nan, "C": 1.0}, "finite"),
        ({"B": math.inf, "C": 0.0}, "finite"),
    ],
)
def test_weights_are_validated_independently_within_each_epoch(
    bad_weights, expected
) -> None:
    request = _request(
        residual_weights_by_epoch={
            "E1": {"B": 0.25, "C": 0.75},
            "E2": bad_weights,
        }
    )
    with pytest.raises(ValueError, match=expected):
        calibration._allocate_temporal_custom_weights(
            _direct(),
            request,
            allow_signed=False,
            tolerance=1e-9,
        )


@pytest.mark.parametrize(
    "weights, expected",
    [
        ({"A": 1.0}, "Tested entities"),
        ({"unknown": 1.0}, "unknown"),
    ],
)
def test_tested_and_unknown_weight_recipients_raise(weights, expected) -> None:
    request = _request(
        residual_weights_by_epoch={"E1": weights, "E2": {"B": 1.0}}
    )
    with pytest.raises(ValueError, match=expected):
        calibration._allocate_temporal_custom_weights(
            _direct(),
            request,
            allow_signed=False,
            tolerance=1e-9,
        )


def test_exact_weights_are_retained_in_immutable_epoch_provenance() -> None:
    result = _result()
    e1 = result.residual_provenance["E1"]
    e2 = result.residual_provenance["E2"]

    assert e1["residual_weights"] == {"B": 0.25, "C": 0.75}
    assert e2["residual_weights"] == {"C": 0.2, "B": 0.8}
    assert e1["adjustments"] == (("B", 5.0), ("C", 15.0))
    assert e2["adjustments"] == (("C", 4.0), ("B", 16.0))
    with pytest.raises(TypeError):
        e1["residual_weights"]["B"] = 1.0


def test_custom_allocation_never_changes_ratio_rows_or_uncovered_epoch() -> None:
    state = _direct()
    result = _result(state=state)

    assert result.epoch_ratio_rows is state.epoch_ratio_rows
    assert result.calibrated_values_by_epoch["E3"] == tuple(
        ESTIMATES["E3"].items()
    )
    assert result.residual_provenance["E3"]["status"] == "not_calibrated"


def test_infeasible_custom_allocation_raises_after_exact_weighting() -> None:
    state = _direct(rows=(_ratio("E1", 3.0), _ratio("E2", 0.8)))
    request = _request(
        residual_weights_by_epoch={"E1": {"B": 1.0}, "E2": {"B": 1.0}}
    )
    allocations = calibration._allocate_temporal_custom_weights(
        state,
        request,
        allow_signed=False,
        tolerance=1e-9,
    )
    with pytest.raises(ValueError, match="nonnegative calibrated estimates"):
        calibration._finalize_temporal_residual_allocations(
            state,
            request,
            allocations,
            allow_signed=False,
            tolerance=1e-9,
        )
