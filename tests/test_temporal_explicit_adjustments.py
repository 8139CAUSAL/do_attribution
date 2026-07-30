"""Exact temporal adjustments and five-policy dispatch for CAL-064."""

import math

import pytest

import do_attribution.calibration as calibration
from do_attribution.calibration import EpochRatioRow, LiftTestEvidence


EPOCHS = ("E1", "E2", "E3")
ESTIMATES = {
    "E1": {"A": 100.0, "B": 60.0, "C": 40.0},
    "E2": {"A": 100.0, "B": 60.0, "C": 40.0},
    "E3": {"A": 100.0, "B": 60.0, "C": 40.0},
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


def _direct(*, rows=None, allow_signed=False):
    return calibration._temporal_direct_state(
        ESTIMATES,
        epoch_ratio_rows=(
            (_ratio("E2"), _ratio("E1")) if rows is None else rows
        ),
        evidence_by_entity=_evidence(),
        allow_signed=allow_signed,
        tolerance=1e-9,
    )


def _request(**overrides):
    values = {
        "epochs": EPOCHS,
        "calibrated_epochs": ("E2", "E1"),
        "residual_policy": "explicit_adjustments",
        "residual_adjustments_by_epoch": {
            "E1": {"B": 5.0, "C": 15.0},
            "E2": {"B": 12.0, "C": 8.0},
        },
    }
    values.update(overrides)
    return calibration._normalize_temporal_residual_request(**values)


def _result(*, state=None, request=None, allow_signed=False):
    state = _direct(allow_signed=allow_signed) if state is None else state
    request = _request() if request is None else request
    return calibration._apply_temporal_residual_request(
        state,
        request,
        allow_signed=allow_signed,
        tolerance=1e-9,
    )


def test_exact_adjustments_balance_each_epoch_independently() -> None:
    result = _result()

    assert result.allocations_by_epoch["E1"].adjustments == (
        ("B", 5.0),
        ("C", 15.0),
    )
    assert result.allocations_by_epoch["E2"].adjustments == (
        ("B", 12.0),
        ("C", 8.0),
    )
    assert result.calibrated_values_by_epoch["E1"] == (
        ("A", 80.0),
        ("B", 65.0),
        ("C", 55.0),
    )
    assert result.calibrated_values_by_epoch["E2"] == (
        ("A", 80.0),
        ("B", 72.0),
        ("C", 48.0),
    )
    assert result.allocated_residuals == {"E1": 20.0, "E2": 20.0, "E3": 0.0}
    assert result.calibrated_totals == result.original_totals


def test_adjustments_are_not_copied_between_epochs() -> None:
    result = _result()
    assert result.residual_provenance["E1"]["adjustments"] == (
        ("B", 5.0),
        ("C", 15.0),
    )
    assert result.residual_provenance["E2"]["adjustments"] == (
        ("B", 12.0),
        ("C", 8.0),
    )
    assert result.residual_provenance["E3"]["adjustments"] == ()


@pytest.mark.parametrize(
    "adjustments, expected",
    [
        ({"E1": {"B": 20.0}}, "missing"),
        (
            {
                "E1": {"B": 20.0},
                "E2": {"B": 20.0},
                "E3": {"B": 0.0},
            },
            "extra",
        ),
    ],
)
def test_adjustment_mapping_exactly_covers_explicit_policy_epochs(
    adjustments, expected
) -> None:
    with pytest.raises(ValueError, match=expected):
        _request(residual_adjustments_by_epoch=adjustments)


@pytest.mark.parametrize(
    "bad_adjustments, expected",
    [
        ({"B": 19.0}, "must sum"),
        ({"B": math.nan}, "finite"),
        ({"B": math.inf}, "finite"),
    ],
)
def test_each_epoch_adjustment_sum_and_values_are_validated(
    bad_adjustments, expected
) -> None:
    request = _request(
        residual_adjustments_by_epoch={
            "E1": {"B": 5.0, "C": 15.0},
            "E2": bad_adjustments,
        }
    )
    with pytest.raises(ValueError, match=expected):
        calibration._allocate_temporal_explicit_adjustments(
            _direct(),
            request,
            allow_signed=False,
            tolerance=1e-9,
        )


@pytest.mark.parametrize(
    "adjustments, expected",
    [
        ({"A": 20.0}, "Tested entities"),
        ({"unknown": 20.0}, "unknown"),
    ],
)
def test_tested_and_unknown_explicit_recipients_raise(adjustments, expected) -> None:
    request = _request(
        residual_adjustments_by_epoch={
            "E1": adjustments,
            "E2": {"B": 20.0},
        }
    )
    with pytest.raises(ValueError, match=expected):
        calibration._allocate_temporal_explicit_adjustments(
            _direct(),
            request,
            allow_signed=False,
            tolerance=1e-9,
        )


def test_infeasible_negative_output_requires_signed_permission() -> None:
    rows = (_ratio("E1", 3.0), _ratio("E2", 0.8))
    request = _request(
        residual_adjustments_by_epoch={
            "E1": {"B": -200.0},
            "E2": {"B": 20.0},
        }
    )
    state = _direct(rows=rows)
    allocations = calibration._allocate_temporal_explicit_adjustments(
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

    signed_state = _direct(rows=rows, allow_signed=True)
    signed_result = _result(
        state=signed_state,
        request=request,
        allow_signed=True,
    )
    assert dict(signed_result.calibrated_values_by_epoch["E1"])["B"] == -140.0


def test_exact_adjustment_provenance_and_ratio_rows_are_immutable() -> None:
    state = _direct()
    result = _result(state=state)
    e1 = result.residual_provenance["E1"]

    assert e1["residual_policy"] == "explicit_adjustments"
    assert e1["adjustments"] == (("B", 5.0), ("C", 15.0))
    assert result.epoch_ratio_rows is state.epoch_ratio_rows
    with pytest.raises(TypeError):
        e1["adjustments"] = ()


def test_complete_dispatch_routes_all_five_policies_once_by_epoch() -> None:
    epochs = ("E1", "E2", "E3", "E4", "E5")
    estimates = {
        epoch: {"A": 100.0, "B": 60.0, "C": 40.0}
        for epoch in epochs
    }
    rows = tuple(_ratio(epoch) for epoch in epochs)
    state = calibration._temporal_direct_state(
        estimates,
        epoch_ratio_rows=rows,
        evidence_by_entity=_evidence(),
        allow_signed=False,
        tolerance=1e-9,
    )
    request = calibration._normalize_temporal_residual_request(
        epochs=epochs,
        calibrated_epochs=epochs,
        residual_policy={
            "E1": "allow_total_change",
            "E2": "proportional_untested",
            "E3": "custom_weights",
            "E4": "residual_bucket",
            "E5": "explicit_adjustments",
        },
        eligible_entities_by_epoch={"E2": ["B", "C"]},
        residual_weights_by_epoch={"E3": {"B": 0.25, "C": 0.75}},
        residual_entity_by_epoch={"E4": "baseline"},
        residual_adjustments_by_epoch={"E5": {"B": 7.0, "C": 13.0}},
    )
    result = calibration._apply_temporal_residual_request(
        state,
        request,
        allow_signed=False,
        tolerance=1e-9,
    )

    assert tuple(result.allocations_by_epoch) == epochs
    assert result.calibrated_totals == {
        "E1": 180.0,
        "E2": 200.0,
        "E3": 200.0,
        "E4": 200.0,
        "E5": 200.0,
    }
    assert result.allocations_by_epoch["E1"].adjustments == ()
    assert result.allocations_by_epoch["E2"].adjustments == (
        ("B", 12.0),
        ("C", 8.0),
    )
    assert result.allocations_by_epoch["E3"].adjustments == (
        ("B", 5.0),
        ("C", 15.0),
    )
    assert result.allocations_by_epoch["E4"].adjustments == (("baseline", 20.0),)
    assert result.allocations_by_epoch["E5"].adjustments == (
        ("B", 7.0),
        ("C", 13.0),
    )
