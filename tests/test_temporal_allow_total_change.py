"""Per-epoch temporal ``allow_total_change`` tests for CAL-060."""

import pytest

import do_attribution.calibration as calibration
from do_attribution.calibration import EpochRatioRow, LiftTestEvidence


EPOCHS = ("E1", "E2", "E3")
ESTIMATES = {
    "E1": {"A": 100.0, "B": 50.0},
    "E2": {"A": 120.0, "B": 80.0},
    "E3": {"A": 90.0, "B": 110.0},
}


def _evidence():
    item = LiftTestEvidence(
        tested_entity="A",
        observed_lift=80.0,
        matched_mta_estimate=100.0,
        test_id="a-test",
    )
    return {"A": (item,)}


def _ratio(epoch, ratio, *, entity="A", source_ids=()):
    return EpochRatioRow(
        tested_entity=entity,
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
        source_evidence_ids=source_ids,
        evidence_weights=None,
        left_source_evidence_id=None,
        right_source_evidence_id=None,
        inside_observed_support=True,
        ratio_bound_lower=None,
        ratio_bound_upper=None,
    )


def _direct(**overrides):
    values = {
        "estimates_by_epoch": ESTIMATES,
        "epoch_ratio_rows": (_ratio("E3", 1.2), _ratio("E1", 0.8)),
        "evidence_by_entity": _evidence(),
        "allow_signed": False,
        "tolerance": 1e-9,
    }
    values.update(overrides)
    return calibration._temporal_direct_state(**values)


def _request(**overrides):
    values = {
        "epochs": EPOCHS,
        "calibrated_epochs": ("E3", "E1"),
        "residual_policy": "allow_total_change",
    }
    values.update(overrides)
    return calibration._normalize_temporal_residual_request(**values)


def _result(state=None, request=None):
    state = _direct() if state is None else state
    request = _request() if request is None else request
    allocations = calibration._allocate_temporal_allow_total_change(
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


def test_direct_ratios_apply_only_to_the_declared_entity_epoch_pairs() -> None:
    state = _direct()

    e1 = state.states_by_epoch["E1"]
    e2 = state.states_by_epoch["E2"]
    e3 = state.states_by_epoch["E3"]
    assert [row.direct_calibrated_estimate for row in e1.rows] == [80.0, 50.0]
    assert [row.direct_test_adjustment for row in e1.rows] == [-20.0, 0.0]
    assert [row.direct_calibrated_estimate for row in e2.rows] == [120.0, 80.0]
    assert [row.direct_test_adjustment for row in e2.rows] == [0.0, 0.0]
    assert [row.direct_calibrated_estimate for row in e3.rows] == [108.0, 110.0]
    assert [row.direct_test_adjustment for row in e3.rows] == [18.0, 0.0]
    assert state.calibrated_epochs == ("E1", "E3")
    assert state.uncalibrated_epochs == ("E2",)


def test_allow_total_change_allocates_nothing_and_uses_raw_epoch_totals() -> None:
    result = _result()

    assert result.original_totals == {"E1": 150.0, "E2": 200.0, "E3": 200.0}
    assert result.raw_post_test_totals == {
        "E1": 130.0,
        "E2": 200.0,
        "E3": 218.0,
    }
    assert result.required_residuals == {
        "E1": 20.0,
        "E2": 0.0,
        "E3": -18.0,
    }
    assert result.allocated_residuals == {"E1": 0.0, "E2": 0.0, "E3": 0.0}
    assert result.calibrated_totals == result.raw_post_test_totals
    assert all(not allocation.adjustments for allocation in result.allocations_by_epoch.values())


def test_untested_values_and_the_entire_uncovered_epoch_remain_unchanged() -> None:
    result = _result()

    assert result.calibrated_values_by_epoch["E1"] == (
        ("A", 80.0),
        ("B", 50.0),
    )
    assert result.calibrated_values_by_epoch["E2"] == (
        ("A", 120.0),
        ("B", 80.0),
    )
    assert result.calibrated_values_by_epoch["E3"] == (
        ("A", 108.0),
        ("B", 110.0),
    )
    assert result.residual_provenance["E2"]["status"] == "not_calibrated"
    assert result.residual_provenance["E2"]["residual_policy"] is None
    assert result.residual_provenance["E2"]["adjustments"] == ()


def test_authoritative_epoch_order_and_ratio_rows_are_not_rewritten() -> None:
    ratio_rows = [_ratio("E3", 1.2), _ratio("E1", 0.8)]
    state = _direct(epoch_ratio_rows=ratio_rows)
    result = _result(state=state)

    assert state.epochs == EPOCHS
    assert tuple(result.original_totals) == EPOCHS
    assert result.epoch_ratio_rows == tuple(ratio_rows)
    assert result.epoch_ratio_rows is state.epoch_ratio_rows


def test_explicit_ratio_rows_keep_empty_source_provenance_but_direct_roles_are_known() -> None:
    state = _direct()
    assert all(row.source_evidence_ids == () for row in state.epoch_ratio_rows)
    assert state.states_by_epoch["E1"].rows[0].evidence_ids == ("a-test",)
    assert state.states_by_epoch["E3"].rows[0].evidence_ids == ("a-test",)
    assert state.states_by_epoch["E2"].rows[0].evidence_ids == ()


def test_allow_total_change_handler_processes_only_its_selected_epochs() -> None:
    request = _request(
        residual_policy={
            "E1": "allow_total_change",
            "E3": "proportional_untested",
        },
        eligible_entities_by_epoch={"E3": ["B"]},
    )
    allocations = calibration._allocate_temporal_allow_total_change(
        _direct(),
        request,
        allow_signed=False,
        tolerance=1e-9,
    )
    assert tuple(allocations) == ("E1",)
    with pytest.raises(ValueError, match="missing=.*E3"):
        calibration._finalize_temporal_residual_allocations(
            _direct(),
            request,
            allocations,
            allow_signed=False,
            tolerance=1e-9,
        )


def test_empty_ratio_schedule_keeps_every_epoch_fully_unchanged() -> None:
    state = _direct(epoch_ratio_rows=())
    request = _request(calibrated_epochs=(), residual_policy={})
    result = _result(state=state, request=request)

    assert state.calibrated_epochs == ()
    assert state.uncalibrated_epochs == EPOCHS
    assert result.allocations_by_epoch == {}
    assert result.calibrated_values_by_epoch == {
        epoch: tuple(estimates.items()) for epoch, estimates in ESTIMATES.items()
    }
    assert result.calibrated_totals == result.original_totals


@pytest.mark.parametrize(
    "row, expected",
    [
        (_ratio("unknown", 0.9), "unknown epoch"),
        (_ratio("E1", 0.9, entity="missing"), "must exist"),
    ],
)
def test_ratio_rows_must_target_known_epoch_estimate_pairs(row, expected) -> None:
    with pytest.raises(ValueError, match=expected):
        _direct(epoch_ratio_rows=(row,))


def test_ratio_source_evidence_must_belong_to_the_tested_entity() -> None:
    b = LiftTestEvidence(
        tested_entity="B",
        observed_lift=55.0,
        matched_mta_estimate=50.0,
        test_id="b-test",
    )
    evidence = dict(_evidence())
    evidence["B"] = (b,)
    with pytest.raises(ValueError, match="must belong"):
        _direct(
            epoch_ratio_rows=(_ratio("E1", 0.8, source_ids=("b-test",)),),
            evidence_by_entity=evidence,
        )


def test_signed_effective_ratios_require_explicit_permission() -> None:
    row = _ratio("E1", -0.5)
    with pytest.raises(ValueError, match="allow_signed=True"):
        _direct(epoch_ratio_rows=(row,))

    state = _direct(
        epoch_ratio_rows=(row,),
        allow_signed=True,
    )
    request = _request(calibrated_epochs=("E1",))
    allocations = calibration._allocate_temporal_allow_total_change(
        state,
        request,
        allow_signed=True,
        tolerance=1e-9,
    )
    result = calibration._finalize_temporal_residual_allocations(
        state,
        request,
        allocations,
        allow_signed=True,
        tolerance=1e-9,
    )
    assert result.calibrated_values_by_epoch["E1"][0] == ("A", -50.0)


def test_caller_inputs_and_nested_result_state_are_immutable() -> None:
    estimates = {epoch: dict(values) for epoch, values in ESTIMATES.items()}
    ratio_rows = [_ratio("E1", 0.8), _ratio("E3", 1.2)]
    state = _direct(
        estimates_by_epoch=estimates,
        epoch_ratio_rows=ratio_rows,
    )
    result = _result(state=state)
    estimates["E1"]["A"] = 999.0
    ratio_rows.reverse()

    assert state.states_by_epoch["E1"].rows[0].original_estimate == 100.0
    assert [row.epoch for row in result.epoch_ratio_rows] == ["E1", "E3"]
    with pytest.raises(TypeError):
        result.calibrated_totals["E1"] = 0.0
    with pytest.raises(TypeError):
        result.residual_provenance["E1"]["new"] = "value"
