"""Epoch-local proportional temporal residual allocation for CAL-061."""

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
    index = EPOCHS.index(epoch)
    return EpochRatioRow(
        tested_entity="A",
        epoch=epoch,
        epoch_start=float(index * 10),
        epoch_end=float((index + 1) * 10),
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


def _direct(*, estimates=ESTIMATES, rows=None, allow_signed=False):
    if rows is None:
        rows = (_ratio("E2"), _ratio("E1"))
    return calibration._temporal_direct_state(
        estimates,
        epoch_ratio_rows=rows,
        evidence_by_entity=_evidence(),
        allow_signed=allow_signed,
        tolerance=1e-9,
    )


def _request(**overrides):
    values = {
        "epochs": EPOCHS,
        "calibrated_epochs": ("E2", "E1"),
        "residual_policy": "proportional_untested",
        "eligible_entities": ("B", "C"),
    }
    values.update(overrides)
    return calibration._normalize_temporal_residual_request(**values)


def _result(*, state=None, request=None, allow_signed=False):
    state = _direct(allow_signed=allow_signed) if state is None else state
    request = _request() if request is None else request
    allocations = calibration._allocate_temporal_proportional_untested(
        state,
        request,
        allow_signed=allow_signed,
        tolerance=1e-9,
    )
    return calibration._finalize_temporal_residual_allocations(
        state,
        request,
        allocations,
        allow_signed=allow_signed,
        tolerance=1e-9,
    )


def test_changing_mix_drives_independent_epoch_proportional_allocations() -> None:
    result = _result()

    assert result.required_residuals == {"E1": 20.0, "E2": 20.0, "E3": 0.0}
    assert result.allocations_by_epoch["E1"].adjustments == (
        ("B", 6.0),
        ("C", 14.0),
    )
    assert result.allocations_by_epoch["E2"].adjustments == (
        ("B", 16.0),
        ("C", 4.0),
    )
    assert result.calibrated_values_by_epoch["E1"] == (
        ("A", 80.0),
        ("B", 36.0),
        ("C", 84.0),
    )
    assert result.calibrated_values_by_epoch["E2"] == (
        ("A", 80.0),
        ("B", 96.0),
        ("C", 24.0),
    )


def test_each_epoch_preserves_relative_proportions_and_original_total() -> None:
    result = _result()

    for epoch in ("E1", "E2"):
        values = dict(result.calibrated_values_by_epoch[epoch])
        assert values["B"] / values["C"] == pytest.approx(
            ESTIMATES[epoch]["B"] / ESTIMATES[epoch]["C"]
        )
        assert result.allocated_residuals[epoch] == pytest.approx(20.0)
        assert result.calibrated_totals[epoch] == pytest.approx(
            result.original_totals[epoch]
        )


def test_ratio_rows_and_authoritative_windows_are_identical_when_mix_changes() -> None:
    state = _direct()
    result = _result(state=state)

    assert result.epoch_ratio_rows is state.epoch_ratio_rows
    assert [(row.epoch_start, row.epoch_end) for row in result.epoch_ratio_rows] == [
        (10.0, 20.0),
        (0.0, 10.0),
    ]
    assert [row.effective_ratio for row in result.epoch_ratio_rows] == [0.8, 0.8]
    assert result.calibrated_values_by_epoch["E3"] == tuple(
        ESTIMATES["E3"].items()
    )


def test_per_epoch_recipient_sets_do_not_leak_across_epochs() -> None:
    request = _request(
        eligible_entities=None,
        eligible_entities_by_epoch={"E1": ["B", "C"], "E2": ["B"]},
    )
    result = _result(request=request)

    assert result.allocations_by_epoch["E1"].adjustments == (
        ("B", 6.0),
        ("C", 14.0),
    )
    assert result.allocations_by_epoch["E2"].adjustments == (("B", 20.0),)
    assert dict(result.calibrated_values_by_epoch["E2"])["C"] == 20.0


def test_all_untested_uses_each_epoch_current_entity_membership_and_order() -> None:
    estimates = {
        "E1": {"A": 100.0, "B": 25.0, "C": 75.0},
        "E2": {"A": 100.0, "B": 60.0, "D": 40.0},
        "E3": ESTIMATES["E3"],
    }
    state = _direct(estimates=estimates)
    request = _request(eligible_entities=None, all_untested=True)
    result = _result(state=state, request=request)

    assert result.allocations_by_epoch["E1"].adjustments == (
        ("B", 5.0),
        ("C", 15.0),
    )
    assert result.allocations_by_epoch["E2"].adjustments == (
        ("B", 12.0),
        ("D", 8.0),
    )


def test_residual_provenance_records_exact_epoch_local_logic() -> None:
    result = _result()
    e1 = result.residual_provenance["E1"]
    e2 = result.residual_provenance["E2"]

    assert e1["residual_policy"] == "proportional_untested"
    assert e1["eligible_entities"] == ("B", "C")
    assert e1["recipients"] == ("B", "C")
    assert e1["adjustments"] == (("B", 6.0), ("C", 14.0))
    assert e2["adjustments"] == (("B", 16.0), ("C", 4.0))
    with pytest.raises(TypeError):
        e1["new"] = "value"


def test_mixed_policy_epochs_combine_without_cross_policy_allocation() -> None:
    state = _direct()
    request = _request(
        residual_policy={
            "E1": "allow_total_change",
            "E2": "proportional_untested",
        },
        eligible_entities=None,
        eligible_entities_by_epoch={"E2": ["B", "C"]},
    )
    allocations = dict(
        calibration._allocate_temporal_allow_total_change(
            state,
            request,
            allow_signed=False,
            tolerance=1e-9,
        )
    )
    allocations.update(
        calibration._allocate_temporal_proportional_untested(
            state,
            request,
            allow_signed=False,
            tolerance=1e-9,
        )
    )
    result = calibration._finalize_temporal_residual_allocations(
        state,
        request,
        allocations,
        allow_signed=False,
        tolerance=1e-9,
    )

    assert result.calibrated_totals["E1"] == 180.0
    assert result.allocated_residuals["E1"] == 0.0
    assert result.calibrated_totals["E2"] == 200.0
    assert result.allocated_residuals["E2"] == 20.0


@pytest.mark.parametrize(
    "eligible, expected",
    [
        (("A",), "Tested entities"),
        (("unknown",), "unknown"),
    ],
)
def test_tested_and_unknown_explicit_recipients_raise(eligible, expected) -> None:
    request = _request(eligible_entities=eligible)
    with pytest.raises(ValueError, match=expected):
        calibration._allocate_temporal_proportional_untested(
            _direct(),
            request,
            allow_signed=False,
            tolerance=1e-9,
        )


def test_zero_epoch_local_eligible_mass_raises() -> None:
    estimates = {
        "E1": {"A": 100.0, "B": 0.0, "C": 0.0},
        "E2": ESTIMATES["E2"],
        "E3": ESTIMATES["E3"],
    }
    with pytest.raises(ValueError, match="zero eligible mass"):
        calibration._allocate_temporal_proportional_untested(
            _direct(estimates=estimates),
            _request(),
            allow_signed=False,
            tolerance=1e-9,
        )


def test_infeasible_negative_epoch_output_raises_at_finalization() -> None:
    state = _direct(rows=(_ratio("E1", 3.0), _ratio("E2", 0.8)))
    request = _request()
    allocations = calibration._allocate_temporal_proportional_untested(
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
