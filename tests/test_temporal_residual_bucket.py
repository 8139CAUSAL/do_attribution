"""Explicit epoch-local temporal residual buckets for CAL-063."""

import pytest

import do_attribution.calibration as calibration
from do_attribution.calibration import EpochRatioRow, LiftTestEvidence


EPOCHS = ("E1", "E2", "E3")
ESTIMATES = {
    "E1": {"A": 100.0, "B": 100.0},
    "E2": {"A": 100.0, "B": 100.0},
    "E3": {"A": 100.0, "B": 100.0},
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
        "residual_policy": "residual_bucket",
        "residual_entity_by_epoch": {
            "E1": "baseline-e1",
            "E2": "baseline-e2",
        },
    }
    values.update(overrides)
    return calibration._normalize_temporal_residual_request(**values)


def _result(*, state=None, request=None, allow_signed=False):
    state = _direct(allow_signed=allow_signed) if state is None else state
    request = _request() if request is None else request
    allocations = calibration._allocate_temporal_residual_bucket(
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


def test_each_explicit_bucket_receives_its_exact_epoch_residual() -> None:
    result = _result()

    assert result.allocations_by_epoch["E1"].adjustments == (
        ("baseline-e1", 20.0),
    )
    assert result.allocations_by_epoch["E2"].adjustments == (
        ("baseline-e2", 20.0),
    )
    assert result.allocated_residuals == {"E1": 20.0, "E2": 20.0, "E3": 0.0}
    assert result.calibrated_totals == result.original_totals


def test_new_bucket_is_appended_after_each_epoch_existing_entity_order() -> None:
    result = _result()

    assert result.calibrated_values_by_epoch["E1"] == (
        ("A", 80.0),
        ("B", 100.0),
        ("baseline-e1", 20.0),
    )
    assert result.calibrated_values_by_epoch["E2"] == (
        ("A", 80.0),
        ("B", 100.0),
        ("baseline-e2", 20.0),
    )


def test_one_global_bucket_name_is_valid_for_uniform_bucket_policy() -> None:
    request = _request(
        residual_entity_by_epoch=None,
        residual_entity="baseline",
    )
    result = _result(request=request)

    assert result.allocations_by_epoch["E1"].new_entities == ("baseline",)
    assert result.allocations_by_epoch["E2"].new_entities == ("baseline",)
    assert result.calibrated_values_by_epoch["E1"][-1] == ("baseline", 20.0)
    assert result.calibrated_values_by_epoch["E2"][-1] == ("baseline", 20.0)


@pytest.mark.parametrize(
    "entities, expected",
    [
        ({"E1": "base"}, "missing"),
        (
            {"E1": "base-1", "E2": "base-2", "E3": "base-3"},
            "extra",
        ),
    ],
)
def test_bucket_names_exactly_cover_bucket_policy_epochs(entities, expected) -> None:
    with pytest.raises(ValueError, match=expected):
        _request(residual_entity_by_epoch=entities)


def test_bucket_is_never_inferred_when_no_name_is_supplied() -> None:
    with pytest.raises(ValueError, match="require residual_entity"):
        _request(residual_entity_by_epoch=None)


def test_existing_untested_collision_requires_explicit_permission_per_epoch() -> None:
    request = _request(
        residual_entity_by_epoch=None,
        residual_entity="B",
    )
    with pytest.raises(ValueError, match="collides with an existing entity"):
        calibration._allocate_temporal_residual_bucket(
            _direct(),
            request,
            allow_signed=False,
            tolerance=1e-9,
        )

    allowed = _request(
        residual_entity_by_epoch=None,
        residual_entity="B",
        allow_bucket_collision=True,
    )
    result = _result(request=allowed)
    assert result.allocations_by_epoch["E1"].new_entities == ()
    assert result.calibrated_values_by_epoch["E1"] == (
        ("A", 80.0),
        ("B", 120.0),
    )


def test_tested_bucket_collision_always_raises_even_when_opted_in() -> None:
    request = _request(
        residual_entity_by_epoch=None,
        residual_entity="A",
        allow_bucket_collision=True,
    )
    with pytest.raises(ValueError, match="Tested entities"):
        calibration._allocate_temporal_residual_bucket(
            _direct(),
            request,
            allow_signed=False,
            tolerance=1e-9,
        )


@pytest.mark.parametrize("bucket", ["", "   ", []])
def test_bucket_names_must_be_nonempty_and_hashable(bucket) -> None:
    with pytest.raises((TypeError, ValueError), match="residual_entity"):
        _request(
            residual_entity_by_epoch={"E1": bucket, "E2": "valid"}
        )


def test_negative_new_bucket_requires_signed_output_permission() -> None:
    rows = (_ratio("E1", 1.2), _ratio("E2", 0.8))
    state = _direct(rows=rows)
    request = _request()
    allocations = calibration._allocate_temporal_residual_bucket(
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
    assert signed_result.calibrated_values_by_epoch["E1"][-1] == (
        "baseline-e1",
        -20.0,
    )


def test_bucket_provenance_is_exact_epoch_local_and_immutable() -> None:
    result = _result()
    e1 = result.residual_provenance["E1"]
    e2 = result.residual_provenance["E2"]

    assert e1["residual_entity"] == "baseline-e1"
    assert e1["recipients"] == ("baseline-e1",)
    assert e1["adjustments"] == (("baseline-e1", 20.0),)
    assert e2["residual_entity"] == "baseline-e2"
    with pytest.raises(TypeError):
        e1["residual_entity"] = "changed"


def test_ratio_rows_and_uncovered_epoch_remain_unchanged() -> None:
    state = _direct()
    result = _result(state=state)

    assert result.epoch_ratio_rows is state.epoch_ratio_rows
    assert result.calibrated_values_by_epoch["E3"] == tuple(
        ESTIMATES["E3"].items()
    )
    assert result.residual_provenance["E3"]["status"] == "not_calibrated"
