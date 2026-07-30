"""Direct final epoch-ratio rows for CAL-054."""

import pytest

from do_attribution.calibration import (
    LiftTestEvidence,
    _build_constant_schedule,
    _build_explicit_ratio_schedule,
    _build_explicit_strength_schedule,
    _build_observed_only_schedule,
    _reduce_direct_epoch_ratios,
    _resolve_evidence_conflicts,
    _validate_epoch_ratio_reduction_request,
    _validate_evidence_conflict_request,
)


EPOCHS = ("E1", "E2", "E3")
WINDOWS = {
    "E1": (0.0, 10.0),
    "E2": (10.0, 20.0),
    "E3": (20.0, 30.0),
}


def _evidence(entity, test_id, coordinate, lift, matched=100.0):
    return LiftTestEvidence(
        tested_entity=entity,
        observed_lift=lift,
        matched_mta_estimate=matched,
        test_id=test_id,
        outcome_name="sales",
        population="all-customers",
        unit="orders",
        coordinate_name="index",
        coordinate_value=coordinate,
    )


def _records():
    return [
        _evidence("A", "a-first", 2.0, 80.0),
        _evidence("B", "b-only", 12.0, 120.0),
        _evidence("A", "a-second", 4.0, 60.0),
    ]


def _observed(policy, *, support=None):
    schedule = _build_observed_only_schedule(
        _records(),
        epochs=EPOCHS,
        epoch_windows=WINDOWS,
    )
    conflict_request = _validate_evidence_conflict_request(
        schedule,
        evidence_conflict_policy={"select": "select_evidence", "weighted": "weighted_ratio", "pooled": "pooled_lift"}[policy],
    )
    selected = (
        {("A", "E1"): "a-second"} if policy == "select" else None
    )
    weights = (
        {("A", "E1"): {"a-first": 0.25, "a-second": 0.75}}
        if policy == "weighted"
        else None
    )
    conflicts = _resolve_evidence_conflicts(
        conflict_request,
        selected_evidence=selected,
        evidence_weights=weights,
    )
    request = _validate_epoch_ratio_reduction_request(
        ratio_policy="observed_only",
        epoch_ratio_policy="direct",
        epochs=EPOCHS,
        calibrated_pairs=(("B", "E2"), ("A", "E1")),
        epoch_windows=WINDOWS,
        epoch_support=support,
    )
    return request, schedule, conflicts


@pytest.mark.parametrize(
    "policy, expected_ratio, expected_source",
    [
        ("select", 0.6, "selected_from_multiple"),
        ("weighted", 0.65, "weighted_from_multiple"),
        ("pooled", 0.7, "pooled_from_multiple"),
    ],
)
def test_observed_conflicts_emit_one_explicit_unchanged_resolution(
    policy, expected_ratio, expected_source
) -> None:
    request, schedule, conflicts = _observed(policy)
    result = _reduce_direct_epoch_ratios(
        request,
        schedule,
        conflict_resolutions=conflicts,
    )
    assert [(row.tested_entity, row.epoch) for row in result.rows] == [
        ("A", "E1"),
        ("B", "E2"),
    ]
    a, b = result.rows
    assert a.effective_ratio == pytest.approx(expected_ratio)
    assert a.ratio_source == expected_source
    assert a.evidence_conflict_policy == {
        "select": "select_evidence",
        "weighted": "weighted_ratio",
        "pooled": "pooled_lift",
    }[policy]
    assert b.effective_ratio == 1.2
    assert b.ratio_source == "observed"
    assert b.evidence_conflict_policy is None
    assert b.source_evidence_ids == ("b-only",)


def test_weighted_conflict_preserves_weights_and_full_resolution_provenance() -> None:
    request, schedule, conflicts = _observed("weighted")
    result = _reduce_direct_epoch_ratios(
        request,
        schedule,
        conflict_resolutions=conflicts,
    )
    row = result.rows[0]
    assert row.evidence_weights == {"a-first": 0.25, "a-second": 0.75}
    provenance = result.provenance_by_pair[("A", "E1")]
    assert provenance["conflict_resolution"]["candidate_evidence_ids"] == (
        "a-first",
        "a-second",
    )
    assert provenance["conflict_resolution"][
        "evidence_combination_weights"
    ] == {"a-first": 0.25, "a-second": 0.75}


def test_direct_rows_preserve_authoritative_windows_and_optional_support() -> None:
    request, schedule, conflicts = _observed(
        "select", support={"E1": 11.0, "E2": 22.0}
    )
    result = _reduce_direct_epoch_ratios(
        request,
        schedule,
        conflict_resolutions=conflicts,
    )
    assert result.rows[0].epoch_start == 0.0
    assert result.rows[0].epoch_end == 10.0
    assert result.rows[0].epoch_support == 11.0
    assert result.rows[1].epoch_support == 22.0
    assert result.rows[0].epoch_coordinate is None
    assert result.rows[0].ratio_bound_lower is None
    assert result.rows[0].ratio_bound_upper is None


def test_constant_ratios_and_source_ids_pass_through_unchanged() -> None:
    evidence = _records()
    schedule = _build_constant_schedule(
        evidence,
        epochs=EPOCHS,
        source_evidence={"A": "a-first"},
        tested_entity_epochs={"A": ("E1", "E3")},
    )
    request = _validate_epoch_ratio_reduction_request(
        ratio_policy="constant",
        epoch_ratio_policy="direct",
        epochs=EPOCHS,
        calibrated_pairs=(("A", "E3"), ("A", "E1")),
    )
    result = _reduce_direct_epoch_ratios(request, schedule)
    assert [(row.epoch, row.effective_ratio) for row in result.rows] == [
        ("E1", 0.8),
        ("E3", 0.8),
    ]
    assert all(row.ratio_source == "constant" for row in result.rows)
    assert all(row.source_evidence_ids == ("a-first",) for row in result.rows)
    assert all(row.epoch_start is None for row in result.rows)


def test_explicit_ratios_pass_through_without_evidence_ids() -> None:
    schedule = _build_explicit_ratio_schedule(
        _records(),
        epochs=EPOCHS,
        ratio_schedule={("B", "E2"): 1.1, ("A", "E1"): 0.9},
    )
    request = _validate_epoch_ratio_reduction_request(
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        epochs=EPOCHS,
        calibrated_pairs=(("B", "E2"), ("A", "E1")),
    )
    result = _reduce_direct_epoch_ratios(request, schedule)
    assert [(row.tested_entity, row.epoch, row.effective_ratio) for row in result.rows] == [
        ("A", "E1", 0.9),
        ("B", "E2", 1.1),
    ]
    assert all(row.source_evidence_ids == () for row in result.rows)
    assert all(row.ratio_source == "explicit" for row in result.rows)


def test_explicit_strength_preserves_source_and_attenuation_provenance() -> None:
    schedule = _build_explicit_strength_schedule(
        _records(),
        epochs=EPOCHS,
        source_evidence={"A": "a-first", "B": "b-only"},
        evidence_strength={("A", "E1"): 0.5, ("B", "E2"): 0.25},
    )
    request = _validate_epoch_ratio_reduction_request(
        ratio_policy="explicit_strength",
        epoch_ratio_policy="direct",
        epochs=EPOCHS,
        calibrated_pairs=(("A", "E1"), ("B", "E2")),
    )
    result = _reduce_direct_epoch_ratios(request, schedule)
    assert result.rows[0].effective_ratio == 0.9
    assert result.rows[1].effective_ratio == 1.05
    assert result.rows[0].source_evidence_ids == ("a-first",)
    assert result.provenance_by_pair[("A", "E1")]["evidence_strength"] == 0.5
    assert result.rows[0].ratio_source == "strength_derived"


def test_uncalibrated_pairs_emit_no_rows_and_remain_explicitly_recorded() -> None:
    request, schedule, conflicts = _observed("select")
    result = _reduce_direct_epoch_ratios(
        request,
        schedule,
        conflict_resolutions=conflicts,
    )
    row_pairs = {(row.tested_entity, row.epoch) for row in result.rows}
    assert ("A", "E2") not in row_pairs
    assert ("B", "E1") not in row_pairs
    assert ("A", "E2") in result.uncalibrated_pairs
    assert ("B", "E1") in result.uncalibrated_pairs
    assert ("A", "E3") in result.uncalibrated_pairs
    assert ("B", "E3") in result.uncalibrated_pairs


def test_schedule_and_request_pair_coverage_must_match_exactly() -> None:
    schedule = _build_explicit_ratio_schedule(
        _records(),
        epochs=EPOCHS,
        ratio_schedule={("A", "E1"): 0.9},
    )
    request = _validate_epoch_ratio_reduction_request(
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        epochs=EPOCHS,
        calibrated_pairs=(("A", "E1"), ("B", "E2")),
    )
    with pytest.raises(ValueError, match="extra"):
        _reduce_direct_epoch_ratios(request, schedule)


def test_observed_conflicts_require_exact_resolution_coverage() -> None:
    request, schedule, conflicts = _observed("select")
    with pytest.raises(ValueError, match="missing"):
        _reduce_direct_epoch_ratios(request, schedule)
    with pytest.raises(ValueError, match="extra"):
        _reduce_direct_epoch_ratios(
            request,
            schedule,
            conflict_resolutions={
                **dict(conflicts),
                ("B", "E2"): conflicts[("A", "E1")],
            },
        )


def test_mismatched_schedule_type_or_nondirect_request_raises() -> None:
    schedule = _build_explicit_ratio_schedule(
        _records(),
        epochs=EPOCHS,
        ratio_schedule={("A", "E1"): 0.9},
    )
    constant_request = _validate_epoch_ratio_reduction_request(
        ratio_policy="constant",
        epoch_ratio_policy="direct",
        epochs=EPOCHS,
        calibrated_pairs=(("A", "E1"),),
    )
    with pytest.raises(TypeError, match="matching direct schedule"):
        _reduce_direct_epoch_ratios(constant_request, schedule)


def test_direct_result_is_deterministic_deeply_read_only_and_input_isolated() -> None:
    request, schedule, conflicts = _observed("weighted")
    first = _reduce_direct_epoch_ratios(
        request, schedule, conflict_resolutions=conflicts
    )
    second = _reduce_direct_epoch_ratios(
        request, schedule, conflict_resolutions=conflicts
    )
    assert first == second
    with pytest.raises(TypeError):
        first.provenance_by_pair[("A", "E1")]["ratio_source"] = "changed"
    with pytest.raises(TypeError):
        first.rows[0].evidence_weights["a-first"] = 1.0
