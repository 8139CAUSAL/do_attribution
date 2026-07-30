"""Conflict-group detection and explicit policy validation for CAL-048."""

import pytest

from do_attribution.calibration import (
    LiftTestEvidence,
    _build_observed_only_schedule,
    _validate_evidence_conflict_request,
)


def _evidence(entity, test_id, coordinate, ratio=0.9):
    return LiftTestEvidence(
        tested_entity=entity,
        observed_lift=ratio * 100.0,
        matched_mta_estimate=100.0,
        test_id=test_id,
        coordinate_name="index",
        coordinate_value=coordinate,
    )


def _schedule():
    evidence = [
        _evidence("A", "a-e2-first", 12.0, 0.8),
        _evidence("B", "b-e1-first", 2.0, 1.1),
        _evidence("A", "a-e1-only", 5.0, 0.95),
        _evidence("B", "b-e1-second", 4.0, 1.2),
        _evidence("A", "a-e2-second", 15.0, 0.7),
        _evidence("B", "b-unmatched", 30.0, 0.6),
    ]
    return _build_observed_only_schedule(
        evidence,
        epochs=("E1", "E2"),
        epoch_windows={"E1": (0.0, 10.0), "E2": (10.0, 20.0)},
    )


def test_omission_raises_for_every_unresolved_direct_conflict_group() -> None:
    with pytest.raises(ValueError, match="evidence_conflict_policy is required"):
        _validate_evidence_conflict_request(
            _schedule(),
            evidence_conflict_policy=None,
        )


def test_scalar_policy_expands_in_authoritative_pair_order() -> None:
    request = _validate_evidence_conflict_request(
        _schedule(),
        evidence_conflict_policy="select_evidence",
    )

    assert tuple(request.conflict_groups) == (("B", "E1"), ("A", "E2"))
    assert request.policy_by_pair == {
        ("B", "E1"): "select_evidence",
        ("A", "E2"): "select_evidence",
    }
    assert [
        point.evidence_id for point in request.conflict_groups[("B", "E1")]
    ] == ["b-e1-first", "b-e1-second"]
    assert [
        point.evidence_id for point in request.conflict_groups[("A", "E2")]
    ] == ["a-e2-first", "a-e2-second"]


def test_exact_mapping_is_reordered_without_selecting_or_discarding() -> None:
    schedule = _schedule()
    request = _validate_evidence_conflict_request(
        schedule,
        evidence_conflict_policy={
            ("A", "E2"): "weighted_ratio",
            ("B", "E1"): "pooled_lift",
        },
    )

    assert tuple(request.policy_by_pair) == (("B", "E1"), ("A", "E2"))
    assert request.policy_by_pair == {
        ("B", "E1"): "pooled_lift",
        ("A", "E2"): "weighted_ratio",
    }
    assert request.evidence_points == schedule.evidence_points
    assert [point.evidence_id for point in request.evidence_points] == [
        "a-e2-first",
        "b-e1-first",
        "a-e1-only",
        "b-e1-second",
        "a-e2-second",
        "b-unmatched",
    ]
    assert schedule.status_by_pair[("B", "E1")] == "conflict_pending"
    assert schedule.status_by_pair[("A", "E2")] == "conflict_pending"
    assert len(schedule.candidates[("B", "E1")]) == 2
    assert len(schedule.candidates[("A", "E2")]) == 2


@pytest.mark.parametrize(
    "policies, expected",
    [
        ({("B", "E1"): "select_evidence"}, "missing"),
        (
            {
                ("B", "E1"): "select_evidence",
                ("A", "E2"): "weighted_ratio",
                ("A", "E1"): "pooled_lift",
            },
            "extra",
        ),
    ],
)
def test_mapping_keys_must_exactly_cover_conflict_groups(policies, expected) -> None:
    with pytest.raises(ValueError, match=expected):
        _validate_evidence_conflict_request(
            _schedule(),
            evidence_conflict_policy=policies,
        )


@pytest.mark.parametrize(
    "policies",
    [
        "mean_ratio",
        [],
        {
            ("B", "E1"): "select_evidence",
            ("A", "E2"): "mean_ratio",
        },
    ],
)
def test_unknown_conflict_policies_raise(policies) -> None:
    with pytest.raises(ValueError, match="must be one of"):
        _validate_evidence_conflict_request(
            _schedule(),
            evidence_conflict_policy=policies,
        )


def test_no_conflicts_accepts_no_policy_and_rejects_extra_policy() -> None:
    schedule = _build_observed_only_schedule(
        [_evidence("A", "a-only", 5.0)],
        epochs=("E1",),
        epoch_windows={"E1": (0.0, 10.0)},
    )
    request = _validate_evidence_conflict_request(
        schedule,
        evidence_conflict_policy=None,
    )
    empty_mapping_request = _validate_evidence_conflict_request(
        schedule,
        evidence_conflict_policy={},
    )

    assert request.conflict_groups == {}
    assert request.policy_by_pair == {}
    assert empty_mapping_request == request
    with pytest.raises(ValueError, match="not applicable"):
        _validate_evidence_conflict_request(
            schedule,
            evidence_conflict_policy="select_evidence",
        )
    with pytest.raises(ValueError, match="extra"):
        _validate_evidence_conflict_request(
            schedule,
            evidence_conflict_policy={("A", "E1"): "select_evidence"},
        )


def test_conflict_request_is_deterministic_and_deeply_read_only() -> None:
    first = _validate_evidence_conflict_request(
        _schedule(),
        evidence_conflict_policy="pooled_lift",
    )
    second = _validate_evidence_conflict_request(
        _schedule(),
        evidence_conflict_policy="pooled_lift",
    )
    assert first == second

    with pytest.raises(TypeError):
        first.policy_by_pair[("B", "E1")] = "select_evidence"
    with pytest.raises(TypeError):
        first.conflict_groups[("B", "E1")] = ()


def test_unmatched_and_single_observations_never_form_conflict_groups() -> None:
    request = _validate_evidence_conflict_request(
        _schedule(),
        evidence_conflict_policy={
            ("B", "E1"): "select_evidence",
            ("A", "E2"): "select_evidence",
        },
    )
    grouped_ids = {
        point.evidence_id
        for group in request.conflict_groups.values()
        for point in group
    }
    assert "a-e1-only" not in grouped_ids
    assert "b-unmatched" not in grouped_ids
    assert request.evidence_points[-1].matched_epoch is None
