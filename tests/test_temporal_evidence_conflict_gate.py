"""Complete applied and observational conflict-policy gate for CAL-052."""

import pytest

from do_attribution.calibration import (
    LiftTestEvidence,
    _build_evidence_conflict_diagnostics,
    _build_observed_only_schedule,
    _resolve_evidence_conflicts,
    _validate_evidence_conflict_request,
)


def _evidence(entity, test_id, coordinate, lift, matched):
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


def _request(policies=None):
    evidence = [
        _evidence("A", "a-first", 2.0, 80.0, 100.0),
        _evidence("A", "a-second", 4.0, 60.0, 100.0),
        _evidence("B", "b-first", 12.0, 50.0, 100.0),
        _evidence("B", "b-second", 14.0, 150.0, 100.0),
        _evidence("C", "c-first", 22.0, 90.0, 100.0),
        _evidence("C", "c-second", 24.0, 100.0, 200.0),
    ]
    schedule = _build_observed_only_schedule(
        evidence,
        epochs=("E1", "E2", "E3"),
        epoch_windows={
            "E1": (0.0, 10.0),
            "E2": (10.0, 20.0),
            "E3": (20.0, 30.0),
        },
    )
    return _validate_evidence_conflict_request(
        schedule,
        evidence_conflict_policy=(
            {
                ("A", "E1"): "select_evidence",
                ("B", "E2"): "weighted_ratio",
                ("C", "E3"): "pooled_lift",
            }
            if policies is None
            else policies
        ),
    )


def _applied(request=None):
    return _resolve_evidence_conflicts(
        _request() if request is None else request,
        selected_evidence={("A", "E1"): "a-second"},
        evidence_weights={
            ("B", "E2"): {"b-first": 0.25, "b-second": 0.75}
        },
    )


def test_all_three_applied_policies_dispatch_once_in_authoritative_order() -> None:
    resolutions = _applied()
    assert tuple(resolutions) == (
        ("A", "E1"),
        ("B", "E2"),
        ("C", "E3"),
    )
    assert resolutions[("A", "E1")].effective_ratio == 0.6
    assert resolutions[("B", "E2")].effective_ratio == 1.25
    assert resolutions[("C", "E3")].effective_ratio == pytest.approx(
        190.0 / 300.0
    )
    assert [row.evidence_conflict_policy for row in resolutions.values()] == [
        "select_evidence",
        "weighted_ratio",
        "pooled_lift",
    ]
    assert [row.ratio_source for row in resolutions.values()] == [
        "selected_from_multiple",
        "weighted_from_multiple",
        "pooled_from_multiple",
    ]


def test_applied_dispatch_preserves_complete_policy_specific_provenance() -> None:
    resolutions = _applied()
    selected = resolutions[("A", "E1")]
    weighted = resolutions[("B", "E2")]
    pooled = resolutions[("C", "E3")]
    assert selected.provenance["candidate_evidence_ids"] == (
        "a-first",
        "a-second",
    )
    assert selected.provenance["selected_evidence_id"] == "a-second"
    assert weighted.provenance["evidence_combination_weights"] == {
        "b-first": 0.25,
        "b-second": 0.75,
    }
    assert pooled.provenance["observed_lifts"] == {
        "c-first": 90.0,
        "c-second": 100.0,
    }
    assert pooled.provenance["matched_mta_estimates"] == {
        "c-first": 100.0,
        "c-second": 200.0,
    }


def test_applied_dispatch_rejects_missing_or_irrelevant_policy_inputs() -> None:
    with pytest.raises(ValueError, match="selected_evidence is required"):
        _resolve_evidence_conflicts(
            _request(),
            evidence_weights={
                ("B", "E2"): {"b-first": 0.25, "b-second": 0.75}
            },
        )
    with pytest.raises(ValueError, match="evidence_weights are required"):
        _resolve_evidence_conflicts(
            _request(),
            selected_evidence={("A", "E1"): "a-second"},
        )
    with pytest.raises(ValueError, match="extra"):
        _resolve_evidence_conflicts(
            _request("pooled_lift"),
            selected_evidence={("A", "E1"): "a-first"},
        )


def test_requested_diagnostics_do_not_change_applied_result_or_policy() -> None:
    request = _request()
    applied_before = _applied(request)
    policy_before = dict(request.policy_by_pair)
    diagnostics = _build_evidence_conflict_diagnostics(
        request,
        diagnostic_policies={
            ("A", "E1"): (
                "select_evidence",
                "weighted_ratio",
                "pooled_lift",
            )
        },
        selected_evidence={("A", "E1"): "a-first"},
        evidence_weights={
            ("A", "E1"): {"a-first": 0.25, "a-second": 0.75}
        },
    )
    applied_after = _applied(request)

    alternatives = diagnostics.alternatives_by_pair[("A", "E1")]
    assert tuple(alternatives) == (
        "select_evidence",
        "weighted_ratio",
        "pooled_lift",
    )
    assert alternatives["select_evidence"].effective_ratio == 0.8
    assert alternatives["weighted_ratio"].effective_ratio == pytest.approx(0.65)
    assert alternatives["pooled_lift"].effective_ratio == pytest.approx(0.7)
    assert applied_before[("A", "E1")].effective_ratio == 0.6
    assert applied_after == applied_before
    assert dict(request.policy_by_pair) == policy_before
    assert diagnostics.applied_policy_by_pair == policy_before


def test_diagnostic_state_is_ready_for_later_exports_and_plots_without_recompute() -> None:
    request = _request()
    diagnostics = _build_evidence_conflict_diagnostics(
        request,
        diagnostic_policies={("C", "E3"): ("pooled_lift",)},
    )
    assert diagnostics.evidence_points == request.evidence_points
    assert [point.evidence_id for point in diagnostics.evidence_points] == [
        "a-first",
        "a-second",
        "b-first",
        "b-second",
        "c-first",
        "c-second",
    ]
    assert [
        point.evidence_id
        for point in diagnostics.conflict_groups[("C", "E3")]
    ] == ["c-first", "c-second"]
    pooled = diagnostics.alternatives_by_pair[("C", "E3")]["pooled_lift"]
    assert pooled.provenance["pooled_observed_lift"] == 190.0
    assert pooled.provenance["pooled_matched_mta_estimate"] == 300.0


@pytest.mark.parametrize(
    "diagnostic_policies, expected",
    [
        ({("unknown", "E1"): ("pooled_lift",)}, "extra"),
        ({("A", "E1"): ()}, "must not be empty"),
        (
            {("A", "E1"): ("pooled_lift", "pooled_lift")},
            "must be unique",
        ),
        ({("A", "E1"): ("mean_ratio",)}, "unknown policies"),
    ],
)
def test_invalid_diagnostic_requests_raise(diagnostic_policies, expected) -> None:
    with pytest.raises(ValueError, match=expected):
        _build_evidence_conflict_diagnostics(
            _request(),
            diagnostic_policies=diagnostic_policies,
        )


def test_diagnostic_policy_specific_inputs_remain_mandatory() -> None:
    with pytest.raises(ValueError, match="selected_evidence is required"):
        _build_evidence_conflict_diagnostics(
            _request(),
            diagnostic_policies={("A", "E1"): ("select_evidence",)},
        )
    with pytest.raises(ValueError, match="evidence_weights are required"):
        _build_evidence_conflict_diagnostics(
            _request(),
            diagnostic_policies={("B", "E2"): ("weighted_ratio",)},
        )


def test_applied_and_diagnostic_state_are_deterministic_and_deeply_read_only() -> None:
    applied_first = _applied()
    applied_second = _applied()
    assert applied_first == applied_second
    diagnostics = _build_evidence_conflict_diagnostics(
        _request(),
        diagnostic_policies={("C", "E3"): ("pooled_lift",)},
    )
    with pytest.raises(TypeError):
        applied_first[("A", "E1")] = applied_second[("A", "E1")]
    with pytest.raises(TypeError):
        diagnostics.applied_policy_by_pair[("A", "E1")] = "pooled_lift"
    with pytest.raises(TypeError):
        diagnostics.alternatives_by_pair[("C", "E3")]["pooled_lift"] = (
            applied_first[("C", "E3")]
        )
