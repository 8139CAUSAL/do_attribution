"""Exact selected-evidence conflict resolution for CAL-049."""

import pytest

from do_attribution.calibration import (
    LiftTestEvidence,
    _build_observed_only_schedule,
    _resolve_select_evidence,
    _validate_evidence_conflict_request,
)


def _evidence(entity, test_id, coordinate, ratio):
    return LiftTestEvidence(
        tested_entity=entity,
        observed_lift=ratio * 100.0,
        matched_mta_estimate=100.0,
        test_id=test_id,
        coordinate_name="index",
        coordinate_value=coordinate,
    )


def _request(policies=None):
    schedule = _build_observed_only_schedule(
        [
            _evidence("A", "a-first", 2.0, 0.8),
            _evidence("B", "b-first", 12.0, 1.1),
            _evidence("A", "a-second", 4.0, 0.6),
            _evidence("B", "b-second", 14.0, 1.3),
        ],
        epochs=("E1", "E2"),
        epoch_windows={"E1": (0.0, 10.0), "E2": (10.0, 20.0)},
    )
    return _validate_evidence_conflict_request(
        schedule,
        evidence_conflict_policy=(
            "select_evidence" if policies is None else policies
        ),
    )


def test_selected_ratio_is_exact_and_records_selected_and_candidate_ids() -> None:
    request = _request()
    resolutions = _resolve_select_evidence(
        request,
        selected_evidence={
            ("A", "E1"): "a-second",
            ("B", "E2"): "b-first",
        },
    )

    a = resolutions[("A", "E1")]
    b = resolutions[("B", "E2")]
    assert a.effective_ratio == 0.6
    assert b.effective_ratio == 1.1
    assert a.evidence_conflict_policy == "select_evidence"
    assert a.ratio_source == "selected_from_multiple"
    assert a.source_evidence_ids == ("a-second",)
    assert a.provenance == {
        "selected_evidence_id": "a-second",
        "candidate_evidence_ids": ("a-first", "a-second"),
        "candidate_observed_ratios": {"a-first": 0.8, "a-second": 0.6},
    }


def test_nonselected_observations_remain_visible_in_request_state() -> None:
    request = _request()
    _resolve_select_evidence(
        request,
        selected_evidence={
            ("A", "E1"): "a-second",
            ("B", "E2"): "b-second",
        },
    )
    assert [point.evidence_id for point in request.evidence_points] == [
        "a-first",
        "b-first",
        "a-second",
        "b-second",
    ]
    assert [
        point.evidence_id for point in request.conflict_groups[("A", "E1")]
    ] == ["a-first", "a-second"]


@pytest.mark.parametrize("selected_id", ["missing", "b-first"])
def test_invalid_or_cross_group_selected_ids_raise(selected_id) -> None:
    with pytest.raises(ValueError, match="must belong"):
        _resolve_select_evidence(
            _request(),
            selected_evidence={
                ("A", "E1"): selected_id,
                ("B", "E2"): "b-first",
            },
        )


@pytest.mark.parametrize(
    "selected, expected",
    [
        (None, "required"),
        ({("A", "E1"): "a-first"}, "missing"),
        (
            {
                ("A", "E1"): "a-first",
                ("B", "E2"): "b-first",
                ("A", "E2"): "a-second",
            },
            "extra",
        ),
    ],
)
def test_selection_mapping_must_exactly_cover_selected_groups(
    selected, expected
) -> None:
    with pytest.raises(ValueError, match=expected):
        _resolve_select_evidence(
            _request(),
            selected_evidence=selected,
        )


def test_mixed_policy_request_requires_selection_only_for_selected_groups() -> None:
    request = _request(
        {
            ("A", "E1"): "select_evidence",
            ("B", "E2"): "weighted_ratio",
        }
    )
    resolutions = _resolve_select_evidence(
        request,
        selected_evidence={("A", "E1"): "a-first"},
    )
    assert tuple(resolutions) == (("A", "E1"),)

    with pytest.raises(ValueError, match="extra"):
        _resolve_select_evidence(
            request,
            selected_evidence={
                ("A", "E1"): "a-first",
                ("B", "E2"): "b-first",
            },
        )


def test_no_selected_groups_accepts_omission_and_rejects_selection_input() -> None:
    request = _request("pooled_lift")
    assert _resolve_select_evidence(request, selected_evidence=None) == {}
    with pytest.raises(ValueError, match="extra"):
        _resolve_select_evidence(
            request,
            selected_evidence={("A", "E1"): "a-first"},
        )


def test_resolution_is_deterministic_and_deeply_read_only() -> None:
    selected = {
        ("A", "E1"): "a-first",
        ("B", "E2"): "b-second",
    }
    first = _resolve_select_evidence(_request(), selected_evidence=selected)
    selected[("A", "E1")] = "a-second"
    second = _resolve_select_evidence(
        _request(),
        selected_evidence={
            ("A", "E1"): "a-first",
            ("B", "E2"): "b-second",
        },
    )
    assert first == second
    assert first[("A", "E1")].effective_ratio == 0.8

    with pytest.raises(TypeError):
        first[("A", "E1")] = second[("A", "E1")]
    with pytest.raises(TypeError):
        first[("A", "E1")].provenance["selected_evidence_id"] = "changed"
    with pytest.raises(TypeError):
        first[("A", "E1")].provenance["candidate_observed_ratios"][
            "a-first"
        ] = 99.0
