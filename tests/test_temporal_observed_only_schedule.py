"""Observed-only schedule construction for CAL-041."""

import pytest

from do_attribution.calibration import (
    LiftTestEvidence,
    _build_observed_only_schedule,
)


def _evidence(entity: str, test_id: str, coordinate: float, ratio: float):
    return LiftTestEvidence(
        tested_entity=entity,
        observed_lift=ratio * 100.0,
        matched_mta_estimate=100.0,
        test_id=test_id,
        coordinate_name="time",
        coordinate_value=coordinate,
        metadata={"source": {"id": test_id}},
    )


def _schedule():
    evidence = [
        _evidence("A", "a-e1", 0.5, 0.8),
        _evidence("A", "a-e2-first", 1.25, 1.1),
        _evidence("B", "b-e3", 2.5, 0.9),
        _evidence("A", "a-e2-second", 1.75, 1.2),
        _evidence("A", "a-outside", 4.0, 0.7),
    ]
    windows = {"E1": (0.0, 1.0), "E2": (1.0, 2.0), "E3": (2.0, 3.0)}
    return evidence, windows, _build_observed_only_schedule(
        evidence,
        epochs=("E1", "E2", "E3"),
        epoch_windows=windows,
    )


def test_observed_rows_and_candidates_preserve_every_evidence_point() -> None:
    _, _, schedule = _schedule()
    expected_ids = [
        "a-e1",
        "a-e2-first",
        "b-e3",
        "a-e2-second",
        "a-outside",
    ]

    assert [point.evidence_id for point in schedule.evidence_points] == expected_ids
    assert [
        row.source_evidence_ids[0] for row in schedule.coordinate_schedule
    ] == expected_ids
    assert [row.schedule_ratio for row in schedule.coordinate_schedule] == [
        0.8,
        1.1,
        0.9,
        1.2,
        0.7,
    ]
    assert all(row.schedule_source == "observed" for row in schedule.coordinate_schedule)
    assert all(row.inside_observed_support for row in schedule.coordinate_schedule)
    assert schedule.evidence_points[-1].matched_epoch is None


def test_only_matched_pairs_become_candidates_in_authoritative_order() -> None:
    _, _, schedule = _schedule()
    assert tuple(schedule.candidates) == (
        ("A", "E1"),
        ("A", "E2"),
        ("B", "E3"),
    )
    assert tuple(schedule.status_by_pair) == (
        ("A", "E1"),
        ("B", "E1"),
        ("A", "E2"),
        ("B", "E2"),
        ("A", "E3"),
        ("B", "E3"),
    )
    assert schedule.status_by_pair == {
        ("A", "E1"): "observed",
        ("B", "E1"): "not_calibrated",
        ("A", "E2"): "conflict_pending",
        ("B", "E2"): "not_calibrated",
        ("A", "E3"): "not_calibrated",
        ("B", "E3"): "observed",
    }


def test_multiple_observations_remain_unresolved_and_ordered() -> None:
    _, _, schedule = _schedule()
    conflict = schedule.candidates[("A", "E2")]
    assert [point.evidence_id for point in conflict] == [
        "a-e2-first",
        "a-e2-second",
    ]
    assert [point.observed_ratio for point in conflict] == [1.1, 1.2]
    assert all(
        point.conflict_group == ("a-e2-first", "a-e2-second")
        for point in conflict
    )
    assert not hasattr(schedule, "effective_ratio")
    assert not hasattr(schedule, "selected_evidence")


def test_schedule_state_and_caller_inputs_are_immutable_and_isolated() -> None:
    evidence, windows, schedule = _schedule()
    evidence.reverse()
    windows["E1"] = (10.0, 11.0)

    assert schedule.evidence_points[0].evidence_id == "a-e1"
    assert schedule.evidence_points[0].matched_epoch == "E1"
    with pytest.raises(TypeError):
        schedule.candidates[("A", "E3")] = ()
    with pytest.raises(TypeError):
        schedule.status_by_pair[("A", "E3")] = "observed"
    with pytest.raises(TypeError):
        schedule.evidence_points[0].metadata["new"] = "value"


def test_observed_only_requires_windows_and_coordinates() -> None:
    evidence = [_evidence("A", "a", 0.5, 0.9)]
    with pytest.raises(ValueError, match="epoch_windows"):
        _build_observed_only_schedule(
            evidence,
            epochs=("E1",),
            epoch_windows=None,
        )
    coordinate_less = LiftTestEvidence(
        tested_entity="A",
        observed_lift=90.0,
        matched_mta_estimate=100.0,
        test_id="a-static",
    )
    with pytest.raises(ValueError, match="explicit evidence coordinates"):
        _build_observed_only_schedule(
            [coordinate_less],
            epochs=("E1",),
            epoch_windows={"E1": (0.0, 1.0)},
        )
