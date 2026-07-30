"""Cross-policy ratio-schedule gate for CAL-047."""

from datetime import date
import inspect

import pytest

from do_attribution.calibration import (
    LiftTestEvidence,
    _build_constant_schedule,
    _build_coordinate_linear_schedule,
    _build_explicit_ratio_schedule,
    _build_explicit_strength_schedule,
    _build_observed_only_schedule,
    _evaluate_coordinate_linear_schedule,
    _normalize_authoritative_epoch_windows,
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


def _records():
    return [
        _evidence("B", "b-left", 0.0, 1.0),
        _evidence("A", "a-left", 0.0, 0.8),
        _evidence("B", "b-right", 10.0, 0.5),
        _evidence("A", "a-right", 10.0, 1.2),
    ]


def test_all_five_policies_keep_multiple_entity_schedules_independent() -> None:
    evidence = _records()
    epochs = ("E1", "E2")
    windows = {"E1": (0.0, 5.0), "E2": (5.0, 10.0)}

    observed = _build_observed_only_schedule(
        evidence,
        epochs=epochs,
        epoch_windows=windows,
    )
    constant = _build_constant_schedule(
        evidence,
        epochs=epochs,
        source_evidence={"A": "a-left", "B": "b-right"},
        tested_entity_epochs={"A": ["E2"], "B": ["E1"]},
    )
    explicit = _build_explicit_ratio_schedule(
        evidence,
        epochs=epochs,
        ratio_schedule={("A", "E1"): 0.9, ("B", "E1"): 1.1},
    )
    strength = _build_explicit_strength_schedule(
        evidence,
        epochs=epochs,
        source_evidence={"A": "a-left", "B": "b-right"},
        evidence_strength={("A", "E2"): 0.5, ("B", "E2"): 0.5},
    )
    coordinate = _build_coordinate_linear_schedule(evidence)

    assert observed.status_by_pair[("A", "E1")] == "observed"
    assert observed.status_by_pair[("B", "E2")] == "observed"
    assert constant.ratios_by_pair == {("B", "E1"): 0.5, ("A", "E2"): 0.8}
    assert explicit.ratios_by_pair == {("B", "E1"): 1.1, ("A", "E1"): 0.9}
    assert strength.ratios_by_pair[("A", "E2")] == pytest.approx(0.9)
    assert strength.ratios_by_pair[("B", "E2")] == pytest.approx(0.75)
    assert _evaluate_coordinate_linear_schedule(
        coordinate, tested_entity="A", coordinate_value=5.0
    ).schedule_ratio == pytest.approx(1.0)
    assert _evaluate_coordinate_linear_schedule(
        coordinate, tested_entity="B", coordinate_value=5.0
    ).schedule_ratio == pytest.approx(0.75)


def test_missing_pairs_are_never_filled_and_provenance_is_complete() -> None:
    evidence = _records()
    schedule = _build_explicit_ratio_schedule(
        evidence,
        epochs=("E1", "E2"),
        ratio_schedule={("A", "E1"): 0.9},
    )
    assert schedule.ratios_by_pair == {("A", "E1"): 0.9}
    assert schedule.status_by_pair[("A", "E2")] == "not_calibrated"
    assert schedule.status_by_pair[("B", "E1")] == "not_calibrated"
    assert schedule.provenance_by_pair[("A", "E1")] == {
        "ratio_source": "explicit",
        "source_evidence_ids": (),
    }

    coordinate = _build_coordinate_linear_schedule(evidence)
    unsupported = _evaluate_coordinate_linear_schedule(
        coordinate,
        tested_entity="A",
        coordinate_value=20.0,
    )
    assert unsupported.schedule_ratio is None
    assert unsupported.schedule_source == "unsupported"
    assert unsupported.source_evidence_ids == ()


def test_schedules_are_deterministic_immutable_and_cannot_change_epoch_grid() -> None:
    evidence = _records()
    first = _build_coordinate_linear_schedule(evidence)
    second = _build_coordinate_linear_schedule(evidence)
    assert first == second
    assert [row.source_evidence_ids for row in first.anchor_rows] == [
        ("b-left",),
        ("b-right",),
        ("a-left",),
        ("a-right",),
    ]
    assert "epoch" not in inspect.signature(_build_coordinate_linear_schedule).parameters

    windows = {"E1": (date(2026, 1, 1), date(2026, 2, 1))}
    before = _normalize_authoritative_epoch_windows(windows, epochs=("E1",))
    _build_coordinate_linear_schedule(evidence)
    after = _normalize_authoritative_epoch_windows(windows, epochs=("E1",))
    assert before == after
    with pytest.raises(TypeError):
        first.normalized_anchors_by_entity["A"] = ()


def test_string_epoch_labels_cannot_enter_coordinate_schedule_arithmetic() -> None:
    with pytest.raises((TypeError, ValueError), match="coordinate_value"):
        _evidence("A", "a-q1", "Q1", 0.9)
