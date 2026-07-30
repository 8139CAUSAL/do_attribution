"""Exact explicit ratio-schedule tests for CAL-043."""

import math

import pytest

from do_attribution.calibration import (
    LiftTestEvidence,
    _build_explicit_ratio_schedule,
)


def _evidence(entity: str, test_id: str):
    return LiftTestEvidence(
        tested_entity=entity,
        observed_lift=100.0,
        matched_mta_estimate=100.0,
        test_id=test_id,
    )


def _records():
    return [_evidence("B", "b-test"), _evidence("A", "a-test"), _evidence("C", "c-test")]


def test_explicit_ratios_apply_exactly_in_authoritative_pair_order() -> None:
    schedule = _build_explicit_ratio_schedule(
        _records(),
        epochs=("E1", "E2", "E3"),
        ratio_schedule=[
            (("A", "E3"), 0.0),
            (("A", "E1"), 1.2),
            (("B", "E1"), 0.8),
            (("B", "E2"), 1.0),
        ],
    )

    assert tuple(schedule.ratios_by_pair) == (
        ("B", "E1"),
        ("A", "E1"),
        ("B", "E2"),
        ("A", "E3"),
    )
    assert schedule.ratios_by_pair == {
        ("B", "E1"): 0.8,
        ("A", "E1"): 1.2,
        ("B", "E2"): 1.0,
        ("A", "E3"): 0.0,
    }
    assert all(
        schedule.status_by_pair[pair] == "explicit"
        for pair in schedule.ratios_by_pair
    )
    assert schedule.status_by_pair[("C", "E2")] == "not_calibrated"
    assert schedule.provenance_by_pair[("C", "E2")] == {
        "ratio_source": "not_calibrated"
    }


def test_optional_tested_entity_epoch_declaration_must_match_exactly() -> None:
    exact = _build_explicit_ratio_schedule(
        _records(),
        epochs=("E1", "E2"),
        ratio_schedule={("A", "E1"): 0.9, ("A", "E2"): 1.1},
        tested_entity_epochs={"A": ["E2", "E1"]},
    )
    assert exact.ratios_by_pair == {("A", "E1"): 0.9, ("A", "E2"): 1.1}

    with pytest.raises(ValueError, match=r"missing=.*\('A', 'E2'\)"):
        _build_explicit_ratio_schedule(
            _records(),
            epochs=("E1", "E2"),
            ratio_schedule={("A", "E1"): 0.9},
            tested_entity_epochs={"A": ["E1", "E2"]},
        )
    with pytest.raises(ValueError, match=r"extra=.*\('A', 'E2'\)"):
        _build_explicit_ratio_schedule(
            _records(),
            epochs=("E1", "E2"),
            ratio_schedule={("A", "E1"): 0.9, ("A", "E2"): 1.1},
            tested_entity_epochs={"A": ["E1"]},
        )


def test_sequence_duplicate_pairs_raise_before_mapping_conversion() -> None:
    with pytest.raises(ValueError, match="duplicate key"):
        _build_explicit_ratio_schedule(
            _records(),
            epochs=("E1",),
            ratio_schedule=[(("A", "E1"), 0.9), (("A", "E1"), 1.1)],
        )


@pytest.mark.parametrize(
    ("ratio_schedule", "message"),
    [
        ({("D", "E1"): 0.9}, "unknown tested entity"),
        ({("A", "E9"): 0.9}, "unknown epoch"),
        ({"A": 0.9}, "keys must be"),
        ({("A", "E1"): math.nan}, "must be finite"),
        ({("A", "E1"): math.inf}, "must be finite"),
        ({("A", "E1"): True}, "real number"),
        ({("A", "E1"): -0.1}, "allow_signed"),
    ],
)
def test_invalid_pairs_numbers_and_unsigned_ratios_raise(ratio_schedule, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _build_explicit_ratio_schedule(
            _records(),
            epochs=("E1",),
            ratio_schedule=ratio_schedule,
        )


def test_signed_ratios_require_and_honor_explicit_permission() -> None:
    schedule = _build_explicit_ratio_schedule(
        _records(),
        epochs=("E1",),
        ratio_schedule={("A", "E1"): -0.25},
        allow_signed=True,
    )
    assert schedule.ratios_by_pair[("A", "E1")] == -0.25
    with pytest.raises(TypeError, match="allow_signed must be a bool"):
        _build_explicit_ratio_schedule(
            _records(),
            epochs=("E1",),
            ratio_schedule={("A", "E1"): 0.9},
            allow_signed=1,
        )


def test_schedule_is_immutable_isolated_and_never_fills_missing_pairs() -> None:
    ratios = [(('A', "E1"), 0.9)]
    declarations = {"A": ["E1"]}
    schedule = _build_explicit_ratio_schedule(
        _records(),
        epochs=("E1", "E2"),
        ratio_schedule=ratios,
        tested_entity_epochs=declarations,
    )
    ratios.append((("A", "E2"), 1.1))
    declarations["A"].append("E2")

    assert schedule.ratios_by_pair == {("A", "E1"): 0.9}
    assert schedule.status_by_pair[("A", "E2")] == "not_calibrated"
    with pytest.raises(TypeError):
        schedule.ratios_by_pair[("A", "E2")] = 1.1
    with pytest.raises(TypeError):
        schedule.provenance_by_pair[("A", "E1")]["ratio_source"] = "changed"
