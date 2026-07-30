"""Explicit evidence-strength schedule tests for CAL-044."""

import math

import pytest

from do_attribution.calibration import (
    LiftTestEvidence,
    _build_explicit_strength_schedule,
)


def _evidence(entity: str, test_id: str, ratio: float):
    return LiftTestEvidence(
        tested_entity=entity,
        observed_lift=ratio * 100.0,
        matched_mta_estimate=100.0,
        test_id=test_id,
    )


def _records():
    return [
        _evidence("B", "b-test", 1.2),
        _evidence("A", "a-first", 0.8),
        _evidence("A", "a-selected", 0.9),
        _evidence("C", "c-unused", 0.7),
    ]


def test_documented_point_nine_strength_identity_is_exact() -> None:
    strengths = [1.0, 0.75, 0.5, 0.25, 0.0]
    schedule = _build_explicit_strength_schedule(
        _records(),
        epochs=("E1", "E2", "E3", "E4", "E5"),
        source_evidence={"A": "a-selected"},
        evidence_strength={
            ("A", f"E{index}"): strength
            for index, strength in enumerate(strengths, start=1)
        },
    )

    assert [schedule.ratios_by_pair[("A", f"E{index}")] for index in range(1, 6)] == pytest.approx(
        [0.9, 0.925, 0.95, 0.975, 1.0]
    )
    assert schedule.strengths_by_pair[("A", "E1")] == 1.0
    assert schedule.ratios_by_pair[("A", "E1")] == 0.9
    assert schedule.strengths_by_pair[("A", "E5")] == 0.0
    assert schedule.ratios_by_pair[("A", "E5")] == 1.0
    assert schedule.provenance_by_pair[("A", "E3")] == {
        "ratio_source": "strength_derived",
        "source_evidence_id": "a-selected",
        "observed_ratio": 0.9,
        "evidence_strength": 0.5,
        "effective_ratio": 0.95,
    }


def test_sources_and_strengths_remain_independent_by_entity() -> None:
    schedule = _build_explicit_strength_schedule(
        _records(),
        epochs=("E1", "E2"),
        source_evidence={"A": "a-selected", "B": "b-test"},
        evidence_strength={("A", "E1"): 0.5, ("B", "E1"): 0.5},
    )
    assert tuple(schedule.ratios_by_pair) == (("B", "E1"), ("A", "E1"))
    assert schedule.ratios_by_pair[("A", "E1")] == pytest.approx(0.95)
    assert schedule.ratios_by_pair[("B", "E1")] == pytest.approx(1.1)
    assert schedule.source_evidence_ids_by_pair[("A", "E1")] == ("a-selected",)
    assert schedule.status_by_pair[("C", "E2")] == "not_calibrated"


@pytest.mark.parametrize(
    ("source_evidence", "strengths", "message"),
    [
        ({"B": "b-test"}, {("A", "E1"): 0.5}, "exactly cover"),
        ({"A": "missing"}, {("A", "E1"): 0.5}, "unknown evidence ID"),
        ({"A": "b-test"}, {("A", "E1"): 0.5}, "belongs to tested entity"),
        ({"A": "a-selected"}, {("D", "E1"): 0.5}, "unknown tested entity"),
        ({"A": "a-selected"}, {("A", "E9"): 0.5}, "unknown epoch"),
        ({"A": "a-selected"}, {"A": 0.5}, "keys must be"),
        ({"A": "a-selected"}, {("A", "E1"): -0.1}, "nonnegative"),
        ({"A": "a-selected"}, {("A", "E1"): 1.1}, r"\[0, 1\]"),
        ({"A": "a-selected"}, {("A", "E1"): math.nan}, "finite"),
        ({"A": "a-selected"}, {("A", "E1"): True}, "real number"),
    ],
)
def test_invalid_sources_pairs_and_strength_values_raise(
    source_evidence, strengths, message
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _build_explicit_strength_schedule(
            _records(),
            epochs=("E1",),
            source_evidence=source_evidence,
            evidence_strength=strengths,
        )


def test_optional_epoch_declarations_require_exact_strength_coverage() -> None:
    with pytest.raises(ValueError, match="must exactly cover"):
        _build_explicit_strength_schedule(
            _records(),
            epochs=("E1", "E2"),
            source_evidence={"A": "a-selected"},
            evidence_strength={("A", "E1"): 0.5},
            tested_entity_epochs={"A": ["E1", "E2"]},
        )


def test_duplicate_strength_pairs_raise_before_mapping_conversion() -> None:
    with pytest.raises(ValueError, match="duplicate key"):
        _build_explicit_strength_schedule(
            _records(),
            epochs=("E1",),
            source_evidence={"A": "a-selected"},
            evidence_strength=[(("A", "E1"), 0.5), (("A", "E1"), 0.75)],
        )


def test_signed_source_evidence_requires_explicit_permission() -> None:
    signed = [_evidence("A", "a-signed", -0.5)]
    with pytest.raises(ValueError, match="allow_signed=True"):
        _build_explicit_strength_schedule(
            signed,
            epochs=("E1",),
            source_evidence={"A": "a-signed"},
            evidence_strength={("A", "E1"): 1.0},
        )
    schedule = _build_explicit_strength_schedule(
        signed,
        epochs=("E1",),
        source_evidence={"A": "a-signed"},
        evidence_strength={("A", "E1"): 1.0},
        allow_signed=True,
    )
    assert schedule.ratios_by_pair[("A", "E1")] == -0.5


def test_strength_schedule_is_immutable_isolated_and_never_fills() -> None:
    strengths = [(('A', "E1"), 0.5)]
    schedule = _build_explicit_strength_schedule(
        _records(),
        epochs=("E1", "E2"),
        source_evidence={"A": "a-selected"},
        evidence_strength=strengths,
    )
    strengths.append((("A", "E2"), 1.0))

    assert schedule.ratios_by_pair == {("A", "E1"): pytest.approx(0.95)}
    assert schedule.status_by_pair[("A", "E2")] == "not_calibrated"
    with pytest.raises(TypeError):
        schedule.strengths_by_pair[("A", "E2")] = 1.0
    with pytest.raises(TypeError):
        schedule.provenance_by_pair[("A", "E1")]["evidence_strength"] = 1.0
