"""Explicitly sourced constant schedule tests for CAL-042."""

import pytest

from do_attribution.calibration import LiftTestEvidence, _build_constant_schedule


def _evidence(entity: str, test_id: str, ratio: float):
    return LiftTestEvidence(
        tested_entity=entity,
        observed_lift=ratio * 100.0,
        matched_mta_estimate=100.0,
        test_id=test_id,
        coordinate_name="time",
        coordinate_value=ratio,
    )


def _records():
    return [
        _evidence("B", "b-first", 0.8),
        _evidence("A", "a-first", 1.1),
        _evidence("A", "a-selected", 1.2),
        _evidence("C", "c-unused", 0.7),
    ]


def test_constant_schedule_uses_only_explicit_sources_and_epochs() -> None:
    schedule = _build_constant_schedule(
        _records(),
        epochs=("E1", "E2", "E3"),
        source_evidence={"A": "a-selected", "B": "b-first"},
        tested_entity_epochs={"A": ["E3", "E1"], "B": ["E2", "E1"]},
    )

    assert [row.tested_entity for row in schedule.coordinate_schedule] == ["B", "A"]
    assert [row.schedule_ratio for row in schedule.coordinate_schedule] == [0.8, 1.2]
    assert [row.source_evidence_ids for row in schedule.coordinate_schedule] == [
        ("b-first",),
        ("a-selected",),
    ]
    assert tuple(schedule.ratios_by_pair) == (
        ("B", "E1"),
        ("A", "E1"),
        ("B", "E2"),
        ("A", "E3"),
    )
    assert schedule.ratios_by_pair == {
        ("B", "E1"): 0.8,
        ("A", "E1"): 1.2,
        ("B", "E2"): 0.8,
        ("A", "E3"): 1.2,
    }


def test_constant_schedule_records_transport_and_not_calibrated_provenance() -> None:
    schedule = _build_constant_schedule(
        _records(),
        epochs=("E1", "E2"),
        source_evidence={"A": "a-selected"},
        tested_entity_epochs={"A": ["E2"]},
    )

    assert schedule.status_by_pair[("A", "E2")] == "constant"
    assert schedule.provenance_by_pair[("A", "E2")] == {
        "ratio_source": "constant",
        "source_evidence_id": "a-selected",
        "transport_assumption": True,
    }
    assert schedule.source_evidence_ids_by_pair[("A", "E2")] == ("a-selected",)
    for pair in (("B", "E1"), ("A", "E1"), ("C", "E1"), ("B", "E2"), ("C", "E2")):
        assert schedule.status_by_pair[pair] == "not_calibrated"
        assert schedule.provenance_by_pair[pair] == {
            "ratio_source": "not_calibrated"
        }


@pytest.mark.parametrize(
    ("source_evidence", "tested_entity_epochs", "message"),
    [
        ({"A": "a-selected"}, {"B": ["E1"]}, "exact entity coverage"),
        ({"A": "missing"}, {"A": ["E1"]}, "unknown evidence ID"),
        ({"A": "b-first"}, {"A": ["E1"]}, "belongs to tested entity"),
        ({"D": "a-selected"}, {"D": ["E1"]}, "without evidence"),
        ({"A": "a-selected"}, {"A": ["E9"]}, "unknown epochs"),
        ({"A": "a-selected"}, {"A": ["E1", "E1"]}, "duplicate key"),
        ({"A": "a-selected"}, {"A": []}, "must not be empty"),
        ({"A": "a-selected"}, {"A": "E1"}, "sequence of hashable keys"),
    ],
)
def test_invalid_sources_ownership_and_epoch_declarations_raise(
    source_evidence, tested_entity_epochs, message
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _build_constant_schedule(
            _records(),
            epochs=("E1", "E2"),
            source_evidence=source_evidence,
            tested_entity_epochs=tested_entity_epochs,
        )


def test_constant_schedule_is_immutable_and_isolated_from_caller_inputs() -> None:
    evidence = _records()
    sources = {"A": "a-selected"}
    declarations = {"A": ["E1"]}
    schedule = _build_constant_schedule(
        evidence,
        epochs=("E1", "E2"),
        source_evidence=sources,
        tested_entity_epochs=declarations,
    )
    evidence.reverse()
    sources["A"] = "a-first"
    declarations["A"].append("E2")

    assert schedule.ratios_by_pair == {("A", "E1"): 1.2}
    with pytest.raises(TypeError):
        schedule.ratios_by_pair[("A", "E2")] = 1.1
    with pytest.raises(TypeError):
        schedule.provenance_by_pair[("A", "E1")]["transport_assumption"] = False
