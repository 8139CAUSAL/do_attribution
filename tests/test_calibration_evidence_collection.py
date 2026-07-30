"""Evidence collection and calibrator-construction tests for CAL-012/CAL-019."""

from dataclasses import dataclass

import pytest

from do_attribution.calibration import (
    AttributionCalibration,
    LiftTestEvidence,
    TemporalAttributionCalibration,
    _conflict_groups_by_epoch,
)


def _evidence(entity: str, test_id: str, ratio: float = 1.0) -> LiftTestEvidence:
    return LiftTestEvidence(
        tested_entity=entity,
        observed_lift=ratio * 100.0,
        matched_mta_estimate=100.0,
        test_id=test_id,
        metadata={"source": {"ids": [test_id]}},
    )


def test_duplicate_ids_raise_with_identifier() -> None:
    with pytest.raises(ValueError, match="duplicate test_id 'same'"):
        TemporalAttributionCalibration(
            evidence=[_evidence("A", "same"), _evidence("B", "same")]
        )


def test_static_constructor_rejects_duplicate_evidence_ids() -> None:
    with pytest.raises(ValueError, match="duplicate test_id 'same'"):
        AttributionCalibration(
            evidence=[_evidence("A", "same"), _evidence("B", "same")]
        )


def test_static_same_entity_evidence_is_ambiguous_without_static_conflict_api() -> None:
    with pytest.raises(ValueError, match="exactly one evidence record.*'A'"):
        AttributionCalibration(
            evidence=[_evidence("A", "a-1"), _evidence("A", "a-2")]
        )


def test_static_constructor_preserves_order_and_builds_immutable_indexes() -> None:
    records = [
        _evidence("B", "b-1", 0.8),
        _evidence("A", "a-1", 1.2),
        _evidence("C", "c-1", 1.0),
    ]

    calibrator = AttributionCalibration(evidence=records)
    records.reverse()

    assert [item.test_id for item in calibrator._evidence] == [
        "b-1",
        "a-1",
        "c-1",
    ]
    assert list(calibrator._evidence_by_entity) == ["B", "A", "C"]
    assert [item.test_id for item in calibrator._evidence_by_entity["B"]] == [
        "b-1"
    ]
    assert list(calibrator._evidence_by_id) == ["b-1", "a-1", "c-1"]
    assert calibrator._evidence_by_id["a-1"].tested_entity == "A"

    with pytest.raises(KeyError, match="unknown-evidence"):
        calibrator._evidence_by_id["unknown-evidence"]
    with pytest.raises(TypeError):
        calibrator._evidence_by_entity["D"] = (_evidence("D", "d-1"),)
    with pytest.raises(TypeError):
        calibrator._evidence_by_id["d-1"] = _evidence("D", "d-1")


def test_static_constructor_stores_evidence_only_and_selects_no_policy() -> None:
    calibrator = AttributionCalibration(evidence=[_evidence("A", "a-1")])

    assert set(vars(calibrator)) == {
        "_evidence",
        "_evidence_by_entity",
        "_evidence_by_id",
    }
    assert not any("policy" in name for name in vars(calibrator))


def test_temporal_collection_retains_multiple_entities_and_observations_in_order() -> None:
    records = [
        _evidence("B", "b-1"),
        _evidence("A", "a-1"),
        _evidence("B", "b-2"),
    ]
    calibrator = TemporalAttributionCalibration(evidence=records)
    records.reverse()
    assert [item.test_id for item in calibrator._evidence] == ["b-1", "a-1", "b-2"]
    assert list(calibrator._evidence_by_entity) == ["B", "A"]
    assert [item.test_id for item in calibrator._evidence_by_entity["B"]] == [
        "b-1",
        "b-2",
    ]
    with pytest.raises(TypeError):
        calibrator._evidence_by_id["new"] = _evidence("C", "c-1")


def test_empty_and_non_evidence_collections_raise() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        AttributionCalibration(evidence=[])
    with pytest.raises(TypeError, match=r"evidence\[0\]"):
        TemporalAttributionCalibration(evidence=[object()])


@dataclass(frozen=True)
class _MatchedRecord:
    tested_entity: str
    evidence_id: str
    matched_epoch: str | None


def test_conflict_groups_are_identified_without_selection_or_aggregation() -> None:
    records = (
        _MatchedRecord("A", "a-1", "E1"),
        _MatchedRecord("A", "a-2", "E1"),
        _MatchedRecord("B", "b-1", "E1"),
        _MatchedRecord("A", "a-3", "E2"),
        _MatchedRecord("A", "a-outside", None),
    )
    groups = _conflict_groups_by_epoch(records)
    assert list(groups) == [("A", "E1")]
    assert [record.evidence_id for record in groups[("A", "E1")]] == ["a-1", "a-2"]
    assert len(records) == 5
    with pytest.raises(TypeError):
        groups[("A", "E2")] = ()
