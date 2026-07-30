"""Stable normalization and duplicate-detection tests for CAL-009."""

from dataclasses import dataclass

import pytest

from do_attribution.calibration import (
    _normalize_estimates_by_epoch,
    _normalize_pairs,
    _normalize_unique_records,
)


def test_mapping_and_sequence_inputs_preserve_visible_order() -> None:
    mapping = {"B": 2.0, "A": 1.0}
    sequence = [("B", 2.0), ("A", 1.0)]
    assert _normalize_pairs(mapping, name="estimates") == tuple(mapping.items())
    assert _normalize_pairs(sequence, name="estimates") == tuple(sequence)


@pytest.mark.parametrize(
    "pairs",
    [
        [("A", 1.0), ("A", 2.0)],
        [(('A', "E1"), 1.0), (('A', "E1"), 2.0)],
    ],
)
def test_sequence_duplicates_raise_before_mapping_conversion(pairs) -> None:
    with pytest.raises(ValueError, match="duplicate key") as error:
        _normalize_pairs(pairs, name="schedule")
    assert repr(pairs[0][0]) in str(error.value)


def test_empty_and_malformed_pair_inputs_raise() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _normalize_pairs({}, name="estimates")
    with pytest.raises(TypeError, match="two-item pair"):
        _normalize_pairs([("A", 1.0, "extra")], name="estimates")
    with pytest.raises(TypeError, match="mapping or a sequence of pairs"):
        _normalize_pairs("A", name="estimates")


def test_mapping_cannot_recover_an_already_overwritten_key() -> None:
    overwritten = dict([("A", 1.0), ("A", 2.0)])
    assert _normalize_pairs(overwritten, name="estimates") == (("A", 2.0),)
    assert "surviving keys" in (_normalize_pairs.__doc__ or "")
    assert "makes no recovery claim" in (_normalize_pairs.__doc__ or "")


def test_nested_values_are_defensively_copied() -> None:
    nested = {"A": {"weights": [0.25, 0.75]}}
    normalized = _normalize_pairs(nested, name="nested")
    nested["A"]["weights"].append(1.0)
    assert normalized == (("A", {"weights": [0.25, 0.75]}),)


def test_epoch_and_entity_order_are_stable_and_nested_inputs_are_copied() -> None:
    source = {
        "E2": [("B", 2.0), ("A", 1.0)],
        "E1": {"C": 3.0},
    }
    normalized = _normalize_estimates_by_epoch(source)
    source["E2"].append(("C", 4.0))
    source["E1"]["D"] = 5.0
    assert normalized == (
        ("E2", (("B", 2.0), ("A", 1.0))),
        ("E1", (("C", 3.0),)),
    )
    assert _normalize_estimates_by_epoch({"E2": [("B", 2.0)]}) == (
        ("E2", (("B", 2.0),)),
    )


@dataclass
class _Record:
    evidence_id: str
    metadata: dict[str, list[int]]


def test_record_ids_are_unique_ordered_and_deeply_copied() -> None:
    records = [
        _Record("b", {"values": [2]}),
        _Record("a", {"values": [1]}),
    ]
    normalized = _normalize_unique_records(
        records, id_attribute="evidence_id", name="evidence"
    )
    records[0].metadata["values"].append(3)
    assert [record.evidence_id for record in normalized] == ["b", "a"]
    assert normalized[0].metadata == {"values": [2]}


def test_duplicate_and_missing_record_ids_raise() -> None:
    with pytest.raises(ValueError, match="duplicate evidence_id 'same'"):
        _normalize_unique_records(
            [_Record("same", {}), _Record("same", {})],
            id_attribute="evidence_id",
            name="evidence",
        )
    with pytest.raises(TypeError, match="must expose 'evidence_id'"):
        _normalize_unique_records(
            [object()], id_attribute="evidence_id", name="evidence"
        )


def test_repeated_normalization_is_deterministic() -> None:
    source = [("A", {"x": [1]}), ("B", {"x": [2]})]
    assert _normalize_pairs(source, name="source") == _normalize_pairs(
        source, name="source"
    )
