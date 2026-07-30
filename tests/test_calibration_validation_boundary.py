"""Behavior contracts for the calibration validation responsibility."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from types import MappingProxyType

import pytest

from do_attribution.calibration import (
    _coordinate_kind,
    _deep_export_copy,
    _deep_freeze,
    _normalize_estimates_by_epoch,
    _normalize_pairs,
    _normalize_unique_records,
)


def test_pair_normalization_preserves_order_and_defensively_copies() -> None:
    key = ("entity",)
    nested = {"values": [1, 2]}
    result = _normalize_pairs([(key, nested), ("B", 3)], name="values")

    nested["values"].append(4)
    assert result == ((("entity",), {"values": [1, 2]}), ("B", 3))

    with pytest.raises(ValueError, match="duplicate key 'A'"):
        _normalize_pairs([("A", 1), ("A", 2)], name="values")
    with pytest.raises(TypeError, match="must be hashable"):
        _normalize_pairs([(["A"], 1)], name="values")


def test_epoch_normalization_preserves_authoritative_mapping_order() -> None:
    estimates = {
        "E2": [("B", 2.0), ("A", 1.0)],
        "E1": {"A": 3.0},
    }
    assert _normalize_estimates_by_epoch(estimates) == (
        ("E2", (("B", 2.0), ("A", 1.0))),
        ("E1", (("A", 3.0),)),
    )


@dataclass
class _Record:
    identifier: str
    metadata: list[int]


def test_unique_record_normalization_rejects_duplicate_ids_and_copies() -> None:
    records = [_Record("one", [1]), _Record("two", [2])]
    normalized = _normalize_unique_records(
        records,
        id_attribute="identifier",
        name="records",
    )
    records[0].metadata.append(99)
    assert normalized[0].metadata == [1]

    with pytest.raises(ValueError, match="duplicate identifier 'same'"):
        _normalize_unique_records(
            [_Record("same", []), _Record("same", [])],
            id_attribute="identifier",
            name="records",
        )


def test_deep_freeze_and_export_copy_preserve_nested_values_without_aliasing() -> None:
    source = {
        "sequence": [{"value": 1}],
        "members": {"A", "B"},
    }
    frozen = _deep_freeze(source)

    assert isinstance(frozen, MappingProxyType)
    assert isinstance(frozen["sequence"], tuple)
    assert isinstance(frozen["sequence"][0], MappingProxyType)
    assert isinstance(frozen["members"], frozenset)
    with pytest.raises(TypeError):
        frozen["new"] = "value"

    exported = _deep_export_copy(frozen)
    exported["sequence"][0]["value"] = 2
    assert frozen["sequence"][0]["value"] == 1


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.5, "numeric"),
        (date(2024, 1, 1), "date"),
        (datetime(2024, 1, 1, 12, 30), "datetime"),
    ],
)
def test_coordinate_kind_classifies_supported_families(value, expected) -> None:
    assert _coordinate_kind(value, name="coordinate") == expected


@pytest.mark.parametrize("value", [True, "2024-01-01", object()])
def test_coordinate_kind_rejects_unsupported_values(value) -> None:
    with pytest.raises(
        TypeError,
        match="must be a finite number, date, or datetime",
    ):
        _coordinate_kind(value, name="coordinate")
