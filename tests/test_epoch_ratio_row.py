"""Immutable effective epoch-ratio provenance tests for CAL-015."""

from dataclasses import FrozenInstanceError, fields
from datetime import date

import pytest

from do_attribution.calibration import EpochRatioRow, _validate_epoch_ratio_rows


EXPECTED_FIELDS = [
    "tested_entity",
    "epoch",
    "epoch_start",
    "epoch_end",
    "epoch_coordinate",
    "epoch_support",
    "effective_ratio",
    "ratio_policy",
    "epoch_ratio_policy",
    "ratio_source",
    "evidence_conflict_policy",
    "source_evidence_ids",
    "evidence_weights",
    "left_source_evidence_id",
    "right_source_evidence_id",
    "inside_observed_support",
    "ratio_bound_lower",
    "ratio_bound_upper",
]


def _row(**overrides) -> EpochRatioRow:
    values = {
        "tested_entity": "A",
        "epoch": "E1",
        "epoch_start": date(2026, 1, 1),
        "epoch_end": date(2026, 2, 1),
        "epoch_coordinate": date(2026, 1, 15),
        "epoch_support": 120.0,
        "effective_ratio": 0.95,
        "ratio_policy": "coordinate_linear",
        "epoch_ratio_policy": "representative_coordinate",
        "ratio_source": "interpolated",
        "evidence_conflict_policy": None,
        "source_evidence_ids": ("a-left", "a-right"),
        "evidence_weights": None,
        "left_source_evidence_id": "a-left",
        "right_source_evidence_id": "a-right",
        "inside_observed_support": True,
        "ratio_bound_lower": 0.5,
        "ratio_bound_upper": 1.5,
    }
    values.update(overrides)
    return EpochRatioRow(**values)


def test_fields_match_plan_order_and_complete_provenance_is_retained() -> None:
    row = _row()
    assert [item.name for item in fields(row)] == EXPECTED_FIELDS
    assert row.epoch_start == date(2026, 1, 1)
    assert row.epoch_coordinate == date(2026, 1, 15)
    assert row.epoch_support == 120.0
    assert row.ratio_policy == "coordinate_linear"
    assert row.epoch_ratio_policy == "representative_coordinate"
    assert row.ratio_source == "interpolated"
    assert row.source_evidence_ids == ("a-left", "a-right")
    assert row.effective_ratio == 0.95
    assert not hasattr(row, "schedule_ratio")


def test_bounds_raise_without_clipping_the_effective_ratio() -> None:
    row = _row(effective_ratio=1.5, ratio_bound_upper=1.5)
    assert row.effective_ratio == 1.5
    with pytest.raises(ValueError, match="never clip"):
        _row(effective_ratio=1.6)
    with pytest.raises(ValueError, match="supplied together"):
        _row(ratio_bound_upper=None)


def test_weighted_conflict_provenance_is_complete_and_read_only() -> None:
    weights = {"a-left": 0.25, "a-right": 0.75}
    row = _row(
        evidence_conflict_policy="weighted_ratio",
        evidence_weights=weights,
    )
    weights["a-left"] = 1.0
    assert dict(row.evidence_weights) == {"a-left": 0.25, "a-right": 0.75}
    with pytest.raises(TypeError):
        row.evidence_weights["a-left"] = 0.5
    with pytest.raises(ValueError, match="exactly match"):
        _row(
            evidence_conflict_policy="weighted_ratio",
            evidence_weights={"a-left": 1.0},
        )
    with pytest.raises(ValueError, match="only with weighted_ratio"):
        _row(evidence_weights={"a-left": 0.25, "a-right": 0.75})


def test_authoritative_metadata_and_policy_names_are_validated() -> None:
    with pytest.raises(ValueError, match="start < end"):
        _row(epoch_end=date(2026, 1, 1))
    with pytest.raises(ValueError, match="epoch_support must be nonnegative"):
        _row(epoch_support=-1)
    with pytest.raises(ValueError, match="ratio_policy must be one of"):
        _row(ratio_policy="automatic")
    with pytest.raises(ValueError, match="epoch_ratio_policy must be one of"):
        _row(epoch_ratio_policy="midpoint")
    with pytest.raises(ValueError, match="evidence_conflict_policy"):
        _row(evidence_conflict_policy="automatic")


def test_row_and_nested_source_identifiers_are_immutable() -> None:
    source_ids = ["a-left", "a-right"]
    row = _row(source_evidence_ids=source_ids)
    source_ids.append("mutated")
    assert row.source_evidence_ids == ("a-left", "a-right")
    with pytest.raises(FrozenInstanceError):
        row.effective_ratio = 2.0


def test_calibrated_pair_rows_are_unique_and_complete() -> None:
    rows = (
        _row(),
        _row(tested_entity="B", source_evidence_ids=("b-left", "b-right"),
             left_source_evidence_id="b-left", right_source_evidence_id="b-right"),
        _row(epoch="E2"),
    )
    normalized = _validate_epoch_ratio_rows(
        rows,
        expected_pairs=(("A", "E1"), ("B", "E1"), ("A", "E2")),
    )
    assert normalized == rows
    with pytest.raises(ValueError, match="multiple rows"):
        _validate_epoch_ratio_rows((rows[0], rows[0]))
    with pytest.raises(ValueError, match="missing=.*B.*E1"):
        _validate_epoch_ratio_rows(
            (rows[0], rows[2]),
            expected_pairs=(("A", "E1"), ("B", "E1"), ("A", "E2")),
        )
