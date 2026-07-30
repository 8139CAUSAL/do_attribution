"""Immutable coordinate-schedule provenance tests for CAL-014."""

from dataclasses import FrozenInstanceError, fields

import pytest

from do_attribution.calibration import RatioScheduleRow


EXPECTED_FIELDS = [
    "tested_entity",
    "coordinate_name",
    "coordinate_value",
    "schedule_ratio",
    "schedule_source",
    "source_evidence_ids",
    "left_source_evidence_id",
    "right_source_evidence_id",
    "inside_observed_support",
]


def _row(source: str, **overrides) -> RatioScheduleRow:
    values = {
        "tested_entity": "A",
        "coordinate_name": None,
        "coordinate_value": None,
        "schedule_ratio": 0.9,
        "schedule_source": source,
        "source_evidence_ids": (),
        "left_source_evidence_id": None,
        "right_source_evidence_id": None,
        "inside_observed_support": False,
    }
    source_values = {
        "observed": {
            "coordinate_name": "time",
            "coordinate_value": 1.0,
            "source_evidence_ids": ("a-observed",),
            "inside_observed_support": True,
        },
        "explicit": {},
        "strength_derived": {"source_evidence_ids": ("a-source",)},
        "constant": {"source_evidence_ids": ("a-source",)},
        "interpolated": {
            "coordinate_name": "time",
            "coordinate_value": 2.0,
            "source_evidence_ids": ("a-left", "a-right"),
            "left_source_evidence_id": "a-left",
            "right_source_evidence_id": "a-right",
            "inside_observed_support": True,
        },
        "unsupported": {
            "coordinate_name": "time",
            "coordinate_value": 5.0,
            "schedule_ratio": None,
        },
    }
    values.update(source_values.get(source, {}))
    values.update(overrides)
    return RatioScheduleRow(**values)


def test_fields_match_plan_order_and_all_sources_are_distinct() -> None:
    rows = tuple(
        _row(source)
        for source in (
            "observed",
            "explicit",
            "strength_derived",
            "constant",
            "interpolated",
            "unsupported",
        )
    )
    assert [item.name for item in fields(rows[0])] == EXPECTED_FIELDS
    assert tuple(row.schedule_source for row in rows) == (
        "observed",
        "explicit",
        "strength_derived",
        "constant",
        "interpolated",
        "unsupported",
    )


def test_interpolated_row_keeps_both_ordered_anchor_ids() -> None:
    row = _row("interpolated")
    assert row.schedule_ratio == 0.9
    assert row.source_evidence_ids == ("a-left", "a-right")
    assert row.left_source_evidence_id == "a-left"
    assert row.right_source_evidence_id == "a-right"
    assert row.inside_observed_support is True


def test_unsupported_coordinate_has_no_ratio_and_is_outside_support() -> None:
    row = _row("unsupported")
    assert row.coordinate_value == 5.0
    assert row.schedule_ratio is None
    assert row.inside_observed_support is False
    with pytest.raises(ValueError, match="cannot receive a ratio"):
        _row("unsupported", schedule_ratio=1.0)
    with pytest.raises(ValueError, match="outside observed support"):
        _row("unsupported", inside_observed_support=True)


@pytest.mark.parametrize("source", ["observed", "interpolated"])
def test_coordinate_sources_require_explicit_supported_coordinates(source: str) -> None:
    with pytest.raises(ValueError, match="explicit coordinate"):
        _row(source, coordinate_name=None, coordinate_value=None)
    with pytest.raises(ValueError, match="inside observed support"):
        _row(source, inside_observed_support=False)


def test_interpolation_provenance_must_match_two_distinct_anchors() -> None:
    with pytest.raises(ValueError, match="both source anchor IDs"):
        _row("interpolated", right_source_evidence_id=None)
    with pytest.raises(ValueError, match="must be distinct"):
        _row(
            "interpolated",
            source_evidence_ids=("a-left", "a-right"),
            right_source_evidence_id="a-left",
        )
    with pytest.raises(ValueError, match="ordered left and right"):
        _row(
            "interpolated",
            source_evidence_ids=("a-right", "a-left"),
        )


@pytest.mark.parametrize(
    ("source", "ids"),
    [
        ("observed", ()),
        ("explicit", ("unexpected",)),
        ("strength_derived", ()),
        ("constant", ("one", "two")),
        ("unsupported", ("unexpected",)),
    ],
)
def test_source_evidence_count_must_match_schedule_source(
    source: str, ids: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError, match="source_evidence_ids"):
        _row(source, source_evidence_ids=ids)


def test_ratio_and_source_identifiers_are_validated_and_immutable() -> None:
    source_ids = ["a-source"]
    row = _row("constant", source_evidence_ids=source_ids)
    source_ids.append("mutated")
    assert row.source_evidence_ids == ("a-source",)
    assert row.schedule_ratio == 0.9
    with pytest.raises(FrozenInstanceError):
        row.schedule_ratio = 2.0
    with pytest.raises(ValueError, match="schedule_ratio must be finite"):
        _row("explicit", schedule_ratio=float("nan"))
    with pytest.raises(ValueError, match="schedule_source must be one of"):
        _row("automatic")
