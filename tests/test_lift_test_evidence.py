"""Immutable LiftTestEvidence contract tests for CAL-011."""

from dataclasses import FrozenInstanceError, fields
from datetime import date

import pytest

from do_attribution.calibration import LiftTestEvidence


EXPECTED_FIELDS = [
    "tested_entity",
    "observed_lift",
    "matched_mta_estimate",
    "calibration_ratio",
    "test_id",
    "outcome_name",
    "window_start",
    "window_end",
    "population",
    "unit",
    "coordinate_name",
    "coordinate_value",
    "standard_error",
    "metadata",
]


def _evidence(**overrides) -> LiftTestEvidence:
    values = {
        "tested_entity": "A",
        "observed_lift": 360.0,
        "matched_mta_estimate": 400.0,
        "test_id": "geo-a",
        "outcome_name": "conversions",
        "window_start": "2026-01-01",
        "window_end": "2026-01-14",
        "population": "eligible traffic",
        "unit": "incremental conversions",
        "metadata": {"design": {"regions": ["east", "west"]}},
    }
    values.update(overrides)
    return LiftTestEvidence(**values)


def test_fields_and_canonical_ratio_match_plan_order() -> None:
    evidence = _evidence()
    assert [item.name for item in fields(evidence)] == EXPECTED_FIELDS
    assert evidence.calibration_ratio == 0.9


def test_zero_lift_is_valid_and_zero_denominator_raises() -> None:
    assert _evidence(observed_lift=0).calibration_ratio == 0.0
    with pytest.raises(ValueError, match="matched_mta_estimate must be nonzero"):
        _evidence(matched_mta_estimate=0)


def test_signed_evidence_is_retained_for_call_time_permission_check() -> None:
    evidence = _evidence(observed_lift=-20, matched_mta_estimate=10)
    assert evidence.calibration_ratio == -2.0


@pytest.mark.parametrize("field", ["observed_lift", "matched_mta_estimate"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_evidence_numbers_must_be_finite(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=f"{field} must be finite"):
        _evidence(**{field: value})


def test_static_coordinates_may_be_omitted_but_partial_coordinates_raise() -> None:
    evidence = _evidence()
    assert evidence.coordinate_name is None
    assert evidence.coordinate_value is None
    with pytest.raises(ValueError, match="must be supplied together"):
        _evidence(coordinate_name="date")
    with pytest.raises(ValueError, match="must be supplied together"):
        _evidence(coordinate_value=date(2026, 1, 7))


def test_explicit_coordinate_is_preserved_not_inferred_from_window() -> None:
    coordinate = date(2026, 1, 3)
    evidence = _evidence(coordinate_name="date", coordinate_value=coordinate)
    assert evidence.coordinate_value == coordinate
    assert evidence.coordinate_value != date(2026, 1, 7)


@pytest.mark.parametrize("coordinate", [True, "2026-01-03", float("nan")])
def test_coordinate_type_and_finiteness_are_validated(coordinate) -> None:
    with pytest.raises((TypeError, ValueError), match="coordinate_value"):
        _evidence(coordinate_name="date", coordinate_value=coordinate)


def test_standard_error_is_finite_and_nonnegative() -> None:
    assert _evidence(standard_error=0).standard_error == 0.0
    with pytest.raises(ValueError, match="standard_error must be nonnegative"):
        _evidence(standard_error=-0.1)


def test_metadata_is_defensively_copied_and_recursively_read_only() -> None:
    metadata = {"design": {"regions": ["east", "west"]}}
    evidence = _evidence(metadata=metadata)
    metadata["design"]["regions"].append("south")
    assert evidence.metadata["design"]["regions"] == ("east", "west")
    with pytest.raises(TypeError):
        evidence.metadata["new"] = "value"
    with pytest.raises(TypeError):
        evidence.metadata["design"]["new"] = "value"
    with pytest.raises(FrozenInstanceError):
        evidence.observed_lift = 1.0


def test_required_identity_and_string_metadata_validation() -> None:
    with pytest.raises(ValueError, match="tested_entity is required"):
        _evidence(tested_entity=None)
    with pytest.raises(ValueError, match="test_id must be a nonempty string"):
        _evidence(test_id="")
    with pytest.raises(ValueError, match="unit must be a nonempty string or None"):
        _evidence(unit="")
