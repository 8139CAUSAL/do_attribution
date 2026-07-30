"""Policy-neutral scalar epoch-support diagnostics for CAL-037."""

import math

import pytest

from do_attribution.calibration import _normalize_epoch_support


def test_scalar_support_is_optional_and_never_inferred() -> None:
    assert _normalize_epoch_support(None, epochs=("E1", "E2")) is None


def test_scalar_support_is_ordered_immutable_and_may_cover_a_subset() -> None:
    source = {"E2": 20, "E1": 0}
    normalized = _normalize_epoch_support(source, epochs=("E1", "E2", "E3"))
    source["E1"] = 999

    assert normalized is not None
    assert tuple(normalized) == ("E1", "E2")
    assert normalized == {"E1": 0.0, "E2": 20.0}
    with pytest.raises(TypeError):
        normalized["E3"] = 30.0


@pytest.mark.parametrize("value", [-1, math.nan, math.inf, True, "10"])
def test_scalar_support_must_be_finite_nonnegative_real(value) -> None:
    with pytest.raises((TypeError, ValueError), match="epoch_support"):
        _normalize_epoch_support({"E1": value}, epochs=("E1",))


def test_scalar_support_rejects_unknown_epoch_keys() -> None:
    with pytest.raises(ValueError, match="invalid epoch coverage"):
        _normalize_epoch_support({"E2": 10.0}, epochs=("E1",))


def test_scalar_support_output_contains_no_coordinate_weight_schedule() -> None:
    normalized = _normalize_epoch_support({"E1": 100.0}, epochs=("E1",))
    assert normalized == {"E1": 100.0}
    assert all(isinstance(value, float) for value in normalized.values())
