"""Shared numerical validation tests for CAL-010."""

import math

import numpy as np
import pytest

from do_attribution.calibration import (
    _finite_float,
    _validate_numeric_pairs,
    _validate_ratio_bounds,
    _validate_shares,
    _validate_tolerance,
    _validate_total,
    _validate_totals_by_epoch,
    _validate_unit_interval_pairs,
    _validate_unit_sum_weights,
)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, np.float64(np.nan)])
def test_all_nonfinite_real_values_raise(value) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        _finite_float(value, name="field")


@pytest.mark.parametrize("value", [True, False, "1.0", None, complex(1, 0)])
def test_bool_and_nonreal_values_raise(value) -> None:
    with pytest.raises(TypeError, match="must be a real number"):
        _finite_float(value, name="field")


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
def test_tolerance_must_be_finite_and_positive(value) -> None:
    with pytest.raises(ValueError):
        _validate_tolerance(value)
    assert _validate_tolerance(1e-9) == 1e-9


def test_estimates_and_adjustments_honor_signed_permission() -> None:
    pairs = (("A", 0.0), ("B", 2.5), ("C", -1.0))
    with pytest.raises(ValueError, match="'C'.*allow_signed"):
        _validate_numeric_pairs(pairs, name="estimates", allow_signed=False)
    assert _validate_numeric_pairs(pairs, name="estimates", allow_signed=True) == pairs


def test_shares_are_nonnegative_and_sum_to_one_within_tolerance() -> None:
    assert _validate_shares(
        (("A", 0.4), ("B", 0.6 + 5e-10)), tolerance=1e-9, name="shares"
    )[0] == ("A", 0.4)
    with pytest.raises(ValueError, match="sum to one"):
        _validate_shares(
            (("A", 0.4), ("B", 0.5)), tolerance=1e-9, name="shares"
        )
    with pytest.raises(ValueError, match="nonnegative"):
        _validate_shares(
            (("A", 1.1), ("B", -0.1)), tolerance=1e-9, name="shares"
        )


def test_totals_are_explicit_finite_and_signed_only_with_permission() -> None:
    assert _validate_total(0, name="current_total", allow_signed=False) == 0.0
    with pytest.raises(ValueError, match="allow_signed"):
        _validate_total(-1, name="current_total", allow_signed=False)
    assert _validate_total(-1, name="current_total", allow_signed=True) == -1.0


def test_temporal_totals_require_exact_epoch_coverage_and_epoch_order() -> None:
    assert _validate_totals_by_epoch(
        ("E2", "E1"), {"E1": 10, "E2": 20}, allow_signed=False
    ) == (("E2", 20.0), ("E1", 10.0))
    with pytest.raises(ValueError, match=r"missing=\['E2'\].*extra=\['E3'\]"):
        _validate_totals_by_epoch(
            ("E1", "E2"), {"E1": 10, "E3": 30}, allow_signed=False
        )


def test_strengths_use_closed_unit_interval() -> None:
    assert _validate_unit_interval_pairs(
        (("zero", 0), ("one", 1)), name="strength"
    ) == (("zero", 0.0), ("one", 1.0))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _validate_unit_interval_pairs((("too_high", 1.01),), name="strength")


def test_weights_are_finite_nonnegative_and_sum_to_one() -> None:
    assert _validate_unit_sum_weights(
        (("A", 0.25), ("B", 0.75)), tolerance=1e-9, name="weights"
    ) == (("A", 0.25), ("B", 0.75))
    with pytest.raises(ValueError, match="sum to one"):
        _validate_unit_sum_weights(
            (("A", 0.2), ("B", 0.7)), tolerance=1e-9, name="weights"
        )
    with pytest.raises(ValueError, match="nonnegative"):
        _validate_unit_sum_weights(
            (("A", -0.1), ("B", 1.1)), tolerance=1e-9, name="weights"
        )


def test_ratio_bounds_are_optional_finite_and_ordered() -> None:
    assert _validate_ratio_bounds(None) is None
    assert _validate_ratio_bounds((0, 2)) == (0.0, 2.0)
    with pytest.raises(ValueError, match="lower <= upper"):
        _validate_ratio_bounds((2, 1))
    with pytest.raises(ValueError, match="must be finite"):
        _validate_ratio_bounds((0, math.inf))
    with pytest.raises(TypeError, match="two-item tuple"):
        _validate_ratio_bounds([0, 1])
