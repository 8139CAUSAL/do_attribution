"""Focused tests for the shared signed-attribution normalization utility.

These tests exercise ``do_attribution.normalization`` in isolation, before it
is consumed by any attribution model.  They lock down the numerical contract
in Project_Get-Well_Plan.md sections 6.1 and 6.2.
"""

from __future__ import annotations

import math

import pytest

from do_attribution.normalization import (
    NORMALIZATION_AVAILABLE,
    NORMALIZATION_CANCELLATION_TOLERANCE,
    NORMALIZATION_DIAGNOSTIC_KEYS,
    NORMALIZATION_UNAVAILABLE,
    NORMALIZATION_VALIDATION_TOLERANCE,
    REASON_INVALID_SHARE_SUM,
    REASON_NEAR_CANCELLATION,
    REASON_NEGATIVE_SHARE,
    REASON_NONFINITE_RESULT,
    REASON_SHARE_ABOVE_ONE,
    REASON_ZERO_RAW_MAGNITUDE,
    normalize_signed_attribution,
)


# --------------------------------------------------------------------------- #
# Tolerances are the authoritative internal constants                          #
# --------------------------------------------------------------------------- #
def test_tolerances_have_the_documented_values():
    assert NORMALIZATION_CANCELLATION_TOLERANCE == 1e-12
    assert NORMALIZATION_VALIDATION_TOLERANCE == 1e-12


# --------------------------------------------------------------------------- #
# Zero raw magnitude                                                           #
# --------------------------------------------------------------------------- #
def test_zero_raw_magnitude_is_unavailable():
    result = normalize_signed_attribution({"A": 0.0, "B": 0.0})

    assert result.attribution is None
    assert result.available is False
    assert result.normalization_status == NORMALIZATION_UNAVAILABLE
    assert result.normalization_reason == REASON_ZERO_RAW_MAGNITUDE
    assert result.normalization_denominator is None
    assert result.raw_sum == 0.0
    assert result.raw_magnitude == 0.0
    assert result.cancellation_ratio is None
    assert result.raw_values == {"A": 0.0, "B": 0.0}


def test_empty_raw_values_are_zero_magnitude():
    result = normalize_signed_attribution({})

    assert result.attribution is None
    assert result.normalization_reason == REASON_ZERO_RAW_MAGNITUDE


# --------------------------------------------------------------------------- #
# Cancellation threshold: below, at, and above                                 #
# --------------------------------------------------------------------------- #
def test_cancellation_below_threshold_is_unavailable():
    # ratio = 1e-13 / 2.0 = 5e-14, strictly below 1e-12.
    result = normalize_signed_attribution({"A": 1.0 + 1e-13, "B": -1.0})

    assert result.cancellation_ratio < NORMALIZATION_CANCELLATION_TOLERANCE
    assert result.attribution is None
    assert result.normalization_reason == REASON_NEAR_CANCELLATION
    # The refused denominator is still reported.
    assert result.normalization_denominator == pytest.approx(1e-13)


def test_cancellation_exactly_at_threshold_is_unavailable(monkeypatch):
    """The threshold comparison is inclusive: ratio == tolerance is refused.

    A cancellation ratio equal to the literal ``1e-12`` cannot be produced by
    the utility's own arithmetic: ``raw_sum`` is a difference of same-scale
    floats and is therefore a multiple of that scale's ulp, which is far
    coarser than ``float(1e-12)``.  The boundary is instead exercised exactly
    by moving the tolerance to a ratio that is representable without error.
    """
    from do_attribution import normalization as norm

    monkeypatch.setattr(norm, "NORMALIZATION_CANCELLATION_TOLERANCE", 0.5)
    # raw_sum = 1.0, raw_magnitude = 2.0 -> ratio == 0.5 exactly.
    result = norm.normalize_signed_attribution({"A": 1.5, "B": -0.5})

    assert result.cancellation_ratio == 0.5
    assert result.attribution is None
    assert result.normalization_reason == REASON_NEAR_CANCELLATION


def test_cancellation_immediately_above_threshold_passes_the_guard(monkeypatch):
    """A ratio just above the tolerance clears the cancellation guard."""
    from do_attribution import normalization as norm

    monkeypatch.setattr(norm, "NORMALIZATION_CANCELLATION_TOLERANCE", 0.5)
    # raw_sum = 1.5, raw_magnitude = 2.5 -> ratio == 0.6 > 0.5.
    result = norm.normalize_signed_attribution({"A": 2.0, "B": -0.5})

    assert result.cancellation_ratio > 0.5
    # The guard is cleared, so the failure (if any) comes from the share
    # domain rather than from cancellation.
    assert result.normalization_reason != REASON_NEAR_CANCELLATION


def test_cancellation_above_threshold_normalizes():
    # raw_sum = 1e-11, raw_magnitude = 1e-11 => ratio == 1.0.
    result = normalize_signed_attribution({"A": 1e-11})

    assert result.cancellation_ratio > NORMALIZATION_CANCELLATION_TOLERANCE
    assert result.available is True
    assert result.attribution == pytest.approx({"A": 1.0})
    assert result.normalization_reason is None


def test_near_cancellation_never_produces_explosive_values():
    """A float-residue denominator must not manufacture huge shares."""
    result = normalize_signed_attribution({"A": -0.6, "B": 0.6000000000000001})

    assert result.raw_magnitude == pytest.approx(1.2, abs=1e-15)
    assert result.cancellation_ratio < NORMALIZATION_CANCELLATION_TOLERANCE
    assert result.attribution is None
    assert result.normalization_reason == REASON_NEAR_CANCELLATION
    # Raw signed diagnostics survive untouched.
    assert result.raw_values == {"A": -0.6, "B": 0.6000000000000001}


# --------------------------------------------------------------------------- #
# Scale invariance                                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scale", [1e-6, 0.5, 1.0, 3.0, 1e6])
def test_positive_rescaling_leaves_ratio_and_shares_unchanged(scale):
    raw = {"A": 3.0, "B": 1.0}
    base = normalize_signed_attribution(raw)
    scaled = normalize_signed_attribution(
        {channel: value * scale for channel, value in raw.items()}
    )

    assert scaled.cancellation_ratio == pytest.approx(base.cancellation_ratio)
    assert scaled.attribution == pytest.approx(base.attribution)
    assert scaled.normalization_status == base.normalization_status


@pytest.mark.parametrize("scale", [1e-6, 0.5, 3.0, 1e6])
def test_positive_rescaling_preserves_unavailability(scale):
    raw = {"A": 40.5, "B": -39.5}
    base = normalize_signed_attribution(raw)
    scaled = normalize_signed_attribution(
        {channel: value * scale for channel, value in raw.items()}
    )

    assert base.attribution is None
    assert scaled.attribution is None
    assert scaled.normalization_reason == base.normalization_reason
    assert scaled.cancellation_ratio == pytest.approx(base.cancellation_ratio)


# --------------------------------------------------------------------------- #
# Nonfinite input                                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_raw_value_is_unavailable(bad):
    result = normalize_signed_attribution({"A": 1.0, "B": bad})

    assert result.attribution is None
    assert result.normalization_reason == REASON_NONFINITE_RESULT
    assert result.normalization_denominator is None
    assert math.isnan(result.raw_sum)
    assert math.isnan(result.raw_magnitude)
    assert result.cancellation_ratio is None
    # Raw values are preserved, including the offending one.
    assert result.raw_values["A"] == 1.0
    assert math.isnan(result.raw_values["B"]) or math.isinf(result.raw_values["B"])


# --------------------------------------------------------------------------- #
# Candidate-domain rejections                                                  #
# --------------------------------------------------------------------------- #
def test_negative_candidate_share_is_unavailable():
    # raw_sum = 1.0, magnitude = 80.0, ratio = 0.0125 (well above threshold),
    # but the candidate shares are 40.5 and -39.5.
    result = normalize_signed_attribution({"A": 40.5, "B": -39.5})

    assert result.cancellation_ratio == pytest.approx(0.0125)
    assert result.attribution is None
    assert result.normalization_reason == REASON_NEGATIVE_SHARE
    assert result.normalization_denominator == pytest.approx(1.0)


def test_negative_share_is_checked_before_the_upper_bound():
    """Documented reason order: negative_share precedes share_above_one."""
    # raw_sum = 1.5 -> candidate {A: 1.333, B: -0.667, C: 0.333}: both bounds
    # are violated, and the negative one is reported.
    result = normalize_signed_attribution({"A": 2.0, "B": -1.0, "C": 0.5})

    assert result.attribution is None
    assert result.normalization_reason == REASON_NEGATIVE_SHARE


def test_candidate_share_above_one_is_unavailable():
    """One share exceeds one while every negative stays inside tolerance.

    Candidate shares always sum to one, so an upper-bound violation is only
    reachable when the offsetting negatives are individually small enough to
    pass the negative-share check.
    """
    raw = {"A": 1.0 + 5e-12}
    raw.update({f"N{index}": -5e-13 for index in range(10)})

    result = normalize_signed_attribution(raw)

    # Every negative candidate is inside the validation tolerance ...
    assert all(
        value >= -NORMALIZATION_VALIDATION_TOLERANCE
        for channel, value in result.raw_values.items()
        if channel != "A"
    )
    # ... so the upper bound is the first violated constraint.
    assert result.attribution is None
    assert result.normalization_reason == REASON_SHARE_ABOVE_ONE


def test_invalid_candidate_sum_is_unavailable():
    """Canonicalizing many tolerance-sized negatives can break the unit sum.

    Each negative candidate is snapped to zero, and enough of them together
    push the total beyond the unit-sum tolerance.  The utility must refuse
    rather than publish a vector that does not sum to one.
    """
    raw = {f"P{index}": 0.25 + 6.25e-13 for index in range(4)}
    raw.update({f"N{index}": -5e-13 for index in range(5)})

    result = normalize_signed_attribution(raw)

    # No individual bound is violated beyond tolerance ...
    assert result.cancellation_ratio == pytest.approx(1.0)
    # ... but the canonicalized vector no longer sums to one.
    assert result.attribution is None
    assert result.normalization_reason == REASON_INVALID_SHARE_SUM


def test_nonfinite_aggregate_of_finite_values_is_unavailable():
    """Finite raw values that overflow when summed are refused explicitly."""
    result = normalize_signed_attribution({"A": 1e308, "B": 1e308})

    assert result.attribution is None
    assert result.normalization_reason == REASON_NONFINITE_RESULT
    assert result.cancellation_ratio is None


# --------------------------------------------------------------------------- #
# Valid, well-conditioned vectors                                              #
# --------------------------------------------------------------------------- #
def test_well_conditioned_vector_normalizes():
    result = normalize_signed_attribution({"A": 3.0, "B": 1.0})

    assert result.available is True
    assert result.normalization_status == NORMALIZATION_AVAILABLE
    assert result.normalization_reason is None
    assert result.attribution == pytest.approx({"A": 0.75, "B": 0.25})
    assert result.normalization_denominator == pytest.approx(4.0)
    assert result.raw_sum == pytest.approx(4.0)
    assert result.raw_magnitude == pytest.approx(4.0)
    assert result.cancellation_ratio == pytest.approx(1.0)
    assert sum(result.attribution.values()) == pytest.approx(1.0, abs=1e-12)
    assert all(0.0 <= value <= 1.0 for value in result.attribution.values())


def test_single_nonzero_value_gets_the_whole_share():
    result = normalize_signed_attribution({"A": 5.0, "B": 0.0})

    assert result.available is True
    assert result.attribution == pytest.approx({"A": 1.0, "B": 0.0})


def test_all_negative_values_normalize_against_a_negative_denominator():
    """All-negative raw values still yield a valid share vector."""
    result = normalize_signed_attribution({"A": -3.0, "B": -1.0})

    assert result.available is True
    assert result.attribution == pytest.approx({"A": 0.75, "B": 0.25})
    assert result.raw_sum == pytest.approx(-4.0)
    assert result.cancellation_ratio == pytest.approx(1.0)


def test_weak_cancellation_is_still_refused_by_the_share_domain():
    """Clearing the cancellation guard is not sufficient for a valid vector.

    Any materially negative raw value produces a negative candidate share
    once the raw sum is positive, so mixed-sign diagnostics are refused on
    domain grounds even when they are nowhere near cancelling.
    """
    result = normalize_signed_attribution({"A": 1.0, "B": -0.5, "C": 0.5})

    assert result.cancellation_ratio == pytest.approx(0.5)
    assert result.cancellation_ratio > NORMALIZATION_CANCELLATION_TOLERANCE
    assert result.attribution is None
    assert result.normalization_reason == REASON_NEGATIVE_SHARE


def test_all_positive_values_normalize():
    result = normalize_signed_attribution({"A": 2.0, "B": 1.0, "C": 1.0})

    assert result.available is True
    assert result.cancellation_ratio == pytest.approx(1.0)
    assert result.attribution == pytest.approx({"A": 0.5, "B": 0.25, "C": 0.25})
    assert sum(result.attribution.values()) == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Diagnostics contract                                                         #
# --------------------------------------------------------------------------- #
def test_diagnostics_are_populated_when_available():
    result = normalize_signed_attribution({"A": 3.0, "B": 1.0})
    diagnostics = result.diagnostics()

    assert set(diagnostics) == set(NORMALIZATION_DIAGNOSTIC_KEYS)
    assert diagnostics["normalization_status"] == NORMALIZATION_AVAILABLE
    assert diagnostics["normalization_reason"] is None


def test_diagnostics_are_populated_when_unavailable():
    result = normalize_signed_attribution({"A": 40.5, "B": -39.5})
    diagnostics = result.diagnostics()

    assert set(diagnostics) == set(NORMALIZATION_DIAGNOSTIC_KEYS)
    assert diagnostics["normalization_status"] == NORMALIZATION_UNAVAILABLE
    assert diagnostics["normalization_reason"] == REASON_NEGATIVE_SHARE
    assert diagnostics["raw_sum"] == pytest.approx(1.0)
    assert diagnostics["raw_magnitude"] == pytest.approx(80.0)
    assert diagnostics["cancellation_ratio"] == pytest.approx(0.0125)


def test_raw_values_are_never_modified():
    raw = {"A": 40.5, "B": -39.5}
    result = normalize_signed_attribution(raw)

    assert result.raw_values == raw
    assert raw == {"A": 40.5, "B": -39.5}
