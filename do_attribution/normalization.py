"""
do_attribution.normalization
----------------------------

Authoritative normalization policy for **signed** attribution values.

This module is the single location where raw signed model outputs (Shapley
values, Markov removal effects) are converted into normalized attribution
shares.  Raw values and normalized shares are two different quantities with
two different contracts:

* **raw values** are signed diagnostics on the model's own scale.  They may
  be negative, may cancel, and are never shares, percentages, or
  probabilities.
* **normalized attribution** is a share vector.  It is valid only when every
  value is finite, ``0 <= a_i <= 1``, and ``sum(a_i) == 1`` within the
  documented tolerance.

When a valid share vector cannot be produced from the raw values,
normalization is *unavailable*: ``attribution`` is ``None``, the raw values
are preserved unchanged, and a machine-readable status, reason, and
diagnostic set are returned.  Normalization being unavailable is not a fit
failure — the caller keeps its raw diagnostics.

Numerical contract
~~~~~~~~~~~~~~~~~~

For raw values ``r_i``::

    raw_sum           = sum(r_i)
    raw_magnitude     = sum(abs(r_i))
    cancellation_ratio = abs(raw_sum) / raw_magnitude

``cancellation_ratio`` is scale-invariant: multiplying every raw value by the
same positive constant leaves it — and the resulting shares — unchanged.

Normalization is refused when ``raw_magnitude`` is zero, or when
``cancellation_ratio <= NORMALIZATION_CANCELLATION_TOLERANCE``.  Dividing by a
denominator that is negligible relative to the raw scale produces numerically
explosive values, so that division is never performed.

Candidate shares that survive the cancellation check are validated against
the share domain using ``NORMALIZATION_VALIDATION_TOLERANCE``.  Deviations no
larger than that tolerance at the ``0`` and ``1`` boundaries are canonicalized
to exactly ``0.0`` / ``1.0``; materially signed values are never repaired this
way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import math

# --------------------------------------------------------------------- #
# Internal numerical tolerances — the authoritative definitions.         #
# --------------------------------------------------------------------- #
#: Cancellation ratios at or below this value make normalization unavailable.
NORMALIZATION_CANCELLATION_TOLERANCE: float = 1e-12

#: Absolute tolerance for share-domain and unit-sum validation.
NORMALIZATION_VALIDATION_TOLERANCE: float = 1e-12

# --------------------------------------------------------------------- #
# Machine-readable status and reason codes.                             #
# --------------------------------------------------------------------- #
NORMALIZATION_AVAILABLE: str = "available"
NORMALIZATION_UNAVAILABLE: str = "unavailable"

REASON_ZERO_RAW_MAGNITUDE: str = "zero_raw_magnitude"
REASON_NEAR_CANCELLATION: str = "near_cancellation"
REASON_NONFINITE_RESULT: str = "nonfinite_result"
REASON_NEGATIVE_SHARE: str = "negative_share"
REASON_SHARE_ABOVE_ONE: str = "share_above_one"
REASON_INVALID_SHARE_SUM: str = "invalid_share_sum"

#: Every reason code this module can emit, in evaluation order.
NORMALIZATION_REASONS: tuple = (
    REASON_ZERO_RAW_MAGNITUDE,
    REASON_NEAR_CANCELLATION,
    REASON_NONFINITE_RESULT,
    REASON_NEGATIVE_SHARE,
    REASON_SHARE_ABOVE_ONE,
    REASON_INVALID_SHARE_SUM,
)

#: Diagnostic keys published by :meth:`NormalizationResult.diagnostics`.
NORMALIZATION_DIAGNOSTIC_KEYS: tuple = (
    "normalization_status",
    "normalization_reason",
    "normalization_denominator",
    "raw_sum",
    "raw_magnitude",
    "cancellation_ratio",
)


@dataclass(frozen=True)
class NormalizationResult:
    """Outcome of one normalization attempt.

    Attributes
    ----------
    attribution : dict[str, float] or None
        Normalized shares when normalization is available, otherwise
        ``None``.  Never contains raw signed values.
    raw_values : dict[str, float]
        The raw signed inputs, preserved unmodified.
    normalization_status : str
        ``"available"`` or ``"unavailable"``.
    normalization_reason : str or None
        ``None`` when available; otherwise one of
        :data:`NORMALIZATION_REASONS`.
    normalization_denominator : float or None
        The denominator that was (or would have been) used, i.e. the raw
        sum.  ``None`` when no denominator could be formed.
    raw_sum : float
        ``sum(r_i)``.
    raw_magnitude : float
        ``sum(abs(r_i))``.
    cancellation_ratio : float or None
        ``abs(raw_sum) / raw_magnitude``; ``None`` when undefined.
    """

    attribution: Optional[Dict[str, float]]
    raw_values: Dict[str, float]
    normalization_status: str
    normalization_reason: Optional[str]
    normalization_denominator: Optional[float]
    raw_sum: float
    raw_magnitude: float
    cancellation_ratio: Optional[float]

    @property
    def available(self) -> bool:
        """``True`` when a valid normalized share vector was produced."""
        return self.normalization_status == NORMALIZATION_AVAILABLE

    def diagnostics(self) -> Dict[str, Any]:
        """Return the flat, serializable diagnostic mapping.

        The keys are :data:`NORMALIZATION_DIAGNOSTIC_KEYS`.  They are
        populated whether normalization succeeded or not.
        """
        return {
            "normalization_status": self.normalization_status,
            "normalization_reason": self.normalization_reason,
            "normalization_denominator": self.normalization_denominator,
            "raw_sum": self.raw_sum,
            "raw_magnitude": self.raw_magnitude,
            "cancellation_ratio": self.cancellation_ratio,
        }


def _unavailable(
    *,
    raw_values: Dict[str, float],
    reason: str,
    denominator: Optional[float],
    raw_sum: float,
    raw_magnitude: float,
    cancellation_ratio: Optional[float],
) -> NormalizationResult:
    """Build an unavailable result that still carries full diagnostics."""
    return NormalizationResult(
        attribution=None,
        raw_values=raw_values,
        normalization_status=NORMALIZATION_UNAVAILABLE,
        normalization_reason=reason,
        normalization_denominator=denominator,
        raw_sum=raw_sum,
        raw_magnitude=raw_magnitude,
        cancellation_ratio=cancellation_ratio,
    )


def normalize_signed_attribution(
    raw_values: Mapping[str, float],
) -> NormalizationResult:
    """Normalize signed raw attribution values into shares, or refuse.

    Parameters
    ----------
    raw_values : Mapping[str, float]
        Raw signed model output keyed by entity (channel) name.  Preserved
        unmodified on the returned result.

    Returns
    -------
    NormalizationResult
        ``attribution`` is a share mapping when normalization is available
        and ``None`` otherwise.  Diagnostics are always populated.

    Notes
    -----
    An empty ``raw_values`` mapping has zero magnitude and therefore yields
    an unavailable result with reason ``zero_raw_magnitude``.
    """
    raw: Dict[str, float] = {
        str(key): float(value) for key, value in raw_values.items()
    }

    # 1. Validate the raw values.  Nonfinite input cannot produce a valid
    #    share vector, and its sums are meaningless, so refuse immediately
    #    while still reporting what was seen.
    if not all(math.isfinite(value) for value in raw.values()):
        return _unavailable(
            raw_values=raw,
            reason=REASON_NONFINITE_RESULT,
            denominator=None,
            raw_sum=float("nan"),
            raw_magnitude=float("nan"),
            cancellation_ratio=None,
        )

    # 2. Total absolute raw magnitude and 3. the raw sum.  Plain ``sum`` is
    #    used deliberately so the published diagnostics are exactly what a
    #    caller reproduces from the raw values.
    raw_sum = sum(raw.values())
    raw_magnitude = sum(abs(value) for value in raw.values())

    # Finite raw values can still overflow when aggregated.  A nonfinite
    # aggregate cannot yield a meaningful cancellation ratio, so refuse here
    # rather than let ``inf / inf`` produce a NaN ratio downstream.
    if not (math.isfinite(raw_sum) and math.isfinite(raw_magnitude)):
        return _unavailable(
            raw_values=raw,
            reason=REASON_NONFINITE_RESULT,
            denominator=None,
            raw_sum=raw_sum,
            raw_magnitude=raw_magnitude,
            cancellation_ratio=None,
        )

    # A zero magnitude means every raw value is zero: there is no signal to
    # allocate and no denominator to divide by.
    if raw_magnitude == 0.0:
        return _unavailable(
            raw_values=raw,
            reason=REASON_ZERO_RAW_MAGNITUDE,
            denominator=None,
            raw_sum=raw_sum,
            raw_magnitude=raw_magnitude,
            cancellation_ratio=None,
        )

    # 4. Scale-aware cancellation ratio and 5. the cancellation threshold.
    cancellation_ratio = abs(raw_sum) / raw_magnitude
    if cancellation_ratio <= NORMALIZATION_CANCELLATION_TOLERANCE:
        # The denominator is negligible relative to the raw scale; dividing
        # by it would manufacture explosive values from float residue.
        return _unavailable(
            raw_values=raw,
            reason=REASON_NEAR_CANCELLATION,
            denominator=raw_sum,
            raw_sum=raw_sum,
            raw_magnitude=raw_magnitude,
            cancellation_ratio=cancellation_ratio,
        )

    # 6. Candidate shares.
    denominator = raw_sum
    candidate: Dict[str, float] = {
        channel: value / denominator for channel, value in raw.items()
    }

    # 7. Validate the share domain.
    if not all(math.isfinite(value) for value in candidate.values()):
        return _unavailable(
            raw_values=raw,
            reason=REASON_NONFINITE_RESULT,
            denominator=denominator,
            raw_sum=raw_sum,
            raw_magnitude=raw_magnitude,
            cancellation_ratio=cancellation_ratio,
        )

    if any(
        value < -NORMALIZATION_VALIDATION_TOLERANCE for value in candidate.values()
    ):
        return _unavailable(
            raw_values=raw,
            reason=REASON_NEGATIVE_SHARE,
            denominator=denominator,
            raw_sum=raw_sum,
            raw_magnitude=raw_magnitude,
            cancellation_ratio=cancellation_ratio,
        )

    if any(
        value > 1.0 + NORMALIZATION_VALIDATION_TOLERANCE
        for value in candidate.values()
    ):
        return _unavailable(
            raw_values=raw,
            reason=REASON_SHARE_ABOVE_ONE,
            denominator=denominator,
            raw_sum=raw_sum,
            raw_magnitude=raw_magnitude,
            cancellation_ratio=cancellation_ratio,
        )

    # Canonicalize tolerance-sized boundary deviations only.  Values that
    # reach this point are already inside the domain to within tolerance, so
    # this cannot repair a materially signed value.
    canonical: Dict[str, float] = {}
    for channel, value in candidate.items():
        if -NORMALIZATION_VALIDATION_TOLERANCE <= value < 0.0:
            canonical[channel] = 0.0
        elif 1.0 < value <= 1.0 + NORMALIZATION_VALIDATION_TOLERANCE:
            canonical[channel] = 1.0
        else:
            canonical[channel] = value

    share_sum = sum(canonical.values())
    if abs(share_sum - 1.0) > NORMALIZATION_VALIDATION_TOLERANCE:
        return _unavailable(
            raw_values=raw,
            reason=REASON_INVALID_SHARE_SUM,
            denominator=denominator,
            raw_sum=raw_sum,
            raw_magnitude=raw_magnitude,
            cancellation_ratio=cancellation_ratio,
        )

    # 8. Return normalized attribution with 9. diagnostics preserved.
    return NormalizationResult(
        attribution=canonical,
        raw_values=raw,
        normalization_status=NORMALIZATION_AVAILABLE,
        normalization_reason=None,
        normalization_denominator=denominator,
        raw_sum=raw_sum,
        raw_magnitude=raw_magnitude,
        cancellation_ratio=cancellation_ratio,
    )


__all__ = [
    "NORMALIZATION_CANCELLATION_TOLERANCE",
    "NORMALIZATION_VALIDATION_TOLERANCE",
    "NORMALIZATION_AVAILABLE",
    "NORMALIZATION_UNAVAILABLE",
    "REASON_ZERO_RAW_MAGNITUDE",
    "REASON_NEAR_CANCELLATION",
    "REASON_NONFINITE_RESULT",
    "REASON_NEGATIVE_SHARE",
    "REASON_SHARE_ABOVE_ONE",
    "REASON_INVALID_SHARE_SUM",
    "NORMALIZATION_REASONS",
    "NORMALIZATION_DIAGNOSTIC_KEYS",
    "NormalizationResult",
    "normalize_signed_attribution",
]
