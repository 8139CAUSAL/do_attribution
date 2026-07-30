"""
do_attribution.shapley
---------------------------

Coalitional-game-theoretic Shapley-value multi-touch attribution.

For a set of channels ``N``, define the value function ``v(S)`` as the
total conversions on paths whose unique channel set equals exactly ``S``.
Each channel's Shapley value is

    phi_i = sum_{S subset of N\\{i}}  w(s, n) * (v(S u {i}) - v(S))

with the exact factorial weight

    w(s, n) = s! * (n - s - 1)! / n!

The factorial weights ``w(s, n)`` are computed once per coalition size
and cached; coalition-value lookups use :pyclass:`collections.defaultdict`
so unseen coalitions resolve to 0 without explicit ``.get`` calls.
"""

from __future__ import annotations

from collections import defaultdict
from math import factorial
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import pandas as pd

from .base import BaseAttribution
from .normalization import (  # MODIFIED: get-well Unit 1 — shared policy.
    NormalizationResult,
    normalize_signed_attribution,
)
from .reporting import (  # MODIFIED: distilled summary sections
    build_path_diagnostics_section,
    build_shapley_detail_sections,
)
from .utils import (
    DEFAULT_SEPARATOR,
    extract_channels,
    get_powerset,
    split_path,
    strip_conversion_labels_array,  # MODIFIED: P0.
    to_int_array,
    to_string_array,
    validate_data,
)


# MODIFIED (get-well Unit 1): rendering helpers for the normalization state.
def _format_diagnostic(value: Any) -> str:
    """Render a diagnostic scalar without implying it is a share."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _normalization_note(normalization: NormalizationResult) -> Optional[str]:
    """One-line explanation used where an attribution table would go."""
    if normalization.available:
        return None
    return (
        "Normalized attribution unavailable "
        f"({normalization.normalization_reason}); see raw Shapley values"
    )


def _normalization_section(normalization: NormalizationResult) -> tuple:
    """Build the explicit normalization section for summary()/report()."""
    rows = [
        (key.replace("_", " ").capitalize(), _format_diagnostic(value))
        for key, value in normalization.diagnostics().items()
    ]
    if not normalization.available:
        rows.append(
            (
                "Attributed conversions",
                "Not calculated - requires normalized attribution",
            )
        )
    return ("Normalization", ("Diagnostic", "Value"), tuple(rows))


class ShapleyAttribution(BaseAttribution):
    """Shapley-value multi-touch attribution.

    Parameters
    ----------
    separator : str, default=' > '
        Delimiter used to split path strings into channels.
    conversion_labels : iterable of str, optional
        Tokens in the path strings that represent the conversion endpoint
        (e.g. ``"Purchase"``) rather than a real channel.  Stripped before
        any Shapley math, so endpoint tokens never appear as coalition
        members, channels, or attribution keys.  See
        :class:`do_attribution.base.BaseAttribution` for full semantics.

    Attributes
    ----------
    channels : list[str]
        All user-visible channels observed across the paths (sorted).
    value_function : dict[tuple[str, ...], int]
        ``v(S)`` = total conversions for the subset of paths whose unique
        set of channels is exactly ``S``.  Coalition keys are sorted
        tuples.
    """

    def __init__(
        self,
        separator: str = DEFAULT_SEPARATOR,
        conversion_labels: Optional[Iterable[str]] = None,  # MODIFIED: P0.
    ) -> None:
        # MODIFIED: forward conversion_labels to BaseAttribution.
        super().__init__(separator=separator, conversion_labels=conversion_labels)
        self.value_function: Dict[Tuple[str, ...], int] = {}

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def fit(
        self,
        paths: Union[Sequence[str], np.ndarray, pd.Series],
        conversions: Union[Sequence[int], np.ndarray, pd.Series],
    ) -> Dict[str, Any]:
        """Fit the Shapley model and return attribution + diagnostics.

        Returns
        -------
        dict
            Keys:
              - ``'attribution'``    : dict[str, float] of normalized shares
                summing to 1, **or** ``None`` when normalization is
                unavailable.  Always check before indexing.
              - ``'shapley_values'`` : dict[str, float] raw signed Shapley
                values.  Always present.  These are diagnostics on the
                game's own scale — not shares, percentages, or
                probabilities — and are never rescaled by normalization.
              - ``'value_function'`` : dict[tuple, int] coalition -> convs
              - ``'normalization_status'``     : ``'available'`` or
                ``'unavailable'``
              - ``'normalization_reason'``     : ``None`` when available,
                otherwise a machine-readable reason code
              - ``'normalization_denominator'``, ``'raw_sum'``,
                ``'raw_magnitude'``, ``'cancellation_ratio'`` : numerical
                diagnostics, populated in both states

        Notes
        -----
        Under this library's value function — ``v(S)`` = conversions on
        paths whose unique channel set is exactly ``S`` — the Shapley
        efficiency axiom forces ``sum(phi_i) = v(N)``, and ``v(N) = 0``
        whenever no single path touches every channel.  The raw values then
        cancel, and no valid share vector exists.  ``attribution`` is
        ``None`` in that case; the raw values are still returned.

        Attributed conversions must not be computed from ``attribution``
        unless ``normalization_status == 'available'``.
        """
        paths_arr = to_string_array(paths)
        conv_arr = to_int_array(conversions)

        validate_data(paths_arr, conv_arr)
        # MODIFIED (P0): strip conversion-endpoint tokens up front so no
        # downstream code path — channel enumeration, coalition keys, value
        # function lookups, attribution normalisation — ever sees them.
        if self.conversion_labels:
            paths_arr = strip_conversion_labels_array(
                paths_arr, self.conversion_labels, self.separator
            )
        self.channels = extract_channels(paths_arr, self.separator)
        n = len(self.channels)

        if n == 0:
            # MODIFIED (get-well Unit 1): a fit with no channels has no raw
            # game values, so there is nothing to normalise.  Route it
            # through the same policy so the result shape and the
            # normalization contract are uniform across every fit.
            self.value_function = {}
            normalization = normalize_signed_attribution({})
            self._last_attribution = normalization.attribution
            result = {
                "attribution": normalization.attribution,
                "shapley_values": {},
                "value_function": {},
            }
            result.update(normalization.diagnostics())
            self._store_static_reporting(
                paths_arr,
                conv_arr,
                attribution_note=_normalization_note(normalization),
            )
            return result

        # 1. Value function  v(S) = sum of conversions for paths whose
        #    unique channel-set is exactly S.
        self.value_function = self._calculate_value_function(paths_arr, conv_arr)

        # 2. Pre-compute w(s, n) for every coalition size s = 0 .. n-1.
        #    This is the factorial optimisation mentioned in the spec:
        #    we hit this weight 2^(n-1) * n times otherwise.
        n_fact = factorial(n)
        weights: List[float] = [
            (factorial(s) * factorial(n - s - 1)) / n_fact for s in range(n)
        ]

        # 3. Raw Shapley values.  Use a defaultdict so unseen coalitions
        #    contribute 0 without an explicit `.get`.
        vf = defaultdict(int, self.value_function)
        shapley_values: Dict[str, float] = {ch: 0.0 for ch in self.channels}
        channel_set = set(self.channels)

        for ch in self.channels:
            others = list(channel_set - {ch})
            for coalition in get_powerset(others):
                s = len(coalition)
                with_ch = tuple(sorted(coalition + (ch,)))
                without_ch = tuple(sorted(coalition))
                marginal = vf[with_ch] - vf[without_ch]
                if marginal:
                    shapley_values[ch] += weights[s] * marginal

        # 4. Normalise the raw signed game values into attribution shares.
        #
        # MODIFIED (get-well Unit 1): normalization is delegated to the
        # single authoritative policy in do_attribution.normalization.  Raw
        # Shapley values are a signed diagnostic on the game's own scale;
        # normalized attribution is a separate quantity that only exists
        # when the raw values can produce a finite, non-negative, unit-sum
        # share vector.
        #
        # This replaces the previous three-branch fallback, which substituted
        # positive-contribution renormalisation or equal credit whenever the
        # raw sum collapsed.  Those substitutions published weights that were
        # never computed by the game, so they are no longer performed: the
        # model now reports that normalization is unavailable and preserves
        # the raw values instead.  Well-conditioned fits are unaffected and
        # still use the classic v/total formula.
        normalization = normalize_signed_attribution(shapley_values)
        attribution = normalization.attribution

        if not normalization.available:
            warnings.warn(
                "Shapley normalization is unavailable "
                f"({normalization.normalization_reason}); "
                "'attribution' is None. The raw signed Shapley values remain "
                "available under 'shapley_values' and must not be read as "
                "shares, percentages, or attributed conversions. See "
                "do_attribution.normalization for the numerical contract."
            )

        self._last_attribution = attribution
        result = {
            "attribution": attribution,
            "shapley_values": shapley_values,
            "value_function": self.value_function,
        }
        # MODIFIED (get-well Unit 1): machine-readable normalization state
        # travels with the result so downstream consumers and serialization
        # can distinguish "no attribution" from "attribution of zero".
        result.update(normalization.diagnostics())
        # MODIFIED: distilled model-specific sections for summary()/report()
        # — signed raw Shapley values, top coalitions, single-channel
        # baselines, and scalable path diagnostics. Never the 2^C space.
        details = build_shapley_detail_sections(
            shapley_values=shapley_values,
            value_function=self.value_function,
        ) + (build_path_diagnostics_section(paths_arr, self.separator),)
        # MODIFIED (get-well Unit 1): make the normalization state an
        # explicit, visible section of the rendered summary.
        details = details + (_normalization_section(normalization),)
        self._store_static_reporting(
            paths_arr,
            conv_arr,
            model_details=details,
            attribution_note=_normalization_note(normalization),
        )
        return result

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #
    def _calculate_value_function(
        self,
        paths: np.ndarray,
        conversions: np.ndarray,
    ) -> Dict[Tuple[str, ...], int]:
        """v(S) = total conversions whose *unique* channel set equals S."""
        coalition_map: Dict[Tuple[str, ...], int] = {}
        for path, conv in zip(paths, conversions):
            channels = tuple(
                sorted(set(split_path(str(path), self.separator)))
            )
            if not channels:
                continue
            coalition_map[channels] = coalition_map.get(channels, 0) + int(conv)
        return coalition_map


__all__ = ["ShapleyAttribution"]
