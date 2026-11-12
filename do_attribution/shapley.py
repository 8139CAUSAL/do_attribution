"""
do_attribution.shapley
---------------------------

Coalitional-game-theoretic Shapley-value multi-touch attribution.

For a set of channels ``N``, let ``R(T)`` be the total conversions on paths
whose unique channel set equals exactly ``T`` (the observed coalition
conversion table, published as ``value_function``).  Each channel's Shapley
value is

    phi_i = sum_{S subset of N\\{i}}  w(s, n) * (v(S u {i}) - v(S))

with the exact factorial weight

    w(s, n) = s! * (n - s - 1)! / n!

The characteristic function ``v`` is selected by ``characteristic_function``:

``"cumulative"`` (default)
    ``v(S) = sum_{T subset of S} R(T)`` — the conversions a coalition can
    account for using only its own channels.  ``v`` is monotone and
    ``v(N)`` is the total number of observed conversions, so the efficiency
    axiom makes the Shapley values a non-negative allocation of *every*
    conversion.  Because ``v`` is a sum of unanimity games weighted by
    ``R(T)``, the factorial-weight definition above reduces exactly to

        phi_i = sum_{T containing i} R(T) / |T|

    which is what is evaluated: it costs one pass over the observed
    coalitions instead of the ``2^n`` coalition lattice.

``"exact"`` (legacy)
    ``v(S) = R(S)``.  This game is not monotone and efficiency forces
    ``sum(phi_i) = v(N)``, the conversions on paths that touch *every*
    channel.  ``v(N)`` is zero on most real data, so the raw values cancel
    and no share vector exists.  Retained for backward compatibility and
    evaluated by direct enumeration with cached factorial weights.
"""

from __future__ import annotations

from collections import defaultdict
from math import factorial, fsum
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
    prepare_journey_data,
    split_path,
    strip_conversion_labels_array,  # MODIFIED: P0.
)


#: Supported characteristic functions, default first.
SHAPLEY_CHARACTERISTIC_FUNCTIONS: Tuple[str, ...] = ("cumulative", "exact")


def _validate_characteristic_function(value: Any) -> str:
    """Validate the ``characteristic_function`` constructor argument."""
    if not isinstance(value, str):
        raise TypeError(
            "characteristic_function must be a string, got "
            f"{type(value).__name__}"
        )
    if value not in SHAPLEY_CHARACTERISTIC_FUNCTIONS:
        raise ValueError(
            "characteristic_function must be one of "
            f"{list(SHAPLEY_CHARACTERISTIC_FUNCTIONS)}, got {value!r}"
        )
    return value


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
    characteristic_function : {'cumulative', 'exact'}, default='cumulative'
        The coalition game the Shapley values are computed on, defined from
        the observed coalition conversion table ``R(T)`` (``value_function``).

        * ``'cumulative'`` — ``v(S)`` is the conversions on paths whose
          channel set is contained in ``S``.  The Shapley values are
          non-negative and sum to the total observed conversions, so
          normalized attribution exists whenever a path with at least one
          channel converted.
        * ``'exact'`` — ``v(S)`` is the conversions on paths whose channel
          set is exactly ``S`` (the behaviour of releases up to 0.0.2).
          The values are signed and sum to the conversions on paths that
          touch every channel, so normalized attribution is unavailable
          unless such paths exist and no candidate share is negative.

    Attributes
    ----------
    channels : list[str]
        All user-visible channels observed across the paths (sorted).
    value_function : dict[tuple[str, ...], int]
        ``R(T)`` = total conversions for the subset of paths whose unique
        set of channels is exactly ``T``.  Coalition keys are sorted
        tuples.  Under ``'exact'`` this is ``v`` itself; under
        ``'cumulative'``, ``v(S)`` is the sum of these entries over every
        observed coalition contained in ``S``.
    """

    def __init__(
        self,
        separator: str = DEFAULT_SEPARATOR,
        conversion_labels: Optional[Iterable[str]] = None,  # MODIFIED: P0.
        characteristic_function: str = "cumulative",
    ) -> None:
        # MODIFIED: forward conversion_labels to BaseAttribution.
        super().__init__(separator=separator, conversion_labels=conversion_labels)
        self.characteristic_function: str = _validate_characteristic_function(
            characteristic_function
        )
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
              - ``'shapley_values'`` : dict[str, float] raw Shapley values
                on the game's own scale (conversions).  Always present and
                never rescaled by normalization.  Under ``'cumulative'``
                they are non-negative and sum to the total observed
                conversions; under ``'exact'`` they are signed diagnostics
                — not shares, percentages, or probabilities.
              - ``'value_function'`` : dict[tuple, int] exact channel set
                -> conversions (``R(T)``)
              - ``'characteristic_function'`` : the game that was fitted
              - ``'normalization_status'``     : ``'available'`` or
                ``'unavailable'``
              - ``'normalization_reason'``     : ``None`` when available,
                otherwise a machine-readable reason code
              - ``'normalization_denominator'``, ``'raw_sum'``,
                ``'raw_magnitude'``, ``'cancellation_ratio'`` : numerical
                diagnostics, populated in both states

        Notes
        -----
        The Shapley efficiency axiom forces ``sum(phi_i) = v(N)``.

        Under ``'cumulative'``, ``v(N)`` is every observed conversion, so
        normalization is unavailable only when no conversion can be
        credited to a channel (``zero_raw_magnitude``).

        Under ``'exact'``, ``v(N)`` is the conversions on paths whose
        channel set is the full channel set, and ``v(N) = 0`` whenever no
        single path touches every channel.  The raw values then cancel, and
        no valid share vector exists.  ``attribution`` is ``None`` in that
        case; the raw values are still returned.

        Attributed conversions must not be computed from ``attribution``
        unless ``normalization_status == 'available'``.
        """
        # Shared preparation path: container shape and binary outcomes are
        # validated on the supplied values, before any coercion.
        paths_arr, conv_arr = prepare_journey_data(paths, conversions)
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
                "characteristic_function": self.characteristic_function,
            }
            result.update(normalization.diagnostics())
            self._store_static_reporting(
                paths_arr,
                conv_arr,
                attribution_note=_normalization_note(normalization),
            )
            return result

        # 1. Observed coalition conversion table  R(T) = sum of conversions
        #    for paths whose unique channel-set is exactly T.
        self.value_function = self._calculate_value_function(paths_arr, conv_arr)

        # 2-3. Raw Shapley values on the configured coalition game.
        if self.characteristic_function == "cumulative":
            shapley_values = self._cumulative_shapley_values()
        else:
            shapley_values = self._exact_set_shapley_values()

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
            "characteristic_function": self.characteristic_function,
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
    def _cumulative_shapley_values(self) -> Dict[str, float]:
        """Shapley values of ``v(S) = sum_{T subset of S} R(T)``.

        ``v`` is the sum of the unanimity games ``u_T`` weighted by
        ``R(T)``, and the Shapley value of ``u_T`` gives ``1/|T|`` to each
        member of ``T``.  By linearity the factorial-weight definition
        therefore equals ``phi_i = sum_{T containing i} R(T) / |T|``: each
        observed coalition splits its conversions equally among its
        channels.

        A coalition's players are its channels in ``self.channels``, so a
        reserved token that survived in a path is not a player, matching
        :func:`do_attribution.utils.extract_channels`.  Each channel's
        terms are combined with :func:`math.fsum`, which makes the result
        independent of coalition iteration order.
        """
        players = set(self.channels)
        terms: Dict[str, List[float]] = {ch: [] for ch in self.channels}
        for coalition, conversions in self.value_function.items():
            if not conversions:
                continue
            members = [ch for ch in coalition if ch in players]
            if not members:
                continue
            credit = conversions / len(members)
            for ch in members:
                terms[ch].append(credit)
        return {ch: fsum(parts) for ch, parts in terms.items()}

    def _exact_set_shapley_values(self) -> Dict[str, float]:
        """Shapley values of the legacy game ``v(S) = R(S)``."""
        n = len(self.channels)

        # Pre-compute w(s, n) for every coalition size s = 0 .. n-1.
        # This is the factorial optimisation mentioned in the spec:
        # we hit this weight 2^(n-1) * n times otherwise.
        n_fact = factorial(n)
        weights: List[float] = [
            (factorial(s) * factorial(n - s - 1)) / n_fact for s in range(n)
        ]

        # Raw Shapley values.  Use a defaultdict so unseen coalitions
        # contribute 0 without an explicit `.get`.
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
        return shapley_values

    def _calculate_value_function(
        self,
        paths: np.ndarray,
        conversions: np.ndarray,
    ) -> Dict[Tuple[str, ...], int]:
        """R(T) = total conversions whose *unique* channel set equals T."""
        coalition_map: Dict[Tuple[str, ...], int] = {}
        for path, conv in zip(paths, conversions):
            channels = tuple(
                sorted(set(split_path(str(path), self.separator)))
            )
            if not channels:
                continue
            coalition_map[channels] = coalition_map.get(channels, 0) + int(conv)
        return coalition_map


__all__ = ["ShapleyAttribution", "SHAPLEY_CHARACTERISTIC_FUNCTIONS"]
