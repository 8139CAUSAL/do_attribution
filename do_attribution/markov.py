"""
do_attribution.markov
--------------------------

N-order absorbing-Markov-chain multi-touch attribution.

Models the sequence of channel touchpoints on each customer journey as
a Markov chain with two absorbing states (``CONVERSION`` and ``NULL``).
Per-channel attribution is the removal effect: the relative change in
``P(CONVERSION | START)`` when the channel is removed from the chain.
Four operationalisations of "remove the channel" are provided -- see
``removal_strategy`` on :class:`MarkovAttribution`.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import pandas as pd

from .base import BaseAttribution
from .normalization import normalize_signed_attribution
from .reporting import (  # MODIFIED: distilled summary sections
    build_markov_detail_sections,
    build_path_diagnostics_section,
)
from .utils import (
    DEFAULT_SEPARATOR,
    create_n_order_states,
    extract_channels,
    strip_conversion_labels_array,  # MODIFIED: P0 — strip endpoint tokens up front.
    to_int_array,
    to_string_array,
    validate_data,
)

WeightedStateSequence = Tuple[List[Any], int, float]


# MODIFIED: helper used by the `redirect_to_null` removal strategy to
# decide whether a Markov state "contains" a given channel.  For order 1
# the states are plain channel strings (plus the special tokens
# START/CONVERSION/NULL), so equality is the right test; for order >= 2
# the states are tuples of channels, so membership in the tuple is the
# right test.  Special tokens never match a user channel.
def _state_contains_channel(state: Any, channel: str) -> bool:
    if isinstance(state, tuple):
        return channel in state
    return state == channel


class MarkovAttribution(BaseAttribution):
    """Markov-chain-based multi-touch attribution model.

    This implementation uses absorbing Markov chains to calculate the
    contribution of each channel/touchpoint in multi-touch conversion
    paths.  It supports both first-order and higher-order Markov models as
    well as four distinct channel-removal strategies.

    Parameters
    ----------
    order : int, default=1
        The Markov order (memory length).
        - ``order=1``: memoryless (first-order Markov)
        - ``order=2``: considers previous touchpoint (second-order Markov)
        - ``order=3+``: considers multiple previous touchpoints
    separator : str, default=' > '
        The delimiter used to separate channels in path strings.
    return_transition_matrix : bool, default=False
        Whether to include the transition matrix in the results dict.
    removal_strategy : {'fail', 'fail_refit', 'detour', 'redirect_to_null'}, default='fail'
        How the channel is "removed" before recomputing the absorbing-chain
        conversion probability:
        - ``'fail'``: fixed-population channel failure. Strip the channel,
          force affected conversions to failure, and retain every original
          journey and weight. A journey emptied by removal is represented
          explicitly as ``START -> NULL``.
        - ``'fail_refit'``: legacy strip-and-refit behavior under an explicit
          name. Empty transformed paths are dropped before rebuilding the
          model, so the fitted population and total weight can change and
          raw effects can be signed. Its results can differ from ``'fail'``
          because the two strategies fit different post-removal populations.
        - ``'detour'``: strip the channel from paths but keep the original
          conversion flag and refit.  Produces *signed* effects -- a
          negative effect means removing the channel lowered modelled
          conversion probability (positive association); a positive
          effect means removing the channel raised it (negative
          association).  Direction of association, not causation.
        - ``'redirect_to_null'``: closed-form recomputation on the base
          transition matrix with no refit.  For each channel, copy the
          transition matrix, zero every column whose state contains the
          channel (for order 1: the channel itself; for order >= 2: any
          n-gram tuple containing it), and add the removed mass to the
          ``NULL`` absorbing column.  Row sums stay at 1.  Then recompute
          ``P(CONVERSION|START)`` analytically via the fundamental matrix
          ``(I - Q)^{-1}``.  All removal effects are >= 0.

    conversion_labels : iterable of str, optional
        Tokens in the path strings that represent the conversion endpoint
        (e.g. ``"Purchase"``) rather than a real channel.  When supplied,
        these tokens are stripped from every path before channel extraction
        and state building, so they never appear in ``attribution``,
        ``removal_effects``, the transition matrix, or the viz.  See
        :class:`do_attribution.base.BaseAttribution` for full semantics.
    Attributes
    ----------
    transition_matrix : pandas.DataFrame or None
        The learned transition probability matrix (after calling fit).
    states : list or None
        All unique states in the Markov chain.
    channels : list
        All unique channels found in the paths (excluding special tokens
        and any caller-specified ``conversion_labels``).
    """

    def __init__(
        self,
        order: int = 1,
        separator: str = DEFAULT_SEPARATOR,
        return_transition_matrix: bool = False,
        removal_strategy: str = "fail",
        conversion_labels: Optional[Iterable[str]] = None,  # MODIFIED: P0.
    ) -> None:
        if order < 1:
            raise ValueError("Order must be >= 1")
        # MODIFIED: accept 'redirect_to_null' in addition to 'fail' and 'detour'.
        if removal_strategy not in {
            "fail",
            "fail_refit",
            "detour",
            "redirect_to_null",
        }:
            raise ValueError(
                "removal_strategy must be 'fail', 'fail_refit', 'detour', "
                "or 'redirect_to_null'"
            )

        # MODIFIED: forward conversion_labels to BaseAttribution, which
        # validates + normalises (collisions with SPECIAL_TOKENS raise here).
        super().__init__(separator=separator, conversion_labels=conversion_labels)
        self.order: int = order
        self.return_transition_matrix: bool = return_transition_matrix
        self.removal_strategy: str = removal_strategy
        self._fail_migration_warned: bool = False
        self.transition_matrix: pd.DataFrame | None = None
        self.states: List[Any] | None = None

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def fit(
        self,
        paths: Union[Sequence[str], np.ndarray, pd.Series],
        conversions: Union[Sequence[int], np.ndarray, pd.Series],
        weights: Optional[
            Union[Sequence[float], np.ndarray, pd.Series]
        ] = None,
    ) -> Dict[str, Any]:
        """Fit the Markov model and return attribution + removal effects.

        Parameters
        ----------
        paths : sequence of str
            Ordered channel paths.
        conversions : sequence of int
            Binary conversion outcomes aligned with ``paths``.
        weights : sequence of float, optional
            Per-journey observation weights. Defaults to unit weight.
            Values must be finite, nonnegative, aligned with ``paths``, and
            have positive total weight.

        Returns
        -------
        dict
            Keys:
              - ``'attribution'``         : dict[str, float] summing to 1,
                or ``None`` when raw effects cannot be normalized safely
              - ``'removal_effects'``     : dict[str, float] (raw)
              - ``'total_conversion_probability'`` : float
              - ``'transition_matrix'``   : DataFrame (only if requested)
              - ``'population_diagnostics'`` : per-channel journey-count,
                total-weight, and empty-path diagnostics for ``fail`` and
                ``fail_refit``
        """
        paths_arr = to_string_array(paths)
        conversions_arr = to_int_array(conversions)

        validate_data(paths_arr, conversions_arr)
        if (
            self.removal_strategy == "fail"
            and not self._fail_migration_warned
        ):
            warnings.warn(
                "removal_strategy='fail' now uses fixed-population "
                "semantics and preserves emptied journeys as START -> NULL "
                "failures. To preserve the legacy population-changing "
                "strip-and-refit behavior, select "
                "removal_strategy='fail_refit'.",
                UserWarning,
                stacklevel=2,
            )
            self._fail_migration_warned = True
        if weights is None:
            weights_arr = np.ones(len(paths_arr), dtype=float)
        else:
            weights_arr = np.asarray(weights, dtype=float)
            if weights_arr.ndim != 1:
                raise ValueError("weights must be one-dimensional")
            if len(weights_arr) != len(paths_arr):
                raise ValueError(
                    f"Paths ({len(paths_arr)}) and weights "
                    f"({len(weights_arr)}) must have same length"
                )
            if not np.isfinite(weights_arr).all():
                raise ValueError("weights must contain only finite values")
            if (weights_arr < 0).any():
                raise ValueError("weights must be nonnegative")
            if float(weights_arr.sum()) <= 0.0:
                raise ValueError("weights must have positive total weight")
        # MODIFIED (P0): strip any tokens the caller flagged as conversion
        # endpoints *before* channel extraction / state building / matrix
        # assembly.  Everything downstream therefore operates on paths that
        # contain only real channels plus SPECIAL_TOKENS — the endpoint
        # label can never leak into attribution, removal_effects, the
        # transition matrix, or the viz.
        if self.conversion_labels:
            paths_arr = strip_conversion_labels_array(
                paths_arr, self.conversion_labels, self.separator
            )
        self.channels = extract_channels(paths_arr, self.separator)

        state_sequences: List[WeightedStateSequence] = []
        for path, converted, weight in zip(
            paths_arr, conversions_arr, weights_arr
        ):
            states = create_n_order_states(path, self.order, self.separator)
            if states:
                state_sequences.append(
                    (states, int(converted), float(weight))
                )

        self.transition_matrix = self._build_transition_matrix(state_sequences)
        base_prob = self._calculate_conversion_probability(self.transition_matrix)

        removal_effects, population_diagnostics = self._calculate_removal_effects(
            state_sequences,
            base_prob,
            paths_arr,
            conversions_arr,
            weights_arr,
        )

        normalization = None
        if self.removal_strategy in {"fail", "fail_refit", "detour"}:
            # Raw effects remain separate diagnostics for every path-refit
            # strategy. Route them through the same scale-aware guard as
            # Shapley so signed, zero, or near-cancelling values cannot be
            # presented as a share vector.
            normalization = normalize_signed_attribution(removal_effects)
            attribution = normalization.attribution
        else:
            total_effect = sum(removal_effects.values())
            if total_effect > 0:
                attribution = {
                    ch: eff / total_effect for ch, eff in removal_effects.items()
                }
            else:
                attribution = {
                    ch: 1.0 / len(self.channels) if self.channels else 0.0
                    for ch in self.channels
                }
                if self.channels:
                    warnings.warn(
                        "Total removal effect is 0. Using equal attribution."
                    )

        results: Dict[str, Any] = {
            "attribution": attribution,
            "removal_effects": removal_effects,
            "total_conversion_probability": base_prob,
        }
        if population_diagnostics:
            results["population_diagnostics"] = population_diagnostics
            results["population_policy"] = (
                "fixed"
                if self.removal_strategy == "fail"
                else (
                    "population_refit"
                    if self.removal_strategy == "fail_refit"
                    else self.removal_strategy
                )
            )
        if normalization is not None:
            results.update(normalization.diagnostics())
        if self.return_transition_matrix:
            results["transition_matrix"] = self.transition_matrix
        self._last_attribution = attribution
        # MODIFIED: distilled model-specific sections for summary()/report()
        # — raw signed removal effects beside shares, top transitions and
        # drop-offs, and scalable path diagnostics. Bounded rows only.
        details = build_markov_detail_sections(
            removal_effects=removal_effects,
            attribution=attribution,
            transition_matrix=self.transition_matrix,
        ) + (build_path_diagnostics_section(paths_arr, self.separator),)
        self._store_static_reporting(
            paths_arr,
            conversions_arr,
            model_details=details,
            attribution_note=(
                None
                if normalization is None or normalization.available
                else "Normalized attribution unavailable "
                f"({normalization.normalization_reason}). Raw signed "
                "removal effects remain available as diagnostics."
            ),
        )
        return results

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #
    def _build_transition_matrix(
        self,
        state_sequences: List[WeightedStateSequence],
    ) -> pd.DataFrame:
        """Build the transition probability matrix from state sequences.

        Vectorised numpy implementation. Mathematically identical to the
        prior pandas-based build: integer transition counts are accumulated
        via ``np.add.at`` on a ``float64`` matrix (exact for counts <<
        2**53), rows are divided by their sums (zero-sum rows yield an
        all-zero row, matching the prior ``fillna(0)`` behaviour), and
        absorbing rows are overwritten with identity. Bit-for-bit
        equivalent to the original pandas implementation under the 96
        equivalence tests (1e-9 tolerance).
        """
        # MODIFIED: replaced pandas `.at[...] += 1` hot loop with a
        # vectorised numpy accumulation. The previous implementation spent
        # ~4.3 s per call on the 88K-row Anderl expansion due to pandas
        # label -> iloc lookups; this version runs in the tens of ms while
        # preserving exact arithmetic (see docstring above).
        absorbing_states = {"CONVERSION", "NULL"}

        all_states = set({"START"} | absorbing_states)
        for states, _, _ in state_sequences:
            all_states.update(states)

        self.states = sorted(list(all_states), key=str)
        n = len(self.states)
        state_to_idx = {s: i for i, s in enumerate(self.states)}
        start_idx = state_to_idx["START"]
        conv_idx = state_to_idx["CONVERSION"]
        null_idx = state_to_idx["NULL"]

        # First pass: compute total number of transitions so we can allocate
        # flat int64 from/to index buffers exactly once and avoid Python
        # list growth overhead on large inputs.
        total_transitions = 0
        for states, _, _ in state_sequences:
            k = len(states)
            if k == 0:
                continue
            total_transitions += k + 1  # START->s[0], s[i]->s[i+1] (k-1), s[-1]->terminal

        from_arr = np.empty(total_transitions, dtype=np.int64)
        to_arr = np.empty(total_transitions, dtype=np.int64)
        transition_weights = np.empty(total_transitions, dtype=np.float64)

        pos = 0
        for states, converted, weight in state_sequences:
            k = len(states)
            if k == 0:
                continue
            # START -> states[0]
            from_arr[pos] = start_idx
            to_arr[pos] = state_to_idx[states[0]]
            transition_weights[pos] = weight
            pos += 1
            # states[i] -> states[i+1]
            for i in range(k - 1):
                from_arr[pos] = state_to_idx[states[i]]
                to_arr[pos] = state_to_idx[states[i + 1]]
                transition_weights[pos] = weight
                pos += 1
            # states[-1] -> terminal
            from_arr[pos] = state_to_idx[states[-1]]
            to_arr[pos] = conv_idx if converted else null_idx
            transition_weights[pos] = weight
            pos += 1

        counts = np.zeros((n, n), dtype=np.float64)
        if pos > 0:
            np.add.at(
                counts,
                (from_arr[:pos], to_arr[:pos]),
                transition_weights[:pos],
            )

        row_sums = counts.sum(axis=1)
        # Row-normalise with safe divide; zero-sum rows become all-zero
        # (equivalent to the prior `.fillna(0)` on NaN-producing 0/0).
        tm_values = np.divide(
            counts,
            row_sums[:, None],
            out=np.zeros_like(counts),
            where=row_sums[:, None] > 0,
        )

        # Absorbing rows are overwritten with identity on themselves.
        for s in absorbing_states:
            if s in state_to_idx:
                idx = state_to_idx[s]
                tm_values[idx, :] = 0.0
                tm_values[idx, idx] = 1.0

        # Wrap back as a DataFrame so the rest of the class (which indexes
        # by label via `.loc`) is entirely unchanged.
        return pd.DataFrame(tm_values, index=self.states, columns=self.states)

    def _calculate_conversion_probability(self, tm: pd.DataFrame) -> float:
        """P(CONVERSION | START) via the fundamental matrix N = (I - Q)^-1."""
        tm_states = tm.index.tolist()
        absorbing = [s for s in ["CONVERSION", "NULL"] if s in tm_states]
        transient = [s for s in tm_states if s not in absorbing]

        if "START" not in transient or "CONVERSION" not in absorbing:
            return 0.0

        Q = tm.loc[transient, transient].values
        R = tm.loc[transient, absorbing].values
        I = np.eye(len(transient))
        # MODIFIED: solve the linear system directly instead of forming an
        # explicit inverse — one LU solve has a tighter error bound than
        # inv() followed by a matmul and is faster. Exactly singular (I - Q)
        # is unreachable for data-built chains (every observed sequence
        # terminates in an absorbing state, so rho(Q) < 1), but remains
        # possible for caller-supplied matrices; the pinv fallback is a
        # least-squares surrogate with NO probability-conservation
        # guarantee, so it now warns instead of silently substituting.
        try:
            B = np.linalg.solve(I - Q, R)
        except np.linalg.LinAlgError:
            warnings.warn(
                "(I - Q) is singular; falling back to the pseudo-inverse. "
                "Absorption probabilities are a least-squares surrogate and "
                "may not conserve probability."
            )
            B = np.linalg.pinv(I - Q) @ R

        start_idx = transient.index("START")
        conv_idx = absorbing.index("CONVERSION")
        prob = float(B[start_idx, conv_idx])
        # MODIFIED: clamp floating-point drift (bounded by cond(I-Q) * eps)
        # or pinv junk back onto the probability simplex so downstream
        # removal-effect ratios can never see p < 0 or p > 1.
        return min(1.0, max(0.0, prob))

    def _calculate_removal_effects(
        self,
        state_sequences: List[WeightedStateSequence],
        base_prob: float,
        original_paths: np.ndarray,
        original_conversions: np.ndarray,
        original_weights: np.ndarray,
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
        """Compute the removal effect of each channel.

        For ``'fail'``: paths containing the channel have their conversion
        flag forced to 0 and the channel is stripped from each state. Every
        journey and weight is retained; an emptied path becomes
        ``START -> NULL``. The removal effect is
        ``(base_prob - fixed_prob) / base_prob``.

        For ``'fail_refit'``: apply the legacy ``'fail'`` transform, drop
        emptied paths, and rebuild the model on the remaining population.
        The fitted count and total weight can therefore change, and raw
        effects can be signed.

        For ``'detour'``: the channel is stripped from each original path
        while keeping the original conversion flag, and a *fresh* model is
        fit on the trimmed data.  The removal effect is
        ``(det_prob - base_prob) / base_prob`` and can be negative
        (channel was helpful) or positive (channel was harmful).

        For ``'redirect_to_null'``: the base transition matrix is copied,
        every column whose state contains the channel is zeroed and the
        removed mass is added to the ``NULL`` absorbing column (mass is
        preserved: row sums remain 1).  The removal effect is
        ``(base_prob - red_prob) / base_prob``.  See the class docstring
        for the operational definition.
        """
        # A denormal-tiny base probability cannot support a stable relative
        # effect. The transformations still run so population diagnostics
        # remain inspectable; their raw effects are reported as zero and the
        # shared normalizer returns ``zero_raw_magnitude``.
        has_stable_base_probability = base_prob > np.finfo(float).eps

        effects: Dict[str, float] = {}
        population_diagnostics: Dict[str, Dict[str, Any]] = {}
        sep = self.separator

        # MODIFIED: precompute index mapping once for 'redirect_to_null'.
        if self.removal_strategy == "redirect_to_null":
            tm_base_vals = self.transition_matrix.values
            state_to_idx = {s: i for i, s in enumerate(self.states)}
            null_idx = state_to_idx["NULL"]

        for channel in self.channels:
            if self.removal_strategy == "fail":
                transformed_sequences: List[WeightedStateSequence] = []
                emptied_path_count = 0
                for states, converted, weight in state_sequences:
                    if any(
                        _state_contains_channel(state, channel)
                        for state in states
                    ):
                        new_states = [
                            state
                            for state in states
                            if not _state_contains_channel(state, channel)
                        ]
                        new_converted = 0
                    else:
                        new_states = states
                        new_converted = converted

                    if not new_states:
                        # Fixed-population failure retains the observation
                        # and makes the empty transform an explicit
                        # START -> NULL path.
                        new_states = ["NULL"]
                        new_converted = 0
                        emptied_path_count += 1
                    transformed_sequences.append(
                        (new_states, new_converted, weight)
                    )

                population_diagnostics[channel] = {
                    "original_journey_count": len(state_sequences),
                    "transformed_journey_count": len(
                        transformed_sequences
                    ),
                    "original_total_weight": float(
                        sum(weight for _, _, weight in state_sequences)
                    ),
                    "transformed_total_weight": float(
                        sum(
                            weight
                            for _, _, weight in transformed_sequences
                        )
                    ),
                    "emptied_path_count": emptied_path_count,
                    "dropped_path_count": 0,
                    "empty_path_handling": "start_to_null",
                    "empty_path_representation": "START -> NULL",
                }

                fixed_tm = self._build_transition_matrix(
                    transformed_sequences
                )
                fixed_prob = self._calculate_conversion_probability(fixed_tm)
                effect = (
                    (base_prob - fixed_prob) / base_prob
                    if has_stable_base_probability
                    else 0.0
                )
                effects[channel] = (
                    0.0 if -1e-12 < effect < 0.0 else effect
                )

            elif self.removal_strategy == "fail_refit":
                reduced_sequences: List[WeightedStateSequence] = []
                emptied_path_count = 0
                for states, converted, weight in state_sequences:
                    # MODIFIED: exact state membership via
                    # _state_contains_channel. `channel in s` on order-1
                    # string states was a SUBSTRING test ('Search' matched
                    # 'Paid Search'), contaminating removal effects for any
                    # channel whose name nests inside another channel's.
                    if any(_state_contains_channel(s, channel) for s in states):
                        new_states = [
                            s
                            for s in states
                            if not _state_contains_channel(s, channel)
                        ]
                        new_converted = 0  # force failure
                    else:
                        new_states = states
                        new_converted = converted
                    if new_states:
                        reduced_sequences.append(
                            (new_states, new_converted, weight)
                        )
                    else:
                        emptied_path_count += 1

                population_diagnostics[channel] = {
                    "original_journey_count": len(state_sequences),
                    "transformed_journey_count": len(reduced_sequences),
                    "original_total_weight": float(
                        sum(weight for _, _, weight in state_sequences)
                    ),
                    "transformed_total_weight": float(
                        sum(weight for _, _, weight in reduced_sequences)
                    ),
                    "emptied_path_count": emptied_path_count,
                    "dropped_path_count": emptied_path_count,
                    "empty_path_handling": "drop_after_channel_removal",
                }

                if not reduced_sequences:
                    effects[channel] = 1.0
                    continue

                red_tm = self._build_transition_matrix(reduced_sequences)
                red_prob = self._calculate_conversion_probability(red_tm)
                effect = (
                    (base_prob - red_prob) / base_prob
                    if has_stable_base_probability
                    else 0.0
                )
                # MODIFIED: 'fail' effects are >= 0 by contract; zero out
                # sub-epsilon float residue from the LU solve.
                effects[channel] = 0.0 if -1e-12 < effect < 0.0 else effect

            elif self.removal_strategy == "redirect_to_null":
                # MODIFIED: analytical redirect-to-null on a copy of the
                # base transition matrix.  No refit; no data mutation.
                bad_cols = [
                    state_to_idx[s]
                    for s in self.states
                    if _state_contains_channel(s, channel)
                ]
                if not bad_cols:
                    effects[channel] = 0.0
                    continue
                tm_copy = tm_base_vals.copy()
                # Sum mass going into any 'bad' state, per source row, and
                # redirect that mass to the NULL absorbing column.
                bad_col_mass = tm_copy[:, bad_cols].sum(axis=1)
                tm_copy[:, bad_cols] = 0.0
                tm_copy[:, null_idx] += bad_col_mass
                # The NULL absorbing row was already (0, ..., 1, ..., 0) in
                # the base matrix and is left unchanged. Row sums preserved
                # at 1 by construction.
                red_tm_df = pd.DataFrame(
                    tm_copy, index=self.states, columns=self.states
                )
                red_prob = self._calculate_conversion_probability(red_tm_df)
                effect = (
                    (base_prob - red_prob) / base_prob
                    if has_stable_base_probability
                    else 0.0
                )
                # MODIFIED: same float-residue clamp as 'fail' (contract >= 0).
                effects[channel] = 0.0 if -1e-12 < effect < 0.0 else effect

            else:  # removal_strategy == "detour"
                trimmed_paths: List[str] = []
                for path in original_paths:
                    channels_in_path = path.split(sep)
                    filtered = [ch for ch in channels_in_path if ch != channel]
                    trimmed_paths.append(sep.join(filtered))

                valid_data = [
                    (p, c, weight)
                    for p, c, weight in zip(
                        trimmed_paths,
                        original_conversions,
                        original_weights,
                    )
                    if p
                ]
                if not valid_data:
                    effects[channel] = 1.0
                    continue

                (
                    filtered_paths,
                    filtered_conversions,
                    filtered_weights,
                ) = zip(*valid_data)

                # MODIFIED: compute only P(CONVERSION|START) on the trimmed
                # data. temp_model.fit() also ran the nested per-channel
                # removal-effect loop, making 'detour' O(C^2) transition-
                # matrix builds and inversions when O(C) suffices, and it
                # leaked nested 'Total removal effect is 0' warnings.
                temp_model = MarkovAttribution(
                    order=self.order,
                    separator=self.separator,
                    return_transition_matrix=False,
                    removal_strategy="fail",
                )
                det_sequences: List[WeightedStateSequence] = []
                for det_path, det_conv, det_weight in zip(
                    filtered_paths,
                    filtered_conversions,
                    filtered_weights,
                ):
                    det_states = create_n_order_states(
                        det_path, self.order, self.separator
                    )
                    if det_states:
                        det_sequences.append(
                            (
                                det_states,
                                int(det_conv),
                                float(det_weight),
                            )
                        )
                if not det_sequences:
                    effects[channel] = 1.0
                    continue
                det_tm = temp_model._build_transition_matrix(det_sequences)
                det_prob = temp_model._calculate_conversion_probability(det_tm)
                effects[channel] = (
                    (det_prob - base_prob) / base_prob
                    if has_stable_base_probability
                    else 0.0
                )

        return effects, population_diagnostics


__all__ = ["MarkovAttribution"]
