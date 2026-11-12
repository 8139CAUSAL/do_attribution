"""
do_attribution.temporal_shapley
-------------------------------

Time-varying Shapley-value multi-touch attribution.

`TemporalShapleyAttribution` inherits from `ShapleyAttribution` and adds
per-epoch attribution on top of the inherited Shapley math. A temporal fit
is a static fit applied repeatedly on time-sliced paths, plus the stitching
logic that assembles the per-epoch results into trends and an attribution
matrix. Every knob the parent exposes (`separator`, `conversion_labels`) is
inherited unchanged and forwarded via ``super().__init__``.

Scope note
----------
This module dispatches the inherited static fit per user-chosen epoch.
``freq`` is supplied by the caller — either a single pandas alias
(``'Q'``/``'M'``/``'W'``/...) or a ``Mapping[int, str]`` for per-year
mixed granularity. **The class does not select epochs from data.** The
``suggest_freq`` diagnostic in ``temporal_diagnostics`` is a sample-
floor walker, not a data-driven epoch recommender; epoch selection
is the caller's responsibility.

An earlier design proposal (per-channel change-point detection unioned
into a global epoch lattice with staircase stitching) appears in
``temporal_shaply.md`` and has been **retired** — it overfits long-
tailed channels by cutting on neighbour-channel signal noise. The
forward direction is global-signal alignment; design open. See
``TEMPORAL_PUNCHLIST.md`` at the workspace root.

API
---
Per the md's Option 1 decision: ``fit`` requires timestamps on this class;
calling the parent's static two-argument fit by accident is a footgun we
block by signature. Use ``fit_static`` as an explicit escape hatch when
the caller actually wants the static Shapley fit on a temporal instance.

Path-to-epoch assignment
------------------------
``epoch_assignment`` selects the convention by which a path's single
timestamp is interpreted. Default ``'conversion'`` (the path is binned to
the epoch in which the conversion fired) — Shapley is a coalition game
over the converting cohort, and the conversion timestamp is the natural
anchor for that cohort. ``'first_touch'`` is also supported. ``'split'``
(proportional split across epochs the path spans) is reserved for a
future release and will raise ``NotImplementedError`` from ``fit`` /
``fit_by_epoch``.

Note that the Markov sibling class defaults to ``'first_touch'`` instead,
because the Markov chain is anchored at ``START`` and the principled
default there is the epoch in which the path *originated*. The two
defaults reflect a real difference in the underlying math, not a bug.

Lazy global fit (Risk 5)
------------------------
The static "all paths" fit is deferred until ``self.global_attribution``
(or anything that needs the global view) is accessed for the first time.
This avoids paying for an expensive N+1th Shapley combinatoric pass when
the caller only wants per-epoch trends. The result dict returned from
``fit`` therefore does NOT contain ``global_attribution`` — pull it from
the property when you need it.

Example
-------
>>> from do_attribution.temporal_shapley import TemporalShapleyAttribution
>>> import pandas as pd
>>>
>>> model = TemporalShapleyAttribution(
...     freq='Q',
...     min_samples_per_epoch=10,
...     epoch_assignment='conversion',
... )
>>> result = model.fit(
...     paths=df['path'],
...     timestamps=df['conversion_date'],
...     conversions=df['conversions'],
... )
>>> email_trend = model.get_trend('Email')
>>> matrix = model.get_attribution_matrix()
>>> # Lazy global pass triggers here:
>>> static_global = model.global_attribution
>>> # Compare static-global to the column-mean weighted by conversions:
>>> volume_weighted = model.get_volume_weighted_attribution()
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import pandas as pd

# MODIFIED: relative imports matching the rest of the package.
from .shapley import ShapleyAttribution
from .utils import (
    DEFAULT_SEPARATOR,
    prepare_journey_data,
    split_path,
)
from .temporal import assign_paths_to_epochs, recommend_epochs
from ._temporal_helpers import (
    VALID_EPOCH_ASSIGNMENTS as _VALID_EPOCH_ASSIGNMENTS,
    align_index_to_epochs as _align_index_to_epochs,
    bin_timestamps_to_epochs as _bin_timestamps_to_epochs,
    epoch_attribution as _epoch_attribution,
    get_period_key as _get_period_key,
    matrix_to_long_frame as _matrix_to_long_frame,
    pandas_freq_alias as _pandas_freq_alias,
    periods_have_different_freq as _periods_have_different_freq,
    sample_sizes_to_long_frame as _sample_sizes_to_long_frame,
    validate_epoch_assignment as _validate_epoch_assignment,
    validate_freq as _validate_freq,
    validate_freq_alias as _validate_freq_alias,
)
from .reporting import (  # MODIFIED: + distilled epoch section builder
    build_fitted_snapshot,
    build_shapley_epoch_coalition_section,
    render_temporal_report,
)


class TemporalShapleyAttribution(ShapleyAttribution):
    """Time-varying Shapley-value multi-touch attribution.

    Inherits from :class:`ShapleyAttribution`. Each per-epoch compute step
    calls ``super().fit`` on the epoch's slice, so every correctness fix
    made to the parent Shapley code (characteristic function,
    normalisation policy, conversion-label stripping, etc.) applies to the
    temporal variant automatically.

    Parameters
    ----------
    freq : str or Mapping[int, str], default='Q'
        Pandas frequency alias for temporal binning, supplied by the
        caller. Common values: ``'D'``, ``'W'``, ``'M'``, ``'Q'``,
        ``'Y'``. Epoch selection is the caller's responsibility; this
        class does not infer a freq from the data.

        Also accepts a ``Mapping[int, str]`` mapping calendar year ->
        pandas freq alias (e.g. ``{2024: 'Q', 2025: 'M'}``). Under a
        mapping, each year's timestamps are binned with that year's
        freq and the resulting heterogeneous Periods are concatenated
        into the epoch list, sorted by ``start_time``. The mapping
        shape is what ``suggest_freq(timestamps, by='year')`` produces
        — useful when the caller wants a per-year sample-floor sanity
        check, but ``suggest_freq`` is a sample-floor walker, not a
        data-driven epoch recommender (see
        ``temporal_diagnostics``). Mixed-granularity goes through
        :meth:`fit` only — :meth:`fit_by_epoch` raises ``TypeError``
        if ``freq`` is a mapping.
    min_samples_per_epoch : int, default=30
        Minimum number of paths required in an epoch for reliable
        Shapley computation. Epochs with fewer samples will trigger
        a warning and may be excluded from trends. Shapley's per-epoch
        cost scales as ``O(2^n_channels)``; under high cardinality the
        caller may want to raise this floor manually so each per-epoch
        fit has enough conversions per coalition to be identifiable.
    epoch_assignment : {'first_touch', 'conversion', 'split'}, default='conversion'
        Convention by which a path's single ``timestamps[i]`` is
        interpreted (Risk 1 fix from the v1 punch list). The Shapley
        default is ``'conversion'`` because Shapley is a coalition game
        over the converting cohort. ``'first_touch'`` is also supported.
        ``'split'`` (proportional split across epochs the path spans) is
        reserved for a future release and raises
        ``NotImplementedError`` from ``fit`` / ``fit_by_epoch``.
    separator : str, default=' > '
        Delimiter used to split path strings into channels. Forwarded to
        :class:`ShapleyAttribution`.
    conversion_labels : iterable of str, optional
        Tokens in path strings that represent the conversion endpoint
        (e.g. ``"Purchase"``). Forwarded to :class:`ShapleyAttribution`,
        which strips them per-call during the inherited ``fit``.
    characteristic_function : {'cumulative', 'exact'}, default='cumulative'
        Coalition game forwarded to :class:`ShapleyAttribution` and applied
        identically to every epoch and to the lazy global fit.

    Attributes
    ----------
    epochs : list[Period]
        Chronologically sorted list of time periods analysed.
    channels : list[str]
        Union of channels observed across all epochs (set eagerly from
        per-epoch results). Equal to the global static union by
        construction. Surfaced eagerly so callers do not have to trigger
        the lazy global fit just to enumerate channels.
    epoch_results : dict[Period, dict]
        Shapley results keyed by epoch period. Each value is the dict
        returned by the parent's ``fit``
        (``attribution`` / ``shapley_values`` / ``value_function`` /
        ``characteristic_function`` plus the normalization diagnostics).
    global_attribution : dict[str, float]
        Static Shapley attribution on the full path set. **Lazy** — the
        underlying ``super().fit`` call is deferred until first access,
        so per-epoch-only callers do not pay for the N+1th refit.

        .. warning::
           This is the static Shapley fit on the full path set, not a
           mean of per-epoch attributions. Under the legacy ``'exact'``
           game, channels that enter or exit the data across epochs make
           it diverge from :meth:`get_volume_weighted_attribution`. Under
           the default ``'cumulative'`` game the Shapley values are
           additive over disjoint sets of paths, so the two views agree.

    Examples
    --------
    >>> model = TemporalShapleyAttribution(freq='M')
    >>> model.fit(paths, timestamps, conversions)
    >>> email_trend = model.get_trend('Email')
    >>> matrix = model.get_attribution_matrix()
    """

    def __init__(
        self,
        freq: Union[str, Mapping[int, str]] = "Q",
        min_samples_per_epoch: int = 30,
        # MODIFIED (Risk 1): explicit epoch-assignment convention.
        epoch_assignment: str = "conversion",
        separator: str = DEFAULT_SEPARATOR,
        conversion_labels: Optional[Iterable[str]] = None,
        characteristic_function: str = "cumulative",
    ) -> None:
        super().__init__(
            separator=separator,
            conversion_labels=conversion_labels,
            characteristic_function=characteristic_function,
        )

        # MODIFIED (v2 PR 2): validate freq up-front. Accepts str or
        # Mapping[int, str]; mapping enables mixed-granularity per-year
        # fits (consumer of suggest_freq(by='year')).
        self.freq = _validate_freq(freq)
        self.min_samples = min_samples_per_epoch
        # MODIFIED (Risk 1): store + validate epoch_assignment. 'split' is
        # accepted at construction time but raises at fit time.
        self.epoch_assignment = _validate_epoch_assignment(epoch_assignment)

        # Temporal state — populated by fit / fit_by_epoch.
        self.epochs: List[Any] = []
        self.epoch_results: Dict[Any, Dict[str, Any]] = {}
        self._epoch_paths: Dict[Any, np.ndarray] = {}
        self._epoch_conversions: Dict[Any, np.ndarray] = {}

        # MODIFIED (Risk 5): lazy-global state. The cache is populated on
        # first access of the `global_attribution` property; until then the
        # parent's static fit has not run.
        self._global_fit_done: bool = False
        # MODIFIED (get-well Unit 1): ``None`` means the global fit ran but
        # produced no valid share vector; ``{}`` means it has not run.
        self._global_attribution_cache: Optional[Dict[str, float]] = {}
        self._global_shapley_values_cache: Dict[str, float] = {}
        self._global_normalization: Dict[str, Any] = {}  # MODIFIED
        self._reporting_snapshot = None
        self.epoch_recommendation = None

    # ------------------------------------------------------------------ #
    # Risk 1: epoch_assignment guard                                      #
    # ------------------------------------------------------------------ #
    def _check_epoch_assignment_supported(self) -> None:
        """Block 'split' at fit time per the v1 scope tether."""
        if self.epoch_assignment == "split":
            raise NotImplementedError(
                "epoch_assignment='split' (proportional split across "
                "epochs spanned by a path) is reserved for a future "
                "release. v1 supports 'first_touch' and 'conversion' "
                "only. See temporal_shaply.md and temporal_punch_list.md."
            )

    # ------------------------------------------------------------------ #
    # v2 PR 2: per-year freq resolution                                   #
    # ------------------------------------------------------------------ #
    def _resolve_freq(self, year: int) -> str:
        """Return the pandas freq alias to use for ``year``.

        For ``str`` ``self.freq``, returns it unchanged for any year.
        For ``Mapping`` ``self.freq``, looks up the year and raises
        ``KeyError`` with a helpful message if the year is missing —
        catching missing-year configuration before the per-epoch fit
        runs is preferable to a silent KeyError deep in the fit loop.
        """
        if isinstance(self.freq, str):
            return self.freq
        try:
            return self.freq[year]
        except KeyError:
            raise KeyError(
                f"freq mapping has no entry for year {year!r}. Data "
                f"contains timestamps in this year, but the mapping "
                f"only covers {sorted(self.freq.keys())}. Either pass "
                f"a freq for {year!r} explicitly or use "
                f"suggest_freq(timestamps, by='year') to derive the "
                f"full mapping from the data."
            )

    def _is_mapping_freq(self) -> bool:
        """True if ``self.freq`` is a per-year mapping."""
        return isinstance(self.freq, Mapping)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def fit(  # type: ignore[override]
        self,
        paths: Union[Sequence[str], np.ndarray, pd.Series],
        timestamps: Union[Sequence[Any], np.ndarray, pd.Series, pd.DatetimeIndex],
        conversions: Union[Sequence[int], np.ndarray, pd.Series],
    ) -> Dict[str, Any]:
        """Fit the temporal Shapley model.

        Overrides :meth:`ShapleyAttribution.fit` to require ``timestamps``.
        Calling the two-argument static signature on a temporal instance
        therefore raises ``TypeError`` (intentional footgun-block per the
        design doc); use :meth:`fit_static` to reach the inherited static
        fit on this instance.

        Parameters
        ----------
        paths : sequence of str
            Touchpoint paths (e.g., ``"Social > Email > Direct"``).
        timestamps : sequence of datetime
            One timestamp per path. Interpreted per
            ``self.epoch_assignment`` — for the default ``'conversion'``
            this is expected to be the conversion timestamp; for
            ``'first_touch'`` it is the path-origination timestamp.
        conversions : sequence of int
            Conversion counts corresponding to each path.

        Returns
        -------
        dict
            Keys:
              - ``'epoch_attributions'`` : dict[Period, dict] per-epoch results
              - ``'attribution_matrix'`` : pd.DataFrame channels × epochs
              - ``'epochs'`` : list of Period objects analysed
              - ``'channels'`` : list of channels (union of per-epoch results)

            ``global_attribution`` is **not** in the dict. Access via the
            ``model.global_attribution`` property (lazy — see Risk 5 in
            ``temporal_punch_list.md``).
        """
        # MODIFIED (Risk 1): block 'split' immediately.
        self._check_epoch_assignment_supported()

        # Shared preparation path: the supplied outcomes are validated
        # before coercion, ahead of any per-epoch dispatch.
        paths_arr, conv_arr = prepare_journey_data(paths, conversions)

        if len(timestamps) != len(paths_arr):
            raise ValueError(
                f"timestamps length ({len(timestamps)}) does not match "
                f"paths length ({len(paths_arr)})"
            )

        dt_index = pd.DatetimeIndex(pd.to_datetime(timestamps))

        self._reset_state()
        # MODIFIED: exception-safe lifecycle — any failure below leaves the
        # instance fully unfitted instead of holding partial epoch state.
        try:
            return self._fit_binned(dt_index, paths_arr, conv_arr)
        except BaseException:
            self._reset_state()
            raise

    def _fit_binned(
        self,
        dt_index: pd.DatetimeIndex,
        paths_arr: np.ndarray,
        conv_arr: np.ndarray,
    ) -> Dict[str, Any]:
        """Per-year binning + per-epoch dispatch (exception-safe body)."""

        # MODIFIED (v2 PR 2): per-year binning. Each timestamp is binned
        # with its own calendar year's freq, so a Mapping-freq yields
        # epochs of different freqs, sorted by ``start_time`` (see
        # ``_get_period_key``). Epochs are keyed by period, not by year
        # bucket: a period that straddles a year boundary (the weekly
        # period 2020-12-28/2021-01-03) collects its rows from both years
        # and is fitted once on all of them.
        epochs, epoch_codes = _bin_timestamps_to_epochs(
            dt_index, self._resolve_freq
        )

        # Iterate epochs in chronological order so warnings come out in
        # time order, matching v1 behaviour.
        for position, epoch in enumerate(epochs):
            emask = (epoch_codes == position)
            epoch_paths = paths_arr[emask]
            epoch_convs = conv_arr[emask]

            self._epoch_paths[epoch] = epoch_paths
            self._epoch_conversions[epoch] = epoch_convs

            n_samples = len(epoch_paths)
            if n_samples < self.min_samples:
                warnings.warn(
                    f"Epoch {epoch} has only {n_samples} samples "
                    f"(min: {self.min_samples}). Results may be "
                    f"unreliable."
                )

            # Inherited Shapley math. super().fit handles its own
            # validate_data + conversion-label stripping + channel
            # extraction + value function + normalisation.
            result = super().fit(epoch_paths, epoch_convs)
            self.epoch_results[epoch] = result

        # Final epoch list is the wall-clock-sorted set of distinct epochs.
        self.epochs = epochs

        # MODIFIED (Risk 5): set self.channels eagerly from the union of
        # per-epoch attribution dicts. The eager union equals the global
        # static union by construction (every channel that appears
        # anywhere appears in at least one per-epoch result), so we do
        # NOT need to trigger the lazy global fit just to populate
        # self.channels. The lazy global will overwrite to the same
        # value when it eventually runs.
        self.channels = self._channels_from_epochs()
        self._clear_inherited_static_state()  # MODIFIED
        self._reporting_snapshot = self._build_reporting_snapshot(
            self._calendar_epoch_source()
        )

        return {
            "epoch_attributions": self.epoch_results,
            "attribution_matrix": self.get_attribution_matrix(),
            "epochs": self.epochs,
            "channels": list(self.channels),
        }

    def fit_static(
        self,
        paths: Union[Sequence[str], np.ndarray, pd.Series],
        conversions: Union[Sequence[int], np.ndarray, pd.Series],
    ) -> Dict[str, Any]:
        """Run the inherited static :class:`ShapleyAttribution` fit.

        Escape hatch for callers who explicitly want the static Shapley
        fit on a temporal instance. Bypasses the temporal pipeline
        entirely — no binning, no per-epoch loop, no ``epoch_results``
        population. Also pre-populates the lazy global cache so a
        subsequent ``self.global_attribution`` access returns the same
        result without re-running.
        """
        result = super().fit(paths, conversions)
        # Pre-populate the lazy cache so subsequent property access is free.
        # MODIFIED (get-well Unit 1): keep ``None`` rather than collapsing an
        # unavailable normalization into an empty mapping.
        self._cache_global_fit(result)
        return result

    def fit_recommended(
        self,
        paths: Union[Sequence[str], np.ndarray, pd.Series],
        conversions: Union[Sequence[int], np.ndarray, pd.Series],
        anchor_timestamps: Union[Sequence[Any], np.ndarray, pd.Series, pd.DatetimeIndex],
        *,
        recommender_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> "TemporalShapleyAttribution":
        """Recommend global epochs, then fit one independent Shapley model per epoch.

        The recommender receives the same path-level anchors used for final
        dispatch.  ``transition_drift`` is intentionally unavailable here:
        it is a Markov transition-matrix signal, not a Shapley signal.
        """
        params: Dict[str, Any] = dict(recommender_kwargs or {})
        if params.get("signal") == "transition_drift":
            raise ValueError(
                "TemporalShapleyAttribution.fit_recommended does not support "
                "signal='transition_drift'; use conversion_rate or TemporalMarkovAttribution."
            )
        params.setdefault("min_samples", self.min_samples)
        params.setdefault("separator", self.separator)
        params.setdefault("conversion_labels", tuple(self.conversion_labels))
        recommendation = recommend_epochs(paths, conversions, anchor_timestamps, **params)
        labels = assign_paths_to_epochs(anchor_timestamps, recommendation.boundaries)
        # pd.Interval labels are intentionally non-calendar explicit epochs;
        # fit_by_epoch keeps each complete journey intact in one static fit.
        self.fit_by_epoch(paths, labels, conversions)
        snapshot = self._build_reporting_snapshot(
            f"recommended:{self._recommendation_signal_name(recommendation.signal_name)}",
            epoch_labels=list(recommendation.epoch_metrics["epoch_label"].astype(str)),
            boundaries=recommendation.boundaries,
            sample_floor=recommendation.min_samples,
        )
        snapshot = replace(
            snapshot,
            epoch_metrics=recommendation.epoch_metrics.copy(deep=True),
        )
        self._reporting_snapshot = snapshot
        self.epoch_recommendation = replace(
            recommendation,
            reporting_snapshot=snapshot,
        )
        return self

    def fit_by_epoch(
        self,
        paths: Union[Sequence[str], np.ndarray, pd.Series],
        epochs: Union[Sequence[Any], np.ndarray, pd.Series],
        conversions: Union[Sequence[int], np.ndarray, pd.Series],
    ) -> Dict[str, Any]:
        """Fit using pre-assigned epoch labels.

        Unlike :meth:`fit` which derives epochs from timestamps using a
        frequency, this method accepts explicit epoch labels.

        .. note::
           Mixed-granularity per-year freqs (``Mapping[int, str]``) are
           **not** supported here — that path goes through :meth:`fit`
           where the year-bucketed binning logic lives. If
           ``self.freq`` is a mapping, this method raises ``TypeError``.
        """
        # MODIFIED (Risk 1): block 'split' here too.
        self._check_epoch_assignment_supported()

        # Shared preparation path: the supplied outcomes are validated
        # before coercion, ahead of any per-epoch dispatch.
        paths_arr, conv_arr = prepare_journey_data(paths, conversions)

        if isinstance(epochs, (np.ndarray, pd.Series)):
            epochs_arr = np.asarray(epochs).flatten()
        else:
            epochs_arr = np.array(list(epochs))

        # Calendar mappings only apply to ordinary caller-supplied labels.
        # Explicit recommendation intervals are already complete epoch
        # identities and therefore do not need a mapping-frequency cast.
        is_interval_labels = all(isinstance(value, pd.Interval) for value in epochs_arr)
        if self._is_mapping_freq() and not is_interval_labels:
            raise TypeError(
                "fit_by_epoch does not support a Mapping[int, str] "
                "freq (mixed per-year granularity). Pre-assigned epoch "
                "labels are inherently single-freq. Use fit(paths, "
                "timestamps, conversions) for mixed granularity, or "
                "construct the model with a single str freq for "
                "fit_by_epoch."
            )

        if len(epochs_arr) != len(paths_arr):
            raise ValueError(
                f"epochs length ({len(epochs_arr)}) does not match "
                f"paths length ({len(paths_arr)})"
            )

        try:
            period_epochs = pd.to_datetime(epochs_arr).to_period(self.freq)
        except (ValueError, TypeError):
            period_epochs = epochs_arr

        # Explicit non-temporal labels are reporting identities, so retain
        # their caller-supplied first-seen order rather than lexically sorting
        # them through ``np.unique``.
        unique_epochs = list(pd.unique(period_epochs))
        # MODIFIED: reset BEFORE assigning self.epochs (reset now clears the
        # epoch list), and run the per-epoch loop exception-safe.
        self._reset_state()
        if len(unique_epochs) and isinstance(unique_epochs[0], pd.Period):
            self.epochs = sorted(unique_epochs, key=_get_period_key)
        elif len(unique_epochs) and all(
            isinstance(value, pd.Interval) for value in unique_epochs
        ):
            # MODIFIED: Interval labels are temporal identities (the
            # fit_recommended path). Order them chronologically so the
            # fitted epoch order — which now drives every matrix accessor,
            # the reporting snapshot, and to_dataframes() — is stable
            # regardless of the caller's path order.
            self.epochs = sorted(
                unique_epochs, key=lambda interval: (interval.left, interval.right)
            )
        else:
            self.epochs = unique_epochs

        try:
            return self._fit_labelled(period_epochs, paths_arr, conv_arr)
        except BaseException:
            self._reset_state()
            raise

    def _fit_labelled(
        self,
        period_epochs: Any,
        paths_arr: np.ndarray,
        conv_arr: np.ndarray,
    ) -> Dict[str, Any]:
        """Pre-assigned-label dispatch (exception-safe body)."""
        for epoch in self.epochs:
            mask = period_epochs == epoch
            epoch_paths = paths_arr[mask]
            epoch_convs = conv_arr[mask]

            self._epoch_paths[epoch] = epoch_paths
            self._epoch_conversions[epoch] = epoch_convs

            n_samples = len(epoch_paths)
            if n_samples < self.min_samples:
                warnings.warn(
                    f"Epoch {epoch} has only {n_samples} samples "
                    f"(min: {self.min_samples})."
                )

            result = super().fit(epoch_paths, epoch_convs)
            self.epoch_results[epoch] = result

        self.channels = self._channels_from_epochs()
        self._clear_inherited_static_state()  # MODIFIED
        self._reporting_snapshot = self._build_reporting_snapshot("explicit_labels")

        return {
            "epoch_attributions": self.epoch_results,
            "attribution_matrix": self.get_attribution_matrix(),
            "epochs": self.epochs,
            "channels": list(self.channels),
        }

    # ------------------------------------------------------------------ #
    # Risk 5: lazy global fit                                             #
    # ------------------------------------------------------------------ #
    @property
    def global_attribution(self) -> Optional[Dict[str, float]]:
        """Static Shapley attribution on the full (across-epoch) path set.

        .. warning::
           This is the static Shapley fit on the full path set, not a
           mean of per-epoch attributions. Under the legacy ``'exact'``
           game, channels that enter or exit the data across epochs make
           it diverge from :meth:`get_volume_weighted_attribution`. Under
           the default ``'cumulative'`` game the Shapley values are
           additive over disjoint sets of paths, so the two views agree.

        Lazy — the inherited ``super().fit`` call only runs the first
        time this property (or anything that depends on it) is accessed,
        so per-epoch-only callers do not pay for the N+1th refit.

        Returns
        -------
        dict[str, float] or None
            MODIFIED (get-well Unit 1): ``None`` when the global fit could
            not produce a valid share vector.  Check
            :attr:`global_normalization_status` before indexing.  The raw
            signed values remain on :attr:`global_shapley_values`.
        """
        self._compute_global_fit_lazy()
        cache = self._global_attribution_cache
        # Return a copy so callers cannot mutate the cache.
        return None if cache is None else dict(cache)

    @property
    def global_shapley_values(self) -> Dict[str, float]:
        """Raw signed Shapley values from the global static fit (lazy).

        MODIFIED (get-well Unit 1): always populated after the global fit,
        including when :attr:`global_attribution` is ``None``.  These are
        diagnostics on the game's own scale — not shares.
        """
        self._compute_global_fit_lazy()
        return dict(self._global_shapley_values_cache)

    @property
    def global_normalization_status(self) -> str:
        """MODIFIED (get-well Unit 1): ``'available'`` or ``'unavailable'``."""
        self._compute_global_fit_lazy()
        return self._global_normalization.get("normalization_status", "available")

    @property
    def global_normalization_reason(self) -> Optional[str]:
        """MODIFIED (get-well Unit 1): why the global shares are unavailable."""
        self._compute_global_fit_lazy()
        return self._global_normalization.get("normalization_reason")

    @property
    def attribution(self) -> Optional[Dict[str, float]]:
        """Backwards-compat alias for ``global_attribution`` (lazy)."""
        return self.global_attribution

    def _cache_global_fit(self, result: Mapping[str, Any]) -> None:
        """Store the global static fit, preserving its normalization state.

        MODIFIED (get-well Unit 1): an unavailable normalization is cached
        as ``None``, not as ``{}``, so callers can tell "no valid shares"
        apart from "not fitted yet".
        """
        attribution = result.get("attribution")
        self._global_attribution_cache = (
            None if attribution is None else dict(attribution)
        )
        self._global_shapley_values_cache = dict(result.get("shapley_values", {}))
        self._global_normalization = {
            key: result[key]
            for key in (
                "normalization_status",
                "normalization_reason",
                "normalization_denominator",
                "raw_sum",
                "raw_magnitude",
                "cancellation_ratio",
            )
            if key in result
        }
        self._global_fit_done = True

    def _compute_global_fit_lazy(self) -> None:
        """Run the inherited static fit on the union of per-epoch arrays.

        Idempotent: the first call runs ``super().fit`` once and sets
        ``self._global_fit_done``; subsequent calls short-circuit.
        """
        if self._global_fit_done:
            return
        if not self.epoch_results:
            # Nothing has been fit yet. Don't crash — leave the cache
            # empty and emit a warning so the caller knows they accessed
            # the property before fitting.
            warnings.warn(
                "global_attribution accessed before fit() / fit_by_epoch() "
                "was called; returning an empty dict."
            )
            return
        # MODIFIED: guard the empty-array case (possible after a failed
        # refit) — np.concatenate([]) raises ValueError from a property.
        path_arrays = [
            self._epoch_paths[ep] for ep in self.epochs if ep in self._epoch_paths
        ]
        conv_arrays = [
            self._epoch_conversions[ep]
            for ep in self.epochs
            if ep in self._epoch_conversions
        ]
        if not path_arrays:
            self._global_fit_done = True
            return
        all_paths = np.concatenate(path_arrays)
        all_convs = np.concatenate(conv_arrays)
        if len(all_paths) == 0:
            self._global_fit_done = True
            return
        result = super().fit(all_paths, all_convs)
        self._cache_global_fit(result)  # MODIFIED (get-well Unit 1)

    def get_attribution(self) -> Dict[str, float]:
        """Inherited helper — returns the **global** static attribution.

        Overridden so that on a temporal instance it triggers the lazy
        global fit and returns the global static answer (matching the
        pre-Risk-5 contract that ``get_attribution()`` returned the
        last-fit aggregate, not the last per-epoch slice).
        """
        return self.global_attribution

    def summary(
        self,
        output_format: Literal["summary", "report"] = "summary",
    ) -> str:
        """Render deterministic plain-text diagnostics from stored fit state."""

        if self._reporting_snapshot is None:
            raise RuntimeError("summary() requires a fitted temporal model")
        return render_temporal_report(self._reporting_snapshot, output_format)

    # ------------------------------------------------------------------ #
    # Decoupled DataFrame projection (design doc section 6.5)             #
    # ------------------------------------------------------------------ #
    # MODIFIED: standard programmatic export surface per
    # GLOBAL_SIGNAL_EPOCH_DESIGN.md sections 1.6 / 6.5 / 8.9 and
    # temporal_shapley.md "Standard DataFrame projection".
    def to_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Return the fitted temporal result as flat pandas DataFrames.

        Thin façade over the existing ``get_*`` accessors called with their
        default arguments. It does not fit or refit epochs, trigger the lazy
        ``global_attribution`` fit, enumerate a new coalition lattice,
        render or parse ``summary()`` text, or mutate stored state. The
        potentially exponential per-epoch value-function dictionaries remain
        available through ``epoch_results``; they are not materialized here
        because no dedicated flat ``get_*`` accessor exists for them.

        Returns
        -------
        dict[str, pd.DataFrame]
            Deterministic keys, in this order:

            - ``'attribution'`` — ``epoch``, ``channel``, ``attribution``
              (from ``get_attribution_matrix(include_global=False)``).
            - ``'shapley_values'`` — ``epoch``, ``channel``,
              ``shapley_value`` (raw, signed; from
              ``get_shapley_values_matrix()``).
            - ``'epoch_summary'`` — explicit ``epoch`` column plus the
              columns returned by ``get_epoch_summary()``.
            - ``'channel_epoch_sample_sizes'`` — explicit ``epoch`` and
              ``channel`` columns plus the accessor's diagnostic columns.

            Every frame is a flat copy with a fresh ``RangeIndex`` and no
            ``MultiIndex`` or nested values. Epoch and channel identities
            are rendered as ``str`` columns so the tables join against each
            other and serialize to Parquet/SQL (mixed-freq ``Period``
            epochs have no portable native dtype). Rows preserve fitted
            epoch order, then the accessor's deterministic channel order.
            Channels absent from an epoch appear with NaN metric values,
            matching the accessor matrices, so pivoting a metric frame
            reproduces the accessor output exactly.

        Raises
        ------
        RuntimeError
            If no successful temporal fit has been run.
        """
        if not self.epoch_results:
            raise RuntimeError("to_dataframes() requires a fitted temporal model")

        tables: Dict[str, pd.DataFrame] = {}
        tables["attribution"] = _matrix_to_long_frame(
            self.get_attribution_matrix(), "attribution"
        )
        tables["shapley_values"] = _matrix_to_long_frame(
            self.get_shapley_values_matrix(), "shapley_value"
        )
        tables["epoch_summary"] = self.get_epoch_summary().reset_index()
        tables["channel_epoch_sample_sizes"] = _sample_sizes_to_long_frame(
            self.get_channel_epoch_sample_sizes()
        )
        return tables

    def _reset_state(self) -> None:
        """Reset all temporal + lazy-global state. Called at the top of fit."""
        # MODIFIED: also clear self.epochs — previously a mid-fit exception
        # left the OLD epoch list paired with PARTIAL new per-epoch data,
        # and the lazy global fit then silently computed on the mixture
        # (or crashed on np.concatenate([])).
        self.epochs = []
        # MODIFIED: clear inherited static-fit leftovers so introspection
        # cannot read one epoch's value function / attribution as if it
        # were the temporal model's.
        self.value_function = {}
        self._last_attribution = {}
        self._static_reporting_snapshot = None
        self._export_snapshot = None
        self.epoch_results = {}
        self._epoch_paths = {}
        self._epoch_conversions = {}
        self._global_fit_done = False
        self._global_attribution_cache = {}
        self._global_shapley_values_cache = {}
        self._global_normalization = {}  # MODIFIED (get-well Unit 1)
        self._reporting_snapshot = None
        self.epoch_recommendation = None

    def _clear_inherited_static_state(self) -> None:
        """MODIFIED: after the per-epoch loop the inherited static attributes
        (value_function, _last_attribution) hold the LAST epoch's fit — a
        misleading introspection surface. Clear them; the lazy global fit
        repopulates them with full-data values on first access."""
        self.value_function = {}
        self._last_attribution = {}
        self._static_reporting_snapshot = None
        self._export_snapshot = None

    def _calendar_epoch_source(self) -> str:
        return "mixed_calendar" if self._is_mapping_freq() else f"calendar:{self.freq}"

    @staticmethod
    def _recommendation_signal_name(signal_name: Any) -> str:
        if isinstance(signal_name, str):
            return signal_name
        return getattr(signal_name, "__name__", "custom_signal")

    def _build_reporting_snapshot(
        self,
        source: str,
        *,
        epoch_labels: Optional[Sequence[str]] = None,
        boundaries: Optional[pd.DatetimeIndex] = None,
        sample_floor: Optional[int] = None,
    ):
        # MODIFIED: distilled per-epoch top coalitions — bounded rows, never
        # the 2^C coalition space (full values stay programmatic via
        # get_shapley_values_matrix() / epoch_results / to_dataframes()).
        labels = (
            list(epoch_labels)
            if epoch_labels is not None
            else [str(epoch) for epoch in self.epochs]
        )
        details = build_shapley_epoch_coalition_section(
            labels,
            [self.epoch_results.get(epoch, {}) for epoch in self.epochs],
        )
        return build_fitted_snapshot(
            model_name=self.__class__.__name__,
            epoch_source=source,
            epochs=self.epochs,
            epoch_paths=self._epoch_paths,
            epoch_conversions=self._epoch_conversions,
            epoch_attributions={
                epoch: _epoch_attribution(result)
                for epoch, result in self.epoch_results.items()
            },
            sample_floors=self.min_samples,
            sample_floor=self.min_samples if sample_floor is None else sample_floor,
            epoch_labels=epoch_labels,
            boundaries=boundaries,
            model_details=details,  # MODIFIED
        )

    def _channels_from_epochs(self) -> List[str]:
        """Sorted union of channels across all per-epoch results.

        The raw ``shapley_values`` keys are included because ``attribution``
        is ``None`` when an epoch's normalization is unavailable; a channel
        observed only in such epochs must still be enumerated.
        """
        all_channels: set = set()
        for result in self.epoch_results.values():
            all_channels.update(_epoch_attribution(result))
            all_channels.update(result.get("shapley_values", {}))
        return sorted(all_channels)

    # ------------------------------------------------------------------ #
    # Stitching helpers                                                   #
    # ------------------------------------------------------------------ #
    def get_trend(self, channel: str) -> pd.Series:
        """Get the attribution trend for a specific channel over time."""
        if not self.epoch_results:
            raise ValueError("Model has not been fitted. Call fit() first.")

        data = {}
        for epoch in self.epochs:
            result = self.epoch_results.get(epoch, {})
            epoch_attr = _epoch_attribution(result)
            data[epoch] = epoch_attr.get(channel, np.nan)

        return pd.Series(data, name=channel)

    def get_all_trends(self) -> pd.DataFrame:
        """Get attribution trends for all channels."""
        return self.get_attribution_matrix()

    def get_attribution_matrix(
        self,
        include_global: bool = False,
    ) -> pd.DataFrame:
        """Get the channels × epochs attribution matrix.

        Parameters
        ----------
        include_global : bool, default=False
            If True, append two additional rows:
              * ``'Global'`` — the static-fit attribution on the full path
                set (triggers the lazy global fit).
              * ``'VolumeWeighted'`` — the per-epoch attribution
                averaged with weights equal to per-epoch conversion
                totals.

            .. warning::
               Under the legacy ``'exact'`` game these two rows are
               *different* in general. Channels that enter or exit the
               data across epochs pull them apart, and the gap is itself
               a useful diagnostic. Surfacing both side-by-side is
               intentional (Risk 3 in the v1 punch list). Under the
               default ``'cumulative'`` game the two rows agree.

        Returns
        -------
        pd.DataFrame
            Attribution shares (rows=epochs, columns=channels).
        """
        if not self.epoch_results:
            raise ValueError("Model has not been fitted. Call fit() first.")

        data = {}
        for epoch in self.epochs:
            result = self.epoch_results.get(epoch, {})
            epoch_attr = _epoch_attribution(result)
            data[epoch] = epoch_attr

        matrix = pd.DataFrame(data).T

        # Ensure all per-epoch-union channels are present.
        for ch in self.channels:
            if ch not in matrix.columns:
                matrix[ch] = np.nan

        matrix = matrix.reindex(sorted(matrix.columns), axis=1)
        # MODIFIED (v2 PR 2): mixed-freq Period indices break the native
        # sort_index — fall back to a start_time-keyed sort when needed.
        matrix = _align_index_to_epochs(matrix, self.epochs)  # MODIFIED: fitted order, not lexical

        if include_global:
            # Global static (triggers lazy fit).
            global_series = pd.Series(
                self.global_attribution, name="Global"
            ).reindex(matrix.columns)
            # Volume-weighted per-epoch mean — does NOT trigger the lazy fit.
            vw_series = pd.Series(
                self.get_volume_weighted_attribution(), name="VolumeWeighted"
            ).reindex(matrix.columns)
            matrix = pd.concat(
                [matrix, global_series.to_frame().T, vw_series.to_frame().T],
                axis=0,
            )

        return matrix

    def get_shapley_values_matrix(self) -> pd.DataFrame:
        """Get the channels × epochs raw Shapley values matrix."""
        if not self.epoch_results:
            raise ValueError("Model has not been fitted. Call fit() first.")

        data = {}
        for epoch in self.epochs:
            result = self.epoch_results.get(epoch, {})
            shapley_vals = result.get("shapley_values", {})
            data[epoch] = shapley_vals

        matrix = pd.DataFrame(data).T

        for ch in self.channels:
            if ch not in matrix.columns:
                matrix[ch] = np.nan

        matrix = matrix.reindex(sorted(matrix.columns), axis=1)
        # MODIFIED (v2 PR 2): see get_attribution_matrix for rationale.
        matrix = _align_index_to_epochs(matrix, self.epochs)  # MODIFIED: fitted order, not lexical

        return matrix

    # ------------------------------------------------------------------ #
    # Risk 3: volume-weighted reconciliation helper                       #
    # ------------------------------------------------------------------ #
    def get_volume_weighted_attribution(self) -> Dict[str, float]:
        """Volume-weighted mean of per-epoch attribution shares.

        Weights are per-epoch conversion totals. NaN cells (channel
        absent from that epoch) are treated as a zero contribution; the
        result is renormalised to sum to 1 across the surviving
        channels. Designed to sit alongside ``self.global_attribution``
        so the divergence between the static-fit-on-all-data answer and
        the per-epoch column-mean is visible explicitly (Risk 3).

        Returns
        -------
        dict[str, float]
            Channel → volume-weighted attribution share. Sums to 1.
        """
        if not self.epoch_results:
            raise ValueError("Model has not been fitted. Call fit() first.")

        matrix = self.get_attribution_matrix()  # epochs × channels
        weights = pd.Series(
            {
                epoch: float(self._epoch_conversions.get(epoch, np.array([])).sum())
                for epoch in self.epochs
            }
        )
        # Drop epochs with zero conversions OR all-NaN rows (failed/empty fit).
        all_nan_mask = matrix.isna().all(axis=1)
        usable = (weights > 0) & (~all_nan_mask)
        if not usable.any():
            return {ch: 0.0 for ch in self.channels}

        m = matrix.loc[usable].fillna(0.0)
        w = weights.loc[usable].values
        total_w = w.sum()
        if total_w <= 0:
            return {ch: 0.0 for ch in self.channels}
        weighted = (m.values.T @ w) / total_w  # one entry per channel
        out = {ch: float(v) for ch, v in zip(m.columns, weighted)}
        # Renormalise to 1 (NaN cells filled to 0 may have lost mass).
        s = sum(out.values())
        if s > 0:
            out = {k: v / s for k, v in out.items()}
        return out

    # ------------------------------------------------------------------ #
    # Risk 4: per-channel-per-epoch sample-size diagnostic                #
    # ------------------------------------------------------------------ #
    def get_channel_epoch_sample_sizes(
        self,
        min_per_channel_paths: int = 10,
    ) -> pd.DataFrame:
        """Per-channel-per-epoch sample-size diagnostic.

        For each (channel, epoch) cell, report how many paths in the
        epoch contained the channel and how many of those paths
        converted. Cells below the ``min_per_channel_paths`` floor are
        flagged ``thin=True``. A high count of thin cells in an epoch
        is a warning sign that per-epoch Shapley values for those
        channels are dominated by sampling noise.

        Parameters
        ----------
        min_per_channel_paths : int, default=10
            Path-count floor below which a (channel, epoch) cell is
            flagged thin.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with a 2-level index (channel, epoch)
            and columns ``n_paths_containing_channel``,
            ``n_conversions_on_paths_containing_channel``, ``thin``.
        """
        if not self.epoch_results:
            raise ValueError("Model has not been fitted. Call fit() first.")

        rows: List[Dict[str, Any]] = []
        for epoch in self.epochs:
            paths = self._epoch_paths.get(epoch, np.array([]))
            convs = self._epoch_conversions.get(epoch, np.array([]))
            if len(paths) == 0:
                continue

            # Per-path channel sets (with conversion-label tokens stripped
            # for parity with what the Shapley fit actually saw).
            per_path_channels: List[set] = []
            for p in paths:
                tokens = set(split_path(str(p), self.separator))
                if self.conversion_labels:
                    tokens -= self.conversion_labels
                per_path_channels.append(tokens)

            # Channel union for this epoch.
            epoch_channels: set = set().union(*per_path_channels) if per_path_channels else set()
            for ch in sorted(epoch_channels):
                mask = np.array(
                    [ch in chs for chs in per_path_channels], dtype=bool
                )
                n_paths = int(mask.sum())
                n_convs = int(convs[mask].sum()) if mask.any() else 0
                rows.append(
                    {
                        "channel": ch,
                        # MODIFIED (v2 PR 2): stringify epoch to keep the
                        # MultiIndex sortable under mixed-freq Periods.
                        # Pandas factorize/sort on object-dtype Period
                        # arrays raises IncompatibleFrequency when
                        # freqs differ across rows. ``get_epoch_summary``
                        # already uses str(epoch) for its index, so this
                        # is consistent with existing diagnostic output.
                        "epoch": str(epoch),
                        "n_paths_containing_channel": n_paths,
                        "n_conversions_on_paths_containing_channel": n_convs,
                        "thin": n_paths < min_per_channel_paths,
                    }
                )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "n_paths_containing_channel",
                    "n_conversions_on_paths_containing_channel",
                    "thin",
                ]
            )
        return pd.DataFrame(rows).set_index(["channel", "epoch"])

    def get_epoch_summary(
        self,
        min_per_channel_paths: int = 10,
    ) -> pd.DataFrame:
        """Get a summary of each epoch's data quality and total conversions.

        Columns: ``n_paths``, ``n_conversions``, ``n_channels``,
        ``total_shapley``, ``sufficient_data``, and (Risk 4)
        ``n_thin_channels`` — count of channels in this epoch with
        fewer than ``min_per_channel_paths`` containing-paths.
        """
        if not self.epoch_results:
            raise ValueError("Model has not been fitted. Call fit() first.")

        # Compute the channel-epoch diagnostic once and aggregate by epoch.
        per_channel = self.get_channel_epoch_sample_sizes(
            min_per_channel_paths=min_per_channel_paths
        )
        if per_channel.empty:
            thin_per_epoch: Dict[Any, int] = {}
        else:
            thin_per_epoch = (
                per_channel.groupby(level="epoch")["thin"].sum().astype(int).to_dict()
            )

        rows = []
        for epoch in self.epochs:
            paths = self._epoch_paths.get(epoch, np.array([]))
            convs = self._epoch_conversions.get(epoch, np.array([]))
            result = self.epoch_results.get(epoch, {})

            n_channels = len(result.get("shapley_values", {}))
            shapley_vals = result.get("shapley_values", {})
            total_shapley = sum(abs(v) for v in shapley_vals.values())

            rows.append(
                {
                    "epoch": str(epoch),
                    "n_paths": len(paths),
                    "n_conversions": int(convs.sum()) if len(convs) else 0,
                    "n_channels": n_channels,
                    "total_shapley": total_shapley,
                    "sufficient_data": len(paths) >= self.min_samples,
                    # MODIFIED (v2 PR 2): match str-keyed diagnostic.
                    "n_thin_channels": int(thin_per_epoch.get(str(epoch), 0)),
                }
            )

        return pd.DataFrame(rows).set_index("epoch")

    def detect_threshold_crossings(
        self,
        channel: str,
        threshold: float = 0.1,
        skip_cross_granularity: bool = True,
    ) -> List[Tuple[Any, Any, float]]:
        """Detect adjacent epochs where a channel's change exceeds a threshold.

        Parameters
        ----------
        channel : str
            Channel whose trend is examined.
        threshold : float, default=0.1
            Absolute change threshold; pairs with ``|delta| > threshold``
            are reported.
        skip_cross_granularity : bool, default=True
            **v2 PR 2**: under a mixed-granularity (per-year freq) fit
            the epoch list contains Periods of multiple freqs. A
            boundary like ``2024Q4 -> 2025-01`` compares a 3-month
            aggregate to a 1-month aggregate; the implied step is a
            mechanical bin-width artifact, not a behaviour change.
            ``True`` (default) suppresses change reports across such
            boundaries. Set ``False`` to surface them anyway.

        Returns
        -------
        list of (prev_epoch, curr_epoch, delta)
        """
        trend = self.get_trend(channel)
        changes = []

        epochs_list = list(trend.index)
        for i in range(1, len(epochs_list)):
            prev_epoch = epochs_list[i - 1]
            curr_epoch = epochs_list[i]
            prev_val = trend[prev_epoch]
            curr_val = trend[curr_epoch]
            if pd.isna(prev_val) or pd.isna(curr_val):
                continue
            # MODIFIED (v2 PR 2): skip cross-freq boundaries by default.
            if skip_cross_granularity and _periods_have_different_freq(
                prev_epoch, curr_epoch
            ):
                continue
            change = curr_val - prev_val
            if abs(change) > threshold:
                changes.append((prev_epoch, curr_epoch, change))

        return changes

    def detect_significant_changes(
        self,
        channel: str,
        threshold: float = 0.1,
        skip_cross_granularity: bool = True,
    ) -> List[Tuple[Any, Any, float]]:
        """Deprecated alias for :meth:`detect_threshold_crossings`."""
        warnings.warn(
            "detect_significant_changes() performs descriptive threshold "
            "comparison, not statistical significance testing; use "
            "detect_threshold_crossings() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.detect_threshold_crossings(
            channel=channel,
            threshold=threshold,
            skip_cross_granularity=skip_cross_granularity,
        )

    def __repr__(self) -> str:
        # MODIFIED (v2 PR 2): render mapping freqs as a sorted dict so
        # the repr is deterministic and readable for mixed-granularity
        # configurations, e.g. ``freq={2024: 'Q', 2025: 'M'}``.
        if self._is_mapping_freq():
            freq_items = ", ".join(
                f"{k}: {v!r}" for k, v in sorted(self.freq.items())
            )
            freq_repr = "{" + freq_items + "}"
        else:
            freq_repr = repr(self.freq)
        return (
            f"TemporalShapleyAttribution("
            f"freq={freq_repr}, "
            f"min_samples={self.min_samples}, "
            f"epoch_assignment={self.epoch_assignment!r}, "
            f"n_epochs={len(self.epochs)})"
        )


__all__ = ["TemporalShapleyAttribution"]
