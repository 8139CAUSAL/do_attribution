"""
do_attribution.compare
---------------------------

Thin wrapper that fits both the :class:`MarkovAttribution` and
:class:`ShapleyAttribution` models on the same data and returns a
side-by-side comparison so callers can inspect differences in the
attribution distributions.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .base import BaseAttribution
from .markov import MarkovAttribution
from .shapley import ShapleyAttribution
from .utils import DEFAULT_SEPARATOR, extract_channels


class CompareAttribution:
    """Run Markov and Shapley attribution on the same data and compare.

    Parameters
    ----------
    markov_order : int, default=1
        Order of the Markov chain used internally.
    markov_removal_strategy : {'fail', 'detour'}, default='fail'
        Removal strategy forwarded to :class:`MarkovAttribution`.
    separator : str, default=' > '
        Separator used to split path strings.
    extra_models : dict[str, BaseAttribution], optional
        Additional named attribution models to include in the comparison.
        Each model must implement the :class:`BaseAttribution` API.
    conversion_labels : iterable of str, optional
        Tokens that represent the conversion endpoint (e.g. ``"Purchase"``)
        rather than a real channel.  Forwarded to both sub-models so they
        strip the endpoint tokens before any attribution math.  See
        :class:`do_attribution.base.BaseAttribution` for full semantics.
    shapley_characteristic_function : {'cumulative', 'exact'}, default='cumulative'
        Coalition game forwarded to :class:`ShapleyAttribution`.
    """

    def __init__(
        self,
        markov_order: int = 1,
        markov_removal_strategy: str = "fail",
        separator: str = DEFAULT_SEPARATOR,
        extra_models: Optional[Dict[str, BaseAttribution]] = None,
        conversion_labels: Optional[Iterable[str]] = None,  # MODIFIED: P0.
        shapley_characteristic_function: str = "cumulative",
    ) -> None:
        self.separator = separator
        # MODIFIED (P0): store conversion_labels so both extract_channels()
        # calls and forwarded sub-models exclude the endpoint tokens.  Keep
        # a list copy (not a frozenset) so we can re-pass it unchanged.
        self.conversion_labels: Optional[List[str]] = (
            list(conversion_labels) if conversion_labels is not None else None
        )
        self.markov = MarkovAttribution(
            order=markov_order,
            separator=separator,
            removal_strategy=markov_removal_strategy,
            conversion_labels=conversion_labels,  # MODIFIED: P0 forward.
        )
        self.shapley = ShapleyAttribution(
            separator=separator,
            conversion_labels=conversion_labels,  # MODIFIED: P0 forward.
            characteristic_function=shapley_characteristic_function,
        )
        self.extra_models: Dict[str, BaseAttribution] = dict(extra_models or {})

        self._results: Dict[str, Dict[str, Any]] = {}
        self.channels: List[str] = []

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def fit(
        self,
        paths: Union[Sequence[str], np.ndarray, pd.Series],
        conversions: Union[Sequence[int], np.ndarray, pd.Series],
    ) -> Dict[str, Dict[str, Any]]:
        """Fit every model and return their individual result dicts.

        Returns
        -------
        dict
            Keys are the model names (``'markov'``, ``'shapley'``, plus any
            keys from ``extra_models``). Values are the raw result dicts
            returned by each model's own ``fit``.
        """
        # MODIFIED (P0): pass conversion_labels to extract_channels so the
        # endpoint tokens never show up in self.channels / attribution_frame.
        self.channels = extract_channels(
            paths, self.separator, conversion_labels=self.conversion_labels
        )
        self._results = {
            "markov": self.markov.fit(paths, conversions),
            "shapley": self.shapley.fit(paths, conversions),
        }
        for name, model in self.extra_models.items():
            self._results[name] = model.fit(paths, conversions)
        return self._results

    def attribution_frame(self) -> pd.DataFrame:
        """Return a DataFrame with one row per channel and one column per model.

        Columns are the normalised attribution shares (sum to 1 within each
        column).  Channels are sorted alphabetically.

        MODIFIED (get-well Unit 1): a model whose normalization is
        unavailable contributes an all-``NaN`` column rather than a column
        of zeros.  Zeros would read as genuine "this channel got no credit"
        shares and would sum to 0 while still looking like a share vector.
        ``NaN`` marks the absence of a valid share vector and propagates
        visibly through :meth:`difference_frame`.  Raw signed diagnostics
        stay on each model's own fit result.
        """
        if not self._results:
            raise ValueError("No results available. Call fit() first.")

        index = sorted(self.channels)
        frame = pd.DataFrame(index=index)
        for name, res in self._results.items():
            attribution = res.get("attribution")
            if attribution is None:
                frame[name] = pd.Series(np.nan, index=index, dtype=float)
                continue
            # Channels absent from a particular model fill as 0.
            frame[name] = (
                pd.Series(attribution, dtype=float)
                .reindex(index)
                .fillna(0.0)
            )
        return frame.sort_index()

    def unavailable_models(self) -> Dict[str, str]:
        """Return ``{model_name: normalization_reason}`` for refused models.

        MODIFIED (get-well Unit 1): lets callers tell "no valid share
        vector" apart from "all shares happen to be zero" without inspecting
        each model's result dict.
        """
        if not self._results:
            raise ValueError("No results available. Call fit() first.")
        return {
            name: res.get("normalization_reason", "unavailable")
            for name, res in self._results.items()
            if res.get("attribution") is None
        }

    def difference_frame(self, reference: str = "markov") -> pd.DataFrame:
        """Return per-channel differences vs a reference model's share."""
        frame = self.attribution_frame()
        if reference not in frame.columns:
            raise ValueError(
                f"reference={reference!r} not in results {list(frame.columns)}"
            )
        diff = frame.subtract(frame[reference], axis=0)
        diff.columns = [f"{c}_minus_{reference}" for c in diff.columns]
        return diff.drop(columns=[f"{reference}_minus_{reference}"])

    def get_results(self) -> Dict[str, Dict[str, Any]]:
        """Return the raw results dict from the last :meth:`fit`."""
        if not self._results:
            raise ValueError("No results available. Call fit() first.")
        return self._results


__all__ = ["CompareAttribution"]
