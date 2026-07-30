"""
do_attribution.base
------------------------

Abstract base class shared by every attribution model in the library.
Guarantees a single, consistent ``fit`` / ``get_attribution`` API so that
code can switch between Markov-chain and Shapley-value attribution without
any changes other than the constructor name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .utils import DEFAULT_SEPARATOR, _normalize_conversion_labels  # MODIFIED: P0 imports
from .reporting import (
    TemporalReportingSnapshot,
    build_epoch_metrics,
    render_temporal_report,
)


class BaseAttribution(ABC):
    """Common interface for all attribution models.

    Subclasses must implement :meth:`fit`.  They should populate
    ``self.channels`` and ``self._last_attribution`` so that
    :meth:`get_attribution` and :meth:`get_channels` work uniformly.

    Parameters
    ----------
    separator : str, default=' > '
        Delimiter used to split path strings into channels.
    conversion_labels : iterable of str, optional
        Tokens in the path strings that represent the conversion endpoint
        (e.g. ``"Purchase"``) rather than a real marketing channel.  When
        supplied, these tokens are stripped from every path *before* channel
        extraction, state creation, and attribution math, so they never
        appear in ``attribution``, ``removal_effects``, the transition
        matrix, Shapley values, or any downstream viz.  Case-sensitive exact
        match.  Defaults to ``None`` (every token is treated as a channel,
        matching the pre-P0 behaviour).  Raises ``ValueError`` if any label
        collides with :data:`~do_attribution.utils.SPECIAL_TOKENS`.
    """

    def __init__(
        self,
        separator: str = DEFAULT_SEPARATOR,
        conversion_labels: Optional[Iterable[str]] = None,  # MODIFIED: P0 parameter.
    ):
        self.separator: str = separator
        # MODIFIED: normalise + validate once at construction time so every
        # downstream operation can rely on a frozenset[str] (possibly empty).
        self.conversion_labels: "frozenset[str]" = _normalize_conversion_labels(
            conversion_labels
        )
        self.channels: List[str] = []
        # MODIFIED (get-well Unit 1): ``None`` means a fit completed but
        # normalized attribution was unavailable; ``{}`` means no fit yet.
        self._last_attribution: Optional[Dict[str, float]] = {}
        self._static_reporting_snapshot: TemporalReportingSnapshot | None = None

    # ------------------------------------------------------------------ #
    # Required API                                                        #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def fit(
        self,
        paths: Union[Sequence[str], np.ndarray, pd.Series],
        conversions: Union[Sequence[int], np.ndarray, pd.Series],
    ) -> Dict[str, Any]:
        """Fit the model and return the results dict.

        Every subclass must at minimum include an ``'attribution'`` key
        whose value is a ``dict[str, float]`` normalised to sum to 1.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Convenience helpers (concrete)                                      #
    # ------------------------------------------------------------------ #
    def get_attribution(self) -> Dict[str, float]:
        """Return the attribution dict from the last :meth:`fit` call.

        Raises
        ------
        ValueError
            If no fit has been performed, or if the last fit could not
            produce a valid normalized share vector.  In the latter case the
            raw signed diagnostics remain available on the ``fit`` result;
            they are deliberately not returned here because they are not
            shares.
        """
        # MODIFIED (get-well Unit 1): distinguish "not fitted" from "fitted
        # but normalization unavailable" instead of collapsing both into one
        # misleading message.
        if self._last_attribution is None:
            raise ValueError(
                "Normalized attribution is unavailable for the last fit. "
                "Inspect 'normalization_status' and 'normalization_reason' on "
                "the fit result, and use the raw signed values for "
                "diagnostics."
            )
        if not self._last_attribution:
            raise ValueError("No attribution available. Call fit() first.")
        return dict(self._last_attribution)

    def get_channels(self) -> List[str]:
        """Return the sorted channel list learned during the last fit."""
        return list(self.channels)

    def summary(self, output_format: str = "summary") -> str:
        """Render a plain-text summary or report for a fitted static model.

        Static fits use the same deterministic renderer as temporal fits with
        one ``All paths`` cohort and no timestamp boundaries.  Temporal
        subclasses override this method with their multi-epoch snapshots.
        """

        if self._static_reporting_snapshot is None:
            raise RuntimeError("summary() requires a fitted attribution model")
        return render_temporal_report(self._static_reporting_snapshot, output_format)

    def _store_static_reporting(
        self,
        paths: Union[Sequence[str], np.ndarray, pd.Series],
        conversions: Union[Sequence[int], np.ndarray, pd.Series],
        model_details: tuple = (),  # MODIFIED: distilled model sections
        attribution_note: Optional[str] = None,  # MODIFIED: get-well Unit 1
    ) -> None:
        """Store the all-data snapshot after a successful static fit.

        ``attribution_note`` explains why the attribution matrix is absent
        when normalization was unavailable, so the renderer can say so
        instead of implying the model was never fitted.
        """

        path_count = len(paths)
        conversion_count = int(np.asarray(conversions, dtype=int).sum())
        metrics = build_epoch_metrics(
            epoch_labels=["All paths"],
            starts=[pd.NaT],
            ends=[pd.NaT],
            path_counts=[path_count],
            conversion_counts=[conversion_count],
            sample_floors=0,
        )
        # MODIFIED (get-well Unit 1): an unavailable normalization has no
        # attribution matrix to render.  Raw signed values are never
        # substituted into it, because the renderer formats that matrix as
        # percentages.
        attribution_matrix = (
            None
            if self._last_attribution is None
            else pd.DataFrame([dict(self._last_attribution)])
        )
        self._static_reporting_snapshot = TemporalReportingSnapshot(
            model_name=self.__class__.__name__,
            epoch_source="static",
            epoch_labels=("All paths",),
            boundaries=None,
            epoch_metrics=metrics,
            attribution_matrix=attribution_matrix,
            sample_floor=0,
            model_details=tuple(model_details),  # MODIFIED
            attribution_note=attribution_note,  # MODIFIED
        )

    # ------------------------------------------------------------------ #
    # Introspection                                                       #
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        """Short human-readable name of the concrete model."""
        return self.__class__.__name__

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<{self.name} separator={self.separator!r}>"


__all__ = ["BaseAttribution"]
