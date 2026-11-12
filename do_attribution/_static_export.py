"""Flat DataFrame projection helpers for static attribution fits.

This module is intentionally private.  ``MarkovAttribution.to_dataframes``
and ``ShapleyAttribution.to_dataframes`` build their tables with these
helpers from the snapshot each model stores after a successful ``fit``, and
``TemporalMarkovAttribution`` reuses :func:`transition_edge_frame` for its
per-epoch transition table.

Every helper only projects values that a fit already computed: nothing is
refitted, no coalition is enumerated, and no text is rendered or parsed.
Each returned frame is newly allocated, has a fresh ``RangeIndex``, and
holds one scalar per cell, with identities rendered as ``str`` so the
tables join against each other and serialize without loss.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .normalization import NORMALIZATION_DIAGNOSTIC_KEYS


#: Columns of a flat transition-edge table, in order.
TRANSITION_EDGE_COLUMNS: Tuple[str, ...] = ("from_state", "to_state", "probability")

#: Normalization diagnostics exported as text; the rest are numeric.
_TEXT_NORMALIZATION_KEYS = ("normalization_status", "normalization_reason")


def channel_value_frame(
    channels: Sequence[str],
    values: Optional[Mapping[str, float]],
    value_name: str,
) -> pd.DataFrame:
    """Return one row per fitted channel with its value under ``value_name``.

    ``values=None`` (normalized attribution unavailable) keeps every channel
    row and leaves the value missing, so an absent share is never exported
    as zero credit.
    """
    labels = np.array([str(channel) for channel in channels], dtype=object)
    if values is None:
        column = np.full(len(labels), np.nan, dtype=float)
    else:
        column = np.array(
            [values.get(channel, np.nan) for channel in channels], dtype=float
        )
    return pd.DataFrame({"channel": labels, value_name: column})


def transition_edge_frame(matrix: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Flatten a transition matrix into ``from_state``/``to_state`` edges.

    Rows follow the matrix's own state order, row-major, and zero-probability
    cells are retained so each ``from_state`` group stays row-stochastic.
    State identities are rendered with ``str``, so an ``order >= 2`` tuple
    state such as ``('Email', 'Social')`` exports as ``"('Email', 'Social')"``.
    ``matrix=None`` returns an empty frame with the same columns.
    """
    if matrix is None:
        return pd.DataFrame(columns=list(TRANSITION_EDGE_COLUMNS))
    n_from, n_to = matrix.shape
    from_states = np.array([str(state) for state in matrix.index], dtype=object)
    to_states = np.array([str(state) for state in matrix.columns], dtype=object)
    return pd.DataFrame(
        {
            "from_state": np.repeat(from_states, n_to),
            "to_state": np.tile(to_states, n_from),
            "probability": matrix.to_numpy(dtype=float).reshape(-1),
        }
    )


def conversion_labels_cell(conversion_labels: Iterable[str]) -> str:
    """Render the conversion-endpoint labels as one scalar cell.

    The sorted labels are encoded as a JSON list (``"[]"`` when none were
    declared), which stays unambiguous whatever characters a label contains.
    """
    return json.dumps(sorted(conversion_labels))


def normalization_columns(
    diagnostics: Mapping[str, Any],
) -> List[Tuple[str, Any]]:
    """Return the shared normalization diagnostics as ordered scalar cells.

    Status and reason stay text (the reason is ``None`` when attribution is
    available).  The numeric diagnostics are floats, with ``NaN`` where the
    fit recorded ``None`` because the quantity is undefined.
    """
    columns: List[Tuple[str, Any]] = []
    for key in NORMALIZATION_DIAGNOSTIC_KEYS:
        value = diagnostics.get(key)
        if key not in _TEXT_NORMALIZATION_KEYS:
            value = float("nan") if value is None else float(value)
        columns.append((key, value))
    return columns


def fit_diagnostics_frame(columns: Sequence[Tuple[str, Any]]) -> pd.DataFrame:
    """Return a one-row frame from ordered ``(column, scalar)`` pairs.

    Text cells are supplied as ``object`` values, like the identity columns
    of the other export tables, and a missing text cell stays ``None``.
    Numeric and boolean cells keep their own dtype.
    """
    data = {}
    for name, value in columns:
        if value is None or isinstance(value, str):
            data[name] = np.array([value], dtype=object)
        else:
            data[name] = np.array([value])
    return pd.DataFrame(data)


__all__ = [
    "TRANSITION_EDGE_COLUMNS",
    "channel_value_frame",
    "transition_edge_frame",
    "conversion_labels_cell",
    "normalization_columns",
    "fit_diagnostics_frame",
]
