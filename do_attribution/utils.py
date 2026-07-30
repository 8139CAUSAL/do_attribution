"""
do_attribution.utils
-------------------------

Shared data-handling utilities used by every attribution model in this
library. Consolidates the preprocessing logic that previously lived in both
``markov_attribution.utils`` and ``shapley_attribution.utils``.

All public helpers accept a uniform set of inputs (list / numpy array /
pandas Series of strings and ints) so the models can be used with any of
these containers interchangeably.
"""

from __future__ import annotations

from itertools import chain, combinations
from typing import (
    Any,
    Iterable,
    Iterator,
    List,
    Sequence,
    Set,
    Tuple,
    Union,
)
import warnings

import numpy as np
import pandas as pd

# Reserved tokens that must never be interpreted as user channels. They are
# used internally by the Markov model as absorbing / starting states.
SPECIAL_TOKENS: Set[str] = {"START", "CONVERSION", "NULL"}

DEFAULT_SEPARATOR: str = " > "


# --------------------------------------------------------------------------- #
# Input coercion                                                              #
# --------------------------------------------------------------------------- #
def to_string_array(paths: Union[Sequence[str], np.ndarray, pd.Series]) -> np.ndarray:
    """Coerce a list / Series / ndarray of paths into a 1-D string ndarray.

    ``None`` values are replaced with the empty string; all other values are
    passed through :pyfunc:`str`.
    """
    if isinstance(paths, pd.Series):
        paths = paths.tolist()
    elif isinstance(paths, np.ndarray):
        paths = paths.tolist()
    # Pandas nullable containers can materialize ``None`` as ``nan``.  Treat
    # all scalar missing values as the established empty-path representation.
    cleaned = ["" if p is None or (not isinstance(p, (list, tuple, dict)) and pd.isna(p)) else str(p) for p in paths]
    return np.array(cleaned, dtype=object)


def to_int_array(conversions: Union[Sequence[int], np.ndarray, pd.Series]) -> np.ndarray:
    """Coerce conversion flags to a 1-D int ndarray."""
    if isinstance(conversions, pd.Series):
        conversions = conversions.tolist()
    elif isinstance(conversions, np.ndarray):
        conversions = conversions.tolist()
    return np.array(conversions, dtype=int)


# --------------------------------------------------------------------------- #
# Channel extraction & validation                                             #
# --------------------------------------------------------------------------- #
def split_path(path: str, separator: str = DEFAULT_SEPARATOR) -> List[str]:
    """Split a path string into its ordered list of channel touches.

    Whitespace is stripped from each channel, and empty strings are
    filtered out.  Special tokens (``START`` / ``CONVERSION`` / ``NULL``)
    are *not* filtered here; that is the caller's responsibility.
    """
    if not path:
        return []
    return [p.strip() for p in path.split(separator) if p.strip()]


def extract_channels(
    paths: Union[Sequence[str], np.ndarray, pd.Series],
    separator: str = DEFAULT_SEPARATOR,
    conversion_labels: Union[Iterable[str], None] = None,  # MODIFIED: accept conversion endpoint labels so they are excluded from channels (P0).
) -> List[str]:
    """Return the sorted list of unique, user-visible channels.

    Reserved tokens (``START``, ``CONVERSION``, ``NULL``) are removed so that
    downstream attribution math never accidentally credits them.  Whitespace
    is stripped and empty channels are dropped.

    Parameters
    ----------
    conversion_labels : iterable of str, optional
        Tokens that represent the conversion endpoint (e.g. ``"Purchase"``)
        rather than a real marketing channel.  When supplied, these tokens
        are excluded from the returned channel list.  Case-sensitive exact
        match against whitespace-stripped tokens.
    """
    channels: Set[str] = set()
    arr = to_string_array(paths)
    for path in arr:
        if not path:
            continue
        channels.update(split_path(path, separator))
    channels -= SPECIAL_TOKENS
    # MODIFIED: exclude caller-supplied conversion endpoint tokens (P0).
    if conversion_labels is not None:
        channels -= _normalize_conversion_labels(conversion_labels)
    return sorted(channels)


# --------------------------------------------------------------------------- #
# Conversion-label handling (P0)                                              #
# --------------------------------------------------------------------------- #
# MODIFIED: new helpers so the library can be told which tokens in the path
# string are the conversion endpoint (e.g. ``"Purchase"``) rather than a
# real channel.  Without this the library silently attributes credit to the
# endpoint.  See OPEN_ISSUES.md P0.
def _normalize_conversion_labels(
    conversion_labels: Union[Iterable[str], None],
) -> "frozenset[str]":
    """Coerce a user-supplied ``conversion_labels`` argument to a frozenset.

    - ``None`` returns an empty frozenset (current behaviour).
    - A bare string is rejected to avoid the common bug of iterating it
      character-by-character.
    - Any label colliding with :data:`SPECIAL_TOKENS` raises ``ValueError``.
    - Whitespace is stripped on each label (matching the channel
      normalisation rules applied by :func:`split_path`).
    - Empty labels (after strip) are rejected.
    """
    if conversion_labels is None:
        return frozenset()
    if isinstance(conversion_labels, (str, bytes)):
        raise TypeError(
            "conversion_labels must be a list / tuple / set of strings, "
            f"not a bare {type(conversion_labels).__name__}"
        )
    normalized: Set[str] = set()
    for raw in conversion_labels:
        if not isinstance(raw, str):
            raise TypeError(
                f"conversion_labels entries must be str, got {type(raw).__name__}"
            )
        stripped = raw.strip()
        if not stripped:
            raise ValueError("conversion_labels contains an empty label")
        normalized.add(stripped)
    collision = normalized & SPECIAL_TOKENS
    if collision:
        raise ValueError(
            "conversion_labels must not collide with SPECIAL_TOKENS "
            f"({sorted(SPECIAL_TOKENS)}); got {sorted(collision)}"
        )
    return frozenset(normalized)


def strip_conversion_labels(
    path: str,
    conversion_labels: Iterable[str],
    separator: str = DEFAULT_SEPARATOR,
) -> str:
    """Return ``path`` with every token matching ``conversion_labels`` removed.

    Tokens are split on ``separator`` and whitespace-stripped (matching
    :func:`split_path`); any token whose stripped value is in
    ``conversion_labels`` is dropped.  The remaining tokens are rejoined
    with ``separator``.
    """
    labels = _normalize_conversion_labels(conversion_labels)
    if not labels:
        return path
    tokens = split_path(path, separator)
    filtered = [t for t in tokens if t not in labels]
    return separator.join(filtered)


def strip_conversion_labels_array(
    paths: Union[Sequence[str], np.ndarray, pd.Series],
    conversion_labels: Iterable[str],
    separator: str = DEFAULT_SEPARATOR,
) -> np.ndarray:
    """Vectorised :func:`strip_conversion_labels` over an array of paths."""
    labels = _normalize_conversion_labels(conversion_labels)
    arr = to_string_array(paths)
    if not labels:
        return arr
    out = np.empty(len(arr), dtype=object)
    for i, path in enumerate(arr):
        if not path:
            out[i] = ""
            continue
        tokens = split_path(path, separator)
        out[i] = separator.join(t for t in tokens if t not in labels)
    return out


def validate_data(
    paths: Union[Sequence[str], np.ndarray, pd.Series],
    conversions: Union[Sequence[int], np.ndarray, pd.Series],
) -> None:
    """Validate the aligned (paths, conversions) pair.

    Raises
    ------
    ValueError
        If lengths differ, the list is empty, or a conversion flag is not
        exactly ``0`` or ``1``.
    """
    paths_arr = to_string_array(paths)
    conv_arr = to_int_array(conversions)

    if len(paths_arr) != len(conv_arr):
        raise ValueError(
            f"Paths ({len(paths_arr)}) and conversions ({len(conv_arr)}) "
            "must have same length"
        )
    if len(paths_arr) == 0:
        raise ValueError("No paths provided")
    for i, conv in enumerate(conv_arr.tolist()):
        if conv not in (0, 1):
            raise ValueError(
                f"Conversion at index {i} must be 0 or 1, got {conv}"
            )
    empty_count = sum(1 for p in paths_arr if not p or not p.strip())
    if empty_count > 0:
        warnings.warn(
            f"Found {empty_count} empty paths which will be ignored"
        )


# --------------------------------------------------------------------------- #
# Markov-specific state helpers                                               #
# --------------------------------------------------------------------------- #
def create_n_order_states(
    path: str,
    order: int,
    separator: str = DEFAULT_SEPARATOR,
) -> List[Union[str, Tuple[str, ...]]]:
    """Turn a path string into n-order Markov states.

    For ``order == 1`` returns a flat list of channel strings.  For
    ``order >= 2`` returns a list of sliding ``order``-length tuples; if the
    path is shorter than ``order`` a single tuple left-padded with the
    ``START`` sentinel is returned.
    """
    if order < 1:
        raise ValueError("Order must be >= 1")
    if not path:
        return []

    channels = split_path(path, separator)
    if not channels:
        return []

    if order == 1:
        return channels

    if len(channels) < order:
        padded = ["START"] * (order - len(channels)) + channels
        return [tuple(padded)]

    return [
        tuple(channels[i : i + order])
        for i in range(len(channels) - order + 1)
    ]


# --------------------------------------------------------------------------- #
# Shapley-specific combinatorial helpers                                      #
# --------------------------------------------------------------------------- #
def get_powerset(iterable: Iterable[Any]) -> Iterator[Tuple[Any, ...]]:
    """Yield every subset of ``iterable`` as an ordered tuple.

    Example::

        >>> list(get_powerset(['a', 'b']))
        [(), ('a',), ('b',), ('a', 'b')]
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


__all__ = [
    "SPECIAL_TOKENS",
    "DEFAULT_SEPARATOR",
    "to_string_array",
    "to_int_array",
    "split_path",
    "extract_channels",
    "validate_data",
    "create_n_order_states",
    "get_powerset",
    # MODIFIED: export P0 helpers so downstream models + tests can use them.
    "strip_conversion_labels",
    "strip_conversion_labels_array",
]
