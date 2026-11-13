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

from decimal import Decimal
from itertools import chain, combinations
import math
from numbers import Real
from typing import (
    Any,
    Iterable,
    Iterator,
    List,
    Optional,
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
_NESTED_CONTAINER_TYPES = (
    list,
    tuple,
    set,
    frozenset,
    dict,
    np.ndarray,
    pd.Series,
    pd.Index,
    pd.DataFrame,
)

_MAX_VALUE_REPR = 120


def _value_repr(value: Any) -> str:
    """Render an offending input value for an error message."""
    if isinstance(value, np.generic):
        value = value.item()
    text = repr(value)
    if len(text) > _MAX_VALUE_REPR:
        text = text[: _MAX_VALUE_REPR - 3] + "..."
    return text


def _one_dimensional_values(values: Any, name: str) -> List[Any]:
    """Return a one-dimensional input container as a flat Python list.

    The container shape is checked on the supplied object, before any
    element is stringified or cast, so a bare string, a scalar, or a nested
    or multi-dimensional container is rejected instead of being read as a
    collection of journeys or outcomes.

    Raises
    ------
    TypeError
        If ``values`` is ``None``, a bare ``str`` / ``bytes``, or a scalar.
    ValueError
        If ``values`` is not one-dimensional or holds a nested container.
    """
    if isinstance(values, pd.DataFrame):
        raise ValueError(
            f"{name} must be one-dimensional, got a DataFrame with shape "
            f"{values.shape}"
        )
    if isinstance(values, (pd.Series, pd.Index)):
        items = values.tolist()
    elif isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise ValueError(
                f"{name} must be one-dimensional, got an array with shape "
                f"{values.shape}"
            )
        items = values.tolist()
    elif (
        values is None
        or isinstance(values, (str, bytes))
        or not isinstance(values, Iterable)
    ):
        raise TypeError(
            f"{name} must be a one-dimensional list, tuple, NumPy array, or "
            f"pandas Series, not a bare {type(values).__name__}: "
            f"{_value_repr(values)}"
        )
    else:
        items = list(values)
    for index, item in enumerate(items):
        if isinstance(item, _NESTED_CONTAINER_TYPES):
            raise ValueError(
                f"{name} must be one-dimensional; the entry at index "
                f"{index} is a nested {type(item).__name__}: "
                f"{_value_repr(item)}"
            )
    return items


def _string_array(values: List[Any]) -> np.ndarray:
    """Stringify already shape-checked path values."""
    # Pandas nullable containers can materialize ``None`` as ``nan``.  Treat
    # all scalar missing values as the established empty-path representation.
    cleaned = ["" if p is None or (not isinstance(p, (list, tuple, dict)) and pd.isna(p)) else str(p) for p in values]
    return np.array(cleaned, dtype=object)


def to_string_array(paths: Union[Sequence[str], np.ndarray, pd.Series]) -> np.ndarray:
    """Coerce a list / Series / ndarray of paths into a 1-D string ndarray.

    ``None`` values are replaced with the empty string; all other values are
    passed through :pyfunc:`str`.

    The container must be one-dimensional.  A bare string, a scalar, or a
    nested / multi-dimensional container raises (see
    :func:`_one_dimensional_values`) rather than being stringified into a
    collection of journeys.
    """
    return _string_array(_one_dimensional_values(paths, "paths"))


def _is_missing_scalar(value: Any) -> bool:
    """``True`` for ``None``, ``NaN``, ``pd.NA``, ``pd.NaT`` and the like."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError, ArithmeticError):
        return False


def _is_infinite_scalar(value: Any) -> bool:
    try:
        return bool(math.isinf(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _binary_outcome(value: Any, index: int) -> int:
    """Validate one supplied conversion outcome and return it as ``0``/``1``.

    The check runs on the original value, before any integer conversion:

    * booleans, integers, and other real numbers are accepted only when
      they equal exactly ``0`` or ``1`` (``0.0`` / ``1.0`` included);
    * the canonical strings ``"0"`` and ``"1"`` (``str`` or ``bytes``) are
      accepted, preserving what plain integer coercion took; any other
      spelling (``" 1"``, ``"01"``, ``"1.0"``, ``"true"``) is rejected;
    * missing, non-finite, fractional, out-of-range, and non-numeric
      values are rejected.
    """
    if isinstance(value, str):
        if value == "0":
            return 0
        if value == "1":
            return 1
    elif isinstance(value, bytes):
        if value == b"0":
            return 0
        if value == b"1":
            return 1
    elif isinstance(value, (Real, Decimal, np.bool_)):
        try:
            if value == 0:
                return 0
            if value == 1:
                return 1
        except ArithmeticError:
            # A signalling ``Decimal`` NaN refuses comparison.
            pass

    if _is_missing_scalar(value):
        detail = f"missing value {_value_repr(value)}"
    elif _is_infinite_scalar(value):
        detail = f"non-finite value {_value_repr(value)}"
    else:
        detail = _value_repr(value)
    raise ValueError(f"Conversion at index {index} must be 0 or 1, got {detail}")


def _binary_int_array(values: List[Any]) -> np.ndarray:
    """Validate already shape-checked outcomes, then convert them to int."""
    return np.array(
        [_binary_outcome(value, index) for index, value in enumerate(values)],
        dtype=int,
    )


def _plain_binary_array(conversions: Any) -> Optional[np.ndarray]:
    """Vectorised acceptance of an all-0/1 numeric NumPy array or Series.

    Returns the outcomes as a new int array only when ``conversions`` is a
    one-dimensional boolean / integer / float NumPy array (or a Series
    backed by one) whose every value is exactly ``0`` or ``1``.  Anything
    else returns ``None`` so the per-value path decides and reports: this
    shortcut never rejects and never builds an error message.
    """
    values = conversions.to_numpy() if isinstance(conversions, pd.Series) else conversions
    if (
        not isinstance(values, np.ndarray)
        or values.ndim != 1
        or values.dtype.kind not in "biuf"
    ):
        return None
    if not ((values == 0) | (values == 1)).all():
        return None
    return values.astype(int)


def to_int_array(conversions: Union[Sequence[int], np.ndarray, pd.Series]) -> np.ndarray:
    """Validate binary conversion flags and coerce them to a 1-D int ndarray.

    Each supplied value is validated *before* integer conversion, so a
    fractional, out-of-range, missing, or non-finite outcome raises
    ``ValueError`` naming its position and original value instead of being
    truncated into a different dataset.  See :func:`prepare_journey_data`
    for the accepted values.
    """
    plain = _plain_binary_array(conversions)
    if plain is not None:
        return plain
    return _binary_int_array(_one_dimensional_values(conversions, "conversions"))


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


def prepare_journey_data(
    paths: Union[Sequence[str], np.ndarray, pd.Series],
    conversions: Union[Sequence[int], np.ndarray, pd.Series],
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate the aligned (paths, conversions) pair, then coerce both.

    This is the single preparation path shared by every model ``fit`` entry
    point and by :func:`validate_data`.  All checks run on the supplied
    values, before paths are stringified or outcomes are cast to integers:

    1. both containers must be one-dimensional (a bare string, a scalar, or
       a nested / multi-dimensional container is rejected);
    2. the lengths must match and be non-zero;
    3. every conversion outcome must be exactly ``0`` or ``1``.

    Accepted outcomes are booleans, integers, and other real numbers equal
    to exactly ``0`` or ``1`` (so ``0.0`` / ``1.0``), plus the canonical
    strings ``"0"`` and ``"1"``.  Fractional, out-of-range, missing
    (``None``, ``NaN``, ``pd.NA``), non-finite, and otherwise non-binary
    values are rejected; nothing is truncated or rounded.

    Returns
    -------
    paths : numpy.ndarray
        1-D object array of path strings (missing paths become ``""``).
    conversions : numpy.ndarray
        1-D integer array of ``0`` / ``1`` outcomes.

    Raises
    ------
    TypeError
        If either argument is ``None``, a bare string, or a scalar.
    ValueError
        If either argument is not one-dimensional, the lengths differ, no
        paths are supplied, or a conversion outcome is not exactly ``0`` or
        ``1``.  The message names the offending position and the original
        value.
    """
    path_values = _one_dimensional_values(paths, "paths")
    plain = _plain_binary_array(conversions)
    conversion_values = (
        plain
        if plain is not None
        else _one_dimensional_values(conversions, "conversions")
    )

    if len(path_values) != len(conversion_values):
        raise ValueError(
            f"Paths ({len(path_values)}) and conversions "
            f"({len(conversion_values)}) must have same length"
        )
    if len(path_values) == 0:
        raise ValueError("No paths provided")
    conv_arr = (
        plain if plain is not None else _binary_int_array(conversion_values)
    )
    paths_arr = _string_array(path_values)
    empty_count = sum(1 for p in paths_arr if not p or not p.strip())
    if empty_count > 0:
        warnings.warn(
            f"Found {empty_count} empty paths which will be ignored"
        )
    return paths_arr, conv_arr


def _validate_observation_weights(
    weights: Union[Sequence[float], np.ndarray, pd.Series],
    n_observations: int,
) -> np.ndarray:
    """Validate per-journey observation weights and return them as floats.

    Applies the weight contract of :meth:`MarkovAttribution.fit
    <do_attribution.markov.MarkovAttribution.fit>`, check for check and
    message for message, so the weighted Shapley and comparison paths accept
    and reject exactly the inputs Markov does:

    1. the weights are coerced with ``np.asarray(weights, dtype=float)``, so
       a pandas Series is read positionally (its index labels are ignored)
       and booleans become ``0.0`` / ``1.0``;
    2. the result must be one-dimensional and as long as the observations;
    3. every weight must be finite and nonnegative;
    4. the total weight must be positive.

    Raises
    ------
    ValueError
        If any of the checks above fails.
    """
    weights_arr = np.asarray(weights, dtype=float)
    if weights_arr.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    if len(weights_arr) != n_observations:
        raise ValueError(
            f"Paths ({n_observations}) and weights "
            f"({len(weights_arr)}) must have same length"
        )
    if not np.isfinite(weights_arr).all():
        raise ValueError("weights must contain only finite values")
    if (weights_arr < 0).any():
        raise ValueError("weights must be nonnegative")
    if float(weights_arr.sum()) <= 0.0:
        raise ValueError("weights must have positive total weight")
    return weights_arr


def validate_data(
    paths: Union[Sequence[str], np.ndarray, pd.Series],
    conversions: Union[Sequence[int], np.ndarray, pd.Series],
) -> None:
    """Validate the aligned (paths, conversions) pair.

    Runs the shared :func:`prepare_journey_data` checks on the supplied
    values and discards the coerced arrays.

    Raises
    ------
    TypeError
        If either argument is ``None``, a bare string, or a scalar.
    ValueError
        If either argument is not one-dimensional, lengths differ, the list
        is empty, or a conversion flag is not exactly ``0`` or ``1``.
    """
    prepare_journey_data(paths, conversions)


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
    "prepare_journey_data",
    "validate_data",
    "create_n_order_states",
    "get_powerset",
    # MODIFIED: export P0 helpers so downstream models + tests can use them.
    "strip_conversion_labels",
    "strip_conversion_labels_array",
]
