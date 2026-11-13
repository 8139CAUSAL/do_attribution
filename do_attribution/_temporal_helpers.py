"""Shared implementation details for the temporal attribution models.

This module is intentionally private.  The temporal Shapley and Markov
modules continue to expose their historical private helper names by
importing them from here, while all behavior is maintained in one place.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Tuple, Union

import numpy as np
import pandas as pd


VALID_EPOCH_ASSIGNMENTS: "frozenset[str]" = frozenset(
    {"first_touch", "conversion", "split"}
)


def epoch_attribution(result: Mapping[str, Any]) -> Dict[str, float]:
    """Return an epoch's normalized shares, or ``{}`` when unavailable."""
    return dict(result.get("attribution") or {})


def pandas_freq_alias(freq_str: str) -> str:
    """Translate legacy public aliases for pandas offset validation."""
    replacements = {"M": "ME", "Q": "QE", "Y": "YE", "A": "YE"}
    if freq_str in replacements:
        return replacements[freq_str]
    for old, new in (("Q-", "QE-"), ("A-", "YE-")):
        if freq_str.startswith(old):
            return new + freq_str[len(old):]
    return freq_str


def pandas_period_alias(freq_str: str) -> str:
    """Translate annual aliases for pandas 3 period construction.

    Annual periods require ``Y``/``Y-<month>``, while offsets use ``YE``.
    Other period aliases, including ``M`` and ``Q``, stay unchanged.
    """
    if not isinstance(freq_str, str):
        raise TypeError("freq alias must be a str")
    for annual_alias in ("A", "YE"):
        if freq_str == annual_alias:
            return "Y"
        if freq_str.startswith(annual_alias + "-"):
            return "Y" + freq_str[len(annual_alias):]
    return freq_str


def validate_epoch_assignment(value: str) -> str:
    """Validate an epoch-assignment constructor argument."""
    if not isinstance(value, str):
        raise TypeError(
            "epoch_assignment must be a string, got "
            f"{type(value).__name__}"
        )
    if value not in VALID_EPOCH_ASSIGNMENTS:
        raise ValueError(
            "epoch_assignment must be one of "
            f"{sorted(VALID_EPOCH_ASSIGNMENTS)}, got {value!r}"
        )
    return value


def validate_freq_alias(freq_str: str) -> str:
    """Validate one pandas frequency alias and return it unchanged."""
    if not isinstance(freq_str, str):
        raise TypeError(
            f"freq alias must be a str, got {type(freq_str).__name__}"
        )
    try:
        pd.tseries.frequencies.to_offset(pandas_freq_alias(freq_str))
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"freq {freq_str!r} is not a valid pandas frequency alias "
            f"(pd.tseries.frequencies.to_offset failed: {exc})"
        )
    return freq_str


def validate_freq(
    freq: Union[str, Mapping[int, str]],
) -> Union[str, Mapping[int, str]]:
    """Validate a string or per-year mapping of pandas frequency aliases."""
    if isinstance(freq, str):
        return validate_freq_alias(freq)
    if isinstance(freq, Mapping):
        if len(freq) == 0:
            raise ValueError("freq mapping must be non-empty")
        for key, value in freq.items():
            if not isinstance(key, int) or isinstance(key, bool):
                raise TypeError(
                    "freq mapping keys must be int (calendar year), got "
                    f"{type(key).__name__} for key {key!r}"
                )
            validate_freq_alias(value)
        return freq
    raise TypeError(
        "freq must be a str or a Mapping[int, str], got "
        f"{type(freq).__name__}"
    )


def get_period_key(period: pd.Period) -> pd.Timestamp:
    """Return a wall-clock sort key that works across period frequencies."""
    return period.start_time


def bin_timestamps_to_epochs(
    dt_index: pd.DatetimeIndex,
    resolve_freq: Callable[[int], str],
) -> Tuple[List[pd.Period], np.ndarray]:
    """Assign every timestamp to exactly one calendar epoch.

    Each timestamp is binned with the frequency ``resolve_freq`` returns for
    its calendar year, which is what lets a per-year frequency mapping mix
    granularities.  A period is not confined to one calendar year, though:
    the weekly period ``2020-12-28/2021-01-03`` holds timestamps from both
    2020 and 2021.  The same period can therefore come out of two
    consecutive year buckets.  Those rows belong to a single epoch, so
    epochs are keyed by the period itself and the year buckets are merged;
    keying by year bucket would fit the period twice and keep only the
    later bucket's rows.

    Parameters
    ----------
    dt_index : pd.DatetimeIndex
        One timestamp per path.
    resolve_freq : callable
        Maps a calendar year to the pandas period alias used for that year.
        It is called once per year present in ``dt_index``, in ascending
        year order, before any epoch is returned.

    Returns
    -------
    epochs : list[pd.Period]
        The distinct epochs, ordered by ``start_time``.
    codes : np.ndarray
        For every timestamp, the position of its epoch in ``epochs``.
    """
    years = dt_index.year
    codes = np.empty(len(dt_index), dtype=np.intp)
    epoch_ids: Dict[pd.Period, int] = {}
    for year in sorted(set(int(y) for y in years)):
        freq_for_year = resolve_freq(year)
        year_mask = np.asarray(years == year)
        year_periods = dt_index[year_mask].to_period(
            pandas_period_alias(freq_for_year)
        )
        year_codes, year_uniques = pd.factorize(year_periods)
        # Register this year's periods chronologically so that epochs with
        # equal start times keep a deterministic year-then-time order.
        for period in sorted(year_uniques, key=get_period_key):
            epoch_ids.setdefault(period, len(epoch_ids))
        year_ids = np.array(
            [epoch_ids[period] for period in year_uniques], dtype=np.intp
        )
        codes[year_mask] = year_ids[year_codes]

    epochs = sorted(epoch_ids, key=get_period_key)
    position = np.empty(len(epochs), dtype=np.intp)
    for chronological, epoch in enumerate(epochs):
        position[epoch_ids[epoch]] = chronological
    return epochs, position[codes]


def periods_have_different_freq(first: Any, second: Any) -> bool:
    """Return whether two pandas periods have different frequencies."""
    if not (
        isinstance(first, pd.Period)
        and isinstance(second, pd.Period)
    ):
        return False
    return first.freqstr != second.freqstr


def align_index_to_epochs(
    frame: pd.DataFrame,
    epochs: List[Any],
) -> pd.DataFrame:
    """Reindex matrix rows to fitted epoch order."""
    if list(frame.index) == list(epochs):
        return frame
    return frame.reindex(epochs)


def matrix_to_long_frame(
    matrix: pd.DataFrame,
    value_name: str,
) -> pd.DataFrame:
    """Melt an epochs-by-channels matrix into a stable flat frame."""
    epochs = np.array(
        [str(epoch) for epoch in matrix.index],
        dtype=object,
    )
    channels = np.array(
        [str(channel) for channel in matrix.columns],
        dtype=object,
    )
    if matrix.size:
        values = matrix.to_numpy(dtype=float).reshape(-1)
    else:
        values = np.array([], dtype=float)
    return pd.DataFrame(
        {
            "epoch": np.repeat(epochs, len(channels)),
            "channel": np.tile(channels, len(epochs)),
            value_name: values,
        }
    )


def sample_sizes_to_long_frame(
    diagnostic: pd.DataFrame,
) -> pd.DataFrame:
    """Flatten a channel-and-epoch sample-size diagnostic."""
    if diagnostic.empty:
        return pd.DataFrame(
            columns=[
                "epoch",
                "channel",
                "n_paths_containing_channel",
                "n_conversions_on_paths_containing_channel",
                "thin",
            ]
        )
    flat = diagnostic.reset_index()
    flat["epoch"] = flat["epoch"].astype(str)
    flat["channel"] = flat["channel"].astype(str)
    ordered = ["epoch", "channel"] + [
        column
        for column in flat.columns
        if column not in ("epoch", "channel")
    ]
    return flat[ordered].reset_index(drop=True)
