"""
do_attribution.temporal_diagnostics
-----------------------------------

Lightweight, model-agnostic diagnostics for the temporal classes.
Currently exposes one helper, ``suggest_freq``.

What this module is
~~~~~~~~~~~~~~~~~~~
``suggest_freq`` walks a list of pandas freq aliases finest -> coarsest
and returns the finest one under which every resulting epoch contains
at least ``min_samples`` timestamps. With ``by='year'`` it does the
same per calendar year and returns a ``Dict[int, str]`` mapping year
-> chosen freq, suitable for the per-year mapping mode of
``TemporalShapleyAttribution`` / ``TemporalMarkovAttribution``.

What this module is NOT
~~~~~~~~~~~~~~~~~~~~~~~
``suggest_freq`` is a **sample-floor walker**, not a data-driven epoch
recommender. It guards the *total* path count per bucket; it does not
inspect per-channel volume, does not detect change-points, does not
align epoch boundaries to any signal, and does not protect long-tailed
channels from being fragmented under fine bins. Caller is responsible
for choosing the epoch grid; this helper only catches the
"my-buckets-are-too-thin" case.

Earlier project framing described this helper as an 80%-of-the-value
substitute for "data-driven epoch sizing." That framing oversold what
the helper does and is retired. See ``TEMPORAL_PUNCHLIST.md`` at the
workspace root for the open epoch-selection problem and the forward
design direction.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union
import warnings

import pandas as pd

from ._temporal_helpers import pandas_freq_alias as _pandas_freq_alias


# Ordered finest -> coarsest. The first freq in this tuple that produces
# every-epoch >= min_samples is the recommended choice; coarser freqs are
# only returned as a fallback when no fine bin clears the floor.
_DEFAULT_CANDIDATE_FREQS: Tuple[str, ...] = ("D", "W", "M", "Q", "Y")

# MODIFIED (v2): under by='year' we drop 'Y' — every timestamp in a year
# bucket trivially shares one yearly Period, so 'Y' would always pass and
# never be informative. Callers who want yearly granularity for a specific
# year can pass ``freq={2024: 'Y'}`` explicitly to the temporal classes.
_DEFAULT_CANDIDATE_FREQS_BY_YEAR: Tuple[str, ...] = ("D", "W", "M", "Q")


def _pick_finest_passing(
    dt_index: pd.DatetimeIndex,
    min_samples: int,
    candidates: Tuple[str, ...],
) -> Tuple[Optional[str], bool]:
    """Walk candidates finest -> coarsest; return (freq, passed) or (None, False).

    Returns ``(freq, True)`` for the finest candidate where every produced
    period has ``>= min_samples`` timestamps. Returns ``(None, False)`` if
    no candidate passes; the caller decides the fallback.
    """
    for freq in candidates:
        try:
            periods = dt_index.to_period(freq)
        except (ValueError, TypeError) as exc:
            warnings.warn(f"Skipping candidate freq {freq!r}: {exc}")
            continue
        counts = pd.Series(periods).value_counts()
        if len(counts) == 0:
            continue
        if counts.min() >= min_samples:
            return freq, True
    return None, False


def suggest_freq(
    timestamps: Union[Sequence[Any], pd.Series, pd.DatetimeIndex],
    min_samples: int = 30,
    candidate_freqs: Optional[Iterable[str]] = None,
    by: Optional[str] = None,
) -> Union[str, Dict[int, str]]:
    """Return the finest pandas freq under which every epoch clears ``min_samples``.

    This is a **sample-floor sanity check on bucket sizes**, not a
    data-driven epoch recommender. It walks the candidate freq aliases
    finest -> coarsest and returns the first one whose binning
    produces *every* epoch with ``>= min_samples`` total timestamps.
    The floor is on **total path count per epoch**; it does not
    inspect per-channel volume and does not protect long-tailed
    channels from fragmentation under fine bins (use
    ``get_channel_epoch_sample_sizes`` / ``get_epoch_summary`` on the
    fitted temporal class for that).

    With ``by=None`` (default), runs the walk over the *entire*
    timestamp set and returns a single freq alias.

    With ``by='year'``, partitions timestamps by ``dt.year`` and runs
    the same walk *per year*, returning a ``Dict[int, str]`` mapping
    year -> chosen freq. ``'Y'`` is dropped from the default candidate
    list under this mode (it would trivially pass inside any single-
    year bucket). The result is consumable directly by
    ``TemporalShapleyAttribution(freq=...)`` /
    ``TemporalMarkovAttribution(freq=...)`` for the per-year mapping
    fit path.

    If none of the candidates pass:

    * ``by=None``: returns the coarsest candidate + one ``UserWarning``.
    * ``by='year'``: returns ``'Y'`` for that year + one ``UserWarning``
      per failing year.

    Parameters
    ----------
    timestamps : array-like
        One timestamp per path, in any form ``pd.to_datetime`` accepts.
    min_samples : int, default=30
        Minimum total path count required in every epoch for the
        candidate freq to be considered "passing". Under ``by='year'``
        this is the floor *per epoch*, not per year.
    candidate_freqs : iterable of str, optional
        Pandas freq aliases to evaluate, ordered finest -> coarsest. If
        ``None``, defaults to ``('D', 'W', 'M', 'Q', 'Y')`` for
        ``by=None`` and ``('D', 'W', 'M', 'Q')`` for ``by='year'``.
    by : {'year', None}, default=None
        Bucketing mode. ``None`` returns a single freq alias.
        ``'year'`` returns a per-year mapping suitable for the
        ``Mapping[int, str]`` mode of the temporal classes.

    Returns
    -------
    str or dict[int, str]
        The finest passing freq alias (``by=None``) or a year -> freq
        mapping (``by='year'``).

    Raises
    ------
    ValueError
        If ``timestamps`` is empty, ``min_samples`` is non-positive,
        ``candidate_freqs`` is empty, or ``by`` is not one of
        ``{None, 'year'}``.

    Notes
    -----
    Epoch selection is the caller's responsibility. This helper only
    catches the "my-buckets-are-too-thin" case. For an honest summary
    of what is and isn't solved by the temporal module, see
    ``TEMPORAL_PUNCHLIST.md`` at the workspace root.
    """
    if min_samples <= 0:
        raise ValueError(f"min_samples must be positive, got {min_samples}")
    if by not in (None, "year"):
        raise ValueError(
            f"by must be None or 'year', got {by!r}"
        )

    # Resolve candidate list per mode.
    if candidate_freqs is None:
        default = (
            _DEFAULT_CANDIDATE_FREQS_BY_YEAR if by == "year"
            else _DEFAULT_CANDIDATE_FREQS
        )
        candidates: Tuple[str, ...] = default
    else:
        candidates = tuple(candidate_freqs)
    if not candidates:
        raise ValueError("candidate_freqs must be non-empty")

    dt_index = pd.to_datetime(timestamps)
    if len(dt_index) == 0:
        raise ValueError("timestamps is empty")

    if by is None:
        chosen, passed = _pick_finest_passing(dt_index, min_samples, candidates)
        if passed:
            return chosen  # type: ignore[return-value]
        fallback = candidates[-1]
        warnings.warn(
            f"No candidate freq satisfied min_samples={min_samples} on this "
            f"data; returning the coarsest candidate {fallback!r} as a "
            f"fallback. Either lower min_samples, collect more data, or "
            f"accept thin epochs."
        )
        return fallback

    # by == 'year'
    # Bucket the index by calendar year and run finest-passing per bucket.
    years = dt_index.year
    out: Dict[int, str] = {}
    # Iterate in sorted year order for deterministic warnings + dict order.
    for year in sorted(set(int(y) for y in years)):
        bucket = dt_index[years == year]
        chosen, passed = _pick_finest_passing(bucket, min_samples, candidates)
        if passed:
            out[year] = chosen  # type: ignore[assignment]
        else:
            warnings.warn(
                f"No candidate freq satisfied min_samples={min_samples} "
                f"for year {year}; falling back to 'Y' for that year. "
                f"Either lower min_samples, collect more data for "
                f"{year}, or accept thin epochs."
            )
            out[year] = "Y"
    return out


__all__ = ["suggest_freq"]
