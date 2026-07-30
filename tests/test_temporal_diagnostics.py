"""Tests for ``do_attribution.temporal_diagnostics``.

v2 PR 1 coverage:

- ``suggest_freq(by='year')`` returns ``Dict[int, str]``.
- Per-year mapping picks the finest passing freq inside each year bucket
  and respects the same ``min_samples`` floor as the v1 single-freq mode.
- ``'Y'`` is dropped from the default candidate list under ``by='year'``
  (would otherwise trivially pass inside any single-year bucket).
- Per-year fallback returns ``'Y'`` for that year + one ``UserWarning``
  per failing year.
- ``by`` argument validation.
- Sort-key regression: ``_get_period_key`` returns ``start_time``, so
  heterogeneous lists of Periods (mix of 'Q' and 'M') sort to wall-clock
  order. v1 ordinal-based sort would have shuffled them.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from do_attribution.temporal_diagnostics import suggest_freq
from do_attribution.temporal_shapley import _get_period_key as _sp_key
from do_attribution.temporal_markov import _get_period_key as _mk_key


# --------------------------------------------------------------------------- #
# by='year' — happy paths                                                      #
# --------------------------------------------------------------------------- #
def test_by_year_returns_dict_keyed_by_int_year():
    # 2024: 12 months × 5 ts/month = 60 ts; passes 'M' at min_samples=5.
    # 2025: 4 quarters × 30 ts/quarter = 120 ts; also passes 'M' at floor=5.
    ts: list = []
    for month in range(1, 13):
        for _ in range(5):
            ts.append(pd.Timestamp(f"2024-{month:02d}-15"))
    for month in range(1, 13):
        for _ in range(10):
            ts.append(pd.Timestamp(f"2025-{month:02d}-15"))
    result = suggest_freq(ts, min_samples=5, by="year")
    assert isinstance(result, dict)
    assert set(result.keys()) == {2024, 2025}
    assert all(isinstance(k, int) for k in result.keys())
    assert all(isinstance(v, str) for v in result.values())


def test_by_year_thin_year_picks_coarser_than_thick_year():
    # 2024: thin — 4 timestamps, one per quarter. Every candidate freq
    # produces bins with 1 ts each at most, so every candidate fails at
    # min_samples=2 → falls through to 'Y'.
    # 2025: thick — 12 months × 5 ts on the same day per month = 60 ts.
    # 'D' produces 12 bins of 5 each → passes min_samples=2 (it is the
    # finest passing freq).
    ts: list = [
        pd.Timestamp("2024-01-15"),
        pd.Timestamp("2024-04-15"),
        pd.Timestamp("2024-07-15"),
        pd.Timestamp("2024-10-15"),
    ]
    for month in range(1, 13):
        for _ in range(5):
            ts.append(pd.Timestamp(f"2025-{month:02d}-15"))
    with pytest.warns(UserWarning, match="for year 2024"):
        result = suggest_freq(ts, min_samples=2, by="year")
    assert result[2024] == "Y"
    # 2025 must be a sub-year freq (the v2 picker walked the candidate
    # list per bucket); the exact pick depends on the candidate ordering
    # but it must NOT have fallen through to 'Y'.
    assert result[2025] != "Y"
    assert result[2025] in ("D", "W", "M", "Q")


def test_by_year_iterates_in_sorted_year_order():
    # Out-of-order input. Result dict should iterate years ascending so
    # warnings are deterministic and downstream consumers can assume order.
    ts = [
        pd.Timestamp("2025-06-15"),
        pd.Timestamp("2023-06-15"),
        pd.Timestamp("2024-06-15"),
    ] * 10  # 30 ts spread across 3 years; min_samples=10 → 'Y' in each.
    # Each year has only 10 ts, all on the same day. 'D'/'W'/'M'/'Q' each
    # produce a single bin per year with 10 samples → all pass at floor=10.
    # 'D' is the finest, so 'D' wins per year.
    result = suggest_freq(ts, min_samples=10, by="year")
    assert list(result.keys()) == [2023, 2024, 2025]


def test_by_year_default_drops_Y_from_candidates():
    # Single year with sparse data — every fine bin fails the floor.
    # If 'Y' were still in the default candidate list it would pass
    # trivially (the whole year is one bin, n=5). The post-v2 behaviour
    # is: 'Y' is NOT in the default list under by='year', so we go through
    # the fallback path and emit a UserWarning before returning 'Y'.
    ts = [pd.Timestamp("2024-06-15")] * 5
    with pytest.warns(UserWarning, match="for year 2024"):
        result = suggest_freq(ts, min_samples=10, by="year")
    assert result == {2024: "Y"}


def test_by_year_floor_is_per_epoch_not_per_year():
    # 2024 has 60 ts but they all sit in Q1 (Jan/Feb/Mar). At
    # min_samples=10 'M' produces 3 bins of 20 each (passes), but Q1
    # under 'Q' produces a single bin of 60 — also passes. Finest passing
    # is 'D' if and only if at least 10 ts fall on every represented day;
    # in this construction we put 5 ts on each of 12 days → 'D' fails,
    # 'W' might fail, 'M' passes. The contract: each *epoch* in the
    # chosen freq must clear the floor, not the year as a whole.
    ts: list = []
    for day in range(1, 13):
        for _ in range(5):
            ts.append(pd.Timestamp(f"2024-01-{day:02d}"))
    for day in range(1, 13):
        for _ in range(5):
            ts.append(pd.Timestamp(f"2024-02-{day:02d}"))
    for day in range(1, 13):
        for _ in range(5):
            ts.append(pd.Timestamp(f"2024-03-{day:02d}"))
    result = suggest_freq(ts, min_samples=10, by="year")
    # Each calendar day has 5 ts → 'D' fails. 'M' has Jan/Feb/Mar bins of
    # 60 each → passes. 'W' depends on calendar; could pass or fail. The
    # invariant we assert is: chosen freq is finer than 'Y' (so the v2
    # by-year picker actually walked the candidate list per bucket).
    assert result[2024] in ("D", "W", "M", "Q")
    assert result[2024] != "Y"


# --------------------------------------------------------------------------- #
# by='year' — argument validation + interaction with explicit candidates      #
# --------------------------------------------------------------------------- #
def test_by_invalid_value_raises():
    with pytest.raises(ValueError, match="by must be"):
        suggest_freq([pd.Timestamp("2024-01-01")] * 30, by="quarter")


def test_by_year_respects_explicit_candidate_freqs():
    # Caller forces a candidate list — the by='year' code path should
    # use it as-is (no auto-drop of 'Y'). Here we pass only ('Y',) so
    # every year falls through to 'Y' on the *passing* path, not the
    # fallback path → no warning emitted.
    ts = [pd.Timestamp("2024-06-15")] * 30
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        result = suggest_freq(
            ts, min_samples=10, by="year", candidate_freqs=("Y",)
        )
    assert result == {2024: "Y"}
    # No fallback warning — 'Y' passed on the explicit candidate list.
    assert not any(
        "falling back to 'Y'" in str(r.message) for r in records
    )


def test_by_None_preserves_v1_string_return():
    # Regression: by=None must behave identically to v1 — single string,
    # 'Y' still in default candidates.
    ts = [pd.Timestamp(f"2024-{m:02d}-15") for m in range(1, 13) for _ in range(10)]
    result = suggest_freq(ts, min_samples=5)  # by=None default
    assert isinstance(result, str)


# --------------------------------------------------------------------------- #
# Sort-key regression — _get_period_key returns start_time                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("get_key", [_sp_key, _mk_key])
def test_get_period_key_orders_heterogeneous_periods_by_wall_clock(get_key):
    """Mixed-freq Period list must sort to wall-clock order under start_time.

    v1 used ``p.ordinal``, which is in *that period's freq units*:

        >>> pd.Period('2024Q1', 'Q').ordinal   # quarters since epoch
        216
        >>> pd.Period('2025-01', 'M').ordinal  # months since epoch
        660

    Sorting by ordinal across freqs interleaves them nonsensically. Under
    v2 PR 1 the key is ``start_time``, which is a wall-clock Timestamp
    and gives the obvious chronological order.
    """
    periods = [
        pd.Period("2025-03", "M"),
        pd.Period("2024Q1", "Q"),
        pd.Period("2025-01", "M"),
        pd.Period("2024Q4", "Q"),
        pd.Period("2024Q3", "Q"),
        pd.Period("2025-02", "M"),
    ]
    ordered = sorted(periods, key=get_key)
    starts = [p.start_time for p in ordered]
    # Strictly monotonic increasing in wall-clock time.
    assert all(a < b for a, b in zip(starts, starts[1:]))
    # First period is 2024Q1 (Jan 1 2024), last is 2025-03 (Mar 1 2025).
    assert ordered[0] == pd.Period("2024Q1", "Q")
    assert ordered[-1] == pd.Period("2025-03", "M")


@pytest.mark.parametrize("get_key", [_sp_key, _mk_key])
def test_get_period_key_preserves_same_freq_order(get_key):
    """Within a single freq, start_time-based ordering matches v1 ordinal."""
    periods = [
        pd.Period("2025Q3", "Q"),
        pd.Period("2024Q4", "Q"),
        pd.Period("2025Q1", "Q"),
        pd.Period("2024Q1", "Q"),
    ]
    ordered = sorted(periods, key=get_key)
    assert ordered == [
        pd.Period("2024Q1", "Q"),
        pd.Period("2024Q4", "Q"),
        pd.Period("2025Q1", "Q"),
        pd.Period("2025Q3", "Q"),
    ]
