"""v2 PR 2 — mixed-granularity fit-path tests.

Locks down ``freq: str | Mapping[int, str]`` support on
``TemporalShapleyAttribution`` / ``TemporalMarkovAttribution``:

- End-to-end fit with ``freq=suggest_freq(ts, by='year')`` produces
  correctly-sorted heterogeneous epochs (mix of e.g. 'Q' and 'M').
- Volume-weighted attribution sums to 1 under mixed granularity.
- ``detect_threshold_crossings(skip_cross_granularity=...)`` suppresses
  / surfaces change reports across freq-boundary epochs.
- ``_effective_min_samples`` (Markov) runs per-epoch correctly under
  mixed freqs.
- ``fit_by_epoch`` raises ``TypeError`` when ``freq`` is a mapping.
- Lazy global fit succeeds under mixed freqs (concatenation path is
  freq-agnostic).
- ``__repr__`` renders mapping freqs as a sorted dict.
- ``_validate_freq`` rejects bad input at ``__init__``.
- ``_resolve_freq`` raises ``KeyError`` for missing-year configurations.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from do_attribution import (
    TemporalShapleyAttribution,
    TemporalMarkovAttribution,
    suggest_freq,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def mixed_year_data():
    """Two calendar years with very different volumes.

    2024: 4 quarters × ~30 paths/quarter = 120 paths. Each quarter
    spreads its paths across 3 months; ``suggest_freq(by='year')`` will
    pick a reasonably coarse freq for 2024.

    2025: 6 months × ~30 paths/month = 180 paths. ``suggest_freq``
    should pick a finer freq for 2025.

    The resulting freq mapping is heterogeneous, exercising the v2 PR 2
    code path.
    """
    rng = np.random.default_rng(42)
    paths: list = []
    timestamps: list = []
    convs: list = []

    # 2024: 30 paths/quarter spread across the quarter's months.
    quarter_starts_2024 = ["2024-01-15", "2024-04-15", "2024-07-15", "2024-10-15"]
    for qstart in quarter_starts_2024:
        base = pd.Timestamp(qstart)
        for offset in range(30):
            day = base + pd.Timedelta(days=int(offset * 2))
            timestamps.append(day)
            # Path shape: A and B dominate in 2024.
            choice = rng.choice(["A > B", "A", "B > A", "B"], p=[0.4, 0.2, 0.2, 0.2])
            paths.append(str(choice))
            convs.append(int(rng.integers(0, 2)))

    # 2025: 30 paths/month across H1 (6 months).
    for month in range(1, 7):
        base = pd.Timestamp(f"2025-{month:02d}-01")
        for offset in range(30):
            day = base + pd.Timedelta(days=int(offset // 2))
            timestamps.append(day)
            # Channel mix shifts in 2025: B and C dominate.
            choice = rng.choice(["B > C", "C", "B", "C > B"], p=[0.3, 0.3, 0.2, 0.2])
            paths.append(str(choice))
            convs.append(int(rng.integers(0, 2)))

    return paths, timestamps, convs


# --------------------------------------------------------------------------- #
# __init__ validation                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "cls", [TemporalShapleyAttribution, TemporalMarkovAttribution]
)
def test_freq_str_default_is_unchanged(cls):
    model = cls()
    assert model.freq == "Q"
    assert isinstance(model.freq, str)


@pytest.mark.parametrize(
    "cls", [TemporalShapleyAttribution, TemporalMarkovAttribution]
)
def test_freq_mapping_accepted(cls):
    mapping = {2024: "Q", 2025: "M"}
    model = cls(freq=mapping)
    assert model.freq == mapping
    assert model._is_mapping_freq() is True


@pytest.mark.parametrize(
    "cls", [TemporalShapleyAttribution, TemporalMarkovAttribution]
)
def test_freq_invalid_alias_rejected(cls):
    with pytest.raises(ValueError, match="not a valid pandas frequency alias"):
        cls(freq="NOTAFREQ")


@pytest.mark.parametrize(
    "cls", [TemporalShapleyAttribution, TemporalMarkovAttribution]
)
def test_freq_mapping_rejects_non_int_keys(cls):
    with pytest.raises(TypeError, match="mapping keys must be int"):
        cls(freq={"2024": "Q"})


@pytest.mark.parametrize(
    "cls", [TemporalShapleyAttribution, TemporalMarkovAttribution]
)
def test_freq_mapping_rejects_bool_keys(cls):
    # bool is a subclass of int in Python — explicit rejection prevents
    # {True: 'Q'} from sneaking through.
    with pytest.raises(TypeError, match="mapping keys must be int"):
        cls(freq={True: "Q"})


@pytest.mark.parametrize(
    "cls", [TemporalShapleyAttribution, TemporalMarkovAttribution]
)
def test_freq_mapping_rejects_invalid_value_alias(cls):
    with pytest.raises(ValueError, match="not a valid pandas frequency alias"):
        cls(freq={2024: "BOGUS"})


@pytest.mark.parametrize(
    "cls", [TemporalShapleyAttribution, TemporalMarkovAttribution]
)
def test_freq_mapping_rejects_empty(cls):
    with pytest.raises(ValueError, match="mapping must be non-empty"):
        cls(freq={})


@pytest.mark.parametrize(
    "cls", [TemporalShapleyAttribution, TemporalMarkovAttribution]
)
def test_freq_wrong_type_rejected(cls):
    with pytest.raises(TypeError, match="freq must be a str or a Mapping"):
        cls(freq=42)


# --------------------------------------------------------------------------- #
# _resolve_freq                                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "cls", [TemporalShapleyAttribution, TemporalMarkovAttribution]
)
def test_resolve_freq_str_returns_self(cls):
    model = cls(freq="M")
    assert model._resolve_freq(2024) == "M"
    assert model._resolve_freq(2099) == "M"


@pytest.mark.parametrize(
    "cls", [TemporalShapleyAttribution, TemporalMarkovAttribution]
)
def test_resolve_freq_mapping_lookup(cls):
    model = cls(freq={2024: "Q", 2025: "M"})
    assert model._resolve_freq(2024) == "Q"
    assert model._resolve_freq(2025) == "M"


@pytest.mark.parametrize(
    "cls", [TemporalShapleyAttribution, TemporalMarkovAttribution]
)
def test_resolve_freq_missing_year_raises(cls):
    model = cls(freq={2024: "Q", 2025: "M"})
    with pytest.raises(KeyError, match="no entry for year 2026"):
        model._resolve_freq(2026)


# --------------------------------------------------------------------------- #
# End-to-end Shapley fit under mixed granularity                               #
# --------------------------------------------------------------------------- #
def test_shapley_mixed_freq_fit_produces_heterogeneous_epochs(mixed_year_data):
    paths, timestamps, convs = mixed_year_data
    # Force a heterogeneous mapping so we exercise the freq=='Q' for 2024
    # / freq=='M' for 2025 path regardless of what suggest_freq picks.
    mapping = {2024: "Q", 2025: "M"}
    model = TemporalShapleyAttribution(
        freq=mapping, min_samples_per_epoch=5, epoch_assignment="conversion"
    )
    result = model.fit(paths, timestamps, convs)

    # Epochs include both 'Q' Periods (from 2024) and 'M' Periods (2025).
    epoch_freqs = {ep.freqstr for ep in result["epochs"]}
    assert any(f.startswith("Q") for f in epoch_freqs), epoch_freqs
    assert "M" in epoch_freqs, epoch_freqs


def test_shapley_mixed_freq_epochs_sorted_by_wall_clock(mixed_year_data):
    paths, timestamps, convs = mixed_year_data
    model = TemporalShapleyAttribution(
        freq={2024: "Q", 2025: "M"}, min_samples_per_epoch=5
    )
    model.fit(paths, timestamps, convs)
    starts = [ep.start_time for ep in model.epochs]
    # Strictly monotonic increasing (no duplicate epochs).
    assert all(a < b for a, b in zip(starts, starts[1:]))
    # First epoch is in 2024, last in 2025.
    assert model.epochs[0].start_time.year == 2024
    assert model.epochs[-1].start_time.year == 2025


def test_shapley_mixed_freq_volume_weighted_sums_to_one(mixed_year_data):
    paths, timestamps, convs = mixed_year_data
    model = TemporalShapleyAttribution(
        freq={2024: "Q", 2025: "M"}, min_samples_per_epoch=5
    )
    model.fit(paths, timestamps, convs)
    vw = model.get_volume_weighted_attribution()
    assert abs(sum(vw.values()) - 1.0) < 1e-9


def test_shapley_mixed_freq_via_suggest_freq_round_trip():
    # End-to-end: caller hands the suggest_freq output straight to the
    # model. This is the headline UX promise of v2.
    paths: list = []
    timestamps: list = []
    convs: list = []

    # 2024: 8 timestamps per quarter, all on the quarter-start day → 'Q'
    # has 4 bins of 8 each; finer freqs would have 1 bin of 8 (or 4 bins
    # of 8 stuck in a single month each), so 'Q' or 'M' will be picked
    # by suggest_freq depending on the candidate ordering. We assert the
    # mapping has exactly two years.
    for qstart in ["2024-01-15", "2024-04-15", "2024-07-15", "2024-10-15"]:
        base = pd.Timestamp(qstart)
        for _ in range(8):
            timestamps.append(base)
            paths.append("A > B")
            convs.append(1)

    # 2025: 15 timestamps on the 15th of each month → 'M' has 12 bins of
    # 15 each, passes min_samples=5.
    for month in range(1, 13):
        base = pd.Timestamp(f"2025-{month:02d}-15")
        for _ in range(15):
            timestamps.append(base)
            paths.append("B > C")
            convs.append(1)

    freq_mapping = suggest_freq(timestamps, min_samples=5, by="year")
    assert isinstance(freq_mapping, dict)
    assert set(freq_mapping.keys()) == {2024, 2025}

    model = TemporalShapleyAttribution(
        freq=freq_mapping, min_samples_per_epoch=5
    ).fit(paths, timestamps, convs)
    # No exception, and the fit produced epochs across both years.
    epochs = list(model["epochs"])
    assert any(ep.start_time.year == 2024 for ep in epochs)
    assert any(ep.start_time.year == 2025 for ep in epochs)


# --------------------------------------------------------------------------- #
# End-to-end Markov fit under mixed granularity                                #
# --------------------------------------------------------------------------- #
def test_markov_mixed_freq_fit_produces_heterogeneous_epochs(mixed_year_data):
    paths, timestamps, convs = mixed_year_data
    model = TemporalMarkovAttribution(
        freq={2024: "Q", 2025: "M"},
        min_samples_per_epoch=5,
        epoch_assignment="first_touch",
    )
    result = model.fit(paths, timestamps, convs)
    epoch_freqs = {ep.freqstr for ep in result["epochs"]}
    assert any(f.startswith("Q") for f in epoch_freqs), epoch_freqs
    assert "M" in epoch_freqs, epoch_freqs


def test_markov_mixed_freq_volume_weighted_sums_to_one(mixed_year_data):
    paths, timestamps, convs = mixed_year_data
    model = TemporalMarkovAttribution(
        freq={2024: "Q", 2025: "M"}, min_samples_per_epoch=5
    )
    model.fit(paths, timestamps, convs)
    vw = model.get_volume_weighted_attribution()
    assert abs(sum(vw.values()) - 1.0) < 1e-9


def test_markov_effective_min_samples_runs_per_epoch_under_mixed_freq(
    mixed_year_data,
):
    """The strategy/order-aware floor (Risk 2) must run per-epoch under
    mixed granularity. Each epoch may have a different channel
    cardinality, so the floor varies; the get_epoch_summary view should
    reflect that without crashing."""
    paths, timestamps, convs = mixed_year_data
    model = TemporalMarkovAttribution(
        freq={2024: "Q", 2025: "M"},
        min_samples_per_epoch=5,
        order=1,
        removal_strategy="fail",
    )
    model.fit(paths, timestamps, convs)
    summary = model.get_epoch_summary()
    # effective_min_samples column populated for every epoch and is >= base.
    assert (summary["effective_min_samples"] >= 5).all()
    assert summary.shape[0] == len(model.epochs)


# --------------------------------------------------------------------------- #
# Cross-granularity boundary in detect_threshold_crossings                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "cls", [TemporalShapleyAttribution, TemporalMarkovAttribution]
)
def test_detect_threshold_crossings_uses_strict_descriptive_threshold(
    monkeypatch, cls
):
    model = cls()
    trend = pd.Series(
        [0.0, 0.25, 0.51, np.nan, 1.0],
        index=["epoch-1", "epoch-2", "epoch-3", "epoch-4", "epoch-5"],
        name="Email",
    )
    monkeypatch.setattr(model, "get_trend", lambda channel: trend)

    changes = model.detect_threshold_crossings("Email", threshold=0.25)

    assert [(previous, current) for previous, current, _ in changes] == [
        ("epoch-2", "epoch-3")
    ]
    assert changes[0][2] == pytest.approx(0.26)


def test_shapley_significant_changes_alias_warns_and_preserves_arguments(
    monkeypatch,
):
    model = TemporalShapleyAttribution()
    expected = [("epoch-1", "epoch-2", 0.3)]
    received = {}

    def _detect_threshold_crossings(**kwargs):
        received.update(kwargs)
        return expected

    monkeypatch.setattr(
        model, "detect_threshold_crossings", _detect_threshold_crossings
    )

    with pytest.warns(DeprecationWarning, match="descriptive threshold"):
        actual = model.detect_significant_changes(
            "Email", threshold=0.2, skip_cross_granularity=False
        )

    assert actual is expected
    assert received == {
        "channel": "Email",
        "threshold": 0.2,
        "skip_cross_granularity": False,
    }


def test_markov_significant_changes_alias_warns_and_preserves_arguments(
    monkeypatch,
):
    model = TemporalMarkovAttribution()
    expected = [("epoch-1", "epoch-2", -0.3)]
    received = {}

    def _detect_threshold_crossings(**kwargs):
        received.update(kwargs)
        return expected

    monkeypatch.setattr(
        model, "detect_threshold_crossings", _detect_threshold_crossings
    )

    with pytest.warns(DeprecationWarning, match="descriptive threshold"):
        actual = model.detect_significant_changes(
            "Email",
            threshold=0.2,
            on="removal_effect",
            skip_cross_granularity=False,
        )

    assert actual is expected
    assert received == {
        "channel": "Email",
        "threshold": 0.2,
        "on": "removal_effect",
        "skip_cross_granularity": False,
    }


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (TemporalShapleyAttribution, dict(epoch_assignment="conversion")),
        (TemporalMarkovAttribution, dict(epoch_assignment="first_touch")),
    ],
)
def test_detect_threshold_crossings_skips_cross_freq_by_default(
    mixed_year_data, cls, kwargs
):
    """Default skip_cross_granularity=True suppresses the 2024Q-->2025M
    boundary report. Setting it False surfaces the (mechanical) step."""
    paths, timestamps, convs = mixed_year_data
    model = cls(freq={2024: "Q", 2025: "M"}, min_samples_per_epoch=5, **kwargs)
    model.fit(paths, timestamps, convs)

    # Channel B appears in both years (2024 paths use 'A > B', 2025
    # paths use 'B > C'), so its trend has non-NaN values on both sides
    # of the cross-freq boundary and the threshold check is meaningful.
    assert "B" in model.channels, model.channels

    skipped = model.detect_threshold_crossings(
        "B", threshold=0.0, skip_cross_granularity=True
    )
    surfaced = model.detect_threshold_crossings(
        "B", threshold=0.0, skip_cross_granularity=False
    )

    # surfaced must be a superset of skipped — the only difference is
    # cross-freq pairs.
    assert len(surfaced) >= len(skipped)
    cross_pairs_surfaced = [
        (a, b) for (a, b, _) in surfaced if a.freqstr != b.freqstr
    ]
    cross_pairs_skipped = [
        (a, b) for (a, b, _) in skipped if a.freqstr != b.freqstr
    ]
    assert len(cross_pairs_skipped) == 0
    # On this fixture there's exactly one cross-freq pair (Q4 2024 -> M
    # Jan 2025), and channel B's attribution differs by more than 0
    # across that boundary (different path mixes by construction).
    assert len(cross_pairs_surfaced) >= 1


# --------------------------------------------------------------------------- #
# fit_by_epoch with mapping freq                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "cls", [TemporalShapleyAttribution, TemporalMarkovAttribution]
)
def test_fit_by_epoch_rejects_mapping_freq(cls):
    model = cls(freq={2024: "Q", 2025: "M"})
    with pytest.raises(TypeError, match="fit_by_epoch does not support"):
        model.fit_by_epoch(
            paths=["A > B", "B > C"],
            epochs=[2024, 2025],
            conversions=[1, 1],
        )


# --------------------------------------------------------------------------- #
# Lazy global fit under mixed granularity                                      #
# --------------------------------------------------------------------------- #
def test_shapley_lazy_global_fit_succeeds_under_mixed_freq(mixed_year_data):
    paths, timestamps, convs = mixed_year_data
    model = TemporalShapleyAttribution(
        freq={2024: "Q", 2025: "M"}, min_samples_per_epoch=5
    )
    model.fit(paths, timestamps, convs)
    # Triggers the lazy global fit. The static fit consumes raw paths
    # (freq-agnostic concatenation), so mixed freqs must not affect it.
    # This deterministic fixture produces near-cancelling signed global
    # values, so the fit succeeds but normalized shares are unavailable.
    global_attr = model.global_attribution
    assert global_attr is None
    assert model.global_normalization_status == "unavailable"
    assert model.global_normalization_reason == "near_cancellation"
    assert isinstance(model.global_shapley_values, dict)
    assert len(model.global_shapley_values) > 0


def test_markov_lazy_global_fit_succeeds_under_mixed_freq(mixed_year_data):
    paths, timestamps, convs = mixed_year_data
    model = TemporalMarkovAttribution(
        freq={2024: "Q", 2025: "M"}, min_samples_per_epoch=5
    )
    model.fit(paths, timestamps, convs)
    global_attr = model.global_attribution
    assert isinstance(global_attr, dict)
    # Markov's global fit can return an empty dict on degenerate data;
    # the contract is "the call doesn't crash under mixed freqs", not
    # "global_attr is non-empty".
    # If the data is non-degenerate, the result should at least cover
    # the channels seen in per-epoch fits.
    if global_attr:
        for ch in global_attr:
            assert ch in model.channels


# --------------------------------------------------------------------------- #
# __repr__                                                                     #
# --------------------------------------------------------------------------- #
def test_shapley_repr_renders_mapping_freq():
    model = TemporalShapleyAttribution(freq={2025: "M", 2024: "Q"})
    r = repr(model)
    # Sorted by year → 2024 comes before 2025.
    assert "freq={2024: 'Q', 2025: 'M'}" in r


def test_markov_repr_renders_mapping_freq():
    model = TemporalMarkovAttribution(freq={2025: "M", 2024: "Q"})
    r = repr(model)
    assert "freq={2024: 'Q', 2025: 'M'}" in r


def test_shapley_repr_unchanged_for_str_freq():
    model = TemporalShapleyAttribution(freq="Q")
    r = repr(model)
    assert "freq='Q'" in r


# --------------------------------------------------------------------------- #
# v1 regression — str-freq fit path still produces the same epoch list         #
# --------------------------------------------------------------------------- #
def test_str_freq_fit_path_unchanged_after_v2_refactor():
    """The per-year bucketing refactor under v2 PR 2 must not change the
    str-freq behaviour — under a single freq every year is binned with
    the same freq, so the assembled epoch list is identical to the v1
    code path."""
    rng = np.random.default_rng(7)
    paths = [str(rng.choice(["A > B", "B > C", "A"])) for _ in range(120)]
    timestamps = pd.date_range("2024-01-01", periods=120, freq="D")
    convs = [int(rng.integers(0, 2)) for _ in range(120)]

    model = TemporalShapleyAttribution(freq="M", min_samples_per_epoch=5)
    res = model.fit(paths, timestamps, convs)
    # 120 days @ daily start covers ~Jan-Apr 2024 → 4-5 monthly periods.
    # All epochs should be 'M' freq.
    assert all(ep.freqstr == "M" for ep in res["epochs"])
    # And in chronological order.
    starts = [ep.start_time for ep in res["epochs"]]
    assert all(a < b for a, b in zip(starts, starts[1:]))
