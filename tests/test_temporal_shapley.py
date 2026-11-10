"""Temporal Shapley attribution tests.

Locks down the v1 punch-list fixes in temporal_shapley.py:

- Risk 1 — explicit epoch_assignment parameter, 'split' NotImplementedError
- Risk 3 — global vs. volume-weighted divergence + include_global rows
- Risk 4 — per-channel-per-epoch sample-size diagnostic
- Risk 5 — lazy global fit (property triggers exactly once)
- Smoke test for the suggest_freq diagnostic
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from do_attribution import TemporalShapleyAttribution
from do_attribution.temporal_diagnostics import suggest_freq


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def small_temporal_data():
    """Two-quarter dataset; >= min_samples=5 in each quarter.

    Q1 paths emphasise A and B; Q2 paths emphasise B and C. Channels A
    and C therefore enter / exit, so the static-fit and volume-weighted
    answers should diverge by a non-trivial amount.
    """
    rng = np.random.default_rng(42)
    q1_paths = [
        "A > B", "A > B", "A", "B > A", "A > B",
        "A > B", "A", "B > A", "A > B", "A",
    ]
    q1_dates = pd.date_range("2025-01-15", periods=10, freq="D")
    q1_convs = [1, 1, 0, 1, 1, 0, 1, 1, 0, 1]

    q2_paths = [
        "B > C", "C > B", "C", "B > C", "C",
        "B > C", "C > B", "B > C", "C", "C > B",
    ]
    q2_dates = pd.date_range("2025-04-15", periods=10, freq="D")
    q2_convs = [1, 1, 0, 1, 1, 1, 0, 1, 1, 0]

    paths = list(q1_paths) + list(q2_paths)
    dates = list(q1_dates) + list(q2_dates)
    convs = list(q1_convs) + list(q2_convs)
    return paths, dates, convs


# --------------------------------------------------------------------------- #
# Risk 1 — epoch_assignment validation                                         #
# --------------------------------------------------------------------------- #
def test_epoch_assignment_default_is_conversion():
    model = TemporalShapleyAttribution()
    assert model.epoch_assignment == "conversion"


def test_epoch_assignment_first_touch_accepted():
    model = TemporalShapleyAttribution(epoch_assignment="first_touch")
    assert model.epoch_assignment == "first_touch"


def test_epoch_assignment_invalid_raises():
    with pytest.raises(ValueError, match="epoch_assignment"):
        TemporalShapleyAttribution(epoch_assignment="invalid_choice")


def test_epoch_assignment_split_raises_at_fit(small_temporal_data):
    paths, dates, convs = small_temporal_data
    model = TemporalShapleyAttribution(
        freq="Q", min_samples_per_epoch=5, epoch_assignment="split"
    )
    with pytest.raises(NotImplementedError, match="split"):
        model.fit(paths, dates, convs)


def test_epoch_assignment_appears_in_repr():
    model = TemporalShapleyAttribution(epoch_assignment="first_touch")
    assert "epoch_assignment='first_touch'" in repr(model)


# --------------------------------------------------------------------------- #
# Risk 3 — static-on-temporal contract + volume-weighted helper                #
# --------------------------------------------------------------------------- #
def test_volume_weighted_attribution_sums_to_one(small_temporal_data):
    paths, dates, convs = small_temporal_data
    model = TemporalShapleyAttribution(freq="Q", min_samples_per_epoch=5)
    model.fit(paths, dates, convs)

    vw = model.get_volume_weighted_attribution()
    assert sum(vw.values()) == pytest.approx(1.0, abs=1e-9)


def test_global_and_volume_weighted_diverge_when_channels_enter_exit(
    small_temporal_data,
):
    """The static global fit and the volume-weighted mean must not agree.

    MODIFIED (get-well Unit 1): under the legacy ``"exact"`` game, channels
    that enter and exit make the divergence total rather than numerical.
    Because that game's v(S) counts paths whose unique channel set is
    exactly S, a dataset where no single path touches
    every channel forces v(N) = 0, the global raw values cancel, and no
    global share vector exists at all.  Each two-channel epoch is still
    well-conditioned, so the volume-weighted view remains available.

    That contrast is the strongest possible statement of Risk 3: the two
    views are not interchangeable, and here only one of them is even
    defined.
    """
    paths, dates, convs = small_temporal_data
    model = TemporalShapleyAttribution(
        freq="Q", min_samples_per_epoch=5, characteristic_function="exact"
    )
    model.fit(paths, dates, convs)

    g = model.global_attribution
    vw = model.get_volume_weighted_attribution()

    # The global static fit cannot produce shares for this shape of data.
    assert g is None
    assert model.global_normalization_status == "unavailable"
    assert model.global_normalization_reason == "near_cancellation"

    # Its raw signed values remain available as diagnostics.
    raw = model.global_shapley_values
    assert set(raw) == {"A", "B", "C"}
    assert any(value < 0 for value in raw.values())

    # The volume-weighted per-epoch view is available and includes A.
    assert "A" in vw
    assert sum(vw.values()) == pytest.approx(1.0, abs=1e-9)


def test_default_game_global_equals_volume_weighted_view(small_temporal_data):
    """Under the default cumulative game the two aggregate views coincide.

    Cumulative Shapley values are additive over disjoint sets of paths: each
    path splits its conversion among its own channels.  The static global
    fit therefore equals the per-epoch values summed over epochs, and its
    shares equal the conversion-weighted mean of the per-epoch shares — even
    though A and C enter and exit between the two quarters.
    """
    paths, dates, convs = small_temporal_data
    model = TemporalShapleyAttribution(freq="Q", min_samples_per_epoch=5)
    model.fit(paths, dates, convs)

    assert model.characteristic_function == "cumulative"
    assert model.global_normalization_status == "available"
    assert model.global_normalization_reason is None

    global_shares = model.global_attribution
    assert set(global_shares) == {"A", "B", "C"}
    assert sum(global_shares.values()) == pytest.approx(1.0, abs=1e-12)
    assert global_shares == pytest.approx(model.get_volume_weighted_attribution())

    raw = model.global_shapley_values
    assert all(value >= 0.0 for value in raw.values())
    assert sum(raw.values()) == pytest.approx(sum(convs))
    summed_over_epochs = model.get_shapley_values_matrix().fillna(0.0).sum()
    assert raw == pytest.approx(summed_over_epochs.to_dict())


def test_characteristic_function_is_applied_to_every_epoch(small_temporal_data):
    paths, dates, convs = small_temporal_data
    default = TemporalShapleyAttribution(freq="Q", min_samples_per_epoch=5)
    default.fit(paths, dates, convs)
    legacy = TemporalShapleyAttribution(
        freq="Q", min_samples_per_epoch=5, characteristic_function="exact"
    )
    legacy.fit(paths, dates, convs)

    assert legacy.characteristic_function == "exact"
    for epoch in default.epochs:
        assert default.epoch_results[epoch]["characteristic_function"] == "cumulative"
        assert legacy.epoch_results[epoch]["characteristic_function"] == "exact"
        # Per-epoch conversions are fully allocated by the default game.
        assert sum(
            default.epoch_results[epoch]["shapley_values"].values()
        ) == pytest.approx(float(default._epoch_conversions[epoch].sum()))


def test_invalid_characteristic_function_rejected_at_construction():
    with pytest.raises(ValueError, match="characteristic_function must be one of"):
        TemporalShapleyAttribution(characteristic_function="subset")


def test_attribution_matrix_include_global_appends_two_rows(
    small_temporal_data,
):
    paths, dates, convs = small_temporal_data
    model = TemporalShapleyAttribution(freq="Q", min_samples_per_epoch=5)
    model.fit(paths, dates, convs)

    plain = model.get_attribution_matrix()
    augmented = model.get_attribution_matrix(include_global=True)
    assert len(augmented.index) == len(plain.index) + 2
    assert "Global" in augmented.index.astype(str).tolist()
    assert "VolumeWeighted" in augmented.index.astype(str).tolist()


# --------------------------------------------------------------------------- #
# Risk 4 — per-channel-per-epoch sample-size diagnostic                        #
# --------------------------------------------------------------------------- #
def test_channel_epoch_sample_sizes_shape_and_thin_flag(small_temporal_data):
    paths, dates, convs = small_temporal_data
    model = TemporalShapleyAttribution(freq="Q", min_samples_per_epoch=5)
    model.fit(paths, dates, convs)

    sizes = model.get_channel_epoch_sample_sizes(min_per_channel_paths=4)
    assert list(sizes.index.names) == ["channel", "epoch"]
    assert "n_paths_containing_channel" in sizes.columns
    assert "n_conversions_on_paths_containing_channel" in sizes.columns
    assert "thin" in sizes.columns
    # In the small fixture B appears in both quarters; A only Q1; C only Q2.
    # B should never be thin under min_per_channel_paths=4 (>= 4 paths
    # containing B in each quarter).
    if "B" in {ch for ch, _ in sizes.index}:
        b_rows = sizes.xs("B", level="channel")
        assert (b_rows["n_paths_containing_channel"] >= 4).all()


def test_epoch_summary_has_n_thin_channels_column(small_temporal_data):
    paths, dates, convs = small_temporal_data
    model = TemporalShapleyAttribution(freq="Q", min_samples_per_epoch=5)
    model.fit(paths, dates, convs)

    summary = model.get_epoch_summary(min_per_channel_paths=999)
    # With an absurdly high floor every channel-epoch is thin.
    assert "n_thin_channels" in summary.columns
    assert (summary["n_thin_channels"] > 0).all()


# --------------------------------------------------------------------------- #
# Risk 5 — lazy global fit                                                     #
# --------------------------------------------------------------------------- #
def test_global_fit_not_done_immediately_after_fit(small_temporal_data):
    paths, dates, convs = small_temporal_data
    model = TemporalShapleyAttribution(freq="Q", min_samples_per_epoch=5)
    model.fit(paths, dates, convs)
    # Lazy: the global fit must not have run yet.
    assert model._global_fit_done is False
    assert model._global_attribution_cache == {}


def test_global_attribution_property_triggers_fit_once(small_temporal_data):
    paths, dates, convs = small_temporal_data
    model = TemporalShapleyAttribution(freq="Q", min_samples_per_epoch=5)
    model.fit(paths, dates, convs)

    # First access: triggers the lazy fit.
    g1 = model.global_attribution
    assert model._global_fit_done is True
    # MODIFIED (get-well Unit 1): this fixture's global fit cancels, so the
    # laziness contract is asserted on the raw values, which are populated
    # in either normalization state.
    assert model.global_shapley_values, (
        "global raw values should be populated after lazy fit"
    )

    # Second access: cache hit, identical contents, _global_fit_done stays True.
    cache_id_before = id(model._global_attribution_cache)
    g2 = model.global_attribution
    assert g1 == g2
    # Same underlying cache object — the lazy compute did not re-run.
    assert id(model._global_attribution_cache) == cache_id_before


def test_fit_result_does_not_eagerly_include_global(small_temporal_data):
    paths, dates, convs = small_temporal_data
    model = TemporalShapleyAttribution(freq="Q", min_samples_per_epoch=5)
    result = model.fit(paths, dates, convs)
    # global_attribution intentionally absent from the eager fit dict
    # (per Risk 5 — pull it from the property when you actually need it).
    assert "global_attribution" not in result


def test_channels_eager_after_fit(small_temporal_data):
    paths, dates, convs = small_temporal_data
    model = TemporalShapleyAttribution(freq="Q", min_samples_per_epoch=5)
    model.fit(paths, dates, convs)
    # channels is set eagerly from per-epoch union, no lazy fit needed.
    assert set(model.channels) >= {"A", "B", "C"}
    assert model._global_fit_done is False


# --------------------------------------------------------------------------- #
# suggest_freq smoke test                                                      #
# --------------------------------------------------------------------------- #
def test_suggest_freq_picks_finest_passing():
    # 100 timestamps spread across 10 distinct calendar days; with
    # min_samples=5 the finest-passing freq should be 'D' (each day has
    # exactly 10 samples). Lower/higher floors trigger different choices.
    base = pd.Timestamp("2025-06-01")
    timestamps = []
    for d in range(10):
        for _ in range(10):
            timestamps.append(base + pd.Timedelta(days=d))
    chosen = suggest_freq(timestamps, min_samples=5)
    assert chosen == "D"


def test_suggest_freq_walks_to_coarser_when_thin():
    # Only 3 timestamps; min_samples=10. 'D'/'W'/'M'/'Q'/'Y' all fail.
    timestamps = [pd.Timestamp("2025-06-01")] * 3
    with pytest.warns(UserWarning, match="No candidate freq"):
        chosen = suggest_freq(timestamps, min_samples=10)
    assert chosen == "Y"  # coarsest fallback


def test_suggest_freq_rejects_empty_timestamps():
    with pytest.raises(ValueError, match="empty"):
        suggest_freq([], min_samples=10)


def test_suggest_freq_rejects_zero_min_samples():
    with pytest.raises(ValueError, match="min_samples"):
        suggest_freq([pd.Timestamp("2025-01-01")], min_samples=0)
