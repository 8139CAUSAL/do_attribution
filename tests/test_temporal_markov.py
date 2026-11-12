"""Temporal Markov attribution tests.

Locks down the v1 punch-list fixes in temporal_markov.py:

- Risk 1 — explicit epoch_assignment parameter, 'split' NotImplementedError
- Risk 2 — strategy- and order-aware adaptive min_samples floor
- Risk 3 — global vs. volume-weighted divergence
- Risk 4 — per-channel-per-epoch sample-size diagnostic
- Risk 5 — lazy global fit (property triggers exactly once)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from do_attribution import TemporalMarkovAttribution


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def small_temporal_data():
    """Two-quarter dataset large enough to keep (I-Q) well-conditioned."""
    rng = np.random.default_rng(7)
    q1_paths = [
        "A > B", "A > B", "A > C", "B > A", "A > B",
        "A > C", "B > A", "C > B > A", "A > B", "B > A",
    ]
    q1_dates = pd.date_range("2025-01-15", periods=10, freq="D")
    q1_convs = [1, 1, 1, 1, 0, 1, 1, 0, 1, 1]

    q2_paths = [
        "B > C", "C > B", "B > C", "B > C", "C > B",
        "B > C", "C > B", "B > C", "C > B", "B > C",
    ]
    q2_dates = pd.date_range("2025-04-15", periods=10, freq="D")
    q2_convs = [1, 1, 1, 0, 1, 1, 1, 0, 1, 1]

    paths = list(q1_paths) + list(q2_paths)
    dates = list(q1_dates) + list(q2_dates)
    convs = list(q1_convs) + list(q2_convs)
    return paths, dates, convs


def test_detour_near_cancellation_remains_diagnostic_in_temporal_consumers():
    paths = ["A"] * 500 + ["B"] * 500
    dates = pd.to_datetime(["2025-01-15"] * 1000)
    conversions = [1] * 400 + [0] * 100 + [1] * 100 + [0] * 400
    model = TemporalMarkovAttribution(
        freq="Q",
        min_samples_per_epoch=1,
        removal_strategy="detour",
    )

    result = model.fit(paths, dates, conversions)
    epoch = result["epochs"][0]
    epoch_result = result["epoch_attributions"][epoch]

    assert epoch_result["attribution"] is None
    assert epoch_result["normalization_status"] == "unavailable"
    assert epoch_result["normalization_reason"] == "near_cancellation"
    assert epoch_result["removal_effects"] == pytest.approx(
        {"A": -0.6, "B": 0.6000000000000001}
    )
    assert result["channels"] == ["A", "B"]
    assert list(result["attribution_matrix"].columns) == ["A", "B"]
    assert result["attribution_matrix"].loc[epoch].isna().all()
    assert np.isnan(model.get_trend("A").loc[epoch])
    assert model.get_volume_weighted_attribution() == {}

    summary = model.summary()
    assert "Normalized attribution unavailable by epoch" in summary
    assert "near_cancellation" in summary
    assert "Top removal effects per epoch (raw, signed)" in summary
    assert "540431955284459584.00%" not in summary

    assert model.global_attribution is None
    assert model.global_normalization_status == "unavailable"
    assert model.global_normalization_reason == "near_cancellation"
    assert model.global_removal_effects == pytest.approx(
        {"A": -0.6, "B": 0.6000000000000001}
    )


@pytest.mark.parametrize(
    "removal_strategy", ["fail", "fail_refit", "detour", "redirect_to_null"]
)
@pytest.mark.parametrize("fit_mode", ["calendar", "labels"])
def test_zero_conversion_epoch_rows_are_missing_not_equal_shares(
    removal_strategy, fit_mode
):
    # Q1 never converts; Q2 has an effect to normalize under every strategy.
    paths = ["A", "B", "A > B", "A", "B"]
    dates = pd.to_datetime(
        ["2025-01-10", "2025-01-11", "2025-04-10", "2025-04-11", "2025-04-12"]
    )
    conversions = [0, 0, 1, 0, 0]
    model = TemporalMarkovAttribution(
        freq="Q",
        min_samples_per_epoch=1,
        removal_strategy=removal_strategy,
    )

    if fit_mode == "calendar":
        result = model.fit(paths, dates, conversions)
    else:
        result = model.fit_by_epoch(paths, dates, conversions)
    zero_epoch, converting_epoch = result["epochs"]
    zero_result = result["epoch_attributions"][zero_epoch]

    assert zero_result["attribution"] is None
    assert zero_result["normalization_status"] == "unavailable"
    assert zero_result["normalization_reason"] == "zero_raw_magnitude"
    assert zero_result["removal_effects"] == {"A": 0.0, "B": 0.0}
    assert zero_result["total_conversion_probability"] == 0.0

    matrix = result["attribution_matrix"]
    assert list(matrix.columns) == ["A", "B"]
    assert matrix.loc[zero_epoch].isna().all()
    assert matrix.loc[converting_epoch].notna().all()
    assert matrix.loc[converting_epoch].sum() == pytest.approx(1.0)
    assert np.isnan(model.get_trend("A").loc[zero_epoch])
    assert (result["removal_effect_matrix"].loc[zero_epoch] == 0.0).all()

    exported = model.to_dataframes()["attribution"]
    zero_rows = exported[exported["epoch"] == str(zero_epoch)]
    assert list(zero_rows["channel"]) == ["A", "B"]
    assert zero_rows["attribution"].isna().all()
    assert exported.loc[
        exported["epoch"] == str(converting_epoch), "attribution"
    ].sum() == pytest.approx(1.0)

    summary = model.summary()
    assert "Normalized attribution unavailable by epoch" in summary
    assert "zero_raw_magnitude" in summary


# --------------------------------------------------------------------------- #
# Risk 1 — epoch_assignment validation                                         #
# --------------------------------------------------------------------------- #
def test_epoch_assignment_default_is_first_touch():
    model = TemporalMarkovAttribution()
    assert model.epoch_assignment == "first_touch"


def test_epoch_assignment_conversion_accepted():
    model = TemporalMarkovAttribution(epoch_assignment="conversion")
    assert model.epoch_assignment == "conversion"


def test_epoch_assignment_invalid_raises():
    with pytest.raises(ValueError, match="epoch_assignment"):
        TemporalMarkovAttribution(epoch_assignment="bogus")


def test_epoch_assignment_split_raises_at_fit(small_temporal_data):
    paths, dates, convs = small_temporal_data
    model = TemporalMarkovAttribution(
        freq="Q", min_samples_per_epoch=5, epoch_assignment="split"
    )
    with pytest.raises(NotImplementedError, match="split"):
        model.fit(paths, dates, convs)


def test_epoch_assignment_appears_in_repr():
    model = TemporalMarkovAttribution(epoch_assignment="conversion")
    assert "epoch_assignment='conversion'" in repr(model)


# --------------------------------------------------------------------------- #
# Risk 2 — adaptive min_samples floor                                          #
# --------------------------------------------------------------------------- #
def test_effective_min_samples_scales_with_n_channels_under_fail():
    """Under removal_strategy='fail' the floor should scale with the
    per-epoch channel count (each channel triggers a refit)."""
    model = TemporalMarkovAttribution(
        min_samples_per_epoch=10,
        removal_strategy="fail",
        order=1,
    )
    paths_1ch = np.array(["A > A", "A"])
    paths_3ch = np.array(["A > B", "B > C", "A > C"])
    f1 = model._effective_min_samples(paths_1ch)
    f3 = model._effective_min_samples(paths_3ch)
    # base * order_mult(=1) * strategy_mult(=n_channels)
    assert f1 == 10 * 1
    assert f3 == 10 * 3
    assert f3 > f1


def test_effective_min_samples_redirect_to_null_does_not_scale():
    """redirect_to_null is closed-form; no per-channel refits, no extra
    multiplier."""
    model = TemporalMarkovAttribution(
        min_samples_per_epoch=10,
        removal_strategy="redirect_to_null",
        order=1,
    )
    paths_3ch = np.array(["A > B", "B > C", "A > C"])
    assert model._effective_min_samples(paths_3ch) == 10


def test_effective_min_samples_scales_with_order():
    """Higher order should scale the floor linearly (1 + (order - 1))."""
    paths_2ch = np.array(["A > B", "B > A"])
    base_kwargs = dict(
        min_samples_per_epoch=10,
        removal_strategy="redirect_to_null",  # isolate the order axis
    )
    f1 = TemporalMarkovAttribution(order=1, **base_kwargs)._effective_min_samples(
        paths_2ch
    )
    f2 = TemporalMarkovAttribution(order=2, **base_kwargs)._effective_min_samples(
        paths_2ch
    )
    f3 = TemporalMarkovAttribution(order=3, **base_kwargs)._effective_min_samples(
        paths_2ch
    )
    assert f1 == 10 * 1
    assert f2 == 10 * 2
    assert f3 == 10 * 3


def test_effective_min_samples_surfaced_in_summary(small_temporal_data):
    paths, dates, convs = small_temporal_data
    model = TemporalMarkovAttribution(
        freq="Q",
        min_samples_per_epoch=5,
        removal_strategy="fail",
        order=1,
    )
    model.fit(paths, dates, convs)
    summary = model.get_epoch_summary()
    assert "effective_min_samples" in summary.columns
    # Strategy='fail' under order=1 -> floor scales as base * n_channels.
    # In the fixture Q1 has channels {A, B, C} (3) -> 5 * 3 = 15;
    # Q2 has channels {B, C} (2) -> 5 * 2 = 10. Verify the per-epoch
    # adaptive floor strictly exceeds the flat base in at least one
    # epoch (the whole point of Risk 2).
    base = model.min_samples
    assert (summary["effective_min_samples"] > base).any(), (
        f"effective_min_samples should exceed base ({base}) in at least "
        f"one epoch under removal_strategy='fail'; got {summary['effective_min_samples'].to_dict()}"
    )
    # And the Q1 floor should be exactly base * 3 (3 channels).
    assert summary.loc["2025Q1", "effective_min_samples"] == 15


# --------------------------------------------------------------------------- #
# Risk 3 — static-on-temporal contract + volume-weighted helper                #
# --------------------------------------------------------------------------- #
def test_volume_weighted_attribution_sums_to_one(small_temporal_data):
    paths, dates, convs = small_temporal_data
    model = TemporalMarkovAttribution(
        freq="Q",
        min_samples_per_epoch=5,
        removal_strategy="redirect_to_null",
    )
    model.fit(paths, dates, convs)
    vw = model.get_volume_weighted_attribution()
    assert sum(vw.values()) == pytest.approx(1.0, abs=1e-9)


def test_attribution_matrix_include_global_appends_two_rows(small_temporal_data):
    paths, dates, convs = small_temporal_data
    model = TemporalMarkovAttribution(
        freq="Q",
        min_samples_per_epoch=5,
        removal_strategy="redirect_to_null",
    )
    model.fit(paths, dates, convs)

    plain = model.get_attribution_matrix()
    augmented = model.get_attribution_matrix(include_global=True)
    assert len(augmented.index) == len(plain.index) + 2
    idx_strs = augmented.index.astype(str).tolist()
    assert "Global" in idx_strs
    assert "VolumeWeighted" in idx_strs


# --------------------------------------------------------------------------- #
# Risk 4 — per-channel-per-epoch sample-size diagnostic                        #
# --------------------------------------------------------------------------- #
def test_channel_epoch_sample_sizes_shape(small_temporal_data):
    paths, dates, convs = small_temporal_data
    model = TemporalMarkovAttribution(
        freq="Q",
        min_samples_per_epoch=5,
        removal_strategy="redirect_to_null",
    )
    model.fit(paths, dates, convs)

    sizes = model.get_channel_epoch_sample_sizes(min_per_channel_paths=4)
    assert list(sizes.index.names) == ["channel", "epoch"]
    for col in (
        "n_paths_containing_channel",
        "n_conversions_on_paths_containing_channel",
        "thin",
    ):
        assert col in sizes.columns


def test_thin_threshold_flips_thin_flag(small_temporal_data):
    paths, dates, convs = small_temporal_data
    model = TemporalMarkovAttribution(
        freq="Q",
        min_samples_per_epoch=5,
        removal_strategy="redirect_to_null",
    )
    model.fit(paths, dates, convs)

    # With min_per_channel_paths=1, no cell is thin (every channel
    # appearing in an epoch has >= 1 path).
    low = model.get_channel_epoch_sample_sizes(min_per_channel_paths=1)
    assert (~low["thin"]).all()

    # With an absurdly high floor every cell is thin.
    high = model.get_channel_epoch_sample_sizes(min_per_channel_paths=10_000)
    assert high["thin"].all()


# --------------------------------------------------------------------------- #
# Risk 5 — lazy global fit                                                     #
# --------------------------------------------------------------------------- #
def test_global_fit_not_done_immediately_after_fit(small_temporal_data):
    paths, dates, convs = small_temporal_data
    model = TemporalMarkovAttribution(
        freq="Q",
        min_samples_per_epoch=5,
        removal_strategy="redirect_to_null",
    )
    model.fit(paths, dates, convs)
    assert model._global_fit_done is False
    assert model._global_attribution_cache == {}


def test_global_attribution_property_triggers_fit_once(small_temporal_data):
    paths, dates, convs = small_temporal_data
    model = TemporalMarkovAttribution(
        freq="Q",
        min_samples_per_epoch=5,
        removal_strategy="redirect_to_null",
    )
    model.fit(paths, dates, convs)

    g1 = model.global_attribution
    assert model._global_fit_done is True
    assert g1

    cache_id_before = id(model._global_attribution_cache)
    g2 = model.global_attribution
    assert g1 == g2
    assert id(model._global_attribution_cache) == cache_id_before


def test_fit_result_does_not_eagerly_include_global(small_temporal_data):
    paths, dates, convs = small_temporal_data
    model = TemporalMarkovAttribution(
        freq="Q",
        min_samples_per_epoch=5,
        removal_strategy="redirect_to_null",
    )
    result = model.fit(paths, dates, convs)
    assert "global_attribution" not in result
