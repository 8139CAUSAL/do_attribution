"""Calendar epochs that straddle a year boundary are fitted exactly once.

A weekly period such as ``2020-12-28/2021-01-03`` holds timestamps from two
calendar years.  Timestamps are binned per calendar year (so a per-year
frequency mapping can mix granularities), which means that period comes out
of both year buckets.  It must be one epoch containing every one of its
rows: fitting it per bucket duplicated the epoch label and kept only the
later year's rows, silently dropping the rest from every result.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from do_attribution import (
    MarkovAttribution,
    ShapleyAttribution,
    TemporalMarkovAttribution,
    TemporalShapleyAttribution,
)
from do_attribution._temporal_helpers import bin_timestamps_to_epochs


STRADDLING_WEEK = "2020-12-28/2021-01-03"
TEMPORAL_MODELS = (TemporalShapleyAttribution, TemporalMarkovAttribution)


@pytest.fixture
def boundary_data():
    """Daily journeys from 2020-12-21 through 2021-01-10 (three ISO weeks)."""
    days = pd.date_range("2020-12-21", "2021-01-10", freq="D")
    patterns = ["A > B", "A", "B", "B > C", "C", "A > C"]
    paths, timestamps, conversions = [], [], []
    for day_number, day in enumerate(days):
        for repeat in range(4):
            index = day_number * 4 + repeat
            paths.append(patterns[index % len(patterns)])
            timestamps.append(day + pd.Timedelta(hours=repeat))
            conversions.append(1 if index % 3 else 0)
    return paths, pd.DatetimeIndex(timestamps), conversions


def _fit(model_class, paths, timestamps, conversions, **kwargs):
    kwargs.setdefault("freq", "W")
    kwargs.setdefault("min_samples_per_epoch", 1)
    model = model_class(**kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(paths, timestamps, conversions)
    return model


# --------------------------------------------------------------------------- #
# Shared binning helper                                                        #
# --------------------------------------------------------------------------- #
def test_helper_merges_a_period_produced_by_two_year_buckets():
    index = pd.DatetimeIndex(
        ["2020-12-27", "2020-12-28", "2020-12-31", "2021-01-01", "2021-01-03", "2021-01-04"]
    )
    epochs, codes = bin_timestamps_to_epochs(index, lambda year: "W")

    assert [str(epoch) for epoch in epochs] == [
        "2020-12-21/2020-12-27",
        STRADDLING_WEEK,
        "2021-01-04/2021-01-10",
    ]
    assert codes.tolist() == [0, 1, 1, 1, 1, 2]


@pytest.mark.parametrize("freq", ["D", "W", "M", "Q", "Y"])
def test_helper_matches_single_pass_binning_for_one_frequency(freq):
    rng = np.random.default_rng(20201231)
    offsets = rng.integers(0, 3 * 365, size=400)
    index = pd.DatetimeIndex(pd.Timestamp("2019-06-01") + pd.to_timedelta(offsets, unit="D"))

    epochs, codes = bin_timestamps_to_epochs(index, lambda year: freq)

    expected = index.to_period(freq)
    assert len(set(epochs)) == len(epochs)
    assert epochs == sorted(expected.unique(), key=lambda period: period.start_time)
    assert [epochs[code] for code in codes] == list(expected)


def test_helper_resolves_each_year_once_in_ascending_order():
    index = pd.DatetimeIndex(["2022-03-01", "2020-05-01", "2021-07-01", "2020-01-02"])
    seen: list[int] = []

    def resolve(year: int) -> str:
        seen.append(year)
        return "M"

    bin_timestamps_to_epochs(index, resolve)
    assert seen == [2020, 2021, 2022]


# --------------------------------------------------------------------------- #
# Both temporal models                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model_class", TEMPORAL_MODELS)
def test_straddling_week_is_one_epoch_holding_every_row(model_class, boundary_data):
    paths, timestamps, conversions = boundary_data
    model = _fit(model_class, paths, timestamps, conversions)

    labels = [str(epoch) for epoch in model.epochs]
    assert labels == [
        "2020-12-21/2020-12-27",
        STRADDLING_WEEK,
        "2021-01-04/2021-01-10",
    ]
    assert len(set(model.epochs)) == len(model.epochs)

    straddling = model.epochs[1]
    # Seven days x four journeys, four days in 2020 and three in 2021.
    assert len(model._epoch_paths[straddling]) == 28
    in_week = (timestamps >= "2020-12-28") & (timestamps < "2021-01-04")
    assert list(model._epoch_paths[straddling]) == list(np.asarray(paths, dtype=object)[in_week])
    assert model._epoch_conversions[straddling].tolist() == list(
        np.asarray(conversions)[in_week]
    )


@pytest.mark.parametrize("model_class", TEMPORAL_MODELS)
def test_every_path_and_conversion_is_assigned_to_exactly_one_epoch(
    model_class, boundary_data
):
    paths, timestamps, conversions = boundary_data
    model = _fit(model_class, paths, timestamps, conversions)

    assert sum(len(rows) for rows in model._epoch_paths.values()) == len(paths)
    assert sum(int(rows.sum()) for rows in model._epoch_conversions.values()) == sum(
        conversions
    )

    summary = model.get_epoch_summary()
    assert summary.index.is_unique
    assert int(summary["n_paths"].sum()) == len(paths)
    assert int(summary["n_conversions"].sum()) == sum(conversions)

    rate = sum(conversions) / len(paths)
    assert (
        f"Overall: {len(paths)} paths, {sum(conversions)} conversions, "
        f"{rate:.2%} conversion rate"
    ) in model.summary()


@pytest.mark.parametrize("model_class", TEMPORAL_MODELS)
def test_export_tables_have_one_row_per_epoch_and_channel(model_class, boundary_data):
    paths, timestamps, conversions = boundary_data
    model = _fit(model_class, paths, timestamps, conversions)

    matrix = model.get_attribution_matrix()
    assert matrix.index.is_unique
    assert len(matrix) == 3

    tables = model.to_dataframes()
    assert not tables["attribution"].duplicated(["epoch", "channel"]).any()
    assert tables["epoch_summary"]["epoch"].is_unique
    assert not tables["channel_epoch_sample_sizes"].duplicated(["epoch", "channel"]).any()


def test_shapley_straddling_epoch_equals_static_fit_on_the_whole_week(boundary_data):
    paths, timestamps, conversions = boundary_data
    model = _fit(TemporalShapleyAttribution, paths, timestamps, conversions)

    in_week = (timestamps >= "2020-12-28") & (timestamps < "2021-01-04")
    week_paths = list(np.asarray(paths, dtype=object)[in_week])
    week_conversions = list(np.asarray(conversions)[in_week])
    static = ShapleyAttribution().fit(week_paths, week_conversions)

    epoch_result = model.epoch_results[model.epochs[1]]
    assert epoch_result["value_function"] == static["value_function"]
    assert epoch_result["shapley_values"] == static["shapley_values"]
    assert epoch_result["attribution"] == static["attribution"]
    # Every conversion of the week is allocated, not just the 2021 part.
    assert sum(epoch_result["shapley_values"].values()) == pytest.approx(
        sum(week_conversions)
    )


def test_markov_straddling_epoch_equals_static_fit_on_the_whole_week(boundary_data):
    paths, timestamps, conversions = boundary_data
    model = _fit(
        TemporalMarkovAttribution,
        paths,
        timestamps,
        conversions,
        removal_strategy="redirect_to_null",
    )

    in_week = (timestamps >= "2020-12-28") & (timestamps < "2021-01-04")
    static = MarkovAttribution(removal_strategy="redirect_to_null").fit(
        list(np.asarray(paths, dtype=object)[in_week]),
        list(np.asarray(conversions)[in_week]),
    )

    epoch_result = model.epoch_results[model.epochs[1]]
    assert epoch_result["removal_effects"] == pytest.approx(static["removal_effects"])
    assert epoch_result["attribution"] == pytest.approx(static["attribution"])
    assert epoch_result["total_conversion_probability"] == pytest.approx(
        static["total_conversion_probability"]
    )


@pytest.mark.parametrize("model_class", TEMPORAL_MODELS)
def test_lazy_global_fit_uses_every_row(model_class, boundary_data):
    paths, timestamps, conversions = boundary_data
    model = _fit(model_class, paths, timestamps, conversions)
    static_class = (
        ShapleyAttribution if model_class is TemporalShapleyAttribution else MarkovAttribution
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        static = static_class().fit(paths, conversions)
        global_attribution = model.global_attribution

    assert global_attribution == pytest.approx(static["attribution"])


@pytest.mark.parametrize("model_class", TEMPORAL_MODELS)
def test_same_weekly_frequency_in_a_per_year_mapping_merges_too(
    model_class, boundary_data
):
    paths, timestamps, conversions = boundary_data
    by_string = _fit(model_class, paths, timestamps, conversions, freq="W")
    by_mapping = _fit(
        model_class, paths, timestamps, conversions, freq={2020: "W", 2021: "W"}
    )

    assert by_mapping.epochs == by_string.epochs
    for epoch in by_string.epochs:
        assert list(by_mapping._epoch_paths[epoch]) == list(by_string._epoch_paths[epoch])


@pytest.mark.parametrize("model_class", TEMPORAL_MODELS)
def test_mixed_per_year_frequencies_still_partition_the_rows(
    model_class, boundary_data
):
    paths, timestamps, conversions = boundary_data
    model = _fit(
        model_class, paths, timestamps, conversions, freq={2020: "W", 2021: "M"}
    )

    assert [str(epoch) for epoch in model.epochs] == [
        "2020-12-21/2020-12-27",
        STRADDLING_WEEK,
        "2021-01",
    ]
    assert len(set(model.epochs)) == len(model.epochs)
    # The 2020 week keeps its four 2020 days; January takes all ten 2021 days.
    assert [len(model._epoch_paths[epoch]) for epoch in model.epochs] == [28, 16, 40]
    assert sum(len(rows) for rows in model._epoch_paths.values()) == len(paths)


@pytest.mark.parametrize("model_class", TEMPORAL_MODELS)
def test_missing_year_in_mapping_leaves_the_model_unfitted(model_class, boundary_data):
    paths, timestamps, conversions = boundary_data
    model = model_class(freq={2020: "W"}, min_samples_per_epoch=1)

    with pytest.raises(KeyError, match="no entry for year 2021"):
        model.fit(paths, timestamps, conversions)

    assert model.epochs == []
    assert model.epoch_results == {}
