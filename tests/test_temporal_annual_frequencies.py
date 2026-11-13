"""Annual public aliases use pandas 3 period semantics in both fit paths."""

import numpy as np
import pandas as pd
import pytest

from do_attribution import TemporalMarkovAttribution, TemporalShapleyAttribution


MODELS = (TemporalMarkovAttribution, TemporalShapleyAttribution)
DATES = pd.DatetimeIndex(
    [
        "2024-07-01", "2023-12-31", "2023-06-30",
        "2024-06-30", "2024-01-01", "2023-07-01",
    ]
)
PATHS = np.array(
    ["Email", "Search", "Email > Search", "Social", "Email > Social", "Search > Social"]
)
CONVERSIONS = np.ones(len(DATES), dtype=int)


def _model(model_class, freq):
    options = {"freq": freq, "min_samples_per_epoch": 1}
    if model_class is TemporalMarkovAttribution:
        options["removal_strategy"] = "redirect_to_null"
    return model_class(**options)


def _assert_partition(model, epochs, row_groups):
    assert model.epochs == epochs
    for epoch, rows in zip(epochs, row_groups, strict=True):
        np.testing.assert_array_equal(model._epoch_paths[epoch], PATHS[rows])
        np.testing.assert_array_equal(
            model._epoch_conversions[epoch], CONVERSIONS[rows]
        )
    assert model.get_epoch_summary()["n_paths"].tolist() == [
        len(rows) for rows in row_groups
    ]


@pytest.mark.parametrize("model_class", MODELS)
@pytest.mark.parametrize("method", ["fit", "fit_by_epoch"])
@pytest.mark.parametrize("freq", ["A", "A-DEC", "Y", "YE", "YE-DEC"])
def test_calendar_annual_aliases_group_dates_into_calendar_years(
    model_class, method, freq
):
    model = _model(model_class, freq)
    getattr(model, method)(PATHS, DATES, CONVERSIONS)

    _assert_partition(
        model,
        [pd.Period("2023", freq="Y-DEC"), pd.Period("2024", freq="Y-DEC")],
        [[1, 2, 5], [0, 3, 4]],
    )
    assert model.freq == freq


@pytest.mark.parametrize("model_class", MODELS)
@pytest.mark.parametrize("method", ["fit", "fit_by_epoch"])
@pytest.mark.parametrize("freq", ["A-JUN", "YE-JUN"])
def test_fiscal_annual_aliases_preserve_june_end_boundaries(
    model_class, method, freq
):
    model = _model(model_class, freq)
    getattr(model, method)(PATHS, DATES, CONVERSIONS)

    _assert_partition(
        model,
        [pd.Period(year, freq="Y-JUN") for year in ("2023", "2024", "2025")],
        [[2], [1, 3, 4, 5], [0]],
    )
    assert model.epochs[1].start_time == pd.Timestamp("2023-07-01")
    assert model.epochs[1].end_time.normalize() == pd.Timestamp("2024-06-30")
    assert model.freq == freq


@pytest.mark.parametrize("model_class", MODELS)
def test_per_year_annual_aliases_merge_the_same_fiscal_period(model_class):
    model = _model(model_class, {2023: "A-JUN", 2024: "YE-JUN"})
    model.fit(PATHS, DATES, CONVERSIONS)

    _assert_partition(
        model,
        [pd.Period(year, freq="Y-JUN") for year in ("2023", "2024", "2025")],
        [[2], [1, 3, 4, 5], [0]],
    )


@pytest.mark.parametrize("model_class", MODELS)
@pytest.mark.parametrize("method", ["fit", "fit_by_epoch"])
@pytest.mark.parametrize("freq", ["M", "Q", "Q-JUN", "W", "W-MON", "D", "h"])
def test_nonannual_period_grouping_is_unchanged(model_class, method, freq):
    model = _model(model_class, freq)
    getattr(model, method)(PATHS, DATES, CONVERSIONS)

    expected = DATES.to_period(freq)
    epochs = sorted(expected.unique(), key=lambda epoch: epoch.start_time)
    rows = [np.flatnonzero(expected == epoch).tolist() for epoch in epochs]
    _assert_partition(model, epochs, rows)
