"""Contract tests for the decoupled DataFrame projection (design doc 8.9).

Acceptance items covered, per GLOBAL_SIGNAL_EPOCH_DESIGN.md section 8.9 and
the "Standard DataFrame projection" sections of temporal_markov.md /
temporal_shapley.md:

- exact public signature ``to_dataframes() -> dict[str, pd.DataFrame]``,
- unfitted models raise the specified ``RuntimeError``,
- every documented key is present in deterministic order and every value is
  a pandas DataFrame,
- every frame has a fresh RangeIndex, no MultiIndex, and explicit epoch /
  channel / state identifier columns,
- pivoting each flat metric frame reproduces the corresponding ``get_*``
  accessor values and fitted ordering,
- ``epoch_summary`` and channel sample diagnostics equal the default direct
  accessor outputs after index normalization,
- optional transition output uses an empty frame with the documented schema
  when matrices were not retained (the key is never omitted),
- mutating a returned frame cannot mutate model state or a later export,
- calling the method does not trigger ``global_attribution``, fitting,
  recommendation, plotting, or text rendering, and
- repeated calls on unchanged state are byte-for-byte stable under
  deterministic serialization.
"""

from __future__ import annotations

import inspect
import json
import re
import warnings

import numpy as np
import pandas as pd
import pytest

from do_attribution import (
    BaseAttribution,
    MarkovAttribution,
    ShapleyAttribution,
    TemporalMarkovAttribution,
    TemporalShapleyAttribution,
)


MARKOV_KEYS = [
    "attribution",
    "removal_effects",
    "conversion_probability",
    "transition_matrices",
    "epoch_summary",
    "channel_epoch_sample_sizes",
]
SHAPLEY_KEYS = [
    "attribution",
    "shapley_values",
    "epoch_summary",
    "channel_epoch_sample_sizes",
]
TRANSITION_COLUMNS = ["epoch", "from_state", "to_state", "probability"]
SAMPLE_SIZE_COLUMNS = [
    "epoch",
    "channel",
    "n_paths_containing_channel",
    "n_conversions_on_paths_containing_channel",
    "thin",
]


def _dataset():
    """Two quarters of paths with a channel ('Ads') absent from 2024Q1."""
    paths, conversions, timestamps = [], [], []
    q1_paths = ["Email > Social", "Email", "Social > Direct", "Direct"]
    q2_paths = ["Email > Ads", "Ads > Social", "Email", "Ads"]
    for day in range(16):
        paths.append(q1_paths[day % 4])
        conversions.append(1 if day % 3 == 0 else 0)
        timestamps.append(pd.Timestamp("2024-01-02") + pd.Timedelta(days=day))
    for day in range(16):
        paths.append(q2_paths[day % 4])
        conversions.append(1 if day % 2 == 0 else 0)
        timestamps.append(pd.Timestamp("2024-04-02") + pd.Timedelta(days=day))
    return paths, conversions, timestamps


def _fit_markov(**kwargs):
    paths, conversions, timestamps = _dataset()
    model = TemporalMarkovAttribution(
        freq="Q", min_samples_per_epoch=1, **kwargs
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(paths, timestamps, conversions)
    return model


def _fit_shapley():
    paths, conversions, timestamps = _dataset()
    model = TemporalShapleyAttribution(freq="Q", min_samples_per_epoch=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(paths, timestamps, conversions)
    return model


def _stringified(matrix: pd.DataFrame) -> pd.DataFrame:
    normalized = matrix.copy()
    normalized.index = [str(value) for value in matrix.index]
    normalized.columns = [str(value) for value in matrix.columns]
    return normalized


def _assert_flat(frame: pd.DataFrame) -> None:
    assert isinstance(frame, pd.DataFrame)
    assert isinstance(frame.index, pd.RangeIndex)
    assert frame.index.start == 0 and frame.index.step == 1
    assert not isinstance(frame.index, pd.MultiIndex)
    assert not isinstance(frame.columns, pd.MultiIndex)


# --------------------------------------------------------------------------- #
# Signature and unfitted behaviour                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "model_type", [TemporalMarkovAttribution, TemporalShapleyAttribution]
)
def test_signature_is_exactly_no_argument(model_type):
    signature = inspect.signature(model_type.to_dataframes)
    assert list(signature.parameters) == ["self"]


@pytest.mark.parametrize(
    "model_type", [TemporalMarkovAttribution, TemporalShapleyAttribution]
)
def test_unfitted_model_raises_runtime_error(model_type):
    with pytest.raises(
        RuntimeError,
        match=re.escape("to_dataframes() requires a fitted temporal model"),
    ):
        model_type().to_dataframes()


# --------------------------------------------------------------------------- #
# Key sets, order, and flat schemas                                            #
# --------------------------------------------------------------------------- #
def test_markov_keys_order_and_flat_schemas():
    tables = _fit_markov().to_dataframes()
    assert list(tables.keys()) == MARKOV_KEYS
    for frame in tables.values():
        _assert_flat(frame)
    assert list(tables["attribution"].columns) == ["epoch", "channel", "attribution"]
    assert list(tables["removal_effects"].columns) == [
        "epoch",
        "channel",
        "removal_effect",
    ]
    assert list(tables["conversion_probability"].columns) == [
        "epoch",
        "conversion_probability",
    ]
    assert list(tables["transition_matrices"].columns) == TRANSITION_COLUMNS
    assert tables["epoch_summary"].columns[0] == "epoch"
    assert list(tables["channel_epoch_sample_sizes"].columns) == SAMPLE_SIZE_COLUMNS


def test_shapley_keys_order_and_flat_schemas():
    tables = _fit_shapley().to_dataframes()
    assert list(tables.keys()) == SHAPLEY_KEYS
    for frame in tables.values():
        _assert_flat(frame)
    assert list(tables["attribution"].columns) == ["epoch", "channel", "attribution"]
    assert list(tables["shapley_values"].columns) == [
        "epoch",
        "channel",
        "shapley_value",
    ]


def test_identifier_columns_are_strings_across_tables():
    model = _fit_markov(return_transition_matrix=True)
    tables = model.to_dataframes()
    for key in MARKOV_KEYS:
        frame = tables[key]
        if frame.empty:
            continue
        assert frame["epoch"].map(type).eq(str).all(), key
    assert tables["attribution"]["channel"].map(type).eq(str).all()
    assert tables["transition_matrices"]["from_state"].map(type).eq(str).all()
    assert tables["transition_matrices"]["to_state"].map(type).eq(str).all()
    # One consistent join key across every table in the export.
    joined = tables["attribution"].merge(tables["epoch_summary"], on="epoch")
    assert len(joined) == len(tables["attribution"])


# --------------------------------------------------------------------------- #
# Pivoting reproduces the accessors, including fitted ordering                 #
# --------------------------------------------------------------------------- #
def test_pivoted_metric_frames_reproduce_accessors_markov():
    model = _fit_markov()
    tables = model.to_dataframes()
    cases = [
        ("attribution", "attribution", model.get_attribution_matrix()),
        ("removal_effects", "removal_effect", model.get_removal_effect_matrix()),
    ]
    for key, value_name, matrix in cases:
        expected = _stringified(matrix)
        pivoted = (
            tables[key]
            .pivot(index="epoch", columns="channel", values=value_name)
            .reindex(index=expected.index, columns=expected.columns)
        )
        pd.testing.assert_frame_equal(
            pivoted, expected, check_names=False, check_freq=False
        )
        # Fitted epoch order is preserved in the flat rows (epoch-major).
        assert tables[key]["epoch"].drop_duplicates().tolist() == list(expected.index)
    trend = model.get_conversion_probability_trend()
    assert tables["conversion_probability"]["epoch"].tolist() == [
        str(epoch) for epoch in trend.index
    ]
    np.testing.assert_allclose(
        tables["conversion_probability"]["conversion_probability"].to_numpy(),
        trend.to_numpy(dtype=float),
    )


def test_pivoted_metric_frames_reproduce_accessors_shapley():
    model = _fit_shapley()
    tables = model.to_dataframes()
    expected = _stringified(model.get_shapley_values_matrix())
    pivoted = (
        tables["shapley_values"]
        .pivot(index="epoch", columns="channel", values="shapley_value")
        .reindex(index=expected.index, columns=expected.columns)
    )
    pd.testing.assert_frame_equal(pivoted, expected, check_names=False)


def test_nan_grid_cells_are_retained_for_absent_channels():
    tables = _fit_markov().to_dataframes()
    attribution = tables["attribution"]
    q1_label = attribution["epoch"].iloc[0]
    ads_q1 = attribution[
        (attribution["epoch"] == q1_label) & (attribution["channel"] == "Ads")
    ]
    assert len(ads_q1) == 1
    assert np.isnan(ads_q1["attribution"].iloc[0])


def test_diagnostics_equal_direct_accessors_after_index_normalization():
    model = _fit_markov()
    tables = model.to_dataframes()
    expected_summary = model.get_epoch_summary().reset_index()
    pd.testing.assert_frame_equal(tables["epoch_summary"], expected_summary)
    expected_sizes = model.get_channel_epoch_sample_sizes().reset_index()
    expected_sizes = expected_sizes[SAMPLE_SIZE_COLUMNS].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        tables["channel_epoch_sample_sizes"], expected_sizes
    )


# --------------------------------------------------------------------------- #
# Transition matrices: retained and non-retained                              #
# --------------------------------------------------------------------------- #
def test_transition_matrices_empty_schema_when_not_retained():
    tables = _fit_markov(return_transition_matrix=False).to_dataframes()
    frame = tables["transition_matrices"]
    assert frame.empty
    assert list(frame.columns) == TRANSITION_COLUMNS


def test_transition_matrices_flatten_retained_state_row_stochastically():
    model = _fit_markov(return_transition_matrix=True)
    tables = model.to_dataframes()
    frame = tables["transition_matrices"]
    assert not frame.empty
    fitted_labels = [str(epoch) for epoch in model.epochs]
    assert frame["epoch"].drop_duplicates().tolist() == fitted_labels
    # Zero cells retained: every (epoch, from_state) group is a complete
    # matrix row and therefore sums to exactly 1.
    row_sums = frame.groupby(["epoch", "from_state"])["probability"].sum()
    np.testing.assert_allclose(row_sums.to_numpy(), 1.0, atol=1e-12)
    # Cell-level equality with the retained accessor matrix.
    first_epoch = model.epochs[0]
    matrix = model.get_transition_matrix(first_epoch)
    sub = frame[frame["epoch"] == str(first_epoch)]
    pivoted = sub.pivot(index="from_state", columns="to_state", values="probability")
    expected = _stringified(matrix)
    pivoted = pivoted.reindex(index=expected.index, columns=expected.columns)
    pd.testing.assert_frame_equal(pivoted, expected, check_names=False)


# --------------------------------------------------------------------------- #
# Isolation: copies, no lazy fit, no state mutation                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fit", [_fit_markov, _fit_shapley])
def test_no_lazy_global_fit_and_no_state_mutation(fit):
    model = fit()
    assert model._global_fit_done is False
    epoch_results_before = {
        epoch: dict(result) for epoch, result in model.epoch_results.items()
    }
    model.to_dataframes()
    assert model._global_fit_done is False
    assert {
        epoch: dict(result) for epoch, result in model.epoch_results.items()
    } == epoch_results_before


def test_mutating_returned_frames_cannot_corrupt_model_or_later_exports():
    model = _fit_markov(return_transition_matrix=True)
    first = model.to_dataframes()
    for frame in first.values():
        for column in frame.columns:
            frame[column] = "corrupted"
    second = model.to_dataframes()
    accessor_matrix = model.get_attribution_matrix()
    assert not accessor_matrix.isin(["corrupted"]).any().any()
    for key in MARKOV_KEYS:
        assert not second[key].isin(["corrupted"]).any().any()


@pytest.mark.parametrize("fit", [_fit_markov, _fit_shapley])
def test_repeated_calls_are_byte_for_byte_stable(fit):
    model = fit()
    first = model.to_dataframes()
    second = model.to_dataframes()
    assert list(first.keys()) == list(second.keys())
    for key in first:
        assert first[key].to_csv(index=False) == second[key].to_csv(index=False)


# --------------------------------------------------------------------------- #
# Explicit-label and interval-label fits (fit_by_epoch / fit_recommended path) #
# --------------------------------------------------------------------------- #
def test_explicit_string_labels_preserve_first_seen_order():
    model = TemporalShapleyAttribution(min_samples_per_epoch=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_by_epoch(
            ["A", "B", "A > B", "B"],
            ["later", "earlier", "later", "earlier"],
            [1, 0, 1, 1],
        )
    tables = model.to_dataframes()
    assert tables["attribution"]["epoch"].drop_duplicates().tolist() == [
        "later",
        "earlier",
    ]


def test_interval_labels_flatten_like_fit_recommended_output():
    left = pd.Timestamp("2024-01-01")
    middle = pd.Timestamp("2024-02-01")
    right = pd.Timestamp("2024-03-01")
    early = pd.Interval(left, middle, closed="left")
    late = pd.Interval(middle, right, closed="left")
    labels = [early, early, late, late, late]
    model = TemporalMarkovAttribution(min_samples_per_epoch=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_by_epoch(
            ["A > B", "B", "A", "A > B", "B > A"],
            labels,
            [1, 0, 1, 1, 0],
        )
    tables = model.to_dataframes()
    assert tables["attribution"]["epoch"].drop_duplicates().tolist() == [
        str(early),
        str(late),
    ]
    assert tables["epoch_summary"]["epoch"].tolist() == [str(early), str(late)]


def _recommended_dataset():
    """30 daily cohorts with a level shift, for fit_recommended coverage."""
    paths, conversions, anchors = [], [], []
    for day in range(30):
        date = pd.Timestamp("2024-03-01") + pd.Timedelta(days=day)
        rate = 0.2 if day < 15 else 0.7
        for sample in range(12):
            paths.append("A > B" if sample % 2 else "B > A")
            conversions.append(int(sample < round(rate * 12)))
            anchors.append(date)
    return paths, conversions, anchors


def test_projection_after_fit_recommended_markov_and_shapley():
    paths, conversions, anchors = _recommended_dataset()
    markov = TemporalMarkovAttribution(min_samples_per_epoch=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        markov.fit_recommended(
            paths,
            conversions,
            anchors,
            recommender_kwargs={"min_samples": 10},
        )
    tables = markov.to_dataframes()
    assert list(tables.keys()) == MARKOV_KEYS
    assert tables["attribution"]["epoch"].drop_duplicates().tolist() == [
        str(epoch) for epoch in markov.epochs
    ]
    # Interval labels are chronologically ordered on the model itself.
    assert markov.epochs == sorted(
        markov.epochs, key=lambda interval: (interval.left, interval.right)
    )
    assert markov._global_fit_done is False

    shapley = TemporalShapleyAttribution(min_samples_per_epoch=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shapley.fit_recommended(
            paths,
            conversions,
            anchors,
            recommender_kwargs={"min_samples": 10},
        )
    shapley_tables = shapley.to_dataframes()
    assert list(shapley_tables.keys()) == SHAPLEY_KEYS
    assert shapley_tables["epoch_summary"]["epoch"].tolist() == [
        str(epoch) for epoch in shapley.epochs
    ]


# --------------------------------------------------------------------------- #
# Static models: flat export from the last successful fit                      #
# --------------------------------------------------------------------------- #
STATIC_MARKOV_KEYS = [
    "attribution",
    "removal_effects",
    "fit_diagnostics",
    "transition_matrix",
]
STATIC_SHAPLEY_KEYS = ["attribution", "shapley_values", "fit_diagnostics"]
STATIC_TRANSITION_COLUMNS = ["from_state", "to_state", "probability"]
NORMALIZATION_COLUMNS = [
    "normalization_status",
    "normalization_reason",
    "normalization_denominator",
    "raw_sum",
    "raw_magnitude",
    "cancellation_ratio",
]
STATIC_MARKOV_DIAGNOSTIC_COLUMNS = [
    "model",
    "order",
    "removal_strategy",
    "separator",
    "conversion_labels",
    "return_transition_matrix",
    "conversion_probability",
] + NORMALIZATION_COLUMNS
STATIC_SHAPLEY_DIAGNOSTIC_COLUMNS = [
    "model",
    "separator",
    "conversion_labels",
    "characteristic_function",
] + NORMALIZATION_COLUMNS

STATIC_PATHS = [
    "Email > Social > Purchase",
    "Search > Email > Purchase",
    "Social > Purchase",
    "Email > Search",
    "Search > Social > Email > Purchase",
    "Direct > Purchase",
    "Email > Social",
    "Search > Email > Social > Purchase",
    "Social > Email > Purchase",
    "Search",
]
STATIC_CONVERSIONS = [1, 1, 1, 0, 1, 1, 0, 1, 1, 0]
STATIC_CHANNELS = ["Direct", "Email", "Search", "Social"]


def _fit_static_markov(paths=STATIC_PATHS, conversions=STATIC_CONVERSIONS, **kwargs):
    kwargs.setdefault("removal_strategy", "redirect_to_null")
    kwargs.setdefault("conversion_labels", ["Purchase"])
    model = MarkovAttribution(**kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(paths, conversions)
    return model, result


def _fit_static_shapley(paths=STATIC_PATHS, conversions=STATIC_CONVERSIONS, **kwargs):
    kwargs.setdefault("conversion_labels", ["Purchase"])
    model = ShapleyAttribution(**kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(paths, conversions)
    return model, result


def _column_as_dict(frame: pd.DataFrame, value_name: str) -> dict:
    return dict(zip(frame["channel"], frame[value_name]))


def _assert_scalar_cells(frame: pd.DataFrame) -> None:
    for column in frame.columns:
        for value in frame[column]:
            assert not isinstance(
                value, (list, tuple, dict, set, np.ndarray, pd.Series)
            ), (column, value)


def _assert_normalization_cells(row: pd.Series, result: dict) -> None:
    assert row["normalization_status"] == result["normalization_status"]
    if result["normalization_reason"] is None:
        assert pd.isna(row["normalization_reason"])
    else:
        assert row["normalization_reason"] == result["normalization_reason"]
    for key in ("normalization_denominator", "raw_sum", "raw_magnitude", "cancellation_ratio"):
        if result[key] is None:
            assert np.isnan(row[key]), key
        else:
            assert row[key] == result[key], key


@pytest.mark.parametrize("model_type", [MarkovAttribution, ShapleyAttribution])
def test_static_signature_is_exactly_no_argument(model_type):
    signature = inspect.signature(model_type.to_dataframes)
    assert list(signature.parameters) == ["self"]


@pytest.mark.parametrize("model_type", [MarkovAttribution, ShapleyAttribution])
def test_static_unfitted_model_raises_runtime_error(model_type):
    with pytest.raises(
        RuntimeError,
        match=re.escape("to_dataframes() requires a fitted attribution model"),
    ):
        model_type().to_dataframes()


def test_static_markov_keys_order_and_flat_schemas():
    model, _ = _fit_static_markov(return_transition_matrix=True)
    tables = model.to_dataframes()

    assert list(tables.keys()) == STATIC_MARKOV_KEYS
    for frame in tables.values():
        _assert_flat(frame)
        _assert_scalar_cells(frame)
        assert "epoch" not in frame.columns
    assert list(tables["attribution"].columns) == ["channel", "attribution"]
    assert list(tables["removal_effects"].columns) == ["channel", "removal_effect"]
    assert list(tables["fit_diagnostics"].columns) == STATIC_MARKOV_DIAGNOSTIC_COLUMNS
    assert len(tables["fit_diagnostics"]) == 1
    assert list(tables["transition_matrix"].columns) == STATIC_TRANSITION_COLUMNS
    for key in ("attribution", "removal_effects"):
        assert tables[key]["channel"].tolist() == STATIC_CHANNELS
        assert tables[key]["channel"].map(type).eq(str).all()


def test_static_shapley_keys_order_and_flat_schemas():
    model, _ = _fit_static_shapley()
    tables = model.to_dataframes()

    assert list(tables.keys()) == STATIC_SHAPLEY_KEYS
    for frame in tables.values():
        _assert_flat(frame)
        _assert_scalar_cells(frame)
        assert "epoch" not in frame.columns
    assert list(tables["attribution"].columns) == ["channel", "attribution"]
    assert list(tables["shapley_values"].columns) == ["channel", "shapley_value"]
    assert list(tables["fit_diagnostics"].columns) == STATIC_SHAPLEY_DIAGNOSTIC_COLUMNS
    assert len(tables["fit_diagnostics"]) == 1
    for key in ("attribution", "shapley_values"):
        assert tables[key]["channel"].tolist() == STATIC_CHANNELS
        assert tables[key]["channel"].map(type).eq(str).all()


@pytest.mark.parametrize(
    "removal_strategy", ["fail", "fail_refit", "detour", "redirect_to_null"]
)
@pytest.mark.parametrize("order", [1, 2])
def test_static_markov_export_matches_the_saved_fit_result(removal_strategy, order):
    model, result = _fit_static_markov(
        order=order,
        removal_strategy=removal_strategy,
        return_transition_matrix=True,
    )
    tables = model.to_dataframes()

    assert _column_as_dict(tables["removal_effects"], "removal_effect") == (
        result["removal_effects"]
    )
    if result["attribution"] is None:
        assert tables["attribution"]["attribution"].isna().all()
    else:
        assert _column_as_dict(tables["attribution"], "attribution") == (
            result["attribution"]
        )
    row = tables["fit_diagnostics"].iloc[0]
    assert row["model"] == "MarkovAttribution"
    assert row["order"] == order
    assert row["removal_strategy"] == removal_strategy
    assert row["separator"] == " > "
    assert row["conversion_labels"] == '["Purchase"]'
    assert bool(row["return_transition_matrix"]) is True
    assert row["conversion_probability"] == result["total_conversion_probability"]
    _assert_normalization_cells(row, result)


@pytest.mark.parametrize("characteristic_function", ["cumulative", "exact"])
def test_static_shapley_export_matches_the_saved_fit_result(characteristic_function):
    model, result = _fit_static_shapley(
        characteristic_function=characteristic_function
    )
    tables = model.to_dataframes()

    assert _column_as_dict(tables["shapley_values"], "shapley_value") == (
        result["shapley_values"]
    )
    if result["attribution"] is None:
        assert tables["attribution"]["attribution"].isna().all()
    else:
        assert _column_as_dict(tables["attribution"], "attribution") == (
            result["attribution"]
        )
    row = tables["fit_diagnostics"].iloc[0]
    assert row["model"] == "ShapleyAttribution"
    assert row["separator"] == " > "
    assert row["conversion_labels"] == '["Purchase"]'
    assert row["characteristic_function"] == characteristic_function
    _assert_normalization_cells(row, result)


def test_static_exact_game_shapley_exports_unavailable_shares_with_raw_values():
    model, result = _fit_static_shapley(characteristic_function="exact")
    assert result["attribution"] is None  # this dataset cancels
    tables = model.to_dataframes()

    attribution = tables["attribution"]
    assert attribution["channel"].tolist() == STATIC_CHANNELS
    assert attribution["attribution"].isna().all()
    assert not (attribution["attribution"] == 0.0).any()

    raw = tables["shapley_values"]
    assert raw["channel"].tolist() == STATIC_CHANNELS
    assert raw["shapley_value"].notna().all()
    assert (raw["shapley_value"] < 0).any() and (raw["shapley_value"] > 0).any()
    assert _column_as_dict(raw, "shapley_value") == result["shapley_values"]

    row = tables["fit_diagnostics"].iloc[0]
    assert row["normalization_status"] == "unavailable"
    assert row["normalization_reason"] == "near_cancellation"
    assert row["raw_magnitude"] == result["raw_magnitude"]


@pytest.mark.parametrize(
    "removal_strategy", ["fail", "fail_refit", "detour", "redirect_to_null"]
)
def test_static_unavailable_attribution_is_distinguishable_from_zero_credit(
    removal_strategy,
):
    # No conversions: there is no effect to normalize.
    unavailable_model, unavailable_result = _fit_static_markov(
        ["A", "B"], [0, 0], removal_strategy=removal_strategy, conversion_labels=None
    )
    unavailable = unavailable_model.to_dataframes()
    assert unavailable_result["attribution"] is None
    assert unavailable["attribution"]["channel"].tolist() == ["A", "B"]
    assert unavailable["attribution"]["attribution"].isna().all()
    assert unavailable["removal_effects"]["removal_effect"].tolist() == [0.0, 0.0]
    unavailable_row = unavailable["fit_diagnostics"].iloc[0]
    assert unavailable_row["normalization_status"] == "unavailable"
    assert unavailable_row["normalization_reason"] == "zero_raw_magnitude"
    assert np.isnan(unavailable_row["normalization_denominator"])
    assert np.isnan(unavailable_row["cancellation_ratio"])
    assert unavailable_row["conversion_probability"] == 0.0


def test_static_available_zero_share_exports_as_zero_not_missing():
    model, result = _fit_static_markov(
        ["A", "B"], [1, 0], removal_strategy="fail", conversion_labels=None
    )
    assert result["attribution"] == {"A": 1.0, "B": 0.0}
    tables = model.to_dataframes()

    assert tables["attribution"]["attribution"].tolist() == [1.0, 0.0]
    assert tables["attribution"]["attribution"].notna().all()
    row = tables["fit_diagnostics"].iloc[0]
    assert row["normalization_status"] == "available"
    assert pd.isna(row["normalization_reason"])


def test_static_no_channel_fit_exports_empty_channel_tables_with_diagnostics():
    for fit in (_fit_static_markov, _fit_static_shapley):
        model, result = fit(["Purchase", "Purchase"], [1, 0])
        assert result["attribution"] is None
        tables = model.to_dataframes()
        raw_key = list(tables)[1]
        assert tables["attribution"].empty
        assert list(tables["attribution"].columns) == ["channel", "attribution"]
        assert tables[raw_key].empty
        row = tables["fit_diagnostics"].iloc[0]
        assert row["normalization_status"] == "unavailable"
        assert row["normalization_reason"] == "zero_raw_magnitude"


def test_static_transition_table_empty_schema_when_not_requested():
    model, _ = _fit_static_markov(return_transition_matrix=False)
    tables = model.to_dataframes()

    frame = tables["transition_matrix"]
    assert frame.empty
    assert list(frame.columns) == STATIC_TRANSITION_COLUMNS
    assert bool(tables["fit_diagnostics"]["return_transition_matrix"].iloc[0]) is False


@pytest.mark.parametrize("order", [1, 2, 3])
def test_static_transition_table_reproduces_the_fitted_matrix(order):
    model, result = _fit_static_markov(order=order, return_transition_matrix=True)
    matrix = result["transition_matrix"]
    frame = model.to_dataframes()["transition_matrix"]

    assert len(frame) == matrix.shape[0] * matrix.shape[1]
    expected = _stringified(matrix)
    # Row-major in fitted state order, with zero cells retained.
    assert frame["from_state"].drop_duplicates().tolist() == list(expected.index)
    assert frame["to_state"].iloc[: matrix.shape[1]].tolist() == list(expected.columns)
    assert frame["from_state"].map(type).eq(str).all()
    assert frame["to_state"].map(type).eq(str).all()
    pivoted = frame.pivot(
        index="from_state", columns="to_state", values="probability"
    ).reindex(index=expected.index, columns=expected.columns)
    pd.testing.assert_frame_equal(pivoted, expected, check_names=False)
    row_sums = frame.groupby("from_state")["probability"].sum()
    np.testing.assert_allclose(row_sums.to_numpy(), 1.0, atol=1e-12)


def test_static_higher_order_transition_identifiers_are_tuple_strings():
    model, result = _fit_static_markov(order=2, return_transition_matrix=True)
    frame = model.to_dataframes()["transition_matrix"]
    identifiers = set(frame["from_state"]) | set(frame["to_state"])

    assert {"START", "CONVERSION", "NULL"} <= identifiers
    assert "('Email', 'Social')" in identifiers
    assert "('START', 'Direct')" in identifiers  # a short path, left-padded
    assert identifiers == {str(state) for state in result["transition_matrix"].index}
    assert len(identifiers) == len(result["transition_matrix"].index)
    edge = frame[
        (frame["from_state"] == "('START', 'Direct')")
        & (frame["to_state"] == "CONVERSION")
    ]
    assert edge["probability"].tolist() == [
        result["transition_matrix"].loc[[("START", "Direct")], "CONVERSION"].iloc[0]
    ]


@pytest.mark.parametrize("fit", [_fit_static_markov, _fit_static_shapley])
def test_static_mutating_returned_frames_cannot_corrupt_model_or_later_exports(fit):
    kwargs = {"return_transition_matrix": True} if fit is _fit_static_markov else {}
    model, _ = fit(**kwargs)
    reference = {
        key: frame.to_csv(index=False) for key, frame in model.to_dataframes().items()
    }

    first = model.to_dataframes()
    for frame in first.values():
        for column in frame.columns:
            frame[column] = "corrupted"
        frame.drop(frame.index, inplace=True)

    second = model.to_dataframes()
    for key, frame in second.items():
        assert frame.to_csv(index=False) == reference[key]
        assert not frame.isin(["corrupted"]).any().any()
    assert model.get_attribution() == _column_as_dict(
        second["attribution"], "attribution"
    )
    assert "corrupted" not in model.summary()


def test_static_markov_export_is_a_defensive_snapshot_of_the_fit():
    model, result = _fit_static_markov(order=2, return_transition_matrix=True)
    reference = {
        key: frame.to_csv(index=False) for key, frame in model.to_dataframes().items()
    }

    # Mutate everything a caller can reach after the fit.
    result["attribution"]["Email"] = 99.0
    result["removal_effects"].clear()
    result["transition_matrix"].iloc[:, :] = -1.0
    model.transition_matrix.iloc[:, :] = -2.0
    model.channels.append("Injected")
    model.order = 7
    model.removal_strategy = "detour"
    model.separator = "|"

    for key, frame in model.to_dataframes().items():
        assert frame.to_csv(index=False) == reference[key], key


def test_static_shapley_export_is_a_defensive_snapshot_of_the_fit():
    model, result = _fit_static_shapley(characteristic_function="exact")
    reference = {
        key: frame.to_csv(index=False) for key, frame in model.to_dataframes().items()
    }

    result["shapley_values"]["Email"] = 99.0
    result["value_function"].clear()
    model.value_function.clear()
    model.channels.append("Injected")
    model.characteristic_function = "cumulative"

    for key, frame in model.to_dataframes().items():
        assert frame.to_csv(index=False) == reference[key], key


@pytest.mark.parametrize("fit", [_fit_static_markov, _fit_static_shapley])
def test_static_export_reflects_the_last_successful_fit_only(fit):
    model, _ = fit()
    first = {
        key: frame.to_csv(index=False) for key, frame in model.to_dataframes().items()
    }

    with pytest.raises(ValueError):
        model.fit(["A", "B"], [0.5, 1])
    after_failed_fit = model.to_dataframes()
    for key, frame in after_failed_fit.items():
        assert frame.to_csv(index=False) == first[key], key

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(["A > B", "B"], [1, 0])
    refitted = model.to_dataframes()
    assert refitted["attribution"]["channel"].tolist() == ["A", "B"]


def test_static_markov_export_never_refits(monkeypatch):
    model, _ = _fit_static_markov(order=2, return_transition_matrix=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("to_dataframes() must not recompute the fit")

    for name in (
        "fit",
        "summary",
        "_build_transition_matrix",
        "_calculate_conversion_probability",
        "_calculate_removal_effects",
    ):
        monkeypatch.setattr(model, name, forbidden)
    monkeypatch.setattr(
        "do_attribution.markov.normalize_signed_attribution", forbidden
    )

    tables = model.to_dataframes()
    assert list(tables.keys()) == STATIC_MARKOV_KEYS
    assert not tables["transition_matrix"].empty


@pytest.mark.parametrize("characteristic_function", ["cumulative", "exact"])
def test_static_shapley_export_never_refits_or_enumerates_coalitions(
    monkeypatch, characteristic_function
):
    model, _ = _fit_static_shapley(characteristic_function=characteristic_function)

    def forbidden(*args, **kwargs):
        raise AssertionError("to_dataframes() must not recompute the fit")

    for name in (
        "fit",
        "summary",
        "_calculate_value_function",
        "_cumulative_shapley_values",
        "_exact_set_shapley_values",
    ):
        monkeypatch.setattr(model, name, forbidden)
    monkeypatch.setattr("do_attribution.shapley.get_powerset", forbidden)
    monkeypatch.setattr(
        "do_attribution.shapley.normalize_signed_attribution", forbidden
    )

    tables = model.to_dataframes()
    assert list(tables.keys()) == STATIC_SHAPLEY_KEYS
    assert tables["shapley_values"]["shapley_value"].notna().all()


@pytest.mark.parametrize("fit", [_fit_static_markov, _fit_static_shapley])
def test_static_repeated_calls_are_byte_for_byte_stable(fit):
    model, _ = fit()
    first = model.to_dataframes()
    second = model.to_dataframes()
    assert list(first.keys()) == list(second.keys())
    for key in first:
        assert first[key] is not second[key]
        assert first[key].to_csv(index=False) == second[key].to_csv(index=False)


def test_static_export_round_trips_through_csv(tmp_path):
    model, result = _fit_static_markov(return_transition_matrix=True)
    for name, table in model.to_dataframes().items():
        table.to_csv(tmp_path / f"markov_{name}.csv", index=False)

    shares = pd.read_csv(tmp_path / "markov_attribution.csv")
    assert list(shares.columns) == ["channel", "attribution"]
    np.testing.assert_allclose(
        shares["attribution"].to_numpy(),
        [result["attribution"][channel] for channel in STATIC_CHANNELS],
    )
    diagnostics = pd.read_csv(tmp_path / "markov_fit_diagnostics.csv")
    assert len(diagnostics) == 1
    assert diagnostics["normalization_status"].iloc[0] == "available"
    assert json.loads(diagnostics["conversion_labels"].iloc[0]) == ["Purchase"]


def test_static_export_does_not_extend_the_base_attribution_contract():
    assert not hasattr(BaseAttribution, "to_dataframes")
    assert BaseAttribution.__abstractmethods__ == frozenset({"fit"})

    class MinimalModel(BaseAttribution):
        def fit(self, paths, conversions):
            self.channels = ["A"]
            self._last_attribution = {"A": 1.0}
            return {"attribution": dict(self._last_attribution)}

    model = MinimalModel()
    assert model.fit(["A"], [1]) == {"attribution": {"A": 1.0}}
    assert model.get_attribution() == {"A": 1.0}
    assert not hasattr(model, "to_dataframes")


@pytest.mark.parametrize(
    ("model_type", "expected_keys"),
    [
        (TemporalMarkovAttribution, MARKOV_KEYS),
        (TemporalShapleyAttribution, SHAPLEY_KEYS),
    ],
)
def test_temporal_models_keep_their_own_export_after_static_export_was_added(
    model_type, expected_keys
):
    paths, conversions, timestamps = _dataset()
    model = model_type(freq="Q", min_samples_per_epoch=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # The inherited static fit alone is not a temporal fit.
        model.fit_static(paths, conversions)
        with pytest.raises(
            RuntimeError,
            match=re.escape("to_dataframes() requires a fitted temporal model"),
        ):
            model.to_dataframes()
        model.fit(paths, timestamps, conversions)

    tables = model.to_dataframes()
    assert list(tables.keys()) == expected_keys
    assert list(tables["attribution"].columns) == ["epoch", "channel", "attribution"]
    # Per-epoch fits leave no static export snapshot behind.
    assert model._export_snapshot is None


def test_mixed_granularity_periods_flatten_in_wall_clock_order():
    paths = ["A > B", "B", "A"] * 8
    conversions = [1, 0, 1] * 8
    timestamps = list(pd.date_range("2024-11-15", periods=12, freq="7D")) + list(
        pd.date_range("2025-01-05", periods=12, freq="10D")
    )
    model = TemporalMarkovAttribution(
        freq={2024: "Q", 2025: "M"}, min_samples_per_epoch=1
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(paths, timestamps, conversions)
    tables = model.to_dataframes()
    labels = tables["attribution"]["epoch"].drop_duplicates().tolist()
    assert labels == [str(epoch) for epoch in model.epochs]
    assert len(labels) == len(set(labels))
