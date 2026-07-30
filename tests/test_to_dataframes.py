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
import re
import warnings

import numpy as np
import pandas as pd
import pytest

from do_attribution import TemporalMarkovAttribution, TemporalShapleyAttribution


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
