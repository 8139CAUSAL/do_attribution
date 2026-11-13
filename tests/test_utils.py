"""Tests for the shared preprocessing utilities.

Mirrors ``markov-attribution/tests/test_utils.py`` so the unified
library's behaviour on the common utilities is explicitly locked down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from do_attribution import (
    CompareAttribution,
    FirstTouchAttribution,
    LastTouchAttribution,
    MarkovAttribution,
    ShapleyAttribution,
    TemporalMarkovAttribution,
    TemporalShapleyAttribution,
)
from do_attribution.utils import (
    DEFAULT_SEPARATOR,
    SPECIAL_TOKENS,
    _normalize_conversion_labels,  # MODIFIED: P0 regression tests.
    create_n_order_states,
    extract_channels,
    get_powerset,
    prepare_journey_data,
    split_path,
    strip_conversion_labels,  # MODIFIED: P0.
    strip_conversion_labels_array,  # MODIFIED: P0.
    to_int_array,
    to_string_array,
    validate_data,
)


# --------------------------------------------------------------------------- #
# split_path / extract_channels                                                #
# --------------------------------------------------------------------------- #
def test_split_path_basic():
    assert split_path("A > B > C") == ["A", "B", "C"]


def test_split_path_strips_whitespace_and_empties():
    assert split_path("A >  > B") == ["A", "B"]
    assert split_path("") == []


def test_extract_channels_basic():
    paths = ["A > B > C", "B > D", "A"]
    assert extract_channels(paths) == ["A", "B", "C", "D"]


def test_extract_channels_custom_separator():
    assert extract_channels(["A|B|C", "B|D"], separator="|") == ["A", "B", "C", "D"]


def test_extract_channels_empty_and_none():
    assert extract_channels([]) == []
    assert extract_channels(["", "  ", None]) == []


def test_extract_channels_accepts_series_and_array():
    series = pd.Series(["A > B", "A > C"])
    arr = np.array(["A > B", "A > C"])
    assert extract_channels(series) == ["A", "B", "C"]
    assert extract_channels(arr) == ["A", "B", "C"]


def test_extract_channels_removes_special_tokens():
    # The reserved Markov tokens must never be returned as channels.
    paths = ["A > START > B", "CONVERSION > NULL > C"]
    assert extract_channels(paths) == ["A", "B", "C"]
    for token in SPECIAL_TOKENS:
        assert token not in extract_channels(paths)


# --------------------------------------------------------------------------- #
# create_n_order_states                                                        #
# --------------------------------------------------------------------------- #
def test_create_n_order_first_order():
    assert create_n_order_states("A > B > C", order=1) == ["A", "B", "C"]


def test_create_n_order_second_order():
    assert create_n_order_states("A > B > C", order=2) == [("A", "B"), ("B", "C")]


def test_create_n_order_third_order():
    assert create_n_order_states("A > B > C", order=3) == [("A", "B", "C")]


def test_create_n_order_pads_when_path_shorter_than_order():
    assert create_n_order_states("A > B", order=3) == [("START", "A", "B")]


def test_create_n_order_empty_path():
    assert create_n_order_states("", order=1) == []
    assert create_n_order_states("", order=2) == []


def test_create_n_order_rejects_invalid_order():
    with pytest.raises(ValueError):
        create_n_order_states("A > B", order=0)


# --------------------------------------------------------------------------- #
# validate_data                                                                #
# --------------------------------------------------------------------------- #
def test_validate_data_ok():
    validate_data(["A > B", "B > C"], [1, 0])


def test_validate_data_length_mismatch():
    with pytest.raises(ValueError, match="must have same length"):
        validate_data(["A > B"], [1, 0])


def test_validate_data_empty():
    with pytest.raises(ValueError, match="No paths provided"):
        validate_data([], [])


def test_validate_data_bad_conversion():
    with pytest.raises(ValueError, match="must be 0 or 1"):
        validate_data(["A > B"], [2])


def test_validate_data_warns_on_empty_path():
    with pytest.warns(UserWarning, match="empty paths"):
        validate_data(["A > B", "", "B > C"], [1, 0, 1])


# --------------------------------------------------------------------------- #
# get_powerset                                                                 #
# --------------------------------------------------------------------------- #
def test_get_powerset_basic():
    assert list(get_powerset(["a", "b"])) == [(), ("a",), ("b",), ("a", "b")]


def test_get_powerset_empty():
    assert list(get_powerset([])) == [()]


def test_get_powerset_size_2n():
    xs = ["a", "b", "c", "d"]
    assert len(list(get_powerset(xs))) == 2 ** len(xs)


# --------------------------------------------------------------------------- #
# Array coercion                                                               #
# --------------------------------------------------------------------------- #
def test_to_string_array_handles_series():
    s = pd.Series(["A > B", None])
    arr = to_string_array(s)
    assert list(arr) == ["A > B", ""]


def test_to_int_array_handles_series():
    s = pd.Series([1, 0, 1])
    arr = to_int_array(s)
    assert arr.tolist() == [1, 0, 1]


def test_default_separator_roundtrip():
    assert DEFAULT_SEPARATOR == " > "


# --------------------------------------------------------------------------- #
# Binary-outcome validation before coercion                                    #
# --------------------------------------------------------------------------- #
VALID_OUTCOMES = [1, 0, 1, 1, 0, 1, 0, 1]

# Every supported way of spelling VALID_OUTCOMES.  Each must coerce to the
# same integers and therefore produce the same fit as the plain int list.
VALID_OUTCOME_CONTAINERS = {
    "list_int": lambda: list(VALID_OUTCOMES),
    "tuple_int": lambda: tuple(VALID_OUTCOMES),
    "list_bool": lambda: [bool(v) for v in VALID_OUTCOMES],
    "list_float": lambda: [float(v) for v in VALID_OUTCOMES],
    "list_canonical_str": lambda: [str(v) for v in VALID_OUTCOMES],
    "list_mixed": lambda: [
        True, 0, 1.0, "1", np.int64(0), np.float64(1.0), np.False_, 1,
    ],
    "array_int": lambda: np.array(VALID_OUTCOMES),
    "array_int8": lambda: np.array(VALID_OUTCOMES, dtype=np.int8),
    "array_bool": lambda: np.array(VALID_OUTCOMES, dtype=bool),
    "array_float": lambda: np.array(VALID_OUTCOMES, dtype=float),
    "array_float32": lambda: np.array(VALID_OUTCOMES, dtype=np.float32),
    "array_object": lambda: np.array(VALID_OUTCOMES, dtype=object),
    "array_canonical_str": lambda: np.array([str(v) for v in VALID_OUTCOMES]),
    "series_int": lambda: pd.Series(VALID_OUTCOMES),
    "series_bool": lambda: pd.Series([bool(v) for v in VALID_OUTCOMES]),
    "series_float": lambda: pd.Series([float(v) for v in VALID_OUTCOMES]),
    "series_nullable_int": lambda: pd.Series(VALID_OUTCOMES, dtype="Int64"),
    "series_nullable_boolean": lambda: pd.Series(
        [bool(v) for v in VALID_OUTCOMES], dtype="boolean"
    ),
    "series_nullable_float": lambda: pd.Series(
        [float(v) for v in VALID_OUTCOMES], dtype="Float64"
    ),
    "series_canonical_str": lambda: pd.Series([str(v) for v in VALID_OUTCOMES]),
    "series_shifted_index": lambda: pd.Series(
        VALID_OUTCOMES, index=range(10, 18)
    ),
}

# (conversions for two journeys, exception type, message pattern).  The
# pattern names the offending position and the original, uncoerced value.
INVALID_OUTCOMES = {
    "fractional_positive": (
        lambda: [0.9, 1.9], ValueError, r"index 0 must be 0 or 1, got 0\.9$"
    ),
    "fractional_negative": (
        lambda: [-0.9, 1.1], ValueError, r"index 0 must be 0 or 1, got -0\.9$"
    ),
    "fractional_above_one": (
        lambda: [1, 1.9], ValueError, r"index 1 must be 0 or 1, got 1\.9$"
    ),
    "integer_above_one": (
        lambda: [0, 2], ValueError, r"index 1 must be 0 or 1, got 2$"
    ),
    "negative_integer": (
        lambda: [0, -1], ValueError, r"index 1 must be 0 or 1, got -1$"
    ),
    "fractional_array": (
        lambda: np.array([0.0, 0.25]),
        ValueError,
        r"index 1 must be 0 or 1, got 0\.25$",
    ),
    "fractional_series": (
        lambda: pd.Series([0.5, 1.0]),
        ValueError,
        r"index 0 must be 0 or 1, got 0\.5$",
    ),
    "nan": (
        lambda: [0, float("nan")],
        ValueError,
        r"index 1 must be 0 or 1, got missing value nan$",
    ),
    "nan_array": (
        lambda: np.array([np.nan, 1.0]),
        ValueError,
        r"index 0 must be 0 or 1, got missing value nan$",
    ),
    "positive_infinity": (
        lambda: [1, float("inf")],
        ValueError,
        r"index 1 must be 0 or 1, got non-finite value inf$",
    ),
    "negative_infinity_array": (
        lambda: np.array([0.0, -np.inf]),
        ValueError,
        r"index 1 must be 0 or 1, got non-finite value -inf$",
    ),
    "none": (
        lambda: [0, None],
        ValueError,
        r"index 1 must be 0 or 1, got missing value None$",
    ),
    "nullable_int_missing": (
        lambda: pd.Series([1, pd.NA], dtype="Int64"),
        ValueError,
        r"index 1 must be 0 or 1, got missing value <NA>$",
    ),
    "nullable_boolean_missing": (
        lambda: pd.Series([pd.NA, True], dtype="boolean"),
        ValueError,
        r"index 0 must be 0 or 1, got missing value <NA>$",
    ),
    "float_series_missing": (
        lambda: pd.Series([0.0, np.nan]),
        ValueError,
        r"index 1 must be 0 or 1, got missing value nan$",
    ),
    "non_canonical_string": (
        lambda: ["0", "1.0"], ValueError, r"index 1 must be 0 or 1, got '1\.0'$"
    ),
    "scalar_integer": (
        lambda: 1, TypeError, r"conversions must be .* not a bare int: 1$"
    ),
    "scalar_fraction": (
        lambda: 0.5, TypeError, r"conversions must be .* not a bare float: 0\.5$"
    ),
    "bare_string": (
        lambda: "01", TypeError, r"conversions must be .* not a bare str: '01'$"
    ),
    "nested_list": (
        lambda: [[0], [1]],
        ValueError,
        r"conversions must be one-dimensional; the entry at index 0 is a "
        r"nested list: \[0\]$",
    ),
    "nested_array_entries": (
        lambda: [np.array([0, 1]), np.array([1, 0])],
        ValueError,
        r"conversions must be one-dimensional; the entry at index 0 is a "
        r"nested ndarray: array\(\[0, 1\]\)$",
    ),
    "two_dimensional_array": (
        lambda: np.array([[0], [1]]),
        ValueError,
        r"conversions must be one-dimensional, got an array with shape "
        r"\(2, 1\)$",
    ),
}

# (paths, exception type, message pattern) for two aligned outcomes.
INVALID_PATH_CONTAINERS = {
    "bare_string": (
        lambda: "A > B",
        TypeError,
        r"paths must be .* not a bare str: 'A > B'$",
    ),
    "nested_list": (
        lambda: [["A", "B"], ["C"]],
        ValueError,
        r"paths must be one-dimensional; the entry at index 0 is a nested "
        r"list: \['A', 'B'\]$",
    ),
    "two_dimensional_array": (
        lambda: np.array([["A"], ["B"]]),
        ValueError,
        r"paths must be one-dimensional, got an array with shape \(2, 1\)$",
    ),
}

ENTRY_PATHS = ["A > B > C", "A > C", "B > C", "A > B", "C", "A", "B > A", "C > B"]
ENTRY_DATES = pd.to_datetime(
    [
        "2025-01-05", "2025-01-06", "2025-01-07", "2025-01-08",
        "2025-04-05", "2025-04-06", "2025-04-07", "2025-04-08",
    ]
)


def _markov():
    return MarkovAttribution(removal_strategy="redirect_to_null")


def _temporal_markov():
    return TemporalMarkovAttribution(
        freq="Q", min_samples_per_epoch=1, removal_strategy="redirect_to_null"
    )


def _temporal_shapley():
    return TemporalShapleyAttribution(freq="Q", min_samples_per_epoch=1)


def _compare_fit(paths, dates, conversions):
    comparison = CompareAttribution(markov_removal_strategy="redirect_to_null")
    comparison.fit(paths, conversions)
    return comparison.attribution_frame()


def _compare_with_baselines_fit(paths, dates, conversions):
    comparison = CompareAttribution(
        markov_removal_strategy="redirect_to_null", include_baselines=True
    )
    comparison.fit(paths, conversions)
    return comparison.attribution_frame()


# Every public entry point that accepts journey outcomes.  Each returns the
# outputs a caller reads, so a fit can be compared with a reference fit.
ENTRY_POINTS = {
    "markov_fit": lambda paths, dates, conversions: {
        key: value
        for key, value in _markov().fit(paths, conversions).items()
        if key in ("attribution", "removal_effects", "total_conversion_probability")
    },
    "shapley_fit": lambda paths, dates, conversions: {
        key: value
        for key, value in ShapleyAttribution().fit(paths, conversions).items()
        if key in ("attribution", "shapley_values", "value_function")
    },
    "compare_fit": _compare_fit,
    "compare_with_baselines_fit": _compare_with_baselines_fit,
    "first_touch_fit": lambda paths, dates, conversions: {
        key: value
        for key, value in FirstTouchAttribution().fit(paths, conversions).items()
        if key in ("attribution", "credited_conversions")
    },
    "last_touch_fit": lambda paths, dates, conversions: {
        key: value
        for key, value in LastTouchAttribution().fit(paths, conversions).items()
        if key in ("attribution", "credited_conversions")
    },
    "temporal_markov_fit": lambda paths, dates, conversions: (
        _temporal_markov().fit(paths, dates, conversions)["attribution_matrix"]
    ),
    "temporal_markov_fit_by_epoch": lambda paths, dates, conversions: (
        _temporal_markov().fit_by_epoch(paths, dates, conversions)[
            "attribution_matrix"
        ]
    ),
    "temporal_markov_fit_static": lambda paths, dates, conversions: (
        _temporal_markov().fit_static(paths, conversions)["attribution"]
    ),
    "temporal_shapley_fit": lambda paths, dates, conversions: (
        _temporal_shapley().fit(paths, dates, conversions)["attribution_matrix"]
    ),
    "temporal_shapley_fit_by_epoch": lambda paths, dates, conversions: (
        _temporal_shapley().fit_by_epoch(paths, dates, conversions)[
            "attribution_matrix"
        ]
    ),
    "temporal_shapley_fit_static": lambda paths, dates, conversions: (
        _temporal_shapley().fit_static(paths, conversions)["attribution"]
    ),
    "validate_data": lambda paths, dates, conversions: validate_data(
        paths, conversions
    ),
}


def _assert_same_output(actual, expected) -> None:
    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected)
    else:
        assert actual == expected


@pytest.mark.parametrize("container", sorted(VALID_OUTCOME_CONTAINERS))
def test_valid_binary_containers_coerce_to_the_same_integers(container):
    supplied = VALID_OUTCOME_CONTAINERS[container]()

    coerced = to_int_array(supplied)
    assert coerced.dtype == np.dtype(int)
    assert coerced.tolist() == VALID_OUTCOMES

    paths_arr, conv_arr = prepare_journey_data(ENTRY_PATHS, supplied)
    assert paths_arr.tolist() == ENTRY_PATHS
    assert conv_arr.dtype == np.dtype(int)
    assert conv_arr.tolist() == VALID_OUTCOMES
    assert validate_data(ENTRY_PATHS, supplied) is None


@pytest.mark.parametrize("entry_point", sorted(ENTRY_POINTS))
@pytest.mark.parametrize("container", sorted(VALID_OUTCOME_CONTAINERS))
def test_valid_binary_containers_produce_unchanged_fits(entry_point, container):
    fit = ENTRY_POINTS[entry_point]
    reference = fit(ENTRY_PATHS, ENTRY_DATES, list(VALID_OUTCOMES))

    actual = fit(ENTRY_PATHS, ENTRY_DATES, VALID_OUTCOME_CONTAINERS[container]())

    _assert_same_output(actual, reference)


@pytest.mark.parametrize("entry_point", sorted(ENTRY_POINTS))
@pytest.mark.parametrize("case", sorted(INVALID_OUTCOMES))
def test_public_entry_points_reject_invalid_outcomes(entry_point, case):
    build, error, pattern = INVALID_OUTCOMES[case]

    with pytest.raises(error, match=pattern):
        ENTRY_POINTS[entry_point](["A", "B"], ENTRY_DATES[:2], build())


@pytest.mark.parametrize("entry_point", sorted(ENTRY_POINTS))
@pytest.mark.parametrize("case", sorted(INVALID_PATH_CONTAINERS))
def test_public_entry_points_reject_non_one_dimensional_paths(entry_point, case):
    build, error, pattern = INVALID_PATH_CONTAINERS[case]

    with pytest.raises(error, match=pattern):
        ENTRY_POINTS[entry_point](build(), ENTRY_DATES[:2], [0, 1])


@pytest.mark.parametrize("case", sorted(INVALID_OUTCOMES))
def test_to_int_array_validates_the_supplied_values(case):
    build, error, pattern = INVALID_OUTCOMES[case]

    with pytest.raises(error, match=pattern):
        to_int_array(build())


@pytest.mark.parametrize("case", sorted(INVALID_PATH_CONTAINERS))
def test_to_string_array_rejects_non_one_dimensional_paths(case):
    build, error, pattern = INVALID_PATH_CONTAINERS[case]

    with pytest.raises(error, match=pattern):
        to_string_array(build())
    with pytest.raises(error, match=pattern):
        extract_channels(build())


def test_reported_truncation_examples_are_rejected_not_coerced():
    """[0.9, 1.9] used to become [0, 1] and [-0.9, 1.1] one conversion."""
    for outcomes in ([0.9, 1.9], [-0.9, 1.1]):
        with pytest.raises(ValueError, match="must be 0 or 1"):
            validate_data(["A", "B"], outcomes)
        with pytest.raises(ValueError, match="must be 0 or 1"):
            ShapleyAttribution().fit(["A", "B"], outcomes)
        with pytest.raises(ValueError, match="must be 0 or 1"):
            _markov().fit(["A", "B"], outcomes)


@pytest.mark.parametrize(
    "value",
    [" 1", "1 ", "01", "+1", "1.0", "0.0", "1e0", "true", "True", "yes", ""],
)
def test_only_canonical_binary_strings_are_accepted(value):
    assert to_int_array(["0", "1"]).tolist() == [0, 1]
    assert to_int_array([b"0", b"1"]).tolist() == [0, 1]

    with pytest.raises(ValueError, match="index 1 must be 0 or 1, got "):
        to_int_array(["0", value])


@pytest.mark.parametrize("value", [1e-17, 1 - 1e-16, 1 + 1e-15, -1e-300])
def test_near_binary_floats_are_not_rounded_to_outcomes(value):
    with pytest.raises(ValueError, match="index 0 must be 0 or 1"):
        to_int_array([value, 1.0])
    with pytest.raises(ValueError, match="index 0 must be 0 or 1"):
        to_int_array(np.array([value, 1.0]))


def test_alignment_errors_keep_their_precedence_and_messages():
    with pytest.raises(ValueError, match=r"Paths \(1\) and conversions \(2\)"):
        validate_data(["A > B"], [1, 2])
    with pytest.raises(ValueError, match=r"Paths \(1\) and conversions \(2\)"):
        validate_data(["A > B"], np.array([1, 2]))
    with pytest.raises(ValueError, match="No paths provided"):
        prepare_journey_data([], np.array([], dtype=int))


def test_validated_outcomes_do_not_alias_the_supplied_array():
    supplied = np.array([0, 1, 1])

    coerced = to_int_array(supplied)
    coerced[0] = 1

    assert supplied.tolist() == [0, 1, 1]


# --------------------------------------------------------------------------- #
# Conversion-labels handling (P0)                                             #
# --------------------------------------------------------------------------- #
# MODIFIED (P0): full test coverage for the new conversion_labels plumbing.
def test_normalize_conversion_labels_none_returns_empty():
    assert _normalize_conversion_labels(None) == frozenset()


def test_normalize_conversion_labels_strips_whitespace():
    assert _normalize_conversion_labels(["  Purchase  ", "Sale"]) == frozenset(
        {"Purchase", "Sale"}
    )


def test_normalize_conversion_labels_rejects_bare_string():
    # Common bug: passing "Purchase" instead of ["Purchase"] iterates chars.
    with pytest.raises(TypeError, match="not a bare str"):
        _normalize_conversion_labels("Purchase")


def test_normalize_conversion_labels_rejects_non_string_entry():
    with pytest.raises(TypeError, match="entries must be str"):
        _normalize_conversion_labels(["Purchase", 42])


def test_normalize_conversion_labels_rejects_empty_label():
    with pytest.raises(ValueError, match="empty label"):
        _normalize_conversion_labels(["Purchase", "   "])


def test_normalize_conversion_labels_rejects_special_token_collision():
    for tok in SPECIAL_TOKENS:
        with pytest.raises(ValueError, match="SPECIAL_TOKENS"):
            _normalize_conversion_labels([tok])


def test_strip_conversion_labels_removes_tokens():
    assert strip_conversion_labels("A > Purchase > B", ["Purchase"]) == "A > B"
    # Trailing endpoint.
    assert strip_conversion_labels("A > B > Purchase", ["Purchase"]) == "A > B"
    # Multiple labels.
    assert (
        strip_conversion_labels("A > Sale > B > Purchase", ["Purchase", "Sale"])
        == "A > B"
    )


def test_strip_conversion_labels_case_sensitive():
    # Spec: "Case-sensitive exact match."
    assert (
        strip_conversion_labels("A > purchase > B", ["Purchase"])
        == "A > purchase > B"
    )


def test_strip_conversion_labels_empty_labels_is_identity():
    assert strip_conversion_labels("A > B > C", []) == "A > B > C"
    assert strip_conversion_labels("A > B > C", None or []) == "A > B > C"


def test_strip_conversion_labels_array_preserves_length():
    paths = ["A > B > Purchase", "", "Purchase", "A > B"]
    out = strip_conversion_labels_array(paths, ["Purchase"])
    assert len(out) == len(paths)
    assert list(out) == ["A > B", "", "", "A > B"]


def test_extract_channels_excludes_conversion_labels():
    paths = ["A > B > Purchase", "Purchase > C"]
    assert extract_channels(paths, conversion_labels=["Purchase"]) == ["A", "B", "C"]


def test_extract_channels_default_still_includes_purchase():
    # Backwards compat: no conversion_labels => every token is a channel.
    paths = ["A > B > Purchase"]
    assert extract_channels(paths) == ["A", "B", "Purchase"]
