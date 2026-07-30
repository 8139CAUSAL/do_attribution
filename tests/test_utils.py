"""Tests for the shared preprocessing utilities.

Mirrors ``markov-attribution/tests/test_utils.py`` so the unified
library's behaviour on the common utilities is explicitly locked down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from do_attribution.utils import (
    DEFAULT_SEPARATOR,
    SPECIAL_TOKENS,
    _normalize_conversion_labels,  # MODIFIED: P0 regression tests.
    create_n_order_states,
    extract_channels,
    get_powerset,
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
