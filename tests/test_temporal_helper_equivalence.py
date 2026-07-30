"""Equivalence contracts for helpers shared by both temporal models.

These tests intentionally exercise the helpers through both legacy module
locations.  They protect behavior while Unit 12 selects one authoritative
implementation and keeps the existing private import paths as delegates.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from do_attribution import temporal_diagnostics, temporal_markov, temporal_shapley


MODEL_MODULES = (temporal_shapley, temporal_markov)


def _outcome(function: Callable[..., Any], *args: Any) -> tuple[Any, ...]:
    try:
        return ("return", function(*args))
    except Exception as exc:  # noqa: BLE001 - exception parity is the contract
        return ("raise", type(exc), str(exc))


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"attribution": None},
        {"attribution": {}},
        {"attribution": {"Email": 0.25, "Search": 0.75}},
    ],
)
def test_epoch_attribution_helpers_are_equivalent(result):
    outcomes = [module._epoch_attribution(result) for module in MODEL_MODULES]
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("D", "D"),
        ("W", "W"),
        ("M", "ME"),
        ("Q", "QE"),
        ("Y", "YE"),
        ("A", "YE"),
    ],
)
def test_pandas_frequency_alias_helpers_are_equivalent(alias, expected):
    helpers = [
        temporal_shapley._pandas_freq_alias,
        temporal_markov._pandas_freq_alias,
        temporal_diagnostics._pandas_freq_alias,
    ]
    assert [helper(alias) for helper in helpers] == [expected] * len(helpers)


@pytest.mark.parametrize(
    "value",
    ["first_touch", "conversion", "split", "invalid", None, 42],
)
def test_epoch_assignment_validators_are_equivalent(value):
    outcomes = [
        _outcome(module._validate_epoch_assignment, value)
        for module in MODEL_MODULES
    ]
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize(
    "value",
    [
        "D",
        "M",
        "Q-DEC",
        {2024: "Q", 2025: "M"},
        {},
        "NOT-A-FREQUENCY",
        {"2024": "Q"},
        {True: "Q"},
        {2024: 12},
        42,
    ],
)
def test_frequency_validators_are_equivalent(value):
    outcomes = [_outcome(module._validate_freq, value) for module in MODEL_MODULES]
    assert outcomes[0] == outcomes[1]


def test_period_and_index_helpers_are_equivalent():
    periods = [
        pd.Period("2025-02", "M"),
        pd.Period("2024Q4", "Q"),
        pd.Period("2025-01", "M"),
    ]
    expected_order = sorted(periods, key=temporal_shapley._get_period_key)
    assert sorted(periods, key=temporal_markov._get_period_key) == expected_order

    same_frequency = (pd.Period("2024Q4", "Q"), pd.Period("2025Q1", "Q"))
    mixed_frequency = (pd.Period("2024Q4", "Q"), pd.Period("2025-01", "M"))
    labels = ("first", "second")
    for values in (same_frequency, mixed_frequency, labels):
        outcomes = [
            module._periods_have_different_freq(*values)
            for module in MODEL_MODULES
        ]
        assert outcomes[0] == outcomes[1]

    frame = pd.DataFrame(
        {"Email": [0.2, 0.8], "Search": [0.8, 0.2]},
        index=["second", "first"],
    )
    aligned = [
        module._align_index_to_epochs(frame, ["first", "second"])
        for module in MODEL_MODULES
    ]
    pd.testing.assert_frame_equal(aligned[0], aligned[1])


@pytest.mark.parametrize("value_name", ["attribution", "raw_value"])
def test_dataframe_projection_helpers_are_equivalent(value_name):
    matrix = pd.DataFrame(
        [[0.25, np.nan], [0.4, 0.6]],
        index=[pd.Period("2024Q1", "Q"), pd.Period("2024Q2", "Q")],
        columns=["Email", "Search"],
    )
    projected = [
        module._matrix_to_long_frame(matrix, value_name)
        for module in MODEL_MODULES
    ]
    pd.testing.assert_frame_equal(projected[0], projected[1])

    index = pd.MultiIndex.from_product(
        [["Email", "Search"], matrix.index],
        names=["channel", "epoch"],
    )
    diagnostic = pd.DataFrame(
        {
            "n_paths_containing_channel": [3, 4, 5, 6],
            "n_conversions_on_paths_containing_channel": [1, 2, 3, 4],
            "thin": [True, False, False, False],
        },
        index=index,
    )
    flattened = [
        module._sample_sizes_to_long_frame(diagnostic)
        for module in MODEL_MODULES
    ]
    pd.testing.assert_frame_equal(flattened[0], flattened[1])

    empty = diagnostic.iloc[0:0]
    empty_flattened = [
        module._sample_sizes_to_long_frame(empty)
        for module in MODEL_MODULES
    ]
    pd.testing.assert_frame_equal(empty_flattened[0], empty_flattened[1])
