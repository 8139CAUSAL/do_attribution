"""The values quoted in the README are the values the package computes.

Every number checked here is printed by a README code block for the
ten-journey quick-start dataset.  The README documents them as comments
next to the calls, so a change in model behaviour that is not carried into
the README fails here instead of leaving stale output in the documentation.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import pytest

from do_attribution import (
    CompareAttribution,
    MarkovAttribution,
    ShapleyAttribution,
)


README = (Path(__file__).resolve().parents[1] / "README.md").read_text(
    encoding="utf-8"
)
JOURNEYS = [
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
CONVERSIONS = [1, 1, 1, 0, 1, 1, 0, 1, 1, 0]


def _rounded(values: dict) -> str:
    return str({channel: round(value, 4) for channel, value in values.items()})


def test_quick_start_dataset_is_the_one_the_readme_prints() -> None:
    for journey in JOURNEYS:
        assert f'"{journey}",' in README
    assert f"conversions = {CONVERSIONS}" in README


def test_quick_start_markov_output() -> None:
    result = MarkovAttribution(
        order=1,
        removal_strategy="redirect_to_null",
        conversion_labels=["Purchase"],
        return_transition_matrix=True,
    ).fit(JOURNEYS, CONVERSIONS)

    assert _rounded(result["attribution"]) in README
    assert (
        f"P(conversion | start) = {result['total_conversion_probability']:.4f}"
        in README
    )


def test_quick_start_shapley_output() -> None:
    result = ShapleyAttribution(conversion_labels=["Purchase"]).fit(
        JOURNEYS, CONVERSIONS
    )

    assert result["normalization_status"] == "available"
    assert f"# {_rounded(result['attribution'])}" in README
    assert f"# {_rounded(result['shapley_values'])}" in README
    assert result["characteristic_function"] == "cumulative"
    assert result["raw_sum"] == pytest.approx(7.0)
    assert result["cancellation_ratio"] == pytest.approx(1.0)
    assert 'shapley["raw_sum"]                   # 7.0' in README
    assert 'shapley["cancellation_ratio"]        # 1.0' in README


def test_legacy_shapley_game_output() -> None:
    with pytest.warns(UserWarning, match="normalization is unavailable"):
        legacy = ShapleyAttribution(
            conversion_labels=["Purchase"], characteristic_function="exact"
        ).fit(JOURNEYS, CONVERSIONS)

    assert legacy["normalization_status"] == "unavailable"
    assert legacy["normalization_reason"] == "near_cancellation"
    assert f"# {_rounded(legacy['shapley_values'])}" in README
    assert f"# {legacy['raw_magnitude']:.4f}" in README
    assert abs(legacy["raw_sum"]) < 1e-12
    assert legacy["cancellation_ratio"] <= 1e-12


def test_side_by_side_comparison_output() -> None:
    with warnings.catch_warnings():
        # The default Markov strategy emits a one-time migration notice.
        warnings.simplefilter("ignore", UserWarning)
        comparison = CompareAttribution(conversion_labels=["Purchase"])
        comparison.fit(JOURNEYS, CONVERSIONS)

    frame = comparison.attribution_frame()
    assert list(frame.columns) == ["markov", "shapley"]
    for channel, row in frame.iterrows():
        line = f"# {channel:<6}  {row['markov']:.6f}  {row['shapley']:.6f}"
        assert line in README, line
    assert comparison.unavailable_models() == {}
    assert "comparison.unavailable_models()     # {}" in README
