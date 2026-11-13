"""First-touch and last-touch baseline tests.

The baselines are fixed reference allocations.  They follow the built-in
models' conventions: the :class:`BaseAttribution` API, conversion-endpoint
stripping, the shared validation of journeys and binary outcomes, and the
shared normalization contract (``attribution`` is ``None`` when nothing can
be credited to a channel).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from do_attribution import (
    BaseAttribution,
    FirstTouchAttribution,
    LastTouchAttribution,
    ShapleyAttribution,
)


BASELINES = [FirstTouchAttribution, LastTouchAttribution]
NORMALIZATION_KEYS = {
    "normalization_status",
    "normalization_reason",
    "normalization_denominator",
    "raw_sum",
    "raw_magnitude",
    "cancellation_ratio",
}


# --------------------------------------------------------------------------- #
# Allocation rule                                                              #
# --------------------------------------------------------------------------- #
def test_first_touch_credits_the_first_channel_of_each_converting_journey(
    example_paths, example_conversions
):
    result = FirstTouchAttribution(conversion_labels=["Purchase"]).fit(
        example_paths, example_conversions
    )

    # Converting journeys start with: Email, Search, Social, Search, Direct,
    # Search, Social.
    assert result["credited_conversions"] == {
        "Direct": 1.0,
        "Email": 1.0,
        "Search": 3.0,
        "Social": 2.0,
    }
    assert result["attribution"] == pytest.approx(
        {"Direct": 1 / 7, "Email": 1 / 7, "Search": 3 / 7, "Social": 2 / 7}
    )


def test_last_touch_credits_the_last_channel_of_each_converting_journey(
    example_paths, example_conversions
):
    result = LastTouchAttribution(conversion_labels=["Purchase"]).fit(
        example_paths, example_conversions
    )

    # Converting journeys end with: Social, Email, Social, Email, Direct,
    # Social, Email.  Search never closes a converting journey.
    assert result["credited_conversions"] == {
        "Direct": 1.0,
        "Email": 3.0,
        "Search": 0.0,
        "Social": 3.0,
    }
    assert result["attribution"] == pytest.approx(
        {"Direct": 1 / 7, "Email": 3 / 7, "Search": 0.0, "Social": 3 / 7}
    )


@pytest.mark.parametrize("model_type", BASELINES)
def test_result_has_shares_raw_credits_and_normalization_diagnostics(
    model_type, example_paths, example_conversions
):
    model = model_type(conversion_labels=["Purchase"])
    result = model.fit(example_paths, example_conversions)

    assert set(result) == {"attribution", "credited_conversions"} | NORMALIZATION_KEYS
    assert sum(result["attribution"].values()) == pytest.approx(1.0)
    assert all(0.0 <= share <= 1.0 for share in result["attribution"].values())
    # Every conversion on a journey with a channel is credited exactly once.
    assert sum(result["credited_conversions"].values()) == sum(example_conversions)
    assert result["normalization_status"] == "available"
    assert result["normalization_reason"] is None
    assert result["raw_sum"] == 7.0
    assert result["cancellation_ratio"] == 1.0
    assert list(result["attribution"]) == model.get_channels() == [
        "Direct",
        "Email",
        "Search",
        "Social",
    ]
    assert model.get_attribution() == result["attribution"]


def test_first_and_last_touch_differ_only_on_multi_channel_journeys():
    single = ["A", "B", "A"]
    outcomes = [1, 1, 0]
    first = FirstTouchAttribution().fit(single, outcomes)
    last = LastTouchAttribution().fit(single, outcomes)
    assert first["credited_conversions"] == last["credited_conversions"] == {
        "A": 1.0,
        "B": 1.0,
    }

    multi = ["A > B > C", "C > B > A"]
    assert FirstTouchAttribution().fit(multi, [1, 1])["credited_conversions"] == {
        "A": 1.0,
        "B": 0.0,
        "C": 1.0,
    }
    assert LastTouchAttribution().fit(multi, [1, 0])["credited_conversions"] == {
        "A": 0.0,
        "B": 0.0,
        "C": 1.0,
    }


def test_repeated_touches_credit_the_journey_once():
    paths = ["A > B > A", "B > A > B > B"]
    assert FirstTouchAttribution().fit(paths, [1, 1])["credited_conversions"] == {
        "A": 1.0,
        "B": 1.0,
    }
    assert LastTouchAttribution().fit(paths, [1, 1])["credited_conversions"] == {
        "A": 1.0,
        "B": 1.0,
    }


def test_non_converting_journeys_receive_no_credit():
    result = LastTouchAttribution().fit(["A > B", "A > C", "C"], [0, 1, 0])
    assert result["credited_conversions"] == {"A": 0.0, "B": 0.0, "C": 1.0}
    assert result["attribution"] == {"A": 0.0, "B": 0.0, "C": 1.0}


# --------------------------------------------------------------------------- #
# Conversion endpoints, reserved tokens, separators                            #
# --------------------------------------------------------------------------- #
def test_conversion_label_is_never_the_last_touch(
    example_paths, example_paths_no_endpoint, example_conversions
):
    labelled = LastTouchAttribution(conversion_labels=["Purchase"]).fit(
        example_paths, example_conversions
    )
    prestripped = LastTouchAttribution().fit(
        example_paths_no_endpoint, example_conversions
    )
    assert "Purchase" not in labelled["attribution"]
    assert "Purchase" not in labelled["credited_conversions"]
    assert labelled == prestripped

    # Without the declaration the endpoint token is an ordinary last touch.
    undeclared = LastTouchAttribution().fit(example_paths, example_conversions)
    assert undeclared["credited_conversions"]["Purchase"] == 7.0


@pytest.mark.parametrize("model_type", BASELINES)
def test_reserved_tokens_are_not_credited(model_type):
    model = model_type()
    result = model.fit(["START > A > NULL", "CONVERSION > B > CONVERSION"], [1, 1])

    assert model.get_channels() == ["A", "B"]
    assert result["credited_conversions"] == {"A": 1.0, "B": 1.0}


@pytest.mark.parametrize("model_type", BASELINES)
def test_custom_separator_and_whitespace(model_type):
    result = model_type(separator="|").fit(["A | B", " B|C "], [1, 1])
    expected = (
        {"A": 1.0, "B": 1.0, "C": 0.0}
        if model_type is FirstTouchAttribution
        else {"A": 0.0, "B": 1.0, "C": 1.0}
    )
    assert result["credited_conversions"] == expected


@pytest.mark.parametrize("model_type", BASELINES)
def test_conversion_label_validation_matches_the_built_in_models(model_type):
    with pytest.raises(ValueError, match="SPECIAL_TOKENS"):
        model_type(conversion_labels=["CONVERSION"])
    with pytest.raises(TypeError, match="not a bare str"):
        model_type(conversion_labels="Purchase")


# --------------------------------------------------------------------------- #
# Unavailable attribution                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model_type", BASELINES)
@pytest.mark.parametrize(
    ("paths", "conversions", "labels", "expected_credits"),
    [
        # No journey converts.
        (["A > B", "B > C"], [0, 0], None, {"A": 0.0, "B": 0.0, "C": 0.0}),
        # The only conversion sits on an endpoint-only journey.
        (["Purchase", "A > Purchase", "B"], [1, 0, 0], ["Purchase"], {"A": 0.0, "B": 0.0}),
        # No channel survives at all.
        (["Purchase", "Purchase"], [1, 1], ["Purchase"], {}),
    ],
)
def test_nothing_to_credit_returns_unavailable_attribution(
    model_type, paths, conversions, labels, expected_credits
):
    model = model_type(conversion_labels=labels)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = model.fit(paths, conversions)

    assert result["attribution"] is None
    assert result["credited_conversions"] == expected_credits
    assert result["normalization_status"] == "unavailable"
    assert result["normalization_reason"] == "zero_raw_magnitude"
    assert result["normalization_denominator"] is None
    assert result["raw_sum"] == 0.0
    assert result["raw_magnitude"] == 0.0
    assert result["cancellation_ratio"] is None
    with pytest.raises(ValueError, match="unavailable"):
        model.get_attribution()
    # The cohort row reports the unavailable state where shares would go.
    cohort_row = next(
        line
        for line in model.summary().splitlines()
        if line.startswith("| All paths")
    )
    assert cohort_row.rstrip(" |").endswith(
        "Normalized attribution unavailable (zero_raw_magnitude); no "
        "conversion was credited to a channel"
    )


@pytest.mark.parametrize("model_type", BASELINES)
def test_unavailability_matches_shapley_for_the_same_inputs(model_type):
    """The baselines and the default Shapley game credit the same journeys."""
    for paths, conversions in (
        (["A", "B"], [0, 0]),
        (["A > B", "", "B"], [0, 1, 0]),
        (["A > B", "B"], [1, 0]),
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            baseline = model_type().fit(paths, conversions)
            shapley = ShapleyAttribution().fit(paths, conversions)
        assert baseline["normalization_status"] == shapley["normalization_status"]
        assert baseline["normalization_reason"] == shapley["normalization_reason"]
        assert baseline["raw_sum"] == shapley["raw_sum"]


# --------------------------------------------------------------------------- #
# Base-class API and shared validation                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model_type", BASELINES)
def test_baselines_are_base_attribution_models(model_type):
    model = model_type()
    assert isinstance(model, BaseAttribution)
    assert model.name == model_type.__name__
    with pytest.raises(ValueError, match="Call fit"):
        model.get_attribution()
    with pytest.raises(RuntimeError, match="fitted attribution model"):
        model.summary()


@pytest.mark.parametrize(
    ("model_type", "title"),
    [
        (FirstTouchAttribution, "First-touch credited conversions"),
        (LastTouchAttribution, "Last-touch credited conversions"),
    ],
)
def test_summary_and_report_render_from_the_shared_text_engine(
    model_type, title, example_paths, example_conversions
):
    model = model_type(conversion_labels=["Purchase"])
    model.fit(example_paths, example_conversions)

    summary = model.summary()
    assert summary.startswith("Attribution Summary\n")
    assert f"Model: {model_type.__name__}" in summary
    assert "Overall: 10 paths, 7 conversions, 70.00% conversion rate" in summary
    assert title in summary
    assert "Path diagnostics" in summary
    assert "Purchase" not in summary
    report = model.summary(output_format="report")
    assert report.startswith(f"Attribution report for {model_type.__name__}.")
    assert model.summary() == summary


@pytest.mark.parametrize("model_type", BASELINES)
@pytest.mark.parametrize(
    ("conversions", "error", "pattern"),
    [
        ([0.9, 1.9], ValueError, r"index 0 must be 0 or 1, got 0\.9$"),
        ([-0.9, 1.1], ValueError, r"index 0 must be 0 or 1, got -0\.9$"),
        ([0, 2], ValueError, r"index 1 must be 0 or 1, got 2$"),
        ([0, float("nan")], ValueError, r"got missing value nan$"),
        ([1, float("inf")], ValueError, r"got non-finite value inf$"),
        (pd.Series([1, pd.NA], dtype="Int64"), ValueError, r"got missing value <NA>$"),
        (1, TypeError, r"not a bare int: 1$"),
        ([[0], [1]], ValueError, r"index 0 is a nested list: \[0\]$"),
        ([1], ValueError, r"Paths \(2\) and conversions \(1\)"),
    ],
)
def test_outcomes_go_through_the_shared_validation(
    model_type, conversions, error, pattern
):
    with pytest.raises(error, match=pattern):
        model_type().fit(["A", "B"], conversions)


@pytest.mark.parametrize("model_type", BASELINES)
def test_path_container_shape_is_validated(model_type):
    with pytest.raises(TypeError, match="not a bare str: 'A > B'"):
        model_type().fit("A > B", [0, 1])
    with pytest.raises(ValueError, match="No paths provided"):
        model_type().fit([], [])


@pytest.mark.parametrize("model_type", BASELINES)
def test_accepts_lists_arrays_and_series(model_type):
    paths = ["A > B", "B > C", "A"]
    outcomes = [1, 1, 0]
    reference = model_type().fit(paths, outcomes)

    assert model_type().fit(np.array(paths), np.array(outcomes)) == reference
    assert model_type().fit(pd.Series(paths), pd.Series(outcomes)) == reference
    assert model_type().fit(tuple(paths), [True, True, False]) == reference
    assert model_type().fit(paths, [1.0, 1.0, 0.0]) == reference


@pytest.mark.parametrize("model_type", BASELINES)
def test_refit_replaces_the_previous_fit(model_type):
    model = model_type()
    model.fit(["A", "B"], [1, 0])
    result = model.fit(["C > D"], [1])

    assert model.get_channels() == ["C", "D"]
    assert set(result["credited_conversions"]) == {"C", "D"}
    assert set(model.get_attribution()) == {"C", "D"}
