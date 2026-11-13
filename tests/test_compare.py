"""Tests for :class:`CompareAttribution`."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from do_attribution import (
    BaseAttribution,
    CompareAttribution,
    FirstTouchAttribution,
    LastTouchAttribution,
    MarkovAttribution,
    ShapleyAttribution,
)


def test_fit_returns_both_model_results(example_paths, example_conversions):
    cmp_ = CompareAttribution()
    results = cmp_.fit(example_paths, example_conversions)
    assert set(results.keys()) == {"markov", "shapley"}
    # MODIFIED (get-well Unit 1): a model may report that normalization is
    # unavailable.  Where a share vector exists it must sum to 1; where it
    # does not, 'attribution' is None and the raw values remain.
    for name, r in results.items():
        if r["attribution"] is None:
            assert r["normalization_status"] == "unavailable"
            assert r["normalization_reason"]
        else:
            assert abs(sum(r["attribution"].values()) - 1.0) < 1e-10


def test_attribution_frame_columns(example_paths, example_conversions):
    cmp_ = CompareAttribution()
    cmp_.fit(example_paths, example_conversions)
    frame = cmp_.attribution_frame()
    assert list(frame.columns) == ["markov", "shapley"]
    # Index should be sorted alphabetically
    assert list(frame.index) == sorted(frame.index)
    # MODIFIED (get-well Unit 1): available columns sum to 1; a column for a
    # model with no valid share vector is all-NaN rather than all-zero.
    unavailable = cmp_.unavailable_models()
    for col in frame.columns:
        if col in unavailable:
            assert frame[col].isna().all()
        else:
            assert frame[col].sum() == pytest.approx(1.0)


def test_unavailable_model_column_is_nan_not_zero(
    example_paths, example_conversions
):
    """A refused normalization must not masquerade as all-zero shares."""
    cmp_ = CompareAttribution(shapley_characteristic_function="exact")
    cmp_.fit(example_paths, example_conversions)
    frame = cmp_.attribution_frame()
    unavailable = cmp_.unavailable_models()

    assert "shapley" in unavailable  # this fixture cancels
    assert unavailable["shapley"] == "near_cancellation"
    assert frame["shapley"].isna().all()
    assert not (frame["shapley"] == 0.0).any()


@pytest.mark.parametrize(
    "removal_strategy", ["fail", "fail_refit", "detour", "redirect_to_null"]
)
def test_zero_conversion_markov_column_is_missing_not_equal_shares(
    removal_strategy,
):
    """With no removal effect to normalize, Markov is as unavailable as
    Shapley for the same inputs instead of reporting an equal split."""
    cmp_ = CompareAttribution(markov_removal_strategy=removal_strategy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        results = cmp_.fit(["A", "B"], [0, 0])

    assert results["markov"]["attribution"] is None
    assert results["markov"]["removal_effects"] == {"A": 0.0, "B": 0.0}
    assert results["markov"]["total_conversion_probability"] == 0.0
    assert cmp_.unavailable_models() == {
        "markov": "zero_raw_magnitude",
        "shapley": "zero_raw_magnitude",
    }

    frame = cmp_.attribution_frame()
    assert list(frame.index) == ["A", "B"]
    assert frame["markov"].isna().all()
    assert frame["shapley"].isna().all()
    assert not (frame["markov"] == 0.5).any()
    assert cmp_.difference_frame("markov")["shapley_minus_markov"].isna().all()


def test_difference_frame_vs_markov_reference(example_paths, example_conversions):
    cmp_ = CompareAttribution()
    cmp_.fit(example_paths, example_conversions)
    diff = cmp_.difference_frame(reference="markov")
    # Reference column should be dropped
    assert "markov_minus_markov" not in diff.columns
    assert "shapley_minus_markov" in diff.columns


def test_unknown_reference_rejected(example_paths, example_conversions):
    cmp_ = CompareAttribution()
    cmp_.fit(example_paths, example_conversions)
    with pytest.raises(ValueError):
        cmp_.difference_frame(reference="bogus")


def test_extra_models_are_included(example_paths, example_conversions):
    class DummyEqualModel(BaseAttribution):
        def fit(self, paths, conversions):
            from do_attribution import extract_channels
            self.channels = extract_channels(paths, self.separator)
            share = 1.0 / len(self.channels)
            self._last_attribution = {c: share for c in self.channels}
            return {"attribution": dict(self._last_attribution)}

    cmp_ = CompareAttribution(extra_models={"equal": DummyEqualModel()})
    results = cmp_.fit(example_paths, example_conversions)
    assert "equal" in results
    frame = cmp_.attribution_frame()
    assert "equal" in frame.columns
    # Dummy splits evenly
    assert frame["equal"].nunique() == 1


def test_get_results_requires_fit():
    cmp_ = CompareAttribution()
    with pytest.raises(ValueError):
        cmp_.get_results()


def test_markov_detour_kwarg_passed_through():
    cmp_ = CompareAttribution(
        markov_order=2, markov_removal_strategy="detour"
    )
    assert cmp_.markov.order == 2
    assert cmp_.markov.removal_strategy == "detour"
    assert isinstance(cmp_.shapley, ShapleyAttribution)
    assert isinstance(cmp_.markov, MarkovAttribution)


def test_shapley_characteristic_function_kwarg_passed_through():
    assert CompareAttribution().shapley.characteristic_function == "cumulative"
    cmp_ = CompareAttribution(shapley_characteristic_function="exact")
    assert cmp_.shapley.characteristic_function == "exact"


def test_default_comparison_has_no_unavailable_models(
    example_paths, example_conversions
):
    """Both default models produce share vectors on the shared fixture."""
    cmp_ = CompareAttribution(
        markov_removal_strategy="redirect_to_null",
        conversion_labels=["Purchase"],
    )
    cmp_.fit(example_paths, example_conversions)

    assert cmp_.unavailable_models() == {}
    frame = cmp_.attribution_frame()
    assert not frame.isna().any().any()
    assert frame["markov"].sum() == pytest.approx(1.0)
    assert frame["shapley"].sum() == pytest.approx(1.0)
    assert not cmp_.difference_frame("markov").isna().any().any()


# --------------------------------------------------------------------------- #
# Built-in first-touch and last-touch baselines (opt-in)                      #
# --------------------------------------------------------------------------- #
def test_baselines_are_off_by_default(example_paths, example_conversions):
    cmp_ = CompareAttribution(conversion_labels=["Purchase"])
    results = cmp_.fit(example_paths, example_conversions)

    assert cmp_.baselines == {}
    assert list(results) == ["markov", "shapley"]
    assert list(cmp_.attribution_frame().columns) == ["markov", "shapley"]
    assert list(cmp_.difference_frame("markov").columns) == ["shapley_minus_markov"]


def test_include_baselines_adds_first_and_last_touch_columns(
    example_paths, example_conversions
):
    cmp_ = CompareAttribution(conversion_labels=["Purchase"], include_baselines=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        results = cmp_.fit(example_paths, example_conversions)

    assert list(results) == ["markov", "shapley", "first_touch", "last_touch"]
    assert isinstance(cmp_.baselines["first_touch"], FirstTouchAttribution)
    assert isinstance(cmp_.baselines["last_touch"], LastTouchAttribution)
    assert all(
        isinstance(model, BaseAttribution) for model in cmp_.baselines.values()
    )

    frame = cmp_.attribution_frame()
    assert list(frame.columns) == ["markov", "shapley", "first_touch", "last_touch"]
    assert list(frame.index) == ["Direct", "Email", "Search", "Social"]
    assert frame["first_touch"].to_dict() == pytest.approx(
        {"Direct": 1 / 7, "Email": 1 / 7, "Search": 3 / 7, "Social": 2 / 7}
    )
    assert frame["last_touch"].to_dict() == pytest.approx(
        {"Direct": 1 / 7, "Email": 3 / 7, "Search": 0.0, "Social": 3 / 7}
    )
    for column in frame.columns:
        assert frame[column].sum() == pytest.approx(1.0)
    assert cmp_.unavailable_models() == {}


def test_baseline_columns_equal_the_standalone_baseline_fits(
    example_paths, example_conversions
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        without = CompareAttribution(conversion_labels=["Purchase"])
        without.fit(example_paths, example_conversions)
        with_baselines = CompareAttribution(
            conversion_labels=["Purchase"], include_baselines=True
        )
        results = with_baselines.fit(example_paths, example_conversions)

    assert results["first_touch"] == FirstTouchAttribution(
        conversion_labels=["Purchase"]
    ).fit(example_paths, example_conversions)
    assert results["last_touch"] == LastTouchAttribution(
        conversion_labels=["Purchase"]
    ).fit(example_paths, example_conversions)
    # Adding the reference points leaves the two models' columns untouched.
    frame = with_baselines.attribution_frame()
    for column in ("markov", "shapley"):
        assert frame[column].equals(without.attribution_frame()[column])


def test_baselines_inherit_separator_and_conversion_labels():
    cmp_ = CompareAttribution(
        separator="|", conversion_labels=["Sale"], include_baselines=True
    )
    for model in cmp_.baselines.values():
        assert model.separator == "|"
        assert model.conversion_labels == frozenset({"Sale"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cmp_.fit(["A|B|Sale", "B|Sale"], [1, 1])

    frame = cmp_.attribution_frame()
    assert list(frame.index) == ["A", "B"]
    assert frame["first_touch"].to_dict() == {"A": 0.5, "B": 0.5}
    assert frame["last_touch"].to_dict() == {"A": 0.0, "B": 1.0}


def test_models_can_be_read_against_a_baseline_reference(
    example_paths, example_conversions
):
    cmp_ = CompareAttribution(conversion_labels=["Purchase"], include_baselines=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cmp_.fit(example_paths, example_conversions)

    frame = cmp_.attribution_frame()
    diff = cmp_.difference_frame("last_touch")
    assert list(diff.columns) == [
        "markov_minus_last_touch",
        "shapley_minus_last_touch",
        "first_touch_minus_last_touch",
    ]
    assert diff["markov_minus_last_touch"].to_dict() == pytest.approx(
        (frame["markov"] - frame["last_touch"]).to_dict()
    )
    # Search never closes a converting journey but opens three of them.
    assert diff.loc["Search", "first_touch_minus_last_touch"] == pytest.approx(3 / 7)


def test_baseline_columns_are_missing_when_nothing_can_be_credited():
    cmp_ = CompareAttribution(
        markov_removal_strategy="redirect_to_null", include_baselines=True
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        results = cmp_.fit(["A", "B"], [0, 0])

    assert results["first_touch"]["attribution"] is None
    assert results["last_touch"]["credited_conversions"] == {"A": 0.0, "B": 0.0}
    assert cmp_.unavailable_models() == {
        "markov": "zero_raw_magnitude",
        "shapley": "zero_raw_magnitude",
        "first_touch": "zero_raw_magnitude",
        "last_touch": "zero_raw_magnitude",
    }
    frame = cmp_.attribution_frame()
    assert frame.isna().all().all()


def test_baselines_precede_extra_models_and_reject_reserved_names(
    example_paths, example_conversions
):
    extra = FirstTouchAttribution(conversion_labels=["Purchase"])
    cmp_ = CompareAttribution(
        conversion_labels=["Purchase"],
        include_baselines=True,
        extra_models={"custom": extra},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cmp_.fit(example_paths, example_conversions)
    frame = cmp_.attribution_frame()
    assert list(frame.columns) == [
        "markov",
        "shapley",
        "first_touch",
        "last_touch",
        "custom",
    ]
    assert frame["custom"].equals(frame["first_touch"].rename("custom"))

    with pytest.raises(ValueError, match="built-in baseline names"):
        CompareAttribution(
            include_baselines=True,
            extra_models={"last_touch": LastTouchAttribution()},
        )
    with pytest.raises(TypeError, match="include_baselines must be a bool"):
        CompareAttribution(include_baselines="yes")


def test_baseline_classes_still_plug_in_through_extra_models(
    example_paths, example_conversions
):
    """The opt-in flag is a convenience over the existing seam."""
    cmp_ = CompareAttribution(
        conversion_labels=["Purchase"],
        extra_models={
            "last_touch": LastTouchAttribution(conversion_labels=["Purchase"])
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cmp_.fit(example_paths, example_conversions)

    frame = cmp_.attribution_frame()
    assert list(frame.columns) == ["markov", "shapley", "last_touch"]
    assert frame["last_touch"].sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("conversions", "error", "pattern"),
    [
        ([0.9, 1.9], ValueError, r"index 0 must be 0 or 1, got 0\.9$"),
        ([0, float("nan")], ValueError, r"got missing value nan$"),
        (1, TypeError, r"not a bare int: 1$"),
    ],
)
def test_comparison_with_baselines_rejects_invalid_outcomes(
    conversions, error, pattern
):
    cmp_ = CompareAttribution(
        markov_removal_strategy="redirect_to_null", include_baselines=True
    )
    with pytest.raises(error, match=pattern):
        cmp_.fit(["A", "B"], conversions)


# --------------------------------------------------------------------------- #
# P0 — conversion_labels endpoint-contamination regression                    #
# --------------------------------------------------------------------------- #
# MODIFIED (P0): The wrapper must forward conversion_labels to both sub-models
# and to its own extract_channels call so no viewer of any of the results sees
# the endpoint token.
def test_conversion_label_forwarded_to_submodels():
    cmp_ = CompareAttribution(conversion_labels=["Purchase"])
    assert cmp_.markov.conversion_labels == frozenset({"Purchase"})
    assert cmp_.shapley.conversion_labels == frozenset({"Purchase"})


def test_compare_conversion_label_absent(example_paths, example_conversions):
    cmp_ = CompareAttribution(conversion_labels=["Purchase"])
    results = cmp_.fit(example_paths, example_conversions)
    assert "Purchase" not in results["markov"]["attribution"]
    # MODIFIED (get-well Unit 1): assert endpoint stripping on an output that
    # exists in both normalization states.
    assert "Purchase" not in results["shapley"]["shapley_values"]
    if results["shapley"]["attribution"] is not None:
        assert "Purchase" not in results["shapley"]["attribution"]
    assert "Purchase" not in cmp_.channels
    frame = cmp_.attribution_frame()
    assert "Purchase" not in frame.index


def test_compare_conversion_label_collision_rejected():
    # Collision is caught at sub-model construction.
    with pytest.raises(ValueError, match="SPECIAL_TOKENS"):
        CompareAttribution(conversion_labels=["START"])


# --------------------------------------------------------------------------- #
# Weighted comparisons                                                         #
# --------------------------------------------------------------------------- #
# fit(..., weights=w) drops zero-weight journeys once and hands the same
# retained journeys and weights to both models, so a weighted Markov column
# is never compared with an unweighted Shapley column or the reverse.
RTOL = 1e-10
ATOL = 1e-12

# "Ghost" is seen only on the two zero-weight journeys.
WEIGHTED_PATHS = ["A", "A > B", "B > A", "C", "Ghost > A", "A > B > C", "Ghost"]
WEIGHTED_CONVERSIONS = [1, 1, 0, 1, 1, 1, 0]
WEIGHTED_WEIGHTS = [0.5, 1.25, 3.0, 2.0, 0.0, 0.75, 0.0]
RETAINED = [0, 1, 2, 3, 5]


def _quiet_fit(cmp_, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return cmp_.fit(*args, **kwargs)


def _record_fit_calls(monkeypatch, model):
    """Record every fit call on ``model`` while still running the real fit."""
    calls = []
    real_fit = model.fit

    def recording_fit(*args, **kwargs):
        calls.append((args, kwargs))
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(model, "fit", recording_fit)
    return calls


def _assert_frames_close(actual, expected):
    assert list(actual.columns) == list(expected.columns)
    assert list(actual.index) == list(expected.index)
    np.testing.assert_allclose(
        actual.to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
        rtol=RTOL,
        atol=ATOL,
        equal_nan=True,
    )


def test_weighted_comparison_passes_identical_retained_inputs_to_both_models(
    monkeypatch,
):
    cmp_ = CompareAttribution(markov_removal_strategy="redirect_to_null")
    markov_calls = _record_fit_calls(monkeypatch, cmp_.markov)
    shapley_calls = _record_fit_calls(monkeypatch, cmp_.shapley)

    results = cmp_.fit(WEIGHTED_PATHS, WEIGHTED_CONVERSIONS, weights=WEIGHTED_WEIGHTS)

    assert list(results) == ["markov", "shapley"]
    assert len(markov_calls) == len(shapley_calls) == 1
    (markov_args, markov_kwargs), (shapley_args, shapley_kwargs) = (
        markov_calls[0],
        shapley_calls[0],
    )
    # Both models receive explicit weights: neither is fitted unweighted.
    assert set(markov_kwargs) == set(shapley_kwargs) == {"weights"}
    expected = (
        [WEIGHTED_PATHS[i] for i in RETAINED],
        [WEIGHTED_CONVERSIONS[i] for i in RETAINED],
        [WEIGHTED_WEIGHTS[i] for i in RETAINED],
    )
    for args, kwargs in ((markov_args, markov_kwargs), (shapley_args, shapley_kwargs)):
        assert len(args) == 2
        assert list(args[0]) == expected[0]
        assert list(args[1]) == expected[1]
        assert list(kwargs["weights"]) == expected[2]
    # Identical inputs, element for element.
    assert list(markov_args[0]) == list(shapley_args[0])
    assert list(markov_args[1]) == list(shapley_args[1])
    assert list(markov_kwargs["weights"]) == list(shapley_kwargs["weights"])

    # The zero-weight-only channel is in no model and not in the comparison.
    assert cmp_.channels == ["A", "B", "C"]
    assert cmp_.markov.get_channels() == cmp_.shapley.get_channels() == ["A", "B", "C"]
    frame = cmp_.attribution_frame()
    assert list(frame.columns) == ["markov", "shapley"]
    assert list(frame.index) == ["A", "B", "C"]
    assert cmp_.unavailable_models() == {}
    for column in frame.columns:
        assert frame[column].sum() == pytest.approx(1.0)


def test_weighted_comparison_equals_the_standalone_weighted_fits():
    cmp_ = CompareAttribution(markov_removal_strategy="redirect_to_null")
    results = cmp_.fit(WEIGHTED_PATHS, WEIGHTED_CONVERSIONS, weights=WEIGHTED_WEIGHTS)

    retained = (
        [WEIGHTED_PATHS[i] for i in RETAINED],
        [WEIGHTED_CONVERSIONS[i] for i in RETAINED],
    )
    retained_weights = [WEIGHTED_WEIGHTS[i] for i in RETAINED]
    markov = MarkovAttribution(removal_strategy="redirect_to_null").fit(
        *retained, weights=retained_weights
    )
    # Standalone Shapley drops the zero-weight journeys itself.
    shapley = ShapleyAttribution().fit(
        WEIGHTED_PATHS, WEIGHTED_CONVERSIONS, weights=WEIGHTED_WEIGHTS
    )

    for name, standalone, raw_key in (
        ("markov", markov, "removal_effects"),
        ("shapley", shapley, "shapley_values"),
    ):
        for key in ("attribution", raw_key):
            assert set(results[name][key]) == set(standalone[key]) == {"A", "B", "C"}
            np.testing.assert_allclose(
                [results[name][key][c] for c in "ABC"],
                [standalone[key][c] for c in "ABC"],
                rtol=RTOL,
                atol=ATOL,
            )
    # Fractional weighted coalition totals reach the comparison untruncated.
    assert results["shapley"]["value_function"][("A", "B")] == pytest.approx(1.25)


@pytest.mark.parametrize("removal_strategy", ["fail", "redirect_to_null"])
@pytest.mark.parametrize("characteristic_function", ["cumulative", "exact"])
def test_unit_weighted_comparison_matches_the_unweighted_comparison(
    removal_strategy, characteristic_function, example_paths, example_conversions
):
    def comparison():
        return CompareAttribution(
            markov_removal_strategy=removal_strategy,
            conversion_labels=["Purchase"],
            shapley_characteristic_function=characteristic_function,
        )

    plain, unit = comparison(), comparison()
    _quiet_fit(plain, example_paths, example_conversions)
    _quiet_fit(
        unit, example_paths, example_conversions, weights=[1.0] * len(example_paths)
    )

    assert unit.channels == plain.channels
    assert unit.unavailable_models() == plain.unavailable_models()
    _assert_frames_close(unit.attribution_frame(), plain.attribution_frame())
    _assert_frames_close(
        unit.difference_frame("markov"), plain.difference_frame("markov")
    )


@pytest.mark.parametrize("removal_strategy", ["fail", "redirect_to_null"])
def test_frequency_weighted_comparison_matches_replicated_journeys(removal_strategy):
    paths = ["A", "A > B", "B > A", "C", "A > B > C", "Z > A", "B", "Z"]
    conversions = [1, 1, 0, 1, 1, 1, 0, 0]
    frequencies = [3, 2, 4, 1, 2, 0, 5, 0]
    replicated_paths, replicated_conversions = [], []
    for path, converted, frequency in zip(paths, conversions, frequencies):
        replicated_paths.extend([path] * frequency)
        replicated_conversions.extend([converted] * frequency)

    weighted = CompareAttribution(markov_removal_strategy=removal_strategy)
    replicated = CompareAttribution(markov_removal_strategy=removal_strategy)
    _quiet_fit(weighted, paths, conversions, weights=frequencies)
    _quiet_fit(replicated, replicated_paths, replicated_conversions)

    # Z is seen only on zero-frequency journeys: absent from both sides.
    assert weighted.channels == replicated.channels == ["A", "B", "C"]
    assert weighted.unavailable_models() == replicated.unavailable_models() == {}
    _assert_frames_close(weighted.attribution_frame(), replicated.attribution_frame())
    _assert_frames_close(
        weighted.difference_frame("markov"), replicated.difference_frame("markov")
    )


def test_weighted_comparison_accepts_series_and_aligns_weights_by_position():
    expected = CompareAttribution(markov_removal_strategy="redirect_to_null")
    expected.fit(WEIGHTED_PATHS, WEIGHTED_CONVERSIONS, weights=WEIGHTED_WEIGHTS)

    n = len(WEIGHTED_PATHS)
    cmp_ = CompareAttribution(markov_removal_strategy="redirect_to_null")
    cmp_.fit(
        pd.Series(WEIGHTED_PATHS, index=range(50, 50 + n)),
        pd.Series(WEIGHTED_CONVERSIONS),
        # Label alignment would reverse these weights; position must win.
        weights=pd.Series(WEIGHTED_WEIGHTS, index=range(n - 1, -1, -1)),
    )
    assert cmp_.channels == expected.channels
    _assert_frames_close(cmp_.attribution_frame(), expected.attribution_frame())


def test_weighted_comparison_strips_conversion_labels(
    example_paths, example_conversions
):
    # Zero weight on "Direct > Purchase", the only journey touching Direct.
    weights = [2.5, 1.0, 1.0, 4.0, 0.5, 0.0, 2.0, 1.5, 1.0, 3.0]
    cmp_ = CompareAttribution(conversion_labels=["Purchase"])
    results = _quiet_fit(cmp_, example_paths, example_conversions, weights=weights)

    assert cmp_.channels == ["Email", "Search", "Social"]
    frame = cmp_.attribution_frame()
    assert list(frame.index) == ["Email", "Search", "Social"]
    for name in ("markov", "shapley"):
        assert "Purchase" not in results[name]["attribution"]
        assert "Direct" not in results[name]["attribution"]
    assert all(
        "Purchase" not in coalition
        for coalition in results["shapley"]["value_function"]
    )


def _two_argument_model():
    class TwoArgumentModel(BaseAttribution):
        """A custom model with the unchanged two-argument fit interface."""

        def fit(self, paths, conversions):
            self.channels = ["A"]
            self._last_attribution = {"A": 1.0}
            return {"attribution": dict(self._last_attribution)}

    return TwoArgumentModel()


@pytest.mark.parametrize(
    ("kwargs", "named"),
    [
        ({"include_baselines": True}, ["first_touch", "last_touch"]),
        ({"extra_models": "custom"}, ["custom"]),
        (
            {"include_baselines": True, "extra_models": "custom"},
            ["first_touch", "last_touch", "custom"],
        ),
    ],
)
def test_weighted_comparison_rejects_baselines_and_extra_models_before_fitting(
    monkeypatch, kwargs, named
):
    kwargs = dict(kwargs)
    if "extra_models" in kwargs:
        kwargs["extra_models"] = {"custom": _two_argument_model()}
    cmp_ = CompareAttribution(markov_removal_strategy="redirect_to_null", **kwargs)
    calls = {
        name: _record_fit_calls(monkeypatch, model)
        for name, model in {
            "markov": cmp_.markov,
            "shapley": cmp_.shapley,
            **cmp_.baselines,
            **cmp_.extra_models,
        }.items()
    }

    with pytest.raises(ValueError) as error:
        cmp_.fit(WEIGHTED_PATHS, WEIGHTED_CONVERSIONS, weights=WEIGHTED_WEIGHTS)

    message = str(error.value)
    assert "weighted comparison supports only the built-in Markov and Shapley" in message
    assert str(named) in message
    # Actionable: it says how to get a valid weighted comparison.
    assert "include_baselines=False" in message
    assert "without extra_models" in message
    # No model was fitted and no comparison state was written.
    assert all(not model_calls for model_calls in calls.values())
    assert cmp_.channels == []
    with pytest.raises(ValueError, match="Call fit"):
        cmp_.get_results()

    # The same comparison still fits unweighted, calling every model with the
    # unchanged two-argument interface.
    results = cmp_.fit(WEIGHTED_PATHS, WEIGHTED_CONVERSIONS)
    assert list(results) == ["markov", "shapley", *named]
    for model_calls in calls.values():
        assert len(model_calls) == 1
        args, call_kwargs = model_calls[0]
        assert len(args) == 2 and call_kwargs == {}


def test_rejected_weighted_comparison_keeps_the_previous_results():
    cmp_ = CompareAttribution(
        markov_removal_strategy="redirect_to_null", include_baselines=True
    )
    cmp_.fit(WEIGHTED_PATHS, WEIGHTED_CONVERSIONS)
    before = cmp_.attribution_frame()

    with pytest.raises(ValueError, match="weighted comparison supports only"):
        cmp_.fit(["X", "Y"], [1, 0], weights=[1.0, 2.0])

    assert cmp_.channels == ["A", "B", "C", "Ghost"]
    assert cmp_.attribution_frame().equals(before)


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ([1.0] * 3, "must have same length"),
        ([[1.0] * 7], "weights must be one-dimensional"),
        ([1.0, float("nan")] + [1.0] * 5, "weights must contain only finite values"),
        ([1.0, -1.0] + [1.0] * 5, "weights must be nonnegative"),
        ([0.0] * 7, "weights must have positive total weight"),
    ],
)
def test_weighted_comparison_rejects_invalid_weights_before_fitting(
    monkeypatch, weights, message
):
    cmp_ = CompareAttribution(markov_removal_strategy="redirect_to_null")
    markov_calls = _record_fit_calls(monkeypatch, cmp_.markov)
    shapley_calls = _record_fit_calls(monkeypatch, cmp_.shapley)

    with pytest.raises(ValueError, match=message):
        cmp_.fit(WEIGHTED_PATHS, WEIGHTED_CONVERSIONS, weights=weights)

    assert markov_calls == [] and shapley_calls == []
    assert cmp_.channels == []


def test_weighted_comparison_validates_outcomes_on_zero_weight_rows(monkeypatch):
    cmp_ = CompareAttribution(markov_removal_strategy="redirect_to_null")
    markov_calls = _record_fit_calls(monkeypatch, cmp_.markov)
    shapley_calls = _record_fit_calls(monkeypatch, cmp_.shapley)

    with pytest.raises(ValueError, match=r"index 1 must be 0 or 1, got 0\.5$"):
        cmp_.fit(["A", "B"], [1, 0.5], weights=[1.0, 0.0])

    assert markov_calls == [] and shapley_calls == []


@pytest.mark.parametrize(
    "removal_strategy", ["fail", "fail_refit", "detour", "redirect_to_null"]
)
def test_weighted_comparison_without_weighted_conversions_is_unavailable(
    removal_strategy,
):
    """Positive total weight, but every converting journey has weight 0."""
    cmp_ = CompareAttribution(markov_removal_strategy=removal_strategy)
    _quiet_fit(cmp_, ["A", "B", "A > B"], [0, 0, 1], weights=[2.0, 3.0, 0.0])

    assert cmp_.unavailable_models() == {
        "markov": "zero_raw_magnitude",
        "shapley": "zero_raw_magnitude",
    }
    frame = cmp_.attribution_frame()
    assert list(frame.columns) == ["markov", "shapley"]
    assert list(frame.index) == ["A", "B"]
    assert frame.isna().all().all()
    assert cmp_.difference_frame("markov")["shapley_minus_markov"].isna().all()


def test_unweighted_comparison_calls_the_models_without_weights(monkeypatch):
    cmp_ = CompareAttribution(markov_removal_strategy="redirect_to_null")
    markov_calls = _record_fit_calls(monkeypatch, cmp_.markov)
    shapley_calls = _record_fit_calls(monkeypatch, cmp_.shapley)

    cmp_.fit(WEIGHTED_PATHS, WEIGHTED_CONVERSIONS)
    cmp_.fit(WEIGHTED_PATHS, WEIGHTED_CONVERSIONS, weights=None)

    for calls in (markov_calls, shapley_calls):
        assert len(calls) == 2
        for args, kwargs in calls:
            # The caller's own containers, passed through untouched.
            assert args == (WEIGHTED_PATHS, WEIGHTED_CONVERSIONS)
            assert kwargs == {}
    assert cmp_.channels == ["A", "B", "C", "Ghost"]
