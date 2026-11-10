"""Tests for :class:`CompareAttribution`."""

from __future__ import annotations

import pytest

from do_attribution import (
    BaseAttribution,
    CompareAttribution,
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
