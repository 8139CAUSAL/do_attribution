"""Shapley-attribution core tests."""

from __future__ import annotations

from math import comb, factorial

import numpy as np
import pandas as pd
import pytest

from do_attribution import ShapleyAttribution


# --------------------------------------------------------------------------- #
# Normalisation & efficiency                                                   #
# --------------------------------------------------------------------------- #
def test_attribution_sums_to_one():
    paths = ["A > B > C", "A > C", "B > C", "A > B"]
    conversions = [1, 0, 1, 1]
    model = ShapleyAttribution()
    results = model.fit(paths, conversions)
    assert abs(sum(results["attribution"].values()) - 1.0) < 1e-12


def test_efficiency_sum_equals_grand_coalition_value():
    """Shapley values should satisfy phi_1 + ... + phi_n == v(N) - v(emptyset).

    Under this library's value-function convention (v(S) = conversions from
    paths whose unique-channel-set is *exactly* S), v(emptyset) = 0 and
    v(N) equals the number of conversions on paths that touched every
    channel.
    """
    paths = ["A > B > C", "A > C", "B > C", "A > B"]
    conversions = [1, 0, 1, 1]
    res = ShapleyAttribution().fit(paths, conversions)

    grand_coalition = tuple(sorted({"A", "B", "C"}))
    v_N = res["value_function"].get(grand_coalition, 0)
    assert v_N == 1  # only "A > B > C" (conv=1) touches every channel
    assert sum(res["shapley_values"].values()) == pytest.approx(v_N)


def test_symmetry_two_identical_channels():
    # A and B are interchangeable in all paths, so they should receive the
    # same Shapley value.
    paths = ["A > C", "B > C"]
    conversions = [1, 1]
    res = ShapleyAttribution().fit(paths, conversions)
    assert res["shapley_values"]["A"] == pytest.approx(res["shapley_values"]["B"])


# --------------------------------------------------------------------------- #
# Hand-computed example                                                        #
# --------------------------------------------------------------------------- #
def test_hand_computed_two_channel_example():
    # Two channels A, B, two paths: {A} converts once, {A, B} converts twice.
    # v({}) = 0, v({A}) = 1, v({B}) = 0, v({A, B}) = 2.
    # phi_A = 0.5 * (v({A}) - v({})) + 0.5 * (v({A,B}) - v({B}))
    #       = 0.5 * 1 + 0.5 * 2 = 1.5
    # phi_B = 0.5 * (v({B}) - v({})) + 0.5 * (v({A,B}) - v({A}))
    #       = 0.5 * 0 + 0.5 * 1 = 0.5
    paths = ["A", "A > B", "A > B"]
    convs = [1, 1, 1]
    res = ShapleyAttribution().fit(paths, convs)
    assert res["shapley_values"]["A"] == pytest.approx(1.5)
    assert res["shapley_values"]["B"] == pytest.approx(0.5)
    assert res["attribution"]["A"] == pytest.approx(0.75)
    assert res["attribution"]["B"] == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# Value function bookkeeping                                                   #
# --------------------------------------------------------------------------- #
def test_value_function_keys_are_sorted_tuples():
    paths = ["B > A > C", "C > A"]
    convs = [1, 0]
    res = ShapleyAttribution().fit(paths, convs)
    for coalition in res["value_function"]:
        assert coalition == tuple(sorted(coalition))


def test_value_function_counts_match_conversions():
    # Coalition {A} appears twice but converts once; {A, B} appears once &
    # converts once; {A, C} appears once & does not convert.
    paths = ["A", "A", "A > B", "A > C"]
    convs = [1, 0, 1, 0]
    vf = ShapleyAttribution().fit(paths, convs)["value_function"]
    assert vf[("A",)] == 1
    assert vf[("A", "B")] == 1
    assert vf[("A", "C")] == 0


# --------------------------------------------------------------------------- #
# Validation & input flexibility                                               #
# --------------------------------------------------------------------------- #
def test_invalid_conversions_rejected():
    with pytest.raises(ValueError):
        ShapleyAttribution().fit(["A > B"], [2])


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        ShapleyAttribution().fit(["A > B"], [1, 0])


def test_accepts_series_inputs():
    paths = pd.Series(["A > B", "B > C"])
    convs = pd.Series([1, 1])
    res = ShapleyAttribution().fit(paths, convs)
    # MODIFIED (get-well Unit 1): no path touches every channel here, so
    # v(N) = 0 and the raw values cancel; normalized attribution is
    # therefore unavailable.  The input-handling contract under test is the
    # channel set, which is asserted on the always-present raw values.
    assert set(res["shapley_values"].keys()) == {"A", "B", "C"}


def test_accepts_numpy_array_inputs():
    res = ShapleyAttribution().fit(np.array(["A > B"]), np.array([1]))
    assert res["attribution"]["A"] == pytest.approx(0.5)
    assert res["attribution"]["B"] == pytest.approx(0.5)


def test_zero_total_value_warns_and_refuses_normalization():
    """MODIFIED (get-well Unit 1): no signal means no shares, not equal credit.

    With zero conversions every raw Shapley value is zero, so there is no
    denominator and nothing to allocate.  The previous contract assigned
    every channel ``1/3``; those weights were never computed by the game, so
    publishing them silently substituted arbitrary values for a missing
    result.  Normalization is now reported as unavailable instead.
    """
    paths = ["A > B", "B > C"]
    convs = [0, 0]
    with pytest.warns(UserWarning, match="normalization is unavailable"):
        res = ShapleyAttribution().fit(paths, convs)

    assert res["attribution"] is None
    assert res["normalization_status"] == "unavailable"
    assert res["normalization_reason"] == "zero_raw_magnitude"
    assert res["raw_magnitude"] == pytest.approx(0.0)
    # Raw values remain available and are all zero.
    assert res["shapley_values"] == pytest.approx({"A": 0.0, "B": 0.0, "C": 0.0})


def test_factorial_weights_match_spec():
    """Regression test: verify the factorial-weight formula used internally.

    The weight ``w(s, n) = s! * (n - s - 1)! / n!`` applies to each coalition
    S ⊆ N\\{i} of size s; there are C(n-1, s) such coalitions. For any fixed
    player i the weights over all subsets sum to 1, which is equivalent to
    sum over s of C(n-1, s) * w(s, n) = 1.
    """
    paths = ["A", "A > B", "A > B > C"]
    convs = [1, 1, 1]
    res = ShapleyAttribution().fit(paths, convs)
    n = 3
    w = [(factorial(s) * factorial(n - s - 1)) / factorial(n) for s in range(n)]
    # All weights positive.
    assert all(x > 0 for x in w)
    # Weighted by C(n-1, s), they sum to 1 (the correct identity).
    assert sum(comb(n - 1, s) * w[s] for s in range(n)) == pytest.approx(1.0)
    # Sanity: the Shapley values sum to v(N) (efficiency).
    grand = tuple(sorted({"A", "B", "C"}))
    assert sum(res["shapley_values"].values()) == pytest.approx(
        res["value_function"].get(grand, 0)
    )


def test_get_attribution_requires_fit():
    with pytest.raises(ValueError):
        ShapleyAttribution().get_attribution()


# --------------------------------------------------------------------------- #
# P0 — conversion_labels endpoint-contamination regression                    #
# --------------------------------------------------------------------------- #
# MODIFIED (P0): With conversion_labels set, the Shapley value function,
# attribution, and channels list must never contain the endpoint token.
def test_conversion_label_absent_from_shapley_outputs(
    example_paths, example_conversions
):
    model = ShapleyAttribution(conversion_labels=["Purchase"])
    res = model.fit(example_paths, example_conversions)

    # MODIFIED (get-well Unit 1): this fixture cancels, so 'attribution' is
    # None; the endpoint-stripping contract is asserted on every output that
    # is populated regardless of normalization availability.
    if res["attribution"] is not None:
        assert "Purchase" not in res["attribution"]
    assert "Purchase" not in res["shapley_values"]
    assert "Purchase" not in model.get_channels()
    for coalition in res["value_function"]:
        assert "Purchase" not in coalition


def test_conversion_label_matches_prestripping_shapley(
    example_paths, example_paths_no_endpoint, example_conversions
):
    labeled = ShapleyAttribution(conversion_labels=["Purchase"]).fit(
        example_paths, example_conversions
    )
    prestripped = ShapleyAttribution().fit(
        example_paths_no_endpoint, example_conversions
    )
    # MODIFIED (get-well Unit 1): the equivalence claim is about the model
    # arithmetic, which lives in the raw values.  Those are populated in
    # every state, so the comparison no longer depends on normalization
    # being available (it is not, for this fixture).
    assert set(labeled["shapley_values"]) == set(prestripped["shapley_values"])
    for ch in labeled["shapley_values"]:
        assert labeled["shapley_values"][ch] == pytest.approx(
            prestripped["shapley_values"][ch], abs=1e-12
        )
    assert labeled["value_function"] == prestripped["value_function"]
    # The normalization outcome must match too.
    assert (
        labeled["normalization_status"] == prestripped["normalization_status"]
    )
    assert labeled["attribution"] == prestripped["attribution"]


def test_conversion_label_collision_rejected_shapley():
    with pytest.raises(ValueError, match="SPECIAL_TOKENS"):
        ShapleyAttribution(conversion_labels=["NULL"])


def test_conversion_label_bare_string_rejected_shapley():
    with pytest.raises(TypeError, match="not a bare str"):
        ShapleyAttribution(conversion_labels="Purchase")


# --------------------------------------------------------------------------- #
# Normalisation safeguards against near-zero Shapley totals                    #
# --------------------------------------------------------------------------- #
# MODIFIED (get-well Unit 1): The library's value function is
# v(S) = conversions whose unique channel set equals *exactly* S.  Under this
# convention, the Shapley efficiency axiom forces sum(phi_i) = v(N), and
# v(N) = 0 whenever no single path touches every channel.  Dividing by that
# float-epsilon denominator produced shares on the order of +/- 5e17 %.
#
# The earlier fix replaced the explosive division with positive-contribution
# renormalisation: share_i = max(phi_i, 0) / sum_j max(phi_j, 0).  That is
# not the Shapley allocation — it discards the negative values and reweights
# the survivors — so it published numbers the game never produced.  Per
# Project_Get-Well_Plan.md sections 6.1 and 6.2 the model must instead refuse
# to normalise and preserve the raw signed diagnostics.  These tests lock
# down that contract.
def test_near_zero_total_refuses_normalization(
    example_paths, example_conversions
):
    with pytest.warns(UserWarning, match="normalization is unavailable"):
        res = ShapleyAttribution(conversion_labels=["Purchase"]).fit(
            example_paths, example_conversions
        )

    assert res["attribution"] is None
    assert res["normalization_status"] == "unavailable"
    assert res["normalization_reason"] == "near_cancellation"


def test_near_zero_total_preserves_raw_signed_values(
    example_paths, example_conversions
):
    """Raw diagnostics survive intact when normalization is refused."""
    with pytest.warns(UserWarning):
        res = ShapleyAttribution(conversion_labels=["Purchase"]).fit(
            example_paths, example_conversions
        )

    raw = res["shapley_values"]
    assert raw  # populated
    assert all(np.isfinite(value) for value in raw.values())
    # This fixture genuinely contains negative raw values; they are not
    # clipped, zeroed, or reweighted.
    assert any(value < 0 for value in raw.values())
    assert res["raw_sum"] == pytest.approx(sum(raw.values()))
    assert res["raw_magnitude"] == pytest.approx(
        sum(abs(value) for value in raw.values())
    )


def test_near_zero_total_reports_scale_aware_cancellation(
    example_paths, example_conversions
):
    """The refusal is driven by the scale-aware ratio, not the bare sum."""
    with pytest.warns(UserWarning):
        res = ShapleyAttribution(conversion_labels=["Purchase"]).fit(
            example_paths, example_conversions
        )

    # The raw magnitude is substantial even though the sum collapses.
    assert res["raw_magnitude"] > 1.0
    assert res["cancellation_ratio"] <= 1e-12
    assert res["normalization_denominator"] == pytest.approx(res["raw_sum"])


def test_near_zero_total_blocks_attributed_conversions(
    example_paths, example_conversions
):
    """Attributed conversions cannot be derived from unavailable attribution."""
    model = ShapleyAttribution(conversion_labels=["Purchase"])
    with pytest.warns(UserWarning):
        res = model.fit(example_paths, example_conversions)

    assert res["attribution"] is None
    with pytest.raises(ValueError, match="unavailable"):
        model.get_attribution()

    summary = model.summary()
    assert "Normalized attribution unavailable" in summary
    assert "Not calculated - requires normalized attribution" in summary


def test_classic_regime_unchanged():
    """When sum(shapley_values) is non-trivial, legacy v/total normalisation
    must still be used bit-for-bit."""
    paths = ["A > B > C", "A > C", "B > C", "A > B"]
    conversions = [1, 0, 1, 1]
    res = ShapleyAttribution().fit(paths, conversions)
    total = sum(res["shapley_values"].values())
    assert total > 1e-6  # well-conditioned
    for ch, v in res["shapley_values"].items():
        assert res["attribution"][ch] == pytest.approx(v / total, abs=1e-12)


def test_near_zero_total_attribution_shares_not_exploding(
    example_paths, example_conversions
):
    """Sample-data Shapley shares were once +/- 5e17 %.

    MODIFIED (get-well Unit 1): no share of any size is published now, so
    the guarantee is stronger than a magnitude bound — the rendered summary
    contains no percentage-formatted attribution at all.
    """
    model = ShapleyAttribution(conversion_labels=["Purchase"])
    with pytest.warns(UserWarning):
        res = model.fit(example_paths, example_conversions)

    assert res["attribution"] is None

    summary = model.summary()
    # The raw signed values are rendered as signed values, never as shares.
    assert "Raw Shapley values (signed)" in summary
    assert "e+" not in summary  # no exponential blow-up anywhere
    for raw in res["shapley_values"].values():
        assert f"{raw * 100:.2f}%" not in summary


def test_characterization_reported_signed_normalization_case():
    """The reported signed case keeps its raw game and refuses to normalise.

    MODIFIED (get-well Unit 1): this test previously recorded the defect —
    the candidate shares (40.5, -39.5) were published verbatim as
    ``attribution``, rendered as "A 4050.00%", and multiplied out into
    4090.5 attributed conversions from 101 actual ones.  It now asserts the
    corrected contract.

    Note the cancellation ratio here is 0.0125, comfortably above the 1e-12
    guard: this case is caught by the share-domain validation, not by the
    near-cancellation check.  Both gates are required.
    """
    paths = ["A"] * 90 + ["B"] * 10 + ["A > B"]
    conversions = [1] * len(paths)
    model = ShapleyAttribution()

    with pytest.warns(UserWarning, match="normalization is unavailable"):
        result = model.fit(paths, conversions)

    # --- The raw exact-coalition game is unchanged. --------------------- #
    assert result["value_function"] == {
        ("A",): 90,
        ("B",): 10,
        ("A", "B"): 1,
    }
    raw_values = result["shapley_values"]
    assert raw_values == pytest.approx({"A": 40.5, "B": -39.5})

    # --- Diagnostics are published and reproducible. -------------------- #
    assert result["raw_sum"] == pytest.approx(1.0)
    assert result["raw_magnitude"] == pytest.approx(80.0)
    assert result["cancellation_ratio"] == pytest.approx(0.0125)
    assert result["normalization_denominator"] == pytest.approx(1.0)
    assert result["raw_sum"] == pytest.approx(sum(raw_values.values()))

    # --- Normalized attribution is refused, not fabricated. ------------- #
    assert result["attribution"] is None
    assert result["normalization_status"] == "unavailable"
    assert result["normalization_reason"] == "negative_share"

    # --- No attributed conversions can be derived. ---------------------- #
    with pytest.raises(ValueError, match="unavailable"):
        model.get_attribution()

    # --- Reporting shows raw values as signed values, not percentages. -- #
    summary = model.summary()
    assert "Raw Shapley values (signed)" in summary
    assert "A 4050.00%" not in summary
    assert "4050" not in summary
    assert "4090" not in summary
    assert "+40.5000" in summary
    assert "-39.5000" in summary
    assert "Normalized attribution unavailable (negative_share)" in summary
    assert "Not calculated - requires normalized attribution" in summary
