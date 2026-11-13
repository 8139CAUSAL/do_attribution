"""Shapley-attribution core tests."""

from __future__ import annotations

from itertools import combinations
from math import comb, factorial
import warnings

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

    Under the legacy ``"exact"`` game (v(S) = conversions from paths whose
    unique-channel-set is *exactly* S), v(emptyset) = 0 and v(N) equals the
    number of conversions on paths that touched every channel.
    """
    paths = ["A > B > C", "A > C", "B > C", "A > B"]
    conversions = [1, 0, 1, 1]
    res = ShapleyAttribution(characteristic_function="exact").fit(paths, conversions)

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
    # Legacy "exact" game.
    # Two channels A, B, two paths: {A} converts once, {A, B} converts twice.
    # v({}) = 0, v({A}) = 1, v({B}) = 0, v({A, B}) = 2.
    # phi_A = 0.5 * (v({A}) - v({})) + 0.5 * (v({A,B}) - v({B}))
    #       = 0.5 * 1 + 0.5 * 2 = 1.5
    # phi_B = 0.5 * (v({B}) - v({})) + 0.5 * (v({A,B}) - v({A}))
    #       = 0.5 * 0 + 0.5 * 1 = 0.5
    paths = ["A", "A > B", "A > B"]
    convs = [1, 1, 1]
    res = ShapleyAttribution(characteristic_function="exact").fit(paths, convs)
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
    # The input-handling contract under test is the channel set, which is
    # asserted on the always-present raw values and on the shares.
    assert set(res["shapley_values"].keys()) == {"A", "B", "C"}
    assert set(res["attribution"].keys()) == {"A", "B", "C"}


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
    res = ShapleyAttribution(characteristic_function="exact").fit(paths, convs)
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

    # The endpoint-stripping contract is asserted on every output.
    assert res["attribution"] is not None
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
    # every state, so the comparison does not depend on normalization
    # being available.
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
# MODIFIED (get-well Unit 1): The legacy ``"exact"`` game is
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
        res = ShapleyAttribution(
            conversion_labels=["Purchase"], characteristic_function="exact"
        ).fit(
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
        res = ShapleyAttribution(
            conversion_labels=["Purchase"], characteristic_function="exact"
        ).fit(
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
        res = ShapleyAttribution(
            conversion_labels=["Purchase"], characteristic_function="exact"
        ).fit(
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
    model = ShapleyAttribution(
        conversion_labels=["Purchase"], characteristic_function="exact"
    )
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
    model = ShapleyAttribution(
        conversion_labels=["Purchase"], characteristic_function="exact"
    )
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
    model = ShapleyAttribution(characteristic_function="exact")

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


# --------------------------------------------------------------------------- #
# Default "cumulative" characteristic function                                 #
# --------------------------------------------------------------------------- #
# v(S) = conversions on paths whose channel set is contained in S.  v(N) is
# every observed conversion, so the Shapley values are a non-negative
# allocation of all conversions and normalized attribution exists whenever
# any path converted — including when no path touches every channel, which
# is the case the legacy "exact" game cannot normalise.
def _brute_force_cumulative_shapley(channels, coalition_conversions):
    """Factorial-weight Shapley values of v(S) = sum of R(T) for T within S."""
    n = len(channels)

    def v(coalition):
        members = set(coalition)
        return sum(
            conversions
            for key, conversions in coalition_conversions.items()
            if set(key) <= members
        )

    values = {}
    for channel in channels:
        others = [other for other in channels if other != channel]
        total = 0.0
        for size in range(n):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            for subset in combinations(others, size):
                total += weight * (v(subset + (channel,)) - v(subset))
        values[channel] = total
    return values


def test_default_characteristic_function_is_cumulative():
    model = ShapleyAttribution()
    assert model.characteristic_function == "cumulative"
    res = model.fit(["A > B"], [1])
    assert res["characteristic_function"] == "cumulative"
    legacy = ShapleyAttribution(characteristic_function="exact").fit(["A > B"], [1])
    assert legacy["characteristic_function"] == "exact"


def test_cumulative_hand_computed_two_channel_example():
    # R({A}) = 1, R({A, B}) = 2, so v({A}) = 1, v({B}) = 0, v({A, B}) = 3.
    # phi_A = 0.5 * (v({A}) - v({})) + 0.5 * (v({A,B}) - v({B}))
    #       = 0.5 * 1 + 0.5 * 3 = 2
    # phi_B = 0.5 * (v({B}) - v({})) + 0.5 * (v({A,B}) - v({A}))
    #       = 0.5 * 0 + 0.5 * 2 = 1
    res = ShapleyAttribution().fit(["A", "A > B", "A > B"], [1, 1, 1])
    assert res["shapley_values"] == {"A": 2.0, "B": 1.0}
    assert res["attribution"]["A"] == pytest.approx(2 / 3)
    assert res["attribution"]["B"] == pytest.approx(1 / 3)
    assert res["normalization_status"] == "available"
    assert res["normalization_reason"] is None


def test_cumulative_efficiency_allocates_every_conversion():
    paths = ["A > B > C", "A > C", "B > C", "A > B", "C", "A"]
    conversions = [1, 0, 1, 1, 1, 1]
    res = ShapleyAttribution().fit(paths, conversions)
    assert sum(res["shapley_values"].values()) == pytest.approx(sum(conversions))
    assert res["raw_sum"] == pytest.approx(sum(conversions))
    assert res["normalization_denominator"] == pytest.approx(sum(conversions))
    assert res["cancellation_ratio"] == pytest.approx(1.0)
    assert all(value >= 0.0 for value in res["shapley_values"].values())
    assert sum(res["attribution"].values()) == pytest.approx(1.0, abs=1e-12)


def test_cumulative_available_when_no_path_touches_every_channel():
    """The regime that defeats the legacy game is well-defined by default."""
    paths = ["A"] * 90 + ["B"] * 10
    conversions = [1] * 100
    res = ShapleyAttribution().fit(paths, conversions)
    assert ("A", "B") not in res["value_function"]  # v_exact(N) = 0
    assert res["shapley_values"] == {"A": 90.0, "B": 10.0}
    assert res["attribution"] == {"A": 0.9, "B": 0.1}

    with pytest.warns(UserWarning, match="normalization is unavailable"):
        legacy = ShapleyAttribution(characteristic_function="exact").fit(
            paths, conversions
        )
    assert legacy["attribution"] is None
    assert legacy["normalization_reason"] == "near_cancellation"
    # Both games are fitted from the same observed coalition table.
    assert legacy["value_function"] == res["value_function"]


def test_cumulative_allocation_is_stable_under_one_added_path():
    """One extra A > B conversion moves each channel by exactly half of it."""
    base_paths = ["A"] * 90 + ["B"] * 10
    before = ShapleyAttribution().fit(base_paths, [1] * 100)
    after = ShapleyAttribution().fit(base_paths + ["A > B"], [1] * 101)
    assert before["shapley_values"] == {"A": 90.0, "B": 10.0}
    assert after["shapley_values"] == {"A": 90.5, "B": 10.5}
    assert after["attribution"]["A"] == pytest.approx(90.5 / 101)
    assert after["attribution"]["B"] == pytest.approx(10.5 / 101)


def test_cumulative_matches_factorial_weight_definition(
    example_paths_no_endpoint, example_conversions
):
    """The dividend identity equals direct enumeration with w(s, n)."""
    res = ShapleyAttribution().fit(example_paths_no_endpoint, example_conversions)
    expected = _brute_force_cumulative_shapley(
        sorted(res["shapley_values"]), res["value_function"]
    )
    assert set(res["shapley_values"]) == set(expected)
    for channel, value in res["shapley_values"].items():
        assert value == pytest.approx(expected[channel], abs=1e-12)


def test_cumulative_symmetry_and_null_player():
    # A and B are interchangeable; D never appears on a converting path.
    paths = ["A > C", "B > C", "D", "A > D"]
    conversions = [1, 1, 0, 0]
    res = ShapleyAttribution().fit(paths, conversions)
    assert res["shapley_values"]["A"] == pytest.approx(res["shapley_values"]["B"])
    assert res["shapley_values"]["D"] == 0.0
    assert res["attribution"]["D"] == 0.0


def test_cumulative_repeated_touches_count_once_per_channel():
    # "A > B > A" has the channel set {A, B}: each channel gets half.
    res = ShapleyAttribution().fit(["A > B > A"], [1])
    assert res["value_function"] == {("A", "B"): 1}
    assert res["shapley_values"] == {"A": 0.5, "B": 0.5}


def test_cumulative_reserved_tokens_are_not_players():
    # Reserved tokens are excluded from the channel list, so they receive no
    # credit and do not dilute the channels that share a path with them.
    res = ShapleyAttribution().fit(["A > NULL", "B"], [1, 1])
    assert set(res["shapley_values"]) == {"A", "B"}
    assert res["shapley_values"] == {"A": 1.0, "B": 1.0}
    assert res["attribution"] == {"A": 0.5, "B": 0.5}


def test_cumulative_is_invariant_to_path_order():
    paths = ["A > B", "B > C", "A", "C > A > B", "B", "A > C"] * 7
    conversions = [1, 1, 0, 1, 1, 0] * 7
    forward = ShapleyAttribution().fit(paths, conversions)
    backward = ShapleyAttribution().fit(paths[::-1], conversions[::-1])
    assert forward["shapley_values"] == backward["shapley_values"]
    assert forward["attribution"] == backward["attribution"]


def test_cumulative_zero_conversions_refuses_normalization():
    with pytest.warns(UserWarning, match="normalization is unavailable"):
        res = ShapleyAttribution().fit(["A > B", "B > C"], [0, 0])
    assert res["attribution"] is None
    assert res["normalization_reason"] == "zero_raw_magnitude"
    assert res["shapley_values"] == {"A": 0.0, "B": 0.0, "C": 0.0}


def test_cumulative_summary_reports_available_shares(
    example_paths, example_conversions
):
    model = ShapleyAttribution(conversion_labels=["Purchase"])
    res = model.fit(example_paths, example_conversions)
    assert res["normalization_status"] == "available"
    assert model.get_attribution() == res["attribution"]
    summary = model.summary()
    assert "Normalized attribution unavailable" not in summary
    assert "Not calculated - requires normalized attribution" not in summary


@pytest.mark.parametrize("value", ["subset", "", "Cumulative", "EXACT"])
def test_unknown_characteristic_function_rejected(value):
    with pytest.raises(ValueError, match="characteristic_function must be one of"):
        ShapleyAttribution(characteristic_function=value)


@pytest.mark.parametrize("value", [None, 1, ["cumulative"]])
def test_non_string_characteristic_function_rejected(value):
    with pytest.raises(TypeError, match="characteristic_function must be a string"):
        ShapleyAttribution(characteristic_function=value)


# --------------------------------------------------------------------------- #
# Optional per-journey observation weights                                     #
# --------------------------------------------------------------------------- #
# fit(..., weights=w) replaces the coalition table R(T) with the weighted
# R_w(T) = sum of w_j * Y_j over the journeys whose channel set is exactly T,
# and evaluates both characteristic functions on it.  Numerical equivalence
# is asserted to rtol=1e-10 / atol=1e-12 rather than bitwise.
RTOL = 1e-10
ATOL = 1e-12
GAMES = ["cumulative", "exact"]

# "B > D" has zero weight, so D must vanish; the empty path converts with
# weight 4.0, so it counts toward weighted conversions but credits no channel.
WEIGHTED_PATHS = ["A", "A > B", "A > B", "C", "B > C > A", "B > D", ""]
WEIGHTED_CONVERSIONS = [1, 1, 0, 1, 1, 1, 1]
WEIGHTED_WEIGHTS = [0.5, 1.25, 3.0, 2.0, 0.75, 0.0, 4.0]
WEIGHTED_TABLE = {
    ("A",): 0.5,
    ("A", "B"): 1.25,
    ("C",): 2.0,
    ("A", "B", "C"): 0.75,
}


def _assert_mapping_close(actual, expected):
    """Same keys, and values equal to rtol=1e-10 / atol=1e-12."""
    assert set(actual) == set(expected)
    keys = sorted(expected)
    np.testing.assert_allclose(
        [actual[key] for key in keys],
        [expected[key] for key in keys],
        rtol=RTOL,
        atol=ATOL,
    )


def _assert_same_fit(actual, expected):
    """Two fits agree on every mathematical output and on availability."""
    _assert_mapping_close(actual["value_function"], expected["value_function"])
    _assert_mapping_close(actual["shapley_values"], expected["shapley_values"])
    assert actual["normalization_status"] == expected["normalization_status"]
    assert actual["normalization_reason"] == expected["normalization_reason"]
    if expected["attribution"] is None:
        assert actual["attribution"] is None
    else:
        _assert_mapping_close(actual["attribution"], expected["attribution"])


def _fit(characteristic_function, paths, conversions, weights=None, **kwargs):
    """Fit quietly: an unavailable normalization warns by design."""
    model = ShapleyAttribution(
        characteristic_function=characteristic_function, **kwargs
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if weights is None:
            result = model.fit(paths, conversions)
        else:
            result = model.fit(paths, conversions, weights=weights)
    return model, result


@pytest.mark.parametrize("characteristic_function", GAMES)
def test_weighted_value_function_is_the_weighted_coalition_table(
    characteristic_function,
):
    model, res = _fit(
        characteristic_function,
        WEIGHTED_PATHS,
        WEIGHTED_CONVERSIONS,
        WEIGHTED_WEIGHTS,
    )
    _assert_mapping_close(res["value_function"], WEIGHTED_TABLE)
    # Fractional totals survive: nothing is cast to an integer.
    assert all(isinstance(value, float) for value in res["value_function"].values())
    assert res["value_function"][("A", "B")] == pytest.approx(1.25, rel=RTOL)
    assert model.value_function is res["value_function"]


def test_weighted_cumulative_hand_computed_example():
    # phi_A = 0.5 + 1.25 / 2 + 0.75 / 3,  phi_B = 1.25 / 2 + 0.75 / 3,
    # phi_C = 2.0 + 0.75 / 3.  They sum to 4.5, the credited weighted total.
    _, res = _fit(
        "cumulative", WEIGHTED_PATHS, WEIGHTED_CONVERSIONS, WEIGHTED_WEIGHTS
    )
    _assert_mapping_close(
        res["shapley_values"], {"A": 1.375, "B": 0.875, "C": 2.25}
    )
    _assert_mapping_close(
        res["attribution"], {"A": 1.375 / 4.5, "B": 0.875 / 4.5, "C": 2.25 / 4.5}
    )
    assert res["normalization_status"] == "available"


def test_weighted_exact_hand_computed_example():
    # v_w({A}) = 0.5, v_w({B}) = 0, v_w({A, B}) = 1.5.
    # phi_A = 0.5 * (0.5 - 0) + 0.5 * (1.5 - 0)   = 1.0
    # phi_B = 0.5 * (0   - 0) + 0.5 * (1.5 - 0.5) = 0.5
    _, res = _fit("exact", ["A", "A > B"], [1, 1], [0.5, 1.5])
    _assert_mapping_close(res["value_function"], {("A",): 0.5, ("A", "B"): 1.5})
    _assert_mapping_close(res["shapley_values"], {"A": 1.0, "B": 0.5})
    _assert_mapping_close(res["attribution"], {"A": 2 / 3, "B": 1 / 3})


def test_weighted_cumulative_efficiency_and_factorial_weight_definition():
    """sum(phi_i) = sum of R_w(T) over every nonempty coalition T."""
    _, res = _fit(
        "cumulative", WEIGHTED_PATHS, WEIGHTED_CONVERSIONS, WEIGHTED_WEIGHTS
    )
    credited = sum(res["value_function"].values())
    np.testing.assert_allclose(
        sum(res["shapley_values"].values()), credited, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(res["raw_sum"], 4.5, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        sum(res["attribution"].values()), 1.0, rtol=RTOL, atol=ATOL
    )
    # The closed form equals the factorial-weight definition on R_w.
    _assert_mapping_close(
        res["shapley_values"],
        _brute_force_cumulative_shapley(
            sorted(res["shapley_values"]), res["value_function"]
        ),
    )


def test_weighted_exact_efficiency_equals_grand_coalition_value():
    """sum(phi_i) = v_w(N) - v_w(empty), and the empty coalition is worth 0."""
    model, res = _fit(
        "exact", WEIGHTED_PATHS, WEIGHTED_CONVERSIONS, WEIGHTED_WEIGHTS
    )
    grand = tuple(model.channels)
    assert grand == ("A", "B", "C")
    assert () not in res["value_function"]
    np.testing.assert_allclose(
        sum(res["shapley_values"].values()),
        res["value_function"][grand],
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(res["raw_sum"], 0.75, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("characteristic_function", GAMES)
@pytest.mark.parametrize(
    "unit_weights",
    [
        lambda n: [1.0] * n,
        lambda n: [1] * n,
        lambda n: np.ones(n),
        lambda n: pd.Series(np.ones(n)),
    ],
    ids=["float-list", "int-list", "ndarray", "series"],
)
def test_unit_weights_reproduce_the_unweighted_fit(
    characteristic_function, unit_weights, example_paths, example_conversions
):
    kwargs = {"conversion_labels": ["Purchase"]}
    plain_model, plain = _fit(
        characteristic_function, example_paths, example_conversions, **kwargs
    )
    unit_model, unit = _fit(
        characteristic_function,
        example_paths,
        example_conversions,
        unit_weights(len(example_paths)),
        **kwargs,
    )
    _assert_same_fit(unit, plain)
    assert unit_model.channels == plain_model.channels
    # The unweighted path keeps its integer coalition totals.
    assert all(isinstance(value, int) for value in plain["value_function"].values())


@pytest.mark.parametrize("characteristic_function", GAMES)
def test_integer_frequency_weights_reproduce_replicated_journeys(
    characteristic_function,
):
    paths = ["A", "A > B", "B > A", "C", "A > B > C", "Z > A", "B", "Z"]
    conversions = [1, 1, 0, 1, 1, 1, 0, 0]
    frequencies = [3, 2, 4, 1, 2, 0, 5, 0]
    replicated_paths, replicated_conversions = [], []
    for path, converted, frequency in zip(paths, conversions, frequencies):
        replicated_paths.extend([path] * frequency)
        replicated_conversions.extend([converted] * frequency)

    weighted_model, weighted = _fit(
        characteristic_function, paths, conversions, frequencies
    )
    replicated_model, replicated = _fit(
        characteristic_function, replicated_paths, replicated_conversions
    )

    _assert_same_fit(weighted, replicated)
    # Zero-frequency journeys are excluded exactly as if they were never
    # listed: Z, seen only on them, is in neither channel set.
    assert weighted_model.channels == replicated_model.channels == ["A", "B", "C"]


@pytest.mark.parametrize("characteristic_function", GAMES)
def test_zero_weight_only_channel_is_absent_from_every_output(
    characteristic_function,
):
    paths = ["A", "A > B", "Ghost > A", "Ghost"]
    conversions = [1, 1, 1, 0]
    model, res = _fit(characteristic_function, paths, conversions, [1.5, 2.0, 0, 0])

    assert model.channels == ["A", "B"]
    assert model.get_channels() == ["A", "B"]
    assert set(res["shapley_values"]) == {"A", "B"}
    assert set(res["attribution"]) == {"A", "B"}
    assert all("Ghost" not in coalition for coalition in res["value_function"])
    tables = model.to_dataframes()
    assert tables["attribution"]["channel"].tolist() == ["A", "B"]
    assert tables["shapley_values"]["channel"].tolist() == ["A", "B"]
    assert "Ghost" not in model.summary()
    assert "Ghost" not in model.summary(output_format="report")

    # Unweighted, the same journeys do make Ghost a channel.
    plain_model, _ = _fit(characteristic_function, paths, conversions)
    assert plain_model.channels == ["A", "B", "Ghost"]


def test_weighted_fit_matches_the_fit_on_positive_weight_rows_only():
    """Dropping the zero-weight rows by hand changes nothing."""
    keep = [weight > 0 for weight in WEIGHTED_WEIGHTS]
    for characteristic_function in GAMES:
        _, full = _fit(
            characteristic_function,
            WEIGHTED_PATHS,
            WEIGHTED_CONVERSIONS,
            WEIGHTED_WEIGHTS,
        )
        _, retained = _fit(
            characteristic_function,
            [p for p, k in zip(WEIGHTED_PATHS, keep) if k],
            [c for c, k in zip(WEIGHTED_CONVERSIONS, keep) if k],
            [w for w, k in zip(WEIGHTED_WEIGHTS, keep) if k],
        )
        _assert_same_fit(full, retained)


def test_weighted_fit_is_invariant_to_journey_order():
    _, forward = _fit(
        "cumulative", WEIGHTED_PATHS, WEIGHTED_CONVERSIONS, WEIGHTED_WEIGHTS
    )
    _, backward = _fit(
        "cumulative",
        WEIGHTED_PATHS[::-1],
        WEIGHTED_CONVERSIONS[::-1],
        WEIGHTED_WEIGHTS[::-1],
    )
    _assert_same_fit(forward, backward)


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ([1.0, 2.0], "Paths (3) and weights (2) must have same length"),
        ([1.0, 2.0, 3.0, 4.0], "Paths (3) and weights (4) must have same length"),
        ([[1.0, 2.0, 3.0]], "weights must be one-dimensional"),
        (2.0, "weights must be one-dimensional"),
        ([1.0, float("nan"), 1.0], "weights must contain only finite values"),
        ([1.0, float("inf"), 1.0], "weights must contain only finite values"),
        ([1.0, -0.5, 1.0], "weights must be nonnegative"),
        ([0.0, 0.0, 0.0], "weights must have positive total weight"),
        ([False, False, False], "weights must have positive total weight"),
    ],
)
def test_invalid_weights_rejected_exactly_as_markov_rejects_them(weights, message):
    from do_attribution import MarkovAttribution

    paths, conversions = ["A", "A > B", "B"], [1, 1, 0]
    with pytest.raises(ValueError) as shapley_error:
        ShapleyAttribution().fit(paths, conversions, weights=weights)
    assert str(shapley_error.value) == message

    with pytest.raises(ValueError) as markov_error:
        MarkovAttribution(removal_strategy="redirect_to_null").fit(
            paths, conversions, weights=weights
        )
    assert str(markov_error.value) == str(shapley_error.value)


def test_rejected_weights_leave_the_previous_fit_in_place():
    model = ShapleyAttribution()
    first = model.fit(["A", "A > B"], [1, 1])
    summary = model.summary()
    with pytest.raises(ValueError, match="nonnegative"):
        model.fit(["C", "D"], [1, 1], weights=[1.0, -1.0])
    assert model.channels == ["A", "B"]
    assert model.get_attribution() == first["attribution"]
    assert model.summary() == summary


@pytest.mark.parametrize("bad_outcome", [2, 0.5, float("nan")])
def test_outcomes_on_zero_weight_rows_are_still_validated(bad_outcome):
    with pytest.raises(ValueError, match="index 1 must be 0 or 1"):
        ShapleyAttribution().fit(
            ["A", "B"], [1, bad_outcome], weights=[1.0, 0.0]
        )


@pytest.mark.parametrize("characteristic_function", GAMES)
def test_boolean_weights_count_as_zero_one(characteristic_function):
    paths = ["A", "A > B", "B > D", "A > B"]
    conversions = [1, 1, 1, 1]
    model, boolean = _fit(
        characteristic_function, paths, conversions, [True, True, False, True]
    )
    _, numeric = _fit(
        characteristic_function, paths, conversions, [1.0, 1.0, 0.0, 1.0]
    )
    _, selected = _fit(
        characteristic_function, ["A", "A > B", "A > B"], [1, 1, 1]
    )
    _assert_same_fit(boolean, numeric)
    _assert_same_fit(boolean, selected)
    assert model.channels == ["A", "B"]


@pytest.mark.parametrize("characteristic_function", GAMES)
def test_series_weights_align_by_position_not_index_labels(characteristic_function):
    _, expected = _fit(
        characteristic_function,
        WEIGHTED_PATHS,
        WEIGHTED_CONVERSIONS,
        WEIGHTED_WEIGHTS,
    )
    n = len(WEIGHTED_PATHS)
    _, res = _fit(
        characteristic_function,
        pd.Series(WEIGHTED_PATHS, index=range(100, 100 + n)),
        pd.Series(WEIGHTED_CONVERSIONS, index=list("gfedcba")),
        # Label alignment would reverse these weights; position must win.
        pd.Series(WEIGHTED_WEIGHTS, index=range(n - 1, -1, -1)),
    )
    _assert_same_fit(res, expected)


@pytest.mark.parametrize("characteristic_function", GAMES)
def test_no_positively_weighted_conversion_refuses_normalization(
    characteristic_function,
):
    """Positive total weight, but every converting journey has weight 0."""
    model = ShapleyAttribution(characteristic_function=characteristic_function)
    with pytest.warns(UserWarning, match="normalization is unavailable"):
        res = model.fit(
            ["A > B", "B > C", "A > C"], [0, 0, 1], weights=[2.0, 3.5, 0.0]
        )

    assert res["attribution"] is None
    assert res["normalization_status"] == "unavailable"
    assert res["normalization_reason"] == "zero_raw_magnitude"
    assert res["shapley_values"] == {"A": 0.0, "B": 0.0, "C": 0.0}
    assert res["value_function"] == {("A", "B"): 0.0, ("B", "C"): 0.0}
    with pytest.raises(ValueError, match="unavailable"):
        model.get_attribution()
    row = model.to_dataframes()["fit_diagnostics"].iloc[0]
    assert row["total_weight"] == pytest.approx(5.5, rel=RTOL)
    assert row["weighted_conversions"] == 0.0
    summary = model.summary()
    assert "Normalized attribution unavailable (zero_raw_magnitude)" in summary
    assert "Not calculated - requires normalized attribution" in summary


def test_weighted_fit_with_no_channel_on_any_positive_weight_journey():
    model = ShapleyAttribution(conversion_labels=["Purchase"])
    res = model.fit(
        ["Purchase", "", "Email > Purchase"], [1, 1, 1], weights=[1.5, 2.0, 0.0]
    )

    assert model.channels == []
    assert res["attribution"] is None
    assert res["shapley_values"] == {}
    assert res["value_function"] == {}
    assert res["normalization_reason"] == "zero_raw_magnitude"
    tables = model.to_dataframes()
    assert tables["attribution"].empty and tables["shapley_values"].empty
    row = tables["fit_diagnostics"].iloc[0]
    assert bool(row["weights_supplied"]) is True
    assert row["total_weight"] == pytest.approx(3.5, rel=RTOL)
    # Conversions on channel-less paths are weighted conversions all the
    # same; they are simply credited to no channel.
    assert row["weighted_conversions"] == pytest.approx(3.5, rel=RTOL)
    summary = model.summary()
    assert "Observation weights" in summary
    assert "Email" not in summary


@pytest.mark.parametrize("characteristic_function", GAMES)
def test_weighted_fit_strips_conversion_labels(
    characteristic_function,
    example_paths,
    example_paths_no_endpoint,
    example_conversions,
):
    weights = [2.5, 1.0, 1.0, 4.0, 0.5, 0.0, 2.0, 1.5, 1.0, 3.0]
    model, labeled = _fit(
        characteristic_function,
        example_paths,
        example_conversions,
        weights,
        conversion_labels=["Purchase"],
    )
    _, prestripped = _fit(
        characteristic_function, example_paths_no_endpoint, example_conversions, weights
    )
    _assert_same_fit(labeled, prestripped)
    assert "Purchase" not in model.get_channels()
    assert "Purchase" not in labeled["shapley_values"]
    assert all("Purchase" not in coalition for coalition in labeled["value_function"])
    # Direct is seen only on the zero-weight journey.
    assert model.get_channels() == ["Email", "Search", "Social"]


@pytest.mark.parametrize("output_format", ["summary", "report"])
def test_weighted_summary_preserves_fractional_totals_and_raw_values(output_format):
    model, _ = _fit(
        "cumulative", WEIGHTED_PATHS, WEIGHTED_CONVERSIONS, WEIGHTED_WEIGHTS
    )
    text = model.summary(output_format=output_format)

    # Observed counts stay ordinary counts of the seven supplied rows.
    assert "7 paths" in text and "6 conversions" in text
    assert "85.71%" in text
    # Weighted totals are reported separately from them.
    assert "Observation weights" in text
    assert "Top coalitions by weighted conversions" in text
    assert "Top coalitions by conversions" not in text
    if output_format == "summary":
        assert "Overall: 7 paths, 6 conversions, 85.71% conversion rate" in text
        for row in (
            "| Total observation weight      | 11.5  |",
            "| Weighted conversions          | 8.5   |",
            "| Zero-weight journeys excluded | 1     |",
            # Fractional coalition totals: 1.25 is not shown as 1.
            "| A + B     | 1.25                 |",
            "| A + B + C | 0.75                 |",
            "| A         | 0.5                  |",
            "| C         | 2                    |",
            "| C       | +2.2500 |",
            "| A       | +1.3750 |",
            "| B       | +0.8750 |",
        ):
            assert row in text, row
    else:
        assert (
            "Observation weights: Total observation weight, 11.5; Weighted "
            "conversions, 8.5; Zero-weight journeys excluded, 1."
        ) in text
        assert (
            "Top coalitions by weighted conversions: C, 2; A + B, 1.25; "
            "A + B + C, 0.75; A, 0.5."
        ) in text
        assert "Single-channel baselines v({c}): C, 2; A, 0.5." in text
        assert (
            "Raw Shapley values (signed): C, +2.2500; A, +1.3750; B, +0.8750."
        ) in text


@pytest.mark.parametrize("characteristic_function", GAMES)
def test_weighted_summary_never_shows_a_small_nonzero_value_as_zero(
    characteristic_function,
):
    model, res = _fit(
        characteristic_function, ["A", "A > B"], [1, 1], [2e-6, 5e-6]
    )
    assert res["value_function"] == {("A",): 2e-6, ("A", "B"): 5e-6}
    for output_format in ("summary", "report"):
        text = model.summary(output_format=output_format)
        assert "2e-06" in text and "5e-06" in text and "7e-06" in text
        assert "+0.0000" not in text and "-0.0000" not in text
    table = model.summary()
    assert "| A + B     | 5e-06                |" in table
    assert "| A         | 2e-06                |" in table


def test_unweighted_summary_is_unchanged_by_the_weight_option():
    model, _ = _fit(
        "cumulative", WEIGHTED_PATHS, WEIGHTED_CONVERSIONS, WEIGHTED_WEIGHTS
    )
    assert "Observation weights" in model.summary()

    # Refitting the same model without weights restores the unweighted form.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        refit = model.fit(WEIGHTED_PATHS, WEIGHTED_CONVERSIONS)
    fresh_model, fresh = _fit("cumulative", WEIGHTED_PATHS, WEIGHTED_CONVERSIONS)

    assert refit == fresh
    assert model.channels == ["A", "B", "C", "D"]
    assert all(isinstance(value, int) for value in refit["value_function"].values())
    for output_format in ("summary", "report"):
        text = model.summary(output_format=output_format)
        assert text == fresh_model.summary(output_format=output_format)
        assert "Observation weights" not in text
        assert "weighted" not in text.lower()
        assert "Top coalitions by conversions" in text
    row = model.to_dataframes()["fit_diagnostics"].iloc[0]
    assert bool(row["weights_supplied"]) is False
    assert row["total_weight"] == 7.0
    assert row["weighted_conversions"] == 6.0


@pytest.mark.parametrize("characteristic_function", GAMES)
def test_weights_whose_totals_overflow_report_unavailable_instead_of_raising(
    characteristic_function,
):
    """Finite weights pass validation even when their sums leave float range."""
    big = 1.5e308
    model = ShapleyAttribution(characteristic_function=characteristic_function)
    with warnings.catch_warnings():
        # NumPy notes the overflow while the weights are being validated.
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        res = model.fit(["A", "A > B", "A"], [1, 1, 1], weights=[big, big, big])
    assert res["attribution"] is None
    assert res["normalization_status"] == "unavailable"
    assert res["normalization_reason"] == "nonfinite_result"
    assert res["value_function"][("A",)] == float("inf")
    assert res["value_function"][("A", "B")] == big
    row = model.to_dataframes()["fit_diagnostics"].iloc[0]
    assert row["total_weight"] == float("inf")
    assert "Normalized attribution unavailable (nonfinite_result)" in model.summary()
