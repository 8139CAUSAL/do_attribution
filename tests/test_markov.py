"""Markov-attribution core tests.

Port of ``markov-attribution/tests/test_core.py`` with additional
coverage for the detour strategy and higher-order chains.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from do_attribution import MarkovAttribution


# --------------------------------------------------------------------------- #
# Core first-order behaviour                                                   #
# --------------------------------------------------------------------------- #
def test_first_order_basic():
    paths = ["A > B", "A > C", "B > C"]
    conversions = [1, 0, 1]

    model = MarkovAttribution(order=1)
    results = model.fit(paths, conversions)

    assert abs(sum(results["attribution"].values()) - 1.0) < 1e-10
    assert set(results["attribution"].keys()) == {"A", "B", "C"}
    assert results["attribution"]["A"] > 0
    assert results["attribution"]["B"] > 0


def test_single_channel_paths():
    paths = ["A", "B", "A > B"]
    conversions = [1, 0, 1]

    model = MarkovAttribution(order=1)
    results = model.fit(paths, conversions)

    assert results["attribution"]["A"] > results["attribution"]["B"]


def test_no_conversions_returns_unavailable_attribution():
    paths = ["A > B", "B > C"]
    conversions = [0, 0]

    model = MarkovAttribution(order=1)
    with pytest.warns(UserWarning):
        results = model.fit(paths, conversions)

    assert results["total_conversion_probability"] == 0.0
    assert results["removal_effects"] == {"A": 0.0, "B": 0.0, "C": 0.0}
    assert results["attribution"] is None
    assert results["normalization_status"] == "unavailable"
    assert results["normalization_reason"] == "zero_raw_magnitude"
    assert results["normalization_denominator"] is None
    assert results["raw_sum"] == 0.0
    assert results["raw_magnitude"] == 0.0
    assert results["cancellation_ratio"] is None


def test_all_conversions_total_prob_one():
    paths = ["A > B", "B > C", "A > C"]
    conversions = [1, 1, 1]

    model = MarkovAttribution(order=1)
    results = model.fit(paths, conversions)

    assert results["total_conversion_probability"] == 1.0


def test_return_transition_matrix_is_dataframe():
    paths = ["A > B", "A > C"]
    conversions = [1, 1]
    model = MarkovAttribution(order=1, return_transition_matrix=True)
    results = model.fit(paths, conversions)

    tm = results["transition_matrix"]
    assert isinstance(tm, pd.DataFrame)
    # Absorbing states must be self-loops.
    assert tm.at["CONVERSION", "CONVERSION"] == 1.0
    assert tm.at["NULL", "NULL"] == 1.0


# --------------------------------------------------------------------------- #
# Higher-order                                                                 #
# --------------------------------------------------------------------------- #
def test_second_order():
    paths = ["A > B > C", "A > B > D"]
    conversions = [1, 0]

    model = MarkovAttribution(order=2)
    results = model.fit(paths, conversions)

    assert set(results["attribution"].keys()) == {"A", "B", "C", "D"}
    assert results["attribution"]["C"] > results["attribution"]["D"]


def test_third_order_channels_match():
    paths = ["A > B > C > D", "A > B > C > E"]
    conversions = [1, 0]
    model = MarkovAttribution(order=3)
    results = model.fit(paths, conversions)
    assert set(results["attribution"].keys()) == {"A", "B", "C", "D", "E"}


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #
def test_invalid_inputs():
    model = MarkovAttribution(order=1)
    with pytest.raises(ValueError):
        model.fit(["A > B"], [1, 0])
    with pytest.raises(ValueError):
        model.fit(["A > B"], [2])
    with pytest.raises(ValueError):
        MarkovAttribution(order=0)


def test_invalid_removal_strategy_rejected():
    with pytest.raises(ValueError, match="fail_refit"):
        MarkovAttribution(removal_strategy="bogus")


def test_fail_refit_is_valid_and_fail_warning_is_actionable_once():
    assert (
        MarkovAttribution(removal_strategy="fail_refit").removal_strategy
        == "fail_refit"
    )
    model = MarkovAttribution(removal_strategy="fail")
    with pytest.warns(UserWarning, match="removal_strategy='fail_refit'"):
        model.fit(["A", "B"], [1, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(["A", "B"], [1, 0])


# --------------------------------------------------------------------------- #
# Removal effects                                                              #
# --------------------------------------------------------------------------- #
def test_fail_removal_effects_are_non_negative():
    paths = ["A > B > C"] * 10 + ["A > C"] * 5 + ["B > C"] * 3
    conversions = [1] * 10 + [0] * 5 + [1] * 3

    model = MarkovAttribution(order=1, removal_strategy="fail")
    results = model.fit(paths, conversions)

    assert results["removal_effects"]["B"] > 0
    for effect in results["removal_effects"].values():
        assert 0 <= effect <= 1


def test_detour_can_produce_signed_effects():
    """Simulated chain where A/C are highly converting and D is harmful.

    Mirrors the scenario from the original ``testknown2.py``: A and C
    convert with p=0.9, B with p=0.5, D with p=0.1.  With a large enough
    sample the detour strategy should produce a clearly negative effect
    for the helpful channel A and a clearly positive effect for the
    harmful channel D.
    """
    rng = np.random.default_rng(42)

    def next_state(prev: str) -> str:
        if prev == "START":
            return rng.choice(["A", "B", "D"], p=[0.4, 0.4, 0.2])
        if prev in ("A", "C"):
            return rng.choice(["CONVERSION", "NULL"], p=[0.9, 0.1])
        if prev == "B":
            return rng.choice(["CONVERSION", "NULL"], p=[0.5, 0.5])
        if prev == "D":
            return rng.choice(["CONVERSION", "NULL"], p=[0.1, 0.9])
        raise AssertionError(f"unexpected state {prev}")

    paths: list[str] = []
    convs: list[int] = []
    for _ in range(3000):
        touches, s = [], "START"
        while s not in ("CONVERSION", "NULL"):
            s = next_state(s)
            if s not in ("CONVERSION", "NULL"):
                touches.append(s)
        paths.append(" > ".join(touches))
        convs.append(int(s == "CONVERSION"))

    model = MarkovAttribution(order=1, removal_strategy="detour")
    results = model.fit(paths, convs)
    eff = results["removal_effects"]

    # D is harmful => removing it should raise conversion => positive effect
    assert eff["D"] > 0
    # A is helpful => removing it should drop conversion => negative effect
    assert eff["A"] < 0


def test_detour_vs_fail_differ():
    paths = ["A > B > C", "A > C", "B > C", "A > B", "C"]
    convs = [1, 0, 1, 1, 0]
    fail = MarkovAttribution(order=1, removal_strategy="fail").fit(paths, convs)
    detour = MarkovAttribution(order=1, removal_strategy="detour").fit(paths, convs)
    assert fail["attribution"] != detour["attribution"]


def test_characterization_detour_near_cancellation_case():
    """Keep signed diagnostics without manufacturing explosive shares."""
    paths = ["A"] * 500 + ["B"] * 500
    conversions = [1] * 400 + [0] * 100 + [1] * 100 + [0] * 400
    model = MarkovAttribution(order=1, removal_strategy="detour")

    result = model.fit(paths, conversions)
    raw_effects = result["removal_effects"]
    raw_sum = sum(raw_effects.values())
    raw_magnitude = sum(abs(value) for value in raw_effects.values())
    cancellation_ratio = abs(raw_sum) / raw_magnitude
    denominator = raw_sum

    assert result["total_conversion_probability"] == pytest.approx(0.5)
    assert raw_effects == pytest.approx(
        {"A": -0.6, "B": 0.6000000000000001}
    )
    assert raw_sum == pytest.approx(1.1102230246251565e-16)
    assert raw_magnitude == pytest.approx(1.2000000000000002)
    assert cancellation_ratio == pytest.approx(9.25185853854297e-17)
    assert denominator == pytest.approx(1.1102230246251565e-16)
    assert result["attribution"] is None
    assert result["normalization_status"] == "unavailable"
    assert result["normalization_reason"] == "near_cancellation"
    assert result["normalization_denominator"] == pytest.approx(denominator)
    assert result["raw_sum"] == pytest.approx(raw_sum)
    assert result["raw_magnitude"] == pytest.approx(raw_magnitude)
    assert result["cancellation_ratio"] == pytest.approx(cancellation_ratio)

    summary = model.summary()
    assert (
        "Removal effects (raw, signed); normalized attribution unavailable"
        in summary
    )
    assert "Removal effects (raw, signed) vs normalized shares" not in summary
    assert "540431955284459584.00%" not in summary
    assert "| Share |" not in summary


def test_characterization_fail_refit_changes_population():
    """Keep legacy population refitting explicit and normalization safe."""
    paths = ["A"] * 500 + ["B"] * 500
    conversions = [1] * 400 + [0] * 100 + [1] * 100 + [0] * 400
    model = MarkovAttribution(order=1, removal_strategy="fail_refit")
    build_populations = []
    original_build = model._build_transition_matrix

    def recording_build(state_sequences):
        build_populations.append(
            {
                "count": len(state_sequences),
                "weight": sum(weight for _, _, weight in state_sequences),
                "converted": sum(
                    converted * weight
                    for _, converted, weight in state_sequences
                ),
                "states": {
                    state
                    for states, _, _ in state_sequences
                    for state in states
                },
            }
        )
        return original_build(state_sequences)

    model._build_transition_matrix = recording_build
    result = model.fit(paths, conversions)

    raw_effects = result["removal_effects"]
    raw_sum = sum(raw_effects.values())
    raw_magnitude = sum(abs(value) for value in raw_effects.values())
    cancellation_ratio = abs(raw_sum) / raw_magnitude
    candidate = {
        channel: effect / raw_sum for channel, effect in raw_effects.items()
    }

    assert build_populations == [
        {
            "count": 1000,
            "weight": 1000,
            "converted": 500,
            "states": {"A", "B"},
        },
        {
            "count": 500,
            "weight": 500,
            "converted": 100,
            "states": {"B"},
        },
        {
            "count": 500,
            "weight": 500,
            "converted": 400,
            "states": {"A"},
        },
    ]
    assert raw_effects == pytest.approx(
        {"A": 0.6, "B": -0.6000000000000001}
    )
    assert raw_sum == pytest.approx(-1.1102230246251565e-16)
    assert raw_magnitude == pytest.approx(1.2000000000000002)
    assert cancellation_ratio == pytest.approx(9.25185853854297e-17)
    assert candidate == pytest.approx(
        {"A": -5404319552844595.0, "B": 5404319552844596.0}
    )
    assert result["attribution"] is None
    assert result["normalization_status"] == "unavailable"
    assert result["normalization_reason"] == "near_cancellation"
    assert result["population_policy"] == "population_refit"
    for diagnostic in result["population_diagnostics"].values():
        assert diagnostic == {
            "original_journey_count": 1000,
            "transformed_journey_count": 500,
            "original_total_weight": 1000.0,
            "transformed_total_weight": 500.0,
            "emptied_path_count": 500,
            "dropped_path_count": 500,
            "empty_path_handling": "drop_after_channel_removal",
        }


def test_fail_and_fail_refit_answer_different_population_questions():
    paths = ["A"] * 500 + ["B"] * 500
    conversions = [1] * 400 + [0] * 100 + [1] * 100 + [0] * 400

    with pytest.warns(UserWarning, match="fail_refit"):
        fixed = MarkovAttribution(
            order=1, removal_strategy="fail"
        ).fit(paths, conversions)
    refit = MarkovAttribution(
        order=1, removal_strategy="fail_refit"
    ).fit(paths, conversions)

    assert fixed["population_policy"] == "fixed"
    assert refit["population_policy"] == "population_refit"
    assert fixed["removal_effects"] == pytest.approx(
        {"A": 0.8, "B": 0.2}
    )
    assert refit["removal_effects"] == pytest.approx(
        {"A": 0.6, "B": -0.6000000000000001}
    )
    assert fixed["attribution"] == pytest.approx({"A": 0.8, "B": 0.2})
    assert refit["attribution"] is None
    assert fixed["normalization_status"] == "available"
    assert fixed["normalization_reason"] is None
    assert refit["normalization_status"] == "unavailable"
    assert refit["normalization_reason"] == "near_cancellation"

    for diagnostic in fixed["population_diagnostics"].values():
        assert diagnostic["transformed_journey_count"] == 1000
        assert diagnostic["transformed_total_weight"] == 1000.0
        assert diagnostic["dropped_path_count"] == 0
        assert diagnostic["empty_path_representation"] == "START -> NULL"
    for diagnostic in refit["population_diagnostics"].values():
        assert diagnostic["transformed_journey_count"] == 500
        assert diagnostic["transformed_total_weight"] == 500.0
        assert diagnostic["dropped_path_count"] == 500


def test_fixed_fail_only_channel_journeys_become_start_to_null():
    with pytest.warns(UserWarning, match="START -> NULL"):
        result = MarkovAttribution(
            removal_strategy="fail"
        ).fit(["A", "A", "A"], [1, 0, 1])

    assert result["removal_effects"]["A"] == pytest.approx(1.0)
    assert result["population_diagnostics"]["A"] == {
        "original_journey_count": 3,
        "transformed_journey_count": 3,
        "original_total_weight": 3.0,
        "transformed_total_weight": 3.0,
        "emptied_path_count": 3,
        "dropped_path_count": 0,
        "empty_path_handling": "start_to_null",
        "empty_path_representation": "START -> NULL",
    }


def test_fixed_fail_paths_retaining_other_channels_stay_in_population():
    model = MarkovAttribution(removal_strategy="fail")
    builds = []
    original_build = model._build_transition_matrix

    def recording_build(state_sequences):
        builds.append(state_sequences)
        return original_build(state_sequences)

    model._build_transition_matrix = recording_build
    with pytest.warns(UserWarning, match="fail_refit"):
        result = model.fit(["A > B", "B"], [1, 0])

    removed_a_sequences = builds[1]
    assert [states for states, _, _ in removed_a_sequences] == [
        ["B"],
        ["B"],
    ]
    assert [converted for _, converted, _ in removed_a_sequences] == [0, 0]
    assert (
        result["population_diagnostics"]["A"]["emptied_path_count"] == 0
    )
    assert (
        result["population_diagnostics"]["A"]["transformed_journey_count"]
        == 2
    )


def test_fixed_fail_removes_every_repeated_channel_occurrence():
    model = MarkovAttribution(removal_strategy="fail")
    builds = []
    original_build = model._build_transition_matrix

    def recording_build(state_sequences):
        builds.append(state_sequences)
        return original_build(state_sequences)

    model._build_transition_matrix = recording_build
    with pytest.warns(UserWarning, match="fail_refit"):
        model.fit(["A > A > B > A", "B"], [1, 0])

    removed_a_sequences = builds[1]
    assert [states for states, _, _ in removed_a_sequences] == [
        ["B"],
        ["B"],
    ]


def test_fixed_fail_preserves_unaffected_outcomes_and_fails_affected_ones():
    model = MarkovAttribution(removal_strategy="fail")
    builds = []
    original_build = model._build_transition_matrix

    def recording_build(state_sequences):
        builds.append(state_sequences)
        return original_build(state_sequences)

    model._build_transition_matrix = recording_build
    with pytest.warns(UserWarning, match="fail_refit"):
        model.fit(["A > B", "B", "C"], [1, 1, 0])

    removed_a_sequences = builds[1]
    assert [
        converted for _, converted, _ in removed_a_sequences
    ] == [0, 1, 0]


def test_fixed_fail_preserves_non_unit_journey_weights():
    with pytest.warns(UserWarning, match="fail_refit"):
        result = MarkovAttribution(
            removal_strategy="fail"
        ).fit(["A", "B"], [1, 0], weights=[2.0, 8.0])

    assert result["total_conversion_probability"] == pytest.approx(0.2)
    assert result["removal_effects"] == pytest.approx(
        {"A": 1.0, "B": 0.0}
    )
    for diagnostic in result["population_diagnostics"].values():
        assert diagnostic["original_total_weight"] == 10.0
        assert diagnostic["transformed_total_weight"] == 10.0


def test_fixed_fail_zero_conversion_fit_keeps_population_diagnostics():
    with pytest.warns(UserWarning, match="fail_refit"):
        result = MarkovAttribution(
            removal_strategy="fail"
        ).fit(["A", "B"], [0, 0], weights=[2.0, 3.0])

    assert result["removal_effects"] == {"A": 0.0, "B": 0.0}
    assert result["attribution"] is None
    assert result["normalization_reason"] == "zero_raw_magnitude"
    for diagnostic in result["population_diagnostics"].values():
        assert diagnostic["original_journey_count"] == 2
        assert diagnostic["transformed_journey_count"] == 2
        assert diagnostic["original_total_weight"] == 5.0
        assert diagnostic["transformed_total_weight"] == 5.0


# --------------------------------------------------------------------------- #
# redirect_to_null strategy                                                    #
# --------------------------------------------------------------------------- #
def test_redirect_to_null_attribution_sums_to_one():
    paths = ["A > B > C"] * 10 + ["A > C"] * 5 + ["B > C"] * 3
    conversions = [1] * 10 + [0] * 5 + [1] * 3
    model = MarkovAttribution(order=1, removal_strategy="redirect_to_null")
    results = model.fit(paths, conversions)
    assert abs(sum(results["attribution"].values()) - 1.0) < 1e-10
    for effect in results["removal_effects"].values():
        # redirect_to_null effects are non-negative by construction
        # (mass only ever leaves the transient subsystem for NULL).
        assert 0 <= effect <= 1


def test_redirect_to_null_preserves_row_sums():
    """The redirect operation is mass-preserving: column mass moved to NULL
    leaves each row summing to 1. Regression test that the strategy never
    produces a malformed sub-stochastic matrix (which would break the
    (I - Q)^-1 computation silently)."""
    import numpy as np

    from do_attribution.markov import _state_contains_channel

    paths = ["A > B > C", "A > C", "B > C", "A > B", "C"] * 20
    convs = [1, 0, 1, 1, 0] * 20

    for order in (1, 2, 3):
        model = MarkovAttribution(
            order=order,
            removal_strategy="redirect_to_null",
            return_transition_matrix=True,
        )
        res = model.fit(paths, convs)
        tm = res["transition_matrix"].values
        states = res["transition_matrix"].index.tolist()
        null_idx = states.index("NULL")

        for channel in model.channels:
            bad_cols = [
                i for i, s in enumerate(states)
                if _state_contains_channel(s, channel)
            ]
            tm_copy = tm.copy()
            removed = tm_copy[:, bad_cols].sum(axis=1)
            tm_copy[:, bad_cols] = 0.0
            tm_copy[:, null_idx] += removed
            row_sums = tm_copy.sum(axis=1)
            assert np.allclose(row_sums, 1.0, atol=1e-12), (
                f"order={order} channel={channel}: row sums {row_sums}"
            )


def test_redirect_to_null_differs_from_fail_on_anderl_like_data():
    """On data with long paths and skewed channel mix, 'fail' and
    'redirect_to_null' produce materially different attributions. This is
    the methodology gap documented in the README: 'fail' strip-and-refit
    alters the transition matrix structure; 'redirect_to_null' only
    redirects mass on the base matrix."""
    rng = __import__("random").Random(17)
    channels = ["alpha", "beta", "gamma", "delta", "epsilon"]
    paths, convs = [], []
    for _ in range(500):
        n = rng.randint(2, 6)
        path = " > ".join(rng.choice(channels) for _ in range(n))
        paths.append(path)
        convs.append(rng.random() < 0.3)

    fail = MarkovAttribution(order=1, removal_strategy="fail").fit(paths, convs)
    rtn = MarkovAttribution(order=1, removal_strategy="redirect_to_null").fit(
        paths, convs
    )
    # Totals both sum to 1.
    assert abs(sum(fail["attribution"].values()) - 1.0) < 1e-10
    assert abs(sum(rtn["attribution"].values()) - 1.0) < 1e-10
    # They should *not* be identical on this kind of data.
    max_abs = max(
        abs(fail["attribution"][c] - rtn["attribution"][c])
        for c in fail["attribution"]
    )
    assert max_abs > 1e-3


def test_redirect_to_null_single_channel_path():
    """Removing the only channel must drive conversion probability to 0,
    i.e. removal effect = 1.0."""
    paths = ["A", "A", "A"]
    convs = [1, 0, 1]
    model = MarkovAttribution(order=1, removal_strategy="redirect_to_null")
    res = model.fit(paths, convs)
    assert res["removal_effects"]["A"] == pytest.approx(1.0, abs=1e-12)


def test_redirect_to_null_higher_order_runs():
    """Smoke-test higher-order chains under the new strategy."""
    paths = [
        "A > B > C", "A > B > D", "B > C > A", "D > A > C", "A > C > B",
        "B > D > A", "C > A > B", "D > B > A", "A > B > C > D",
        "B > C > D > A",
    ] * 10
    convs = [1, 0, 1, 1, 0, 1, 1, 0, 1, 1] * 10
    for order in (2, 3):
        res = MarkovAttribution(
            order=order, removal_strategy="redirect_to_null"
        ).fit(paths, convs)
        assert abs(sum(res["attribution"].values()) - 1.0) < 1e-10
        for effect in res["removal_effects"].values():
            assert 0 <= effect <= 1


# --------------------------------------------------------------------------- #
# Pandas interop                                                               #
# --------------------------------------------------------------------------- #
def test_accepts_pandas_series_inputs():
    paths = pd.Series(["A > B", "B > C"])
    convs = pd.Series([1, 1])
    model = MarkovAttribution(order=1)
    results = model.fit(paths, convs)
    assert set(results["attribution"].keys()) == {"A", "B", "C"}


def test_accepts_numpy_array_inputs():
    paths = np.array(["A > B", "B > C"])
    convs = np.array([1, 1])
    results = MarkovAttribution(order=1).fit(paths, convs)
    assert set(results["attribution"].keys()) == {"A", "B", "C"}


# --------------------------------------------------------------------------- #
# Base-class API                                                               #
# --------------------------------------------------------------------------- #
def test_get_attribution_requires_fit():
    model = MarkovAttribution()
    with pytest.raises(ValueError):
        model.get_attribution()


def test_get_attribution_after_fit():
    model = MarkovAttribution()
    model.fit(["A > B", "A > C"], [1, 0])
    attr = model.get_attribution()
    assert set(attr.keys()) == {"A", "B", "C"}


# --------------------------------------------------------------------------- #
# P0 — conversion_labels endpoint-contamination regression                    #
# --------------------------------------------------------------------------- #
# MODIFIED (P0): These tests lock down the fix for the silent-endpoint bug
# documented in OPEN_ISSUES.md.  Without ``conversion_labels`` the current
# code produces a non-zero share for the ``"Purchase"`` token; with the
# parameter set, ``"Purchase"`` must not appear anywhere.
def test_conversion_label_absent_from_attribution(
    example_paths, example_conversions
):
    model = MarkovAttribution(order=1, conversion_labels=["Purchase"])
    res = model.fit(example_paths, example_conversions)
    assert set(res["attribution"].keys()) == {"Email", "Social", "Search", "Direct"}
    assert "Purchase" not in res["attribution"]
    assert "Purchase" not in res["removal_effects"]
    assert "Purchase" not in model.get_channels()


def test_without_conversion_label_purchase_leaks_in_on_sample_data(
    example_paths, example_conversions
):
    """Regression: without the P0 parameter the endpoint token leaks in.

    This is the *defect* documented in OPEN_ISSUES.md. It is kept as a
    test so that if someone were to accidentally remove conversion_labels
    support, the test suite would loudly flag the regression (this test
    would still pass, but the companion test above would fail).
    """
    res = MarkovAttribution(order=1).fit(example_paths, example_conversions)
    assert "Purchase" in res["attribution"]
    # Contaminated share is materially non-zero on the sample data.
    assert res["attribution"]["Purchase"] > 0.2


def test_conversion_label_matches_prestripping(
    example_paths, example_paths_no_endpoint, example_conversions
):
    """Acceptance criterion 2: values with ``conversion_labels`` must match
    the values produced by pre-stripping the endpoint from the paths."""
    labeled = MarkovAttribution(
        order=1, conversion_labels=["Purchase"]
    ).fit(example_paths, example_conversions)

    prestripped = MarkovAttribution(order=1).fit(
        example_paths_no_endpoint, example_conversions
    )

    assert set(labeled["attribution"]) == set(prestripped["attribution"])
    for ch in labeled["attribution"]:
        assert labeled["attribution"][ch] == pytest.approx(
            prestripped["attribution"][ch], abs=1e-12
        )
    for ch in labeled["removal_effects"]:
        assert labeled["removal_effects"][ch] == pytest.approx(
            prestripped["removal_effects"][ch], abs=1e-12
        )
    assert labeled["total_conversion_probability"] == pytest.approx(
        prestripped["total_conversion_probability"], abs=1e-12
    )


def test_conversion_label_missing_from_transition_matrix(
    example_paths, example_conversions
):
    model = MarkovAttribution(
        order=1,
        conversion_labels=["Purchase"],
        return_transition_matrix=True,
    )
    res = model.fit(example_paths, example_conversions)
    tm = res["transition_matrix"]
    assert "Purchase" not in tm.index
    assert "Purchase" not in tm.columns


def test_conversion_label_collision_with_special_tokens_rejected():
    with pytest.raises(ValueError, match="SPECIAL_TOKENS"):
        MarkovAttribution(order=1, conversion_labels=["CONVERSION"])


def test_conversion_label_bare_string_rejected():
    # Common caller bug: forgetting to wrap in a list.
    with pytest.raises(TypeError, match="not a bare str"):
        MarkovAttribution(order=1, conversion_labels="Purchase")


def test_conversion_label_default_is_none_backcompat():
    # Without conversion_labels, behaviour is unchanged.
    paths = ["A > B", "B > C", "A > C"]
    convs = [1, 0, 1]
    default = MarkovAttribution(order=1).fit(paths, convs)
    explicit_none = MarkovAttribution(order=1, conversion_labels=None).fit(
        paths, convs
    )
    assert default["attribution"] == explicit_none["attribution"]


def test_conversion_label_higher_order(
    example_paths, example_paths_no_endpoint, example_conversions
):
    for order in (2, 3):
        labeled = MarkovAttribution(
            order=order, conversion_labels=["Purchase"]
        ).fit(example_paths, example_conversions)
        prestripped = MarkovAttribution(order=order).fit(
            example_paths_no_endpoint, example_conversions
        )
        assert "Purchase" not in labeled["attribution"]
        for ch in labeled["attribution"]:
            assert labeled["attribution"][ch] == pytest.approx(
                prestripped["attribution"][ch], abs=1e-12
            )


def test_conversion_label_redirect_to_null_strategy(
    example_paths, example_paths_no_endpoint, example_conversions
):
    labeled = MarkovAttribution(
        order=1,
        conversion_labels=["Purchase"],
        removal_strategy="redirect_to_null",
    ).fit(example_paths, example_conversions)
    prestripped = MarkovAttribution(
        order=1, removal_strategy="redirect_to_null"
    ).fit(example_paths_no_endpoint, example_conversions)
    assert "Purchase" not in labeled["attribution"]
    for ch in labeled["attribution"]:
        assert labeled["attribution"][ch] == pytest.approx(
            prestripped["attribution"][ch], abs=1e-12
        )


def test_conversion_label_detour_strategy(
    example_paths, example_paths_no_endpoint, example_conversions
):
    labeled = MarkovAttribution(
        order=1,
        conversion_labels=["Purchase"],
        removal_strategy="detour",
    ).fit(example_paths, example_conversions)
    prestripped = MarkovAttribution(
        order=1, removal_strategy="detour"
    ).fit(example_paths_no_endpoint, example_conversions)
    assert "Purchase" not in labeled["removal_effects"]
    assert labeled["removal_effects"] == pytest.approx(
        prestripped["removal_effects"], abs=1e-12
    )
    assert labeled["attribution"] == prestripped["attribution"]
    assert (
        labeled["normalization_status"]
        == prestripped["normalization_status"]
    )
    assert (
        labeled["normalization_reason"]
        == prestripped["normalization_reason"]
    )
