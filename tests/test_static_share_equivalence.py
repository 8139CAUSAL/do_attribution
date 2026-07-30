"""Static shares-plus-explicit-total equivalence tests for CAL-030."""

import pytest

from do_attribution.calibration import AttributionCalibration, LiftTestEvidence


def _calibrator() -> AttributionCalibration:
    return AttributionCalibration(
        evidence=[
            LiftTestEvidence(
                tested_entity="A",
                observed_lift=360.0,
                matched_mta_estimate=400.0,
                test_id="a-geo",
            )
        ]
    )


POLICY_CASES = [
    ("allow_total_change", {}),
    ("proportional_untested", {"eligible_entities": ["B", "C"]}),
    ("custom_weights", {"residual_weights": {"B": 0.70, "C": 0.30}}),
    ("residual_bucket", {"residual_entity": "baseline"}),
    (
        "explicit_adjustments",
        {"residual_adjustments": {"B": 25.0, "C": 15.0}},
    ),
]


@pytest.mark.parametrize(("policy", "policy_inputs"), POLICY_CASES)
def test_share_and_absolute_entry_points_are_exactly_equivalent_for_every_policy(
    policy: str, policy_inputs: dict[str, object]
) -> None:
    calibrator = _calibrator()
    estimates_result = calibrator.apply_estimates(
        {"A": 400.0, "B": 350.0, "C": 250.0},
        residual_policy=policy,
        **policy_inputs,
    )
    shares_result = calibrator.apply_shares(
        {"A": 0.40, "B": 0.35, "C": 0.25},
        current_total=1000.0,
        residual_policy=policy,
        **policy_inputs,
    )

    assert shares_result == estimates_result


def test_share_conversion_preserves_pair_order_and_caller_input() -> None:
    attribution = [["C", 0.25], ["A", 0.40], ["B", 0.35]]
    expected = [["C", 0.25], ["A", 0.40], ["B", 0.35]]

    result = _calibrator().apply_shares(
        attribution,
        current_total=1000.0,
        residual_policy="allow_total_change",
    )

    assert [row.entity for row in result.rows] == ["C", "A", "B"]
    assert attribution == expected


def test_share_entry_point_uses_only_the_explicit_total() -> None:
    result = _calibrator().apply_shares(
        {"A": 0.40, "B": 0.35, "C": 0.25},
        current_total=2000.0,
        residual_policy="allow_total_change",
    )
    rows = {row.entity: row for row in result.rows}

    assert result.original_total == 2000.0
    assert rows["A"].original_estimate == 800.0
    assert rows["B"].original_estimate == 700.0
    assert rows["C"].original_estimate == 500.0
    assert rows["A"].calibrated_estimate == 720.0


@pytest.mark.parametrize(
    ("attribution", "message"),
    [
        ({"A": 0.4, "B": 0.5}, "sum to one"),
        ({"A": 1.1, "B": -0.1}, "nonnegative"),
        ({"A": float("nan"), "B": 1.0}, "finite"),
        ({"A": float("inf"), "B": 0.0}, "finite"),
    ],
)
def test_share_validation_rejects_invalid_vectors(
    attribution, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _calibrator().apply_shares(
            attribution,
            current_total=1000.0,
            residual_policy="allow_total_change",
        )


def test_share_sequence_rejects_duplicate_entities() -> None:
    with pytest.raises(ValueError, match="duplicate key 'A'"):
        _calibrator().apply_shares(
            [("A", 0.4), ("A", 0.6)],
            current_total=1000.0,
            residual_policy="allow_total_change",
        )


def test_current_total_must_be_explicit_finite_and_signed_only_by_opt_in() -> None:
    with pytest.raises(ValueError, match="current_total must be finite"):
        _calibrator().apply_shares(
            {"A": 0.4, "B": 0.6},
            current_total=float("inf"),
            residual_policy="allow_total_change",
        )
    with pytest.raises(ValueError, match="current_total must be nonnegative"):
        _calibrator().apply_shares(
            {"A": 0.4, "B": 0.6},
            current_total=-1000.0,
            residual_policy="allow_total_change",
        )

    result = _calibrator().apply_shares(
        {"A": 0.4, "B": 0.6},
        current_total=-1000.0,
        residual_policy="allow_total_change",
        allow_signed=True,
    )
    assert result.original_total == -1000.0
    assert result.calibrated_total == -960.0
