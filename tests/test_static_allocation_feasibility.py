"""Cross-policy static feasibility and signed-output tests for CAL-028."""

from math import isclose

import pytest

from do_attribution.calibration import AttributionCalibration, LiftTestEvidence


def _calibrator(observed_lift: float) -> AttributionCalibration:
    return AttributionCalibration(
        evidence=[
            LiftTestEvidence(
                tested_entity="A",
                observed_lift=observed_lift,
                matched_mta_estimate=400.0,
                test_id="a-geo",
            )
        ]
    )


@pytest.mark.parametrize(
    ("policy", "policy_inputs"),
    [
        ("proportional_untested", {"eligible_entities": ["B", "C"]}),
        ("custom_weights", {"residual_weights": {"B": 0.7, "C": 0.3}}),
        ("residual_bucket", {"residual_entity": "baseline"}),
        (
            "explicit_adjustments",
            {"residual_adjustments": {"B": 250.0, "C": 150.0}},
        ),
    ],
)
def test_zero_lift_closed_allocations_succeed_when_recipient_mass_is_sufficient(
    policy: str, policy_inputs: dict[str, object]
) -> None:
    result = _calibrator(0.0).apply_estimates(
        {"A": 400.0, "B": 350.0, "C": 250.0},
        residual_policy=policy,
        **policy_inputs,
    )
    rows = {row.entity: row for row in result.rows}

    assert rows["A"].calibrated_estimate == 0.0
    assert rows["A"].residual_adjustment == 0.0
    assert result.required_residual == 400.0
    assert result.allocated_residual == 400.0
    assert result.calibrated_total == 1000.0


@pytest.mark.parametrize(
    ("policy", "policy_inputs"),
    [
        ("proportional_untested", {"eligible_entities": ["B", "C"]}),
        ("custom_weights", {"residual_weights": {"B": 1.0}}),
        (
            "residual_bucket",
            {"residual_entity": "B", "allow_bucket_collision": True},
        ),
        ("explicit_adjustments", {"residual_adjustments": {"B": -800.0}}),
    ],
)
def test_fixed_tested_value_above_total_rejects_infeasible_nonnegative_allocation(
    policy: str, policy_inputs: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match="cannot be satisfied simultaneously"):
        _calibrator(1200.0).apply_estimates(
            {"A": 400.0, "B": 350.0, "C": 250.0},
            residual_policy=policy,
            **policy_inputs,
        )


def test_missing_and_zero_mass_recipients_raise_actionable_errors() -> None:
    with pytest.raises(ValueError, match="at least one untested residual recipient"):
        _calibrator(360.0).apply_estimates(
            {"A": 400.0},
            residual_policy="proportional_untested",
            all_untested=True,
        )

    with pytest.raises(ValueError, match="cannot be satisfied simultaneously"):
        _calibrator(360.0).apply_estimates(
            {"A": 400.0, "B": 0.0, "C": 0.0},
            residual_policy="proportional_untested",
            eligible_entities=["B", "C"],
        )


def test_allow_signed_permits_feasible_arithmetic_with_negative_outputs() -> None:
    result = _calibrator(1200.0).apply_estimates(
        {"A": 400.0, "B": 350.0, "C": 250.0},
        residual_policy="custom_weights",
        residual_weights={"B": 1.0},
        allow_signed=True,
    )
    rows = {row.entity: row for row in result.rows}

    assert rows["A"].calibrated_estimate == 1200.0
    assert rows["B"].calibrated_estimate == -450.0
    assert rows["C"].calibrated_estimate == 250.0
    assert result.required_residual == result.allocated_residual == -800.0
    assert result.calibrated_total == 1000.0


@pytest.mark.parametrize("invalid", [float("inf"), float("-inf"), float("nan")])
def test_allow_signed_never_bypasses_finite_estimate_validation(invalid: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        _calibrator(360.0).apply_estimates(
            {"A": 400.0, "B": invalid},
            residual_policy="allow_total_change",
            allow_signed=True,
        )


def test_caller_tolerance_controls_closed_residual_sum_validation() -> None:
    result = _calibrator(360.0).apply_estimates(
        {"A": 400.0, "B": 600.0},
        residual_policy="explicit_adjustments",
        residual_adjustments={"B": 40.0000005},
        tolerance=1e-6,
    )
    assert result.allocated_residual == 40.0000005
    assert isclose(result.allocated_residual, result.required_residual, abs_tol=1e-6)
    assert isclose(result.calibrated_total, result.original_total, abs_tol=1e-6)

    with pytest.raises(ValueError, match="sum to the required residual"):
        _calibrator(360.0).apply_estimates(
            {"A": 400.0, "B": 600.0},
            residual_policy="explicit_adjustments",
            residual_adjustments={"B": 40.0000005},
            tolerance=1e-9,
        )
