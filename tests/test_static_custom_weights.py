"""Static custom residual-weight tests for CAL-025."""

from math import isclose

import pytest

from do_attribution.calibration import AttributionCalibration, LiftTestEvidence


def _calibrator(observed_lift: float = 360.0) -> AttributionCalibration:
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


def _rows(result):
    return {row.entity: row for row in result.rows}


def test_canonical_custom_weights_allocate_70_30_without_inference() -> None:
    result = _calibrator().apply_estimates(
        {"A": 400.0, "B": 350.0, "C": 250.0},
        residual_policy="custom_weights",
        residual_weights={"B": 0.70, "C": 0.30},
    )
    rows = _rows(result)

    assert rows["A"].calibrated_estimate == 360.0
    assert rows["A"].residual_adjustment == 0.0
    assert rows["B"].residual_adjustment == 28.0
    assert rows["C"].residual_adjustment == 12.0
    assert rows["B"].calibrated_estimate == 378.0
    assert rows["C"].calibrated_estimate == 262.0
    assert rows["B"].role == rows["C"].role == "residual_recipient"
    assert rows["B"].evidence_ids == rows["C"].evidence_ids == ()
    assert rows["A"].evidence_ids == ("a-geo",)
    assert all(row.residual_policy == "custom_weights" for row in result.rows)
    assert result.required_residual == result.allocated_residual == 40.0
    assert result.calibrated_total == 1000.0
    assert rows["B"].residual_adjustment / result.required_residual == 0.70
    assert rows["C"].residual_adjustment / result.required_residual == 0.30


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ({"B": 0.5, "C": 0.4}, "sum to one"),
        ({"B": -0.1, "C": 1.1}, "nonnegative"),
        ({"B": float("inf"), "C": 0.0}, "finite"),
        ({"B": float("nan"), "C": 1.0}, "finite"),
        ({"missing": 1.0}, "unknown"),
    ],
)
def test_custom_weights_reject_invalid_mass_values_and_recipients(
    weights: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _calibrator().apply_estimates(
            {"A": 400.0, "B": 350.0, "C": 250.0},
            residual_policy="custom_weights",
            residual_weights=weights,
        )


def test_custom_weights_use_caller_tolerance_only_for_unit_mass_validation() -> None:
    result = _calibrator().apply_estimates(
        {"A": 400.0, "B": 350.0, "C": 250.0},
        residual_policy="custom_weights",
        residual_weights={"B": 0.50000025, "C": 0.5},
        tolerance=1e-6,
    )
    rows = _rows(result)

    assert isclose(rows["B"].residual_adjustment, 40.0 * 0.50000025)
    assert isclose(
        rows["B"].residual_adjustment + rows["C"].residual_adjustment,
        40.0,
    )
    assert result.calibrated_total == 1000.0


def test_custom_weights_reject_tested_recipient_before_allocation() -> None:
    with pytest.raises(ValueError, match="Tested entities cannot receive residual"):
        _calibrator().apply_estimates(
            {"A": 400.0, "B": 600.0},
            residual_policy="custom_weights",
            residual_weights={"A": 1.0},
        )


def test_custom_weights_reject_infeasible_negative_output() -> None:
    with pytest.raises(ValueError, match="cannot be satisfied simultaneously"):
        _calibrator(1200.0).apply_estimates(
            {"A": 400.0, "B": 100.0, "C": 500.0},
            residual_policy="custom_weights",
            residual_weights={"B": 1.0},
        )
