"""Static explicit residual-adjustment tests for CAL-027."""

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


def test_canonical_explicit_adjustments_preserve_exact_supplied_values() -> None:
    adjustments = {"B": 25.0, "C": 15.0}
    result = _calibrator().apply_estimates(
        {"A": 400.0, "B": 350.0, "C": 250.0},
        residual_policy="explicit_adjustments",
        residual_adjustments=adjustments,
    )
    rows = _rows(result)

    assert rows["A"].calibrated_estimate == 360.0
    assert rows["A"].residual_adjustment == 0.0
    assert rows["B"].residual_adjustment == 25.0
    assert rows["C"].residual_adjustment == 15.0
    assert rows["B"].calibrated_estimate == 375.0
    assert rows["C"].calibrated_estimate == 265.0
    assert rows["B"].role == rows["C"].role == "residual_recipient"
    assert rows["B"].evidence_ids == rows["C"].evidence_ids == ()
    assert all(row.residual_policy == "explicit_adjustments" for row in result.rows)
    assert result.required_residual == result.allocated_residual == 40.0
    assert result.calibrated_total == 1000.0
    assert adjustments == {"B": 25.0, "C": 15.0}


def test_explicit_adjustments_preserve_caller_mapping_order_without_inferred_weights() -> None:
    result = _calibrator().apply_estimates(
        [("C", 250.0), ("A", 400.0), ("B", 350.0)],
        residual_policy="explicit_adjustments",
        residual_adjustments={"C": 5.0, "B": 35.0},
    )
    rows = _rows(result)

    assert [row.entity for row in result.rows] == ["C", "A", "B"]
    assert rows["C"].residual_adjustment == 5.0
    assert rows["B"].residual_adjustment == 35.0
    assert rows["C"].calibrated_estimate == 255.0
    assert rows["B"].calibrated_estimate == 385.0


def test_explicit_adjustments_handle_a_feasible_negative_residual() -> None:
    result = _calibrator(500.0).apply_estimates(
        {"A": 400.0, "B": 350.0, "C": 250.0},
        residual_policy="explicit_adjustments",
        residual_adjustments={"B": -60.0, "C": -40.0},
    )
    rows = _rows(result)

    assert result.required_residual == result.allocated_residual == -100.0
    assert rows["A"].calibrated_estimate == 500.0
    assert rows["B"].calibrated_estimate == 290.0
    assert rows["C"].calibrated_estimate == 210.0
    assert result.calibrated_total == 1000.0


@pytest.mark.parametrize(
    ("adjustments", "message"),
    [
        ({"B": 20.0, "C": 10.0}, "sum to the required residual"),
        ({"B": float("inf"), "C": 0.0}, "finite"),
        ({"B": float("nan"), "C": 0.0}, "finite"),
        ({"missing": 40.0}, "unknown"),
    ],
)
def test_explicit_adjustments_reject_invalid_sums_values_and_recipients(
    adjustments: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _calibrator().apply_estimates(
            {"A": 400.0, "B": 350.0, "C": 250.0},
            residual_policy="explicit_adjustments",
            residual_adjustments=adjustments,
        )


def test_explicit_adjustments_reject_tested_and_infeasible_recipients() -> None:
    with pytest.raises(ValueError, match="Tested entities cannot receive residual"):
        _calibrator().apply_estimates(
            {"A": 400.0, "B": 600.0},
            residual_policy="explicit_adjustments",
            residual_adjustments={"A": 40.0},
        )

    with pytest.raises(ValueError, match="cannot be satisfied simultaneously"):
        _calibrator(1200.0).apply_estimates(
            {"A": 400.0, "B": 100.0, "C": 500.0},
            residual_policy="explicit_adjustments",
            residual_adjustments={"B": -800.0},
        )
