"""Static allow-total-change allocation tests for CAL-023."""

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


def _rows(result):
    return {row.entity: row for row in result.rows}


def test_canonical_allow_total_change_keeps_untested_values_and_returns_960() -> None:
    result = _calibrator(360.0).apply_estimates(
        {"A": 400.0, "B": 350.0, "C": 250.0},
        residual_policy="allow_total_change",
    )
    rows = _rows(result)

    assert [row.entity for row in result.rows] == ["A", "B", "C"]
    assert rows["A"].calibrated_estimate == 360.0
    assert rows["A"].direct_test_adjustment == -40.0
    assert rows["A"].residual_adjustment == 0.0
    assert rows["A"].role == "tested"
    assert rows["A"].evidence_ids == ("a-geo",)
    assert rows["B"].calibrated_estimate == 350.0
    assert rows["C"].calibrated_estimate == 250.0
    assert rows["B"].role == rows["C"].role == "untouched"
    assert all(row.residual_adjustment == 0.0 for row in result.rows)
    assert all(row.residual_policy == "allow_total_change" for row in result.rows)
    assert result.original_total == 1000.0
    assert result.raw_post_test_total == 960.0
    assert result.required_residual == 40.0
    assert result.allocated_residual == 0.0
    assert result.calibrated_total == 960.0
    assert result.residual_policy == "allow_total_change"
    assert [item.test_id for item in result.evidence] == ["a-geo"]
    assert result.warnings == ()
    assert isclose(rows["A"].original_share, 0.4)
    assert isclose(rows["A"].calibrated_share, 0.375)
    assert isclose(sum(row.calibrated_share for row in result.rows), 1.0)


@pytest.mark.parametrize(
    ("observed_lift", "expected_a", "expected_total", "expected_residual"),
    [
        (500.0, 500.0, 1100.0, -100.0),
        (0.0, 0.0, 600.0, 400.0),
    ],
)
def test_allow_total_change_handles_increase_and_zero_lift(
    observed_lift: float,
    expected_a: float,
    expected_total: float,
    expected_residual: float,
) -> None:
    result = _calibrator(observed_lift).apply_estimates(
        [("A", 400.0), ("B", 600.0)],
        residual_policy="allow_total_change",
    )
    rows = _rows(result)

    assert rows["A"].calibrated_estimate == expected_a
    assert rows["B"].calibrated_estimate == 600.0
    assert result.raw_post_test_total == expected_total
    assert result.required_residual == expected_residual
    assert result.allocated_residual == 0.0
    assert result.calibrated_total == expected_total


def test_allow_total_change_rejects_residual_controls_through_public_entry_point() -> None:
    with pytest.raises(ValueError, match="does not accept residual-allocation inputs"):
        _calibrator(360.0).apply_estimates(
            {"A": 400.0, "B": 600.0},
            residual_policy="allow_total_change",
            eligible_entities=["B"],
        )


def test_allow_total_change_does_not_mutate_caller_estimates() -> None:
    estimates = [["A", 400.0], ["B", 600.0]]
    expected = [["A", 400.0], ["B", 600.0]]

    _calibrator(360.0).apply_estimates(
        estimates,
        residual_policy="allow_total_change",
    )

    assert estimates == expected
