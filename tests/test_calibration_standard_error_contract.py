"""Contract tests for standard-error evidence metadata."""

import pytest

from do_attribution.calibration import AttributionCalibration, LiftTestEvidence


def _calibrate(standard_error: float):
    evidence = LiftTestEvidence(
        tested_entity="A",
        observed_lift=360.0,
        matched_mta_estimate=400.0,
        test_id=f"a-geo-se-{standard_error:g}",
        standard_error=standard_error,
    )
    result = AttributionCalibration(evidence=[evidence]).apply_estimates(
        {"A": 400.0, "B": 350.0, "C": 250.0},
        residual_policy="proportional_untested",
        eligible_entities=["B", "C"],
    )
    return evidence, result


def test_standard_error_is_nonoperative_point_calibration_metadata():
    low_error_evidence, low_error = _calibrate(1.0)
    high_error_evidence, high_error = _calibrate(1000.0)

    assert low_error_evidence.calibration_ratio == pytest.approx(0.9)
    assert high_error_evidence.calibration_ratio == pytest.approx(0.9)
    assert [row.calibrated_estimate for row in low_error.rows] == pytest.approx(
        [row.calibrated_estimate for row in high_error.rows]
    )
    assert [row.direct_test_adjustment for row in low_error.rows] == pytest.approx(
        [row.direct_test_adjustment for row in high_error.rows]
    )
    assert [row.residual_adjustment for row in low_error.rows] == pytest.approx(
        [row.residual_adjustment for row in high_error.rows]
    )
    assert low_error.raw_post_test_total == pytest.approx(
        high_error.raw_post_test_total
    )
    assert low_error.required_residual == pytest.approx(
        high_error.required_residual
    )
    assert low_error.allocated_residual == pytest.approx(
        high_error.allocated_residual
    )
    assert low_error.calibrated_total == pytest.approx(high_error.calibrated_total)
    assert low_error_evidence.standard_error == pytest.approx(1.0)
    assert high_error_evidence.standard_error == pytest.approx(1000.0)
    assert low_error.evidence[0].standard_error == pytest.approx(1.0)
    assert high_error.evidence[0].standard_error == pytest.approx(1000.0)
    for summary in (low_error.summary(), high_error.summary()):
        normalized = summary.lower()
        assert "propagated uncertainty" not in normalized
        assert "confidence interval" not in normalized
