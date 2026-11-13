"""Contract tests for standard-error evidence metadata."""

from dataclasses import replace

import pandas as pd
import pytest

from do_attribution.calibration import (
    AttributionCalibration,
    LiftTestEvidence,
    TemporalAttributionCalibration,
)


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


# --------------------------------------------------------------------------- #
# The contract in the other direction: the standard error is displayed beside  #
# the evidence it belongs to, and changing it changes those annotations only.  #
# --------------------------------------------------------------------------- #
STATIC_ANNOTATION = "  Observed lift ± standard error: "
TEMPORAL_ANNOTATION = "  observed lift ± standard error = "
REPORTED = "(standard error reported by the experiment; calibration does not use it)"


def _static_result(standard_error):
    evidence = LiftTestEvidence(
        tested_entity="A",
        observed_lift=360.0,
        matched_mta_estimate=400.0,
        test_id="a-geo",
        standard_error=standard_error,
    )
    return AttributionCalibration(evidence=[evidence]).apply_estimates(
        {"A": 400.0, "B": 350.0, "C": 250.0},
        residual_policy="proportional_untested",
        eligible_entities=["B", "C"],
    )


def _temporal_result(standard_errors):
    evidence = [
        LiftTestEvidence(
            tested_entity="A",
            observed_lift=100.0 * ratio,
            matched_mta_estimate=100.0,
            test_id=evidence_id,
            coordinate_name="time",
            coordinate_value=coordinate,
            standard_error=standard_error,
        )
        for evidence_id, ratio, coordinate, standard_error in (
            ("a-left", 0.8, 2.0, standard_errors[0]),
            ("a-right", 1.0, 8.0, standard_errors[1]),
        )
    ]
    return TemporalAttributionCalibration(evidence).apply_estimates(
        {
            "E1": {"A": 100.0, "B": 100.0},
            "E2": {"A": 100.0, "B": 100.0},
        },
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="representative_coordinate",
        residual_policy={"E1": "proportional_untested"},
        epoch_windows={"E1": (0.0, 10.0), "E2": (10.0, 20.0)},
        epoch_coordinates={"E1": 5.0, "E2": 15.0},
        tested_entity_epochs={"A": ["E1", "E2"]},
        eligible_entities=["B"],
        ratio_bounds=(0.5, 1.5),
    )


def _differing_lines(first: str, second: str) -> list[tuple[str, str]]:
    first_lines = first.splitlines()
    second_lines = second.splitlines()
    assert len(first_lines) == len(second_lines)
    return [
        (left, right)
        for left, right in zip(first_lines, second_lines)
        if left != right
    ]


def test_static_standard_error_changes_only_the_evidence_annotation():
    missing = _static_result(None)
    low = _static_result(1.0)
    high = _static_result(1000.0)

    for other in (low, high):
        assert other.rows == missing.rows
        assert other.original_total == missing.original_total
        assert other.raw_post_test_total == missing.raw_post_test_total
        assert other.required_residual == missing.required_residual
        assert other.allocated_residual == missing.allocated_residual
        assert other.calibrated_total == missing.calibrated_total
        assert other.residual_policy == missing.residual_policy
        assert other.warnings == missing.warnings
        pd.testing.assert_frame_equal(other.to_dataframe(), missing.to_dataframe())
        assert [
            replace(record, standard_error=None) for record in other.evidence
        ] == list(missing.evidence)
        assert other.evidence[0].calibration_ratio == missing.evidence[0].calibration_ratio

    assert _differing_lines(low.summary(), high.summary()) == [
        (
            f"{STATIC_ANNOTATION}360.00 ± 1.00 {REPORTED}.",
            f"{STATIC_ANNOTATION}360.00 ± 1000.00 {REPORTED}.",
        )
    ]
    assert _differing_lines(missing.summary(), low.summary()) == [
        (
            f"{STATIC_ANNOTATION}360.00; no uncertainty supplied.",
            f"{STATIC_ANNOTATION}360.00 ± 1.00 {REPORTED}.",
        )
    ]


def test_temporal_standard_error_changes_only_the_evidence_annotations():
    missing = _temporal_result((None, None))
    low = _temporal_result((1.0, 2.0))
    high = _temporal_result((1000.0, 2000.0))
    mixed = _temporal_result((5.0, None))

    numeric_state = (
        "epochs",
        "epoch_windows",
        "epoch_coordinates",
        "epoch_support",
        "evidence_points",
        "coordinate_ratio_schedule",
        "epoch_ratio_schedule",
        "rows",
        "ratio_provenance",
        "epoch_reduction_provenance",
        "residual_provenance",
        "original_totals",
        "raw_post_test_totals",
        "required_residuals",
        "calibrated_totals",
        "ratio_policy",
        "epoch_ratio_policy",
        "residual_policy",
        "warnings",
    )
    for other in (low, high, mixed):
        for name in numeric_state:
            assert getattr(other, name) == getattr(missing, name), name
        assert [
            replace(record, standard_error=None) for record in other.evidence
        ] == list(missing.evidence)

    assert _differing_lines(low.summary(), high.summary()) == [
        (
            f"{TEMPORAL_ANNOTATION}80.00 ± 1.00 {REPORTED}",
            f"{TEMPORAL_ANNOTATION}80.00 ± 1000.00 {REPORTED}",
        ),
        (
            f"{TEMPORAL_ANNOTATION}100.00 ± 2.00 {REPORTED}",
            f"{TEMPORAL_ANNOTATION}100.00 ± 2000.00 {REPORTED}",
        ),
    ]
    assert _differing_lines(missing.summary(), mixed.summary()) == [
        (
            f"{TEMPORAL_ANNOTATION}80.00; no uncertainty supplied",
            f"{TEMPORAL_ANNOTATION}80.00 ± 5.00 {REPORTED}",
        )
    ]
    assert (
        f"{TEMPORAL_ANNOTATION}100.00; no uncertainty supplied"
        in mixed.summary().splitlines()
    )


def test_every_evidence_record_in_a_summary_carries_one_uncertainty_annotation():
    static_summary = _static_result(None).summary()
    temporal_summary = _temporal_result((5.0, None)).summary()

    assert static_summary.count(STATIC_ANNOTATION) == 1
    assert temporal_summary.count(TEMPORAL_ANNOTATION) == 2
    for summary in (static_summary, temporal_summary, _static_result(3.0).summary()):
        normalized = summary.lower()
        assert "propagated uncertainty" not in normalized
        assert "confidence interval" not in normalized
