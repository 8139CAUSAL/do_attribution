"""Independent multiple-tested-entity static tests for CAL-031."""

from math import isclose

import pytest

from do_attribution.calibration import AttributionCalibration, LiftTestEvidence


def _evidence(
    entity: str,
    test_id: str,
    observed_lift: float,
    matched_estimate: float,
) -> LiftTestEvidence:
    return LiftTestEvidence(
        tested_entity=entity,
        observed_lift=observed_lift,
        matched_mta_estimate=matched_estimate,
        test_id=test_id,
    )


def _mixed_calibrator() -> AttributionCalibration:
    return AttributionCalibration(
        evidence=[
            _evidence("B", "b-down", 50.0, 100.0),
            _evidence("A", "a-up", 120.0, 100.0),
        ]
    )


def test_independent_ratios_are_applied_before_one_combined_residual() -> None:
    calibrator = _mixed_calibrator()
    result = calibrator.apply_estimates(
        [("C", 200.0), ("A", 400.0), ("B", 300.0), ("D", 100.0)],
        residual_policy="proportional_untested",
        eligible_entities=["C", "D"],
    )
    rows = {row.entity: row for row in result.rows}

    assert [item.test_id for item in result.evidence] == ["b-down", "a-up"]
    assert [row.entity for row in result.rows] == ["C", "A", "B", "D"]
    assert rows["A"].calibrated_estimate == 400.0 * 1.2
    assert rows["A"].direct_test_adjustment == 80.0
    assert rows["A"].evidence_ids == ("a-up",)
    assert rows["B"].calibrated_estimate == 300.0 * 0.5
    assert rows["B"].direct_test_adjustment == -150.0
    assert rows["B"].evidence_ids == ("b-down",)
    assert rows["A"].residual_adjustment == rows["B"].residual_adjustment == 0.0
    assert rows["A"].role == rows["B"].role == "tested"
    assert rows["C"].direct_test_adjustment == rows["D"].direct_test_adjustment == 0.0
    assert isclose(rows["C"].residual_adjustment, 70.0 * 2.0 / 3.0)
    assert isclose(rows["D"].residual_adjustment, 70.0 / 3.0)
    assert result.original_total == 1000.0
    assert result.raw_post_test_total == 930.0
    assert result.required_residual == 70.0
    assert result.allocated_residual == 70.0
    assert result.calibrated_total == 1000.0
    assert isclose(
        result.required_residual,
        -(rows["A"].direct_test_adjustment + rows["B"].direct_test_adjustment),
    )


def test_mixed_ratios_are_not_pooled_under_allow_total_change() -> None:
    result = _mixed_calibrator().apply_estimates(
        {"A": 400.0, "B": 300.0, "C": 300.0},
        residual_policy="allow_total_change",
    )
    rows = {row.entity: row for row in result.rows}

    assert rows["A"].calibrated_estimate == 480.0
    assert rows["B"].calibrated_estimate == 150.0
    assert rows["C"].calibrated_estimate == 300.0
    assert rows["A"].calibrated_estimate != 400.0 * 0.85
    assert rows["B"].calibrated_estimate != 300.0 * 0.85
    assert result.calibrated_total == 930.0


def test_multiple_tested_share_and_estimate_calls_are_equivalent_and_deterministic() -> None:
    calibrator = _mixed_calibrator()
    policy_inputs = {"eligible_entities": ["C", "D"]}
    estimate_result = calibrator.apply_estimates(
        [("C", 200.0), ("A", 400.0), ("B", 300.0), ("D", 100.0)],
        residual_policy="proportional_untested",
        **policy_inputs,
    )
    share_result = calibrator.apply_shares(
        [("C", 0.2), ("A", 0.4), ("B", 0.3), ("D", 0.1)],
        current_total=1000.0,
        residual_policy="proportional_untested",
        **policy_inputs,
    )
    repeated = calibrator.apply_estimates(
        [("C", 200.0), ("A", 400.0), ("B", 300.0), ("D", 100.0)],
        residual_policy="proportional_untested",
        **policy_inputs,
    )

    assert share_result == estimate_result
    assert repeated == estimate_result


def test_multiple_fixed_tested_values_can_make_a_closed_total_infeasible() -> None:
    calibrator = AttributionCalibration(
        evidence=[
            _evidence("A", "a-double", 800.0, 400.0),
            _evidence("B", "b-double", 600.0, 300.0),
        ]
    )

    with pytest.raises(ValueError, match="cannot be satisfied simultaneously"):
        calibrator.apply_estimates(
            {"A": 400.0, "B": 300.0, "C": 300.0},
            residual_policy="proportional_untested",
            eligible_entities=["C"],
        )


def test_multiple_static_records_for_one_entity_remain_explicitly_ambiguous() -> None:
    with pytest.raises(ValueError, match="exactly one evidence record.*'A'"):
        AttributionCalibration(
            evidence=[
                _evidence("A", "a-one", 90.0, 100.0),
                _evidence("A", "a-two", 110.0, 100.0),
            ]
        )
