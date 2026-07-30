"""Deterministic property-style static calibration gate for CAL-032."""

from dataclasses import FrozenInstanceError
from itertools import product
from math import isclose

import pytest

from do_attribution.calibration import AttributionCalibration, LiftTestEvidence


RATIOS = (0.0, 0.2, 0.55, 0.9, 1.0)


def _calibrator(ratio: float) -> AttributionCalibration:
    return AttributionCalibration(
        evidence=[
            LiftTestEvidence(
                tested_entity="A",
                observed_lift=ratio * 100.0,
                matched_mta_estimate=100.0,
                test_id=f"a-{ratio}",
                metadata={"source": {"ratio": ratio}},
            )
        ]
    )


CLOSED_POLICIES = [
    ("proportional_untested", {"eligible_entities": ["B", "C"]}),
    ("custom_weights", {"residual_weights": {"B": 0.6, "C": 0.4}}),
    ("residual_bucket", {"residual_entity": "baseline"}),
]


@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize(("policy", "policy_inputs"), CLOSED_POLICIES)
def test_closed_policy_invariants_hold_across_ratio_grid(
    ratio: float, policy: str, policy_inputs: dict[str, object]
) -> None:
    result = _calibrator(ratio).apply_estimates(
        {"A": 100.0, "B": 60.0, "C": 40.0},
        residual_policy=policy,
        **policy_inputs,
    )
    rows = {row.entity: row for row in result.rows}

    assert isclose(rows["A"].calibrated_estimate, 100.0 * ratio)
    assert isclose(rows["A"].direct_test_adjustment, 100.0 * ratio - 100.0)
    assert rows["A"].residual_adjustment == 0.0
    assert rows["A"].evidence_ids == (f"a-{ratio}",)
    assert rows["B"].direct_test_adjustment == 0.0
    assert rows["C"].direct_test_adjustment == 0.0
    assert result.original_total == 200.0
    assert isclose(result.raw_post_test_total, 100.0 * ratio + 100.0)
    assert isclose(
        result.required_residual,
        result.original_total - result.raw_post_test_total,
    )
    assert isclose(result.allocated_residual, result.required_residual)
    assert isclose(result.calibrated_total, result.original_total)
    assert isclose(
        sum(row.residual_adjustment for row in result.rows),
        result.allocated_residual,
    )
    assert isclose(sum(row.calibrated_share for row in result.rows), 1.0)
    assert all(row.calibrated_estimate >= 0.0 for row in result.rows)
    assert all(
        isclose(
            row.net_adjustment,
            row.direct_test_adjustment + row.residual_adjustment,
        )
        for row in result.rows
    )


@pytest.mark.parametrize("ratio", RATIOS)
def test_allow_total_change_invariants_hold_across_ratio_grid(ratio: float) -> None:
    result = _calibrator(ratio).apply_estimates(
        {"A": 100.0, "B": 60.0, "C": 40.0},
        residual_policy="allow_total_change",
    )
    rows = {row.entity: row for row in result.rows}

    assert isclose(rows["A"].calibrated_estimate, 100.0 * ratio)
    assert rows["B"].calibrated_estimate == 60.0
    assert rows["C"].calibrated_estimate == 40.0
    assert result.allocated_residual == 0.0
    assert isclose(result.calibrated_total, result.raw_post_test_total)
    assert isclose(sum(row.calibrated_share for row in result.rows), 1.0)


@pytest.mark.parametrize("ratio", RATIOS)
def test_explicit_adjustment_invariants_hold_across_ratio_grid(ratio: float) -> None:
    residual = 100.0 * (1.0 - ratio)
    result = _calibrator(ratio).apply_estimates(
        {"A": 100.0, "B": 60.0, "C": 40.0},
        residual_policy="explicit_adjustments",
        residual_adjustments={"B": residual * 0.6, "C": residual * 0.4},
    )
    rows = {row.entity: row for row in result.rows}

    assert rows["A"].residual_adjustment == 0.0
    assert isclose(rows["B"].residual_adjustment, residual * 0.6)
    assert isclose(rows["C"].residual_adjustment, residual * 0.4)
    assert isclose(result.allocated_residual, result.required_residual)
    assert isclose(result.calibrated_total, 200.0)


@pytest.mark.parametrize(
    ("a_ratio", "b_ratio"),
    tuple(product((0.4, 0.8, 1.1), (0.5, 0.9, 1.2))),
)
def test_multiple_tested_ratios_remain_independent_and_deterministic(
    a_ratio: float, b_ratio: float
) -> None:
    calibrator = AttributionCalibration(
        evidence=[
            LiftTestEvidence(
                tested_entity="B",
                observed_lift=b_ratio * 100.0,
                matched_mta_estimate=100.0,
                test_id="b-test",
            ),
            LiftTestEvidence(
                tested_entity="A",
                observed_lift=a_ratio * 100.0,
                matched_mta_estimate=100.0,
                test_id="a-test",
            ),
        ]
    )
    estimates = [("C", 200.0), ("A", 100.0), ("B", 100.0)]

    first = calibrator.apply_estimates(
        estimates,
        residual_policy="allow_total_change",
    )
    second = calibrator.apply_estimates(
        estimates,
        residual_policy="allow_total_change",
    )
    rows = {row.entity: row for row in first.rows}

    assert first == second
    assert [item.test_id for item in first.evidence] == ["b-test", "a-test"]
    assert [row.entity for row in first.rows] == ["C", "A", "B"]
    assert isclose(rows["A"].calibrated_estimate, 100.0 * a_ratio)
    assert isclose(rows["B"].calibrated_estimate, 100.0 * b_ratio)
    assert rows["A"].evidence_ids == ("a-test",)
    assert rows["B"].evidence_ids == ("b-test",)


def test_caller_state_and_nested_result_state_are_isolated_and_read_only() -> None:
    metadata = {"source": {"ids": ["a-test"]}}
    evidence = LiftTestEvidence(
        tested_entity="A",
        observed_lift=80.0,
        matched_mta_estimate=100.0,
        test_id="a-test",
        metadata=metadata,
    )
    calibrator = AttributionCalibration(evidence=[evidence])
    estimates = [["A", 100.0], ["B", 60.0], ["C", 40.0]]
    recipients = ["B", "C"]

    result = calibrator.apply_estimates(
        estimates,
        residual_policy="proportional_untested",
        eligible_entities=recipients,
    )
    metadata["source"]["ids"].append("mutated")
    estimates.reverse()
    recipients.reverse()

    assert [row.entity for row in result.rows] == ["A", "B", "C"]
    assert result.evidence[0].metadata["source"]["ids"] == ("a-test",)
    assert isinstance(result.rows, tuple)
    assert isinstance(result.evidence, tuple)
    with pytest.raises(FrozenInstanceError):
        result.calibrated_total = -1.0
    with pytest.raises(FrozenInstanceError):
        result.rows[0].calibrated_estimate = -1.0
    with pytest.raises(TypeError):
        result.evidence[0].metadata["new"] = "mutable"
