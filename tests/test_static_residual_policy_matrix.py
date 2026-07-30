"""Exact five-policy static arithmetic matrix for CAL-029."""

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


DECREASE_CASES = [
    (
        "allow_total_change",
        {},
        {"A": 360.0, "B": 350.0, "C": 250.0},
        {"A": "tested", "B": "untouched", "C": "untouched"},
        0.0,
        960.0,
    ),
    (
        "proportional_untested",
        {"eligible_entities": ["B", "C"]},
        {"A": 360.0, "B": 373.3333333333333, "C": 266.6666666666667},
        {"A": "tested", "B": "residual_recipient", "C": "residual_recipient"},
        40.0,
        1000.0,
    ),
    (
        "custom_weights",
        {"residual_weights": {"B": 0.70, "C": 0.30}},
        {"A": 360.0, "B": 378.0, "C": 262.0},
        {"A": "tested", "B": "residual_recipient", "C": "residual_recipient"},
        40.0,
        1000.0,
    ),
    (
        "residual_bucket",
        {"residual_entity": "baseline"},
        {"A": 360.0, "B": 350.0, "C": 250.0, "baseline": 40.0},
        {
            "A": "tested",
            "B": "untouched",
            "C": "untouched",
            "baseline": "residual_bucket",
        },
        40.0,
        1000.0,
    ),
    (
        "explicit_adjustments",
        {"residual_adjustments": {"B": 25.0, "C": 15.0}},
        {"A": 360.0, "B": 375.0, "C": 265.0},
        {"A": "tested", "B": "residual_recipient", "C": "residual_recipient"},
        40.0,
        1000.0,
    ),
]


@pytest.mark.parametrize(
    ("policy", "policy_inputs", "expected", "roles", "allocated", "total"),
    DECREASE_CASES,
)
def test_exact_canonical_decrease_matrix(
    policy: str,
    policy_inputs: dict[str, object],
    expected: dict[str, float],
    roles: dict[str, str],
    allocated: float,
    total: float,
) -> None:
    result = _calibrator(360.0).apply_estimates(
        {"A": 400.0, "B": 350.0, "C": 250.0},
        residual_policy=policy,
        **policy_inputs,
    )
    rows = {row.entity: row for row in result.rows}

    assert list(rows) == list(expected)
    for entity, value in expected.items():
        assert isclose(rows[entity].calibrated_estimate, value)
        assert rows[entity].role == roles[entity]
        assert rows[entity].residual_policy == policy
        assert isclose(
            rows[entity].net_adjustment,
            rows[entity].direct_test_adjustment
            + rows[entity].residual_adjustment,
        )
    assert rows["A"].direct_test_adjustment == -40.0
    assert rows["A"].residual_adjustment == 0.0
    assert rows["A"].calibrated_estimate == 360.0
    assert rows["A"].evidence_ids == ("a-geo",)
    assert rows["A"].calibrated_estimate != 375.0
    assert all(row.direct_test_adjustment == 0.0 for row in result.rows if row.entity != "A")
    assert all(row.evidence_ids == () for row in result.rows if row.entity != "A")
    assert result.original_total == 1000.0
    assert result.raw_post_test_total == 960.0
    assert result.required_residual == 40.0
    assert isclose(result.allocated_residual, allocated)
    assert isclose(result.calibrated_total, total)
    assert isclose(sum(row.calibrated_share for row in result.rows), 1.0)


INCREASE_CASES = [
    (
        "allow_total_change",
        {},
        {"A": 500.0, "B": 350.0, "C": 250.0},
        0.0,
        1100.0,
    ),
    (
        "proportional_untested",
        {"eligible_entities": ["B", "C"]},
        {"A": 500.0, "B": 291.6666666666667, "C": 208.33333333333334},
        -100.0,
        1000.0,
    ),
    (
        "custom_weights",
        {"residual_weights": {"B": 0.70, "C": 0.30}},
        {"A": 500.0, "B": 280.0, "C": 220.0},
        -100.0,
        1000.0,
    ),
    (
        "residual_bucket",
        {"residual_entity": "B", "allow_bucket_collision": True},
        {"A": 500.0, "B": 250.0, "C": 250.0},
        -100.0,
        1000.0,
    ),
    (
        "explicit_adjustments",
        {"residual_adjustments": {"B": -60.0, "C": -40.0}},
        {"A": 500.0, "B": 290.0, "C": 210.0},
        -100.0,
        1000.0,
    ),
]


@pytest.mark.parametrize(
    ("policy", "policy_inputs", "expected", "allocated", "total"),
    INCREASE_CASES,
)
def test_all_policies_cover_the_reverse_residual_direction_without_renormalizing_tested(
    policy: str,
    policy_inputs: dict[str, object],
    expected: dict[str, float],
    allocated: float,
    total: float,
) -> None:
    result = _calibrator(500.0).apply_estimates(
        {"A": 400.0, "B": 350.0, "C": 250.0},
        residual_policy=policy,
        **policy_inputs,
    )
    rows = {row.entity: row for row in result.rows}

    assert rows["A"].calibrated_estimate == 500.0
    assert rows["A"].direct_test_adjustment == 100.0
    assert rows["A"].residual_adjustment == 0.0
    assert rows["A"].role == "tested"
    assert rows["A"].calibrated_estimate != 400.0 * (1000.0 / 1100.0)
    for entity, value in expected.items():
        assert isclose(rows[entity].calibrated_estimate, value)
    assert result.original_total == 1000.0
    assert result.raw_post_test_total == 1100.0
    assert result.required_residual == -100.0
    assert isclose(result.allocated_residual, allocated)
    assert isclose(result.calibrated_total, total)
    assert isclose(sum(row.calibrated_share for row in result.rows), 1.0)
