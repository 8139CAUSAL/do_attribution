"""Seeded generated-input static property tests for CAL-085.

Every generated value in every case derives deterministically from
``random.Random(seed)``, and the seed is the pytest parameter, so any failure
is reproducible by rerunning the failing test id shown in the report. The
canonical plan example is retained below as an explicit recorded example.

These tests deliberately do not assert global scaling or universal share
preservation: closed policies preserve the declared total while changing
calibrated shares, and the tested entity is never renormalized after its
ratio is applied.
"""

from math import isclose
from random import Random

import pytest

from do_attribution.calibration import AttributionCalibration, LiftTestEvidence

SEEDS = (1, 7, 23, 101, 419, 1009, 2718, 31415, 65537, 99991, 123456, 987654)
ABS_TOL = 1e-9
REL_TOL = 1e-9


def _close(left: float, right: float) -> bool:
    return isclose(left, right, rel_tol=REL_TOL, abs_tol=ABS_TOL)


def _evidence(entity: str, ratio: float) -> LiftTestEvidence:
    # matched_mta_estimate == 1.0 makes calibration_ratio == observed_lift
    # exactly, so generated ratios survive construction bit for bit.
    return LiftTestEvidence(
        tested_entity=entity,
        observed_lift=ratio,
        matched_mta_estimate=1.0,
        test_id=f"gen-{entity}",
    )


def _generate_case(seed: int, *, max_ratio: float = 2.2) -> dict:
    """Generate tested entities, ratios, estimates, and recipient sets."""
    rng = Random(seed)
    tested = [f"T{i}" for i in range(rng.randint(1, 3))]
    untested = [f"U{i}" for i in range(rng.randint(2, 5))]
    estimates = {
        entity: rng.uniform(0.5, 10_000.0) for entity in tested + untested
    }
    ratios = {}
    for entity in tested:
        draw = rng.random()
        if draw < 0.1:
            ratios[entity] = 0.0
        elif draw < 0.2:
            ratios[entity] = 1.0
        else:
            ratios[entity] = rng.uniform(0.0, max_ratio)
    eligible = sorted(
        rng.sample(untested, rng.randint(2, len(untested)))
        if len(untested) > 2
        else untested
    )
    return {
        "rng": rng,
        "tested": tested,
        "untested": untested,
        "estimates": estimates,
        "ratios": ratios,
        "eligible": eligible,
    }


def _required_residual(case: dict) -> float:
    return sum(
        case["estimates"][entity] * (1.0 - case["ratios"][entity])
        for entity in case["tested"]
    )


def _rescale_for_feasibility(case: dict, limit: float) -> None:
    """Shrink every (ratio - 1) so the required residual stays above -limit."""
    residual = _required_residual(case)
    if residual >= -limit:
        return
    factor = limit / (-residual) * 0.9
    for entity in case["tested"]:
        ratio = case["ratios"][entity]
        case["ratios"][entity] = 1.0 + (ratio - 1.0) * factor


def _calibrator(case: dict) -> AttributionCalibration:
    return AttributionCalibration(
        evidence=[
            _evidence(entity, case["ratios"][entity])
            for entity in case["tested"]
        ]
    )


def _assert_common_invariants(case: dict, result) -> None:
    rows = {row.entity: row for row in result.rows}
    estimates = case["estimates"]

    # Tested-entity fidelity: ratio times original, fixed during residual
    # allocation, labeled tested, never a residual recipient.
    for entity in case["tested"]:
        row = rows[entity]
        expected = case["ratios"][entity] * estimates[entity]
        assert _close(row.calibrated_estimate, expected)
        assert _close(row.direct_test_adjustment, expected - estimates[entity])
        assert row.residual_adjustment == 0.0
        assert row.role == "tested"
    # Untested entities never carry a direct test adjustment (AC-01/AC-11).
    for entity in case["untested"]:
        assert rows[entity].direct_test_adjustment == 0.0
        assert rows[entity].role in ("residual_recipient", "untouched")
    # Residual identity and accounting.
    assert _close(result.original_total, sum(estimates.values()))
    raw = sum(
        case["ratios"][entity] * estimates[entity] for entity in case["tested"]
    ) + sum(estimates[entity] for entity in case["untested"])
    assert _close(result.raw_post_test_total, raw)
    assert _close(
        result.required_residual,
        result.original_total - result.raw_post_test_total,
    )
    assert _close(result.allocated_residual, sum(r.residual_adjustment for r in result.rows))
    # Net adjustments decompose exactly.
    for row in result.rows:
        assert _close(
            row.net_adjustment,
            row.direct_test_adjustment + row.residual_adjustment,
        )
    # Share sums.
    if result.calibrated_total > 0.0:
        assert _close(sum(row.calibrated_share for row in result.rows), 1.0)
    assert _close(sum(row.original_share for row in result.rows), 1.0)


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_proportional_untested_preserves_total_and_proportions(
    seed: int,
) -> None:
    case = _generate_case(seed)
    eligible_mass = sum(case["estimates"][entity] for entity in case["eligible"])
    _rescale_for_feasibility(case, eligible_mass)
    use_all_untested = case["rng"].random() < 0.5
    if use_all_untested:
        recipients = list(case["untested"])
        recipient_mass = sum(case["estimates"][entity] for entity in recipients)
        _rescale_for_feasibility(case, recipient_mass)
        result = _calibrator(case).apply_estimates(
            dict(case["estimates"]),
            residual_policy="proportional_untested",
            all_untested=True,
        )
    else:
        recipients = case["eligible"]
        result = _calibrator(case).apply_estimates(
            dict(case["estimates"]),
            residual_policy="proportional_untested",
            eligible_entities=list(recipients),
        )
    _assert_common_invariants(case, result)
    rows = {row.entity: row for row in result.rows}
    estimates = case["estimates"]

    # Closed policy: the declared total is preserved.
    assert _close(result.calibrated_total, result.original_total)
    assert _close(result.allocated_residual, result.required_residual)
    # Proportional preservation among eligible untested entities
    # (cross-product form avoids dividing by small originals).
    for left in recipients:
        for right in recipients:
            assert isclose(
                rows[left].calibrated_estimate * estimates[right],
                rows[right].calibrated_estimate * estimates[left],
                rel_tol=1e-8,
                abs_tol=1e-6,
            )
    # Ineligible untested entities remain untouched.
    for entity in case["untested"]:
        if entity not in recipients:
            assert rows[entity].calibrated_estimate == estimates[entity]
            assert rows[entity].residual_adjustment == 0.0
            assert rows[entity].role == "untouched"
    # Feasible construction keeps every calibrated estimate nonnegative.
    assert all(row.calibrated_estimate >= 0.0 for row in result.rows)


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_custom_weights_allocate_residual_exactly(seed: int) -> None:
    case = _generate_case(seed)
    rng = case["rng"]
    recipients = case["eligible"]
    draws = [rng.uniform(0.1, 1.0) for _ in recipients]
    total_draw = sum(draws)
    weights = {
        entity: draw / total_draw for entity, draw in zip(recipients, draws)
    }
    limit = min(
        case["estimates"][entity] / weights[entity] for entity in recipients
    )
    _rescale_for_feasibility(case, limit)
    result = _calibrator(case).apply_estimates(
        dict(case["estimates"]),
        residual_policy="custom_weights",
        residual_weights=dict(weights),
    )
    _assert_common_invariants(case, result)
    rows = {row.entity: row for row in result.rows}

    assert _close(result.calibrated_total, result.original_total)
    assert _close(result.allocated_residual, result.required_residual)
    for entity in recipients:
        assert isclose(
            rows[entity].residual_adjustment,
            result.required_residual * weights[entity],
            rel_tol=1e-8,
            abs_tol=1e-6,
        )
    for entity in case["untested"]:
        if entity not in recipients:
            assert rows[entity].calibrated_estimate == case["estimates"][entity]
    assert all(row.calibrated_estimate >= 0.0 for row in result.rows)


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_residual_bucket_receives_entire_residual(seed: int) -> None:
    case = _generate_case(seed, max_ratio=1.0)  # keep the bucket nonnegative
    result = _calibrator(case).apply_estimates(
        dict(case["estimates"]),
        residual_policy="residual_bucket",
        residual_entity="generated-bucket",
    )
    _assert_common_invariants(case, result)
    rows = {row.entity: row for row in result.rows}

    assert _close(result.calibrated_total, result.original_total)
    assert rows["generated-bucket"].role == "residual_bucket"
    assert _close(
        rows["generated-bucket"].residual_adjustment, result.required_residual
    )
    assert rows["generated-bucket"].original_estimate == 0.0
    for entity in case["untested"]:
        assert rows[entity].calibrated_estimate == case["estimates"][entity]
        assert rows[entity].role == "untouched"
    assert all(row.calibrated_estimate >= 0.0 for row in result.rows)


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_explicit_adjustments_apply_exactly(seed: int) -> None:
    case = _generate_case(seed)
    rng = case["rng"]
    recipients = case["eligible"]
    draws = [rng.uniform(0.1, 1.0) for _ in recipients]
    total_draw = sum(draws)
    proportions = [draw / total_draw for draw in draws]
    limit = min(
        case["estimates"][entity] / proportion
        for entity, proportion in zip(recipients, proportions)
    )
    _rescale_for_feasibility(case, limit)
    residual = _required_residual(case)
    adjustments = {
        entity: residual * proportion
        for entity, proportion in zip(recipients[:-1], proportions[:-1])
    }
    # The final adjustment closes the sum exactly to the required residual.
    adjustments[recipients[-1]] = residual - sum(adjustments.values())
    result = _calibrator(case).apply_estimates(
        dict(case["estimates"]),
        residual_policy="explicit_adjustments",
        residual_adjustments=dict(adjustments),
    )
    _assert_common_invariants(case, result)
    rows = {row.entity: row for row in result.rows}

    assert _close(result.calibrated_total, result.original_total)
    for entity, adjustment in adjustments.items():
        assert _close(rows[entity].residual_adjustment, adjustment)
    assert all(row.calibrated_estimate >= 0.0 for row in result.rows)


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_allow_total_change_moves_only_tested_entities(
    seed: int,
) -> None:
    case = _generate_case(seed)
    result = _calibrator(case).apply_estimates(
        dict(case["estimates"]),
        residual_policy="allow_total_change",
    )
    _assert_common_invariants(case, result)
    rows = {row.entity: row for row in result.rows}

    assert result.allocated_residual == 0.0
    assert _close(result.calibrated_total, result.raw_post_test_total)
    for entity in case["untested"]:
        assert rows[entity].calibrated_estimate == case["estimates"][entity]
        assert rows[entity].role == "untouched"
    # Multiple tested ratios stay independent: each tested entity carries
    # exactly its own evidence ID and its own ratio (AC-01).
    for entity in case["tested"]:
        assert rows[entity].evidence_ids == (f"gen-{entity}",)


def test_recorded_canonical_example_remains_exact() -> None:
    """The plan's recorded 10% example: A=360, B=373.33/266.67 or total 960."""
    calibrator = AttributionCalibration(
        evidence=[
            LiftTestEvidence(
                tested_entity="A",
                observed_lift=360.0,
                matched_mta_estimate=400.0,
                test_id="geo-test-2020-12",
            )
        ]
    )
    estimates = {"A": 400.0, "B": 350.0, "C": 250.0}

    proportional = calibrator.apply_estimates(
        estimates,
        residual_policy="proportional_untested",
        eligible_entities=["B", "C"],
    )
    rows = {row.entity: row for row in proportional.rows}
    assert _close(rows["A"].calibrated_estimate, 360.0)
    assert isclose(rows["B"].calibrated_estimate, 1120.0 / 3.0)
    assert isclose(rows["C"].calibrated_estimate, 800.0 / 3.0)
    assert _close(proportional.calibrated_total, 1000.0)

    open_total = calibrator.apply_estimates(
        estimates,
        residual_policy="allow_total_change",
    )
    open_rows = {row.entity: row for row in open_total.rows}
    assert _close(open_rows["A"].calibrated_estimate, 360.0)
    assert open_rows["B"].calibrated_estimate == 350.0
    assert open_rows["C"].calibrated_estimate == 250.0
    assert _close(open_total.calibrated_total, 960.0)


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_zero_eligible_mass_raises(seed: int) -> None:
    case = _generate_case(seed)
    while _required_residual(case) == 0.0:
        for entity in case["tested"]:
            case["ratios"][entity] = 0.5 * (case["ratios"][entity] + 0.5) * 0.9
    estimates = dict(case["estimates"])
    for entity in case["eligible"]:
        estimates[entity] = 0.0
    case["estimates"] = estimates
    with pytest.raises(ValueError, match="zero eligible mass"):
        _calibrator(case).apply_estimates(
            estimates,
            residual_policy="proportional_untested",
            eligible_entities=list(case["eligible"]),
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_negative_residual_beyond_recipient_mass_raises(
    seed: int,
) -> None:
    case = _generate_case(seed)
    eligible_mass = sum(case["estimates"][entity] for entity in case["eligible"])
    tested_mass = sum(case["estimates"][entity] for entity in case["tested"])
    # Force the required residual below -eligible_mass by construction.
    factor = (1.5 * eligible_mass + tested_mass) / tested_mass
    for entity in case["tested"]:
        case["ratios"][entity] = factor
    assert _required_residual(case) < -eligible_mass
    with pytest.raises(ValueError, match="cannot be satisfied simultaneously"):
        _calibrator(case).apply_estimates(
            dict(case["estimates"]),
            residual_policy="proportional_untested",
            eligible_entities=list(case["eligible"]),
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_unnormalized_custom_weights_raise(seed: int) -> None:
    case = _generate_case(seed)
    rng = case["rng"]
    recipients = case["eligible"]
    draws = [rng.uniform(0.1, 1.0) for _ in recipients]
    total_draw = sum(draws)
    weights = {
        entity: draw / total_draw for entity, draw in zip(recipients, draws)
    }
    first = recipients[0]
    weights[first] = weights[first] + rng.uniform(0.01, 0.25)
    with pytest.raises(ValueError):
        _calibrator(case).apply_estimates(
            dict(case["estimates"]),
            residual_policy="custom_weights",
            residual_weights=weights,
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_explicit_adjustments_sum_mismatch_raises(seed: int) -> None:
    case = _generate_case(seed)
    rng = case["rng"]
    while _required_residual(case) == 0.0:
        for entity in case["tested"]:
            case["ratios"][entity] = 0.5 * (case["ratios"][entity] + 0.5) * 0.9
    residual = _required_residual(case)
    recipients = case["eligible"]
    adjustments = {
        entity: residual / len(recipients) for entity in recipients
    }
    first = recipients[0]
    adjustments[first] = adjustments[first] + max(
        1.0, abs(residual)
    ) * rng.uniform(0.05, 0.5)
    with pytest.raises(ValueError, match="sum to the required residual"):
        _calibrator(case).apply_estimates(
            dict(case["estimates"]),
            residual_policy="explicit_adjustments",
            residual_adjustments=adjustments,
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_empty_recipient_set_raises(seed: int) -> None:
    case = _generate_case(seed)
    with pytest.raises(ValueError, match="eligible_entities must not be empty"):
        _calibrator(case).apply_estimates(
            dict(case["estimates"]),
            residual_policy="proportional_untested",
            eligible_entities=[],
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_signed_evidence_requires_explicit_permission(
    seed: int,
) -> None:
    rng = Random(seed)
    ratio = -rng.uniform(0.05, 1.5)
    calibrator = AttributionCalibration(evidence=[_evidence("T0", ratio)])
    estimates = {"T0": rng.uniform(1.0, 500.0), "U0": rng.uniform(1.0, 500.0)}
    with pytest.raises(ValueError, match="allow_signed=True"):
        calibrator.apply_estimates(
            estimates,
            residual_policy="allow_total_change",
        )
    signed = calibrator.apply_estimates(
        estimates,
        residual_policy="allow_total_change",
        allow_signed=True,
    )
    rows = {row.entity: row for row in signed.rows}
    assert _close(rows["T0"].calibrated_estimate, ratio * estimates["T0"])
