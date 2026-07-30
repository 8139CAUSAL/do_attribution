"""Seeded generated-input temporal property tests for CAL-086.

Every generated value in every case derives deterministically from
``random.Random(seed)``, and the seed is the pytest parameter, so any failure
is reproducible by rerunning the failing test id shown in the report.

The generated grids use an explicit numeric coordinate axis. Expected values
are recomputed independently in this module (linear interpolation, exact
piecewise trapezoid integration, weighted averages, pooled fractions) and the
engine's results must match them; epochs outside observed evidence support
must remain wholly uncalibrated rather than filled or extrapolated.
"""

from math import isclose
from random import Random

import pytest

from do_attribution.calibration import (
    LiftTestEvidence,
    TemporalAttributionCalibration,
)

SEEDS = (1, 7, 23, 101, 419, 1009, 2718, 31415, 65537, 99991, 123456, 987654)
REL_TOL = 1e-9
ABS_TOL = 1e-9


def _close(left: float, right: float) -> bool:
    return isclose(left, right, rel_tol=REL_TOL, abs_tol=ABS_TOL)


def _coordinate_evidence(
    entity: str,
    test_id: str,
    coordinate: float,
    ratio: float,
    *,
    matched: float = 1.0,
) -> LiftTestEvidence:
    return LiftTestEvidence(
        tested_entity=entity,
        observed_lift=ratio * matched,
        matched_mta_estimate=matched,
        test_id=test_id,
        outcome_name="conversions",
        population="generated-population",
        unit="units",
        coordinate_name="x",
        coordinate_value=coordinate,
    )


def _interpolate(anchors: list[tuple[float, float]], x: float) -> float:
    """Reference piecewise-linear evaluation used to check the engine."""
    for (left_x, left_r), (right_x, right_r) in zip(anchors, anchors[1:]):
        if left_x <= x <= right_x:
            return left_r + (x - left_x) / (right_x - left_x) * (right_r - left_r)
    raise AssertionError(f"coordinate {x} outside anchor support")


def _average(anchors: list[tuple[float, float]], start: float, end: float) -> float:
    """Reference exact uniform average of the piecewise-linear schedule."""
    total = 0.0
    for (left_x, _), (right_x, _) in zip(anchors, anchors[1:]):
        lo = max(start, left_x)
        hi = min(end, right_x)
        if hi > lo:
            total += (hi - lo) * (
                _interpolate(anchors, lo) + _interpolate(anchors, hi)
            ) / 2.0
    return total / (end - start)


def _grid(rng: Random, *, entities: tuple[str, ...]) -> dict:
    """Generate an authoritative epoch grid and per-epoch estimates."""
    epoch_count = rng.randint(3, 6)
    boundaries = [0.0]
    for _ in range(epoch_count):
        boundaries.append(boundaries[-1] + rng.uniform(5.0, 15.0))
    epochs = tuple(f"E{i}" for i in range(epoch_count))
    windows = {
        epoch: (boundaries[i], boundaries[i + 1])
        for i, epoch in enumerate(epochs)
    }
    estimates = {
        epoch: {entity: rng.uniform(10.0, 5000.0) for entity in entities}
        for epoch in epochs
    }
    return {
        "epochs": epochs,
        "windows": windows,
        "boundaries": boundaries,
        "estimates": estimates,
    }


def _anchors(
    rng: Random,
    *,
    start: float,
    end: float,
    full_cover: bool,
) -> list[tuple[float, float]]:
    count = rng.randint(2, 4)
    if full_cover:
        interior = sorted(
            rng.uniform(start + 1e-3, end - 1e-3) for _ in range(count - 2)
        )
        coordinates = [start, *interior, end]
    else:
        width = end - start
        lo = start + width * rng.uniform(0.15, 0.3)
        hi = end - width * rng.uniform(0.15, 0.3)
        interior = sorted(rng.uniform(lo, hi) for _ in range(count))
        coordinates = interior
    deduped = []
    for coordinate in coordinates:
        if not deduped or coordinate - deduped[-1] > 1e-6:
            deduped.append(coordinate)
    while len(deduped) < 2:
        deduped.append(deduped[-1] + 1.0)
    return [(coordinate, rng.uniform(0.5, 1.6)) for coordinate in deduped]


def _assert_temporal_invariants(case: dict, result) -> None:
    """Epoch preservation, one-ratio uniqueness, per-epoch accounting."""
    assert result.epochs == case["epochs"]
    pairs = [
        (row.tested_entity, row.epoch) for row in result.epoch_ratio_schedule
    ]
    assert len(pairs) == len(set(pairs))
    calibrated_epochs = {epoch for _, epoch in pairs}
    ratio_by_pair = {
        (row.tested_entity, row.epoch): row.effective_ratio
        for row in result.epoch_ratio_schedule
    }
    for epoch in case["epochs"]:
        rows = {row.entity: row for row in result.rows[epoch]}
        original_total = sum(
            case["estimates"][epoch][entity]
            for entity in case["estimates"][epoch]
        )
        assert _close(result.original_totals[epoch], original_total)
        if epoch not in calibrated_epochs:
            for entity, row in rows.items():
                if entity in case["estimates"][epoch]:
                    assert row.calibrated_estimate == case["estimates"][epoch][entity]
                assert row.role == "untouched"
                assert row.residual_policy is None
            continue
        raw = 0.0
        for entity in case["estimates"][epoch]:
            original = case["estimates"][epoch][entity]
            ratio = ratio_by_pair.get((entity, epoch))
            if ratio is None:
                raw += original
            else:
                expected = ratio * original
                assert _close(rows[entity].calibrated_estimate - rows[entity].residual_adjustment, expected)
                assert _close(
                    rows[entity].direct_test_adjustment, expected - original
                )
                assert rows[entity].residual_adjustment == 0.0
                assert rows[entity].role == "tested"
                raw += expected
        assert _close(result.raw_post_test_totals[epoch], raw)
        assert _close(
            result.required_residuals[epoch],
            result.original_totals[epoch] - result.raw_post_test_totals[epoch],
        )
        allocated = sum(row.residual_adjustment for row in result.rows[epoch])
        if all(
            row.residual_policy == "allow_total_change"
            for row in result.rows[epoch]
        ):
            assert allocated == 0.0
            assert _close(
                result.calibrated_totals[epoch],
                result.raw_post_test_totals[epoch],
            )
        for row in result.rows[epoch]:
            assert _close(
                row.net_adjustment,
                row.direct_test_adjustment + row.residual_adjustment,
            )


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_explicit_strength_identity(seed: int) -> None:
    rng = Random(seed)
    case = _grid(rng, entities=("T0", "U0", "U1"))
    ratio = rng.uniform(0.4, 1.8)
    evidence = LiftTestEvidence(
        tested_entity="T0",
        observed_lift=ratio,
        matched_mta_estimate=1.0,
        test_id="strength-source",
    )
    strengths = {}
    for index, epoch in enumerate(case["epochs"]):
        draw = rng.random()
        if index == 0 or draw < 0.15:
            strengths[("T0", epoch)] = 0.0 if index == 0 else 1.0
        else:
            strengths[("T0", epoch)] = rng.uniform(0.0, 1.0)
    result = TemporalAttributionCalibration([evidence]).apply_estimates(
        case["estimates"],
        ratio_policy="explicit_strength",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        source_evidence={"T0": "strength-source"},
        evidence_strength=strengths,
    )
    _assert_temporal_invariants(case, result)
    observed = evidence.calibration_ratio
    by_epoch = {
        row.epoch: row.effective_ratio for row in result.epoch_ratio_schedule
    }
    assert set(by_epoch) == set(case["epochs"])
    for epoch in case["epochs"]:
        strength = strengths[("T0", epoch)]
        assert _close(by_epoch[epoch], 1.0 + strength * (observed - 1.0))
    # Zero strength is the exact identity ratio 1.0.
    assert by_epoch[case["epochs"][0]] == 1.0


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_interpolation_identity_anchor_fidelity_and_no_extrapolation(
    seed: int,
) -> None:
    rng = Random(seed)
    case = _grid(rng, entities=("T0", "U0", "U1", "U2"))
    start, end = case["boundaries"][0], case["boundaries"][-1]
    anchors = _anchors(rng, start=start, end=end, full_cover=False)
    support_lo, support_hi = anchors[0][0], anchors[-1][0]
    evidence = [
        _coordinate_evidence("T0", f"t0-{index}", coordinate, ratio)
        for index, (coordinate, ratio) in enumerate(anchors)
    ]
    coordinates = {}
    anchor_epoch = None
    for epoch in case["epochs"]:
        window_start, window_end = case["windows"][epoch]
        if anchor_epoch is None and window_start <= support_lo < window_end:
            coordinates[epoch] = support_lo  # exact anchor coordinate
            anchor_epoch = epoch
        else:
            coordinates[epoch] = (window_start + window_end) / 2.0
    result = TemporalAttributionCalibration(evidence).apply_estimates(
        case["estimates"],
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="representative_coordinate",
        residual_policy="allow_total_change",
        epoch_windows=case["windows"],
        epoch_coordinates=coordinates,
    )
    _assert_temporal_invariants(case, result)
    by_epoch = {
        row.epoch: row for row in result.epoch_ratio_schedule
    }
    for epoch in case["epochs"]:
        coordinate = coordinates[epoch]
        inside = support_lo <= coordinate <= support_hi
        if not inside:
            # Outside observed support: the pair stays uncalibrated.
            assert epoch not in by_epoch
            for row in result.rows[epoch]:
                assert row.direct_test_adjustment == 0.0
            continue
        expected = _interpolate(anchors, coordinate)
        row = by_epoch[epoch]
        assert _close(row.effective_ratio, expected)
        assert row.inside_observed_support
        # Endpoint containment between the bracketing anchors.
        left = max(x for x, _ in anchors if x <= coordinate)
        right = min(x for x, _ in anchors if x >= coordinate)
        bracket = [r for x, r in anchors if x in (left, right)]
        assert min(bracket) - ABS_TOL <= row.effective_ratio <= max(bracket) + ABS_TOL
    if anchor_epoch is not None:
        # Anchor fidelity: the schedule reproduces the anchor exactly.
        assert by_epoch[anchor_epoch].effective_ratio == anchors[0][1]


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_coordinate_average_identity(seed: int) -> None:
    rng = Random(seed)
    case = _grid(rng, entities=("T0", "U0", "U1"))
    start, end = case["boundaries"][0], case["boundaries"][-1]
    anchors = _anchors(rng, start=start, end=end, full_cover=True)
    evidence = [
        _coordinate_evidence("T0", f"t0-{index}", coordinate, ratio)
        for index, (coordinate, ratio) in enumerate(anchors)
    ]
    selected = sorted(
        rng.sample(case["epochs"], rng.randint(1, len(case["epochs"])))
    )
    result = TemporalAttributionCalibration(evidence).apply_estimates(
        case["estimates"],
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="coordinate_average",
        residual_policy="allow_total_change",
        epoch_windows=case["windows"],
        tested_entity_epochs={"T0": list(selected)},
    )
    _assert_temporal_invariants(case, result)
    by_epoch = {row.epoch: row for row in result.epoch_ratio_schedule}
    assert set(by_epoch) == set(selected)
    for epoch in selected:
        window_start, window_end = case["windows"][epoch]
        expected = _average(anchors, window_start, window_end)
        assert isclose(
            by_epoch[epoch].effective_ratio, expected, rel_tol=1e-9, abs_tol=1e-9
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_support_weighted_identity(seed: int) -> None:
    rng = Random(seed)
    case = _grid(rng, entities=("T0", "U0", "U1"))
    start, end = case["boundaries"][0], case["boundaries"][-1]
    anchors = _anchors(rng, start=start, end=end, full_cover=True)
    evidence = [
        _coordinate_evidence("T0", f"t0-{index}", coordinate, ratio)
        for index, (coordinate, ratio) in enumerate(anchors)
    ]
    selected = sorted(
        rng.sample(case["epochs"], rng.randint(1, len(case["epochs"])))
    )
    coordinate_weights = {}
    for epoch in selected:
        window_start, window_end = case["windows"][epoch]
        pairs = []
        for _ in range(rng.randint(2, 4)):
            coordinate = rng.uniform(window_start, window_end)
            pairs.append((coordinate, rng.uniform(0.1, 5.0)))
        coordinate_weights[epoch] = pairs
    result = TemporalAttributionCalibration(evidence).apply_estimates(
        case["estimates"],
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="support_weighted_average",
        residual_policy="allow_total_change",
        epoch_windows=case["windows"],
        tested_entity_epochs={"T0": list(selected)},
        epoch_coordinate_weights=coordinate_weights,
    )
    _assert_temporal_invariants(case, result)
    by_epoch = {row.epoch: row for row in result.epoch_ratio_schedule}
    assert set(by_epoch) == set(selected)
    for epoch in selected:
        pairs = coordinate_weights[epoch]
        weight_total = sum(weight for _, weight in pairs)
        expected = (
            sum(
                weight * _interpolate(anchors, coordinate)
                for coordinate, weight in pairs
            )
            / weight_total
        )
        assert isclose(
            by_epoch[epoch].effective_ratio, expected, rel_tol=1e-9, abs_tol=1e-9
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_weighted_evidence_identity_on_shared_grid(seed: int) -> None:
    rng = Random(seed)
    case = _grid(rng, entities=("T0", "T1", "U0", "U1"))
    first_epoch = case["epochs"][0]
    second_epoch = case["epochs"][1]
    first_start, first_end = case["windows"][first_epoch]
    second_start, second_end = case["windows"][second_epoch]
    group_size = rng.randint(2, 3)
    group = []
    for index in range(group_size):
        coordinate = rng.uniform(
            first_start + 1e-3, first_end - 1e-3
        )
        group.append(
            _coordinate_evidence(
                "T0", f"t0-conflict-{index}", coordinate, rng.uniform(0.6, 1.5)
            )
        )
    solo_ratio = rng.uniform(0.6, 1.5)
    solo = _coordinate_evidence(
        "T1",
        "t1-solo",
        rng.uniform(second_start + 1e-3, second_end - 1e-3),
        solo_ratio,
    )
    draws = [rng.uniform(0.1, 1.0) for _ in group]
    total_draw = sum(draws)
    weights = {
        item.test_id: draw / total_draw for item, draw in zip(group, draws)
    }
    result = TemporalAttributionCalibration([*group, solo]).apply_estimates(
        case["estimates"],
        ratio_policy="observed_only",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        epoch_windows=case["windows"],
        evidence_conflict_policy={("T0", first_epoch): "weighted_ratio"},
        evidence_weights={("T0", first_epoch): weights},
    )
    _assert_temporal_invariants(case, result)
    by_pair = {
        (row.tested_entity, row.epoch): row
        for row in result.epoch_ratio_schedule
    }
    expected = sum(
        weights[item.test_id] * item.calibration_ratio for item in group
    )
    weighted_row = by_pair[("T0", first_epoch)]
    assert _close(weighted_row.effective_ratio, expected)
    assert sorted(weighted_row.source_evidence_ids) == sorted(
        item.test_id for item in group
    )
    # The second tested entity on the same authoritative grid keeps its own
    # directly observed ratio and its own evidence ID.
    solo_row = by_pair[("T1", second_epoch)]
    assert solo_row.effective_ratio == solo.calibration_ratio
    assert solo_row.source_evidence_ids == ("t1-solo",)


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_pooled_lift_identity(seed: int) -> None:
    rng = Random(seed)
    case = _grid(rng, entities=("T0", "U0", "U1"))
    first_epoch = case["epochs"][0]
    window_start, window_end = case["windows"][first_epoch]
    group = []
    for index in range(rng.randint(2, 3)):
        matched = rng.uniform(50.0, 400.0)
        ratio = rng.uniform(0.6, 1.5)
        group.append(
            _coordinate_evidence(
                "T0",
                f"t0-pooled-{index}",
                rng.uniform(window_start + 1e-3, window_end - 1e-3),
                ratio,
                matched=matched,
            )
        )
    result = TemporalAttributionCalibration(list(group)).apply_estimates(
        case["estimates"],
        ratio_policy="observed_only",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        epoch_windows=case["windows"],
        evidence_conflict_policy={("T0", first_epoch): "pooled_lift"},
    )
    _assert_temporal_invariants(case, result)
    row = next(iter(result.epoch_ratio_schedule))
    pooled_numerator = sum(item.observed_lift for item in group)
    pooled_denominator = sum(item.matched_mta_estimate for item in group)
    assert _close(row.effective_ratio, pooled_numerator / pooled_denominator)
    assert sorted(row.source_evidence_ids) == sorted(
        item.test_id for item in group
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_share_estimate_equivalence(seed: int) -> None:
    rng = Random(seed)
    case = _grid(rng, entities=("T0", "U0", "U1", "U2"))
    start, end = case["boundaries"][0], case["boundaries"][-1]
    anchors = _anchors(rng, start=start, end=end, full_cover=True)
    evidence = [
        _coordinate_evidence("T0", f"t0-{index}", coordinate, ratio)
        for index, (coordinate, ratio) in enumerate(anchors)
    ]
    coordinates = {
        epoch: (case["windows"][epoch][0] + case["windows"][epoch][1]) / 2.0
        for epoch in case["epochs"]
    }
    totals = {
        epoch: sum(case["estimates"][epoch].values())
        for epoch in case["epochs"]
    }
    shares = {
        epoch: {
            entity: value / totals[epoch]
            for entity, value in case["estimates"][epoch].items()
        }
        for epoch in case["epochs"]
    }
    share_estimates = {
        epoch: {
            entity: share * totals[epoch]
            for entity, share in shares[epoch].items()
        }
        for epoch in case["epochs"]
    }
    common = dict(
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="representative_coordinate",
        residual_policy="proportional_untested",
        epoch_windows=case["windows"],
        epoch_coordinates=coordinates,
        eligible_entities=["U0", "U1", "U2"],
    )
    from_estimates = TemporalAttributionCalibration(evidence).apply_estimates(
        share_estimates,
        **common,
    )
    from_shares = TemporalAttributionCalibration(evidence).apply_shares(
        shares,
        totals_by_epoch=totals,
        **common,
    )
    assert from_shares.epochs == from_estimates.epochs
    estimate_ratios = {
        (row.tested_entity, row.epoch): row.effective_ratio
        for row in from_estimates.epoch_ratio_schedule
    }
    share_ratios = {
        (row.tested_entity, row.epoch): row.effective_ratio
        for row in from_shares.epoch_ratio_schedule
    }
    assert share_ratios == estimate_ratios
    for epoch in case["epochs"]:
        left_rows = {row.entity: row for row in from_estimates.rows[epoch]}
        right_rows = {row.entity: row for row in from_shares.rows[epoch]}
        assert set(left_rows) == set(right_rows)
        for entity, left in left_rows.items():
            right = right_rows[entity]
            assert isclose(
                left.calibrated_estimate,
                right.calibrated_estimate,
                rel_tol=1e-9,
                abs_tol=1e-6,
            )
            assert left.role == right.role
        assert isclose(
            from_estimates.calibrated_totals[epoch],
            from_shares.calibrated_totals[epoch],
            rel_tol=1e-9,
            abs_tol=1e-6,
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_partial_support_average_raises(seed: int) -> None:
    rng = Random(seed)
    case = _grid(rng, entities=("T0", "U0", "U1"))
    start, end = case["boundaries"][0], case["boundaries"][-1]
    anchors = _anchors(rng, start=start, end=end, full_cover=False)
    support_lo = anchors[0][0]
    straddling = None
    for epoch in case["epochs"]:
        window_start, window_end = case["windows"][epoch]
        if window_start < support_lo < window_end:
            straddling = epoch
            break
    if straddling is None:
        pytest.skip("generated support boundary landed on a window boundary")
    evidence = [
        _coordinate_evidence("T0", f"t0-{index}", coordinate, ratio)
        for index, (coordinate, ratio) in enumerate(anchors)
    ]
    with pytest.raises(ValueError):
        TemporalAttributionCalibration(evidence).apply_estimates(
            case["estimates"],
            ratio_policy="coordinate_linear",
            epoch_ratio_policy="coordinate_average",
            residual_policy="allow_total_change",
            epoch_windows=case["windows"],
            tested_entity_epochs={"T0": [straddling]},
        )
