"""Temporal share/estimate equivalence across policies for CAL-065."""

from collections.abc import Mapping

import pytest

from do_attribution.calibration import (
    LiftTestEvidence,
    TemporalAttributionCalibration,
)


def _evidence(test_id="a-test", ratio=0.8, coordinate=None):
    kwargs = {}
    if coordinate is not None:
        kwargs = {"coordinate_name": "index", "coordinate_value": coordinate}
    return LiftTestEvidence(
        tested_entity="A",
        observed_lift=ratio * 100.0,
        matched_mta_estimate=100.0,
        test_id=test_id,
        **kwargs,
    )


SHARES = {
    "E1": {"A": 0.5, "B": 0.3, "C": 0.2},
    "E2": {"A": 0.5, "B": 0.4, "C": 0.1},
}
TOTALS = {"E1": 200.0, "E2": 200.0}


def _estimates(shares=SHARES, totals=TOTALS):
    return {
        epoch: {
            entity: share * totals[epoch]
            for entity, share in (
                epoch_shares.items()
                if isinstance(epoch_shares, Mapping)
                else epoch_shares
            )
        }
        for epoch, epoch_shares in shares.items()
    }


def _assert_equivalent(evidence, *, shares=SHARES, totals=TOTALS, **kwargs):
    calibrator = TemporalAttributionCalibration(evidence)
    from_shares = calibrator.apply_shares(
        shares,
        totals_by_epoch=totals,
        **kwargs,
    )
    from_estimates = calibrator.apply_estimates(
        _estimates(shares, totals),
        **kwargs,
    )
    assert from_shares == from_estimates
    return from_shares


@pytest.mark.parametrize(
    "residual_policy, controls",
    [
        ("allow_total_change", {}),
        ("proportional_untested", {"eligible_entities": ["B", "C"]}),
        ("custom_weights", {"residual_weights": {"B": 0.25, "C": 0.75}}),
        ("residual_bucket", {"residual_entity": "baseline"}),
        (
            "explicit_adjustments",
            {
                "residual_adjustments_by_epoch": {
                    "E1": {"B": 5.0, "C": 15.0},
                    "E2": {"B": 12.0, "C": 8.0},
                }
            },
        ),
    ],
)
def test_share_equivalence_for_all_five_residual_policies(
    residual_policy, controls
) -> None:
    result = _assert_equivalent(
        [_evidence()],
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy=residual_policy,
        ratio_schedule=[(("A", "E2"), 0.8), (("A", "E1"), 0.8)],
        ratio_bounds=(0.5, 1.5),
        tolerance=1e-8,
        **controls,
    )
    assert result.epochs == ("E1", "E2")


def test_share_equivalence_for_constant_ratio_policy() -> None:
    _assert_equivalent(
        [_evidence()],
        ratio_policy="constant",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        source_evidence={"A": "a-test"},
        tested_entity_epochs={"A": ["E1", "E2"]},
    )


def test_share_equivalence_for_explicit_strength_policy() -> None:
    _assert_equivalent(
        [_evidence()],
        ratio_policy="explicit_strength",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        source_evidence={"A": "a-test"},
        evidence_strength={("A", "E1"): 0.5, ("A", "E2"): 1.0},
    )


def test_share_equivalence_for_observed_conflict_weighting() -> None:
    evidence = [
        _evidence("first", 0.8, 2.0),
        _evidence("second", 1.0, 3.0),
    ]
    _assert_equivalent(
        evidence,
        shares={"E1": SHARES["E1"]},
        totals={"E1": 200.0},
        ratio_policy="observed_only",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        epoch_windows={"E1": (0.0, 5.0)},
        evidence_conflict_policy="weighted_ratio",
        evidence_weights={("A", "E1"): {"first": 0.25, "second": 0.75}},
    )


@pytest.mark.parametrize(
    "epoch_ratio_policy, metadata",
    [
        (
            "representative_coordinate",
            {"epoch_coordinates": {"E1": 5.0}},
        ),
        ("coordinate_average", {}),
        (
            "support_weighted_average",
            {"epoch_coordinate_weights": {"E1": [(2.0, 1.0), (8.0, 3.0)]}},
        ),
    ],
)
def test_share_equivalence_for_all_coordinate_epoch_reductions(
    epoch_ratio_policy, metadata
) -> None:
    _assert_equivalent(
        [_evidence("left", 0.8, 0.0), _evidence("right", 1.2, 10.0)],
        shares={"E1": SHARES["E1"]},
        totals={"E1": 200.0},
        ratio_policy="coordinate_linear",
        epoch_ratio_policy=epoch_ratio_policy,
        residual_policy="allow_total_change",
        epoch_windows={"E1": (2.0, 8.0)},
        tested_entity_epochs={"A": ["E1"]},
        **metadata,
    )


def test_share_conversion_preserves_epoch_and_entity_order() -> None:
    shares = {
        "late": [("C", 0.2), ("A", 0.5), ("B", 0.3)],
        "early": [("B", 0.4), ("A", 0.5), ("C", 0.1)],
    }
    totals = {"late": 200.0, "early": 200.0}
    result = _assert_equivalent(
        [_evidence()],
        shares=shares,
        totals=totals,
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        ratio_schedule={("A", "late"): 0.8},
    )
    assert result.epochs == ("late", "early")
    assert [row.entity for row in result.rows["late"]] == ["C", "A", "B"]


@pytest.mark.parametrize(
    "shares, totals, expected",
    [
        (
            {"E1": {"A": 0.5, "B": 0.4}},
            {"E1": 200.0},
            "sum to one",
        ),
        (
            SHARES,
            {"E1": 200.0},
            "missing=.*E2",
        ),
        (
            SHARES,
            {"E1": 200.0, "E2": 200.0, "E3": 1.0},
            "extra=.*E3",
        ),
    ],
)
def test_invalid_temporal_shares_and_total_coverage_raise(shares, totals, expected) -> None:
    with pytest.raises(ValueError, match=expected):
        TemporalAttributionCalibration([_evidence()]).apply_shares(
            shares,
            totals_by_epoch=totals,
            ratio_policy="explicit_ratio",
            epoch_ratio_policy="direct",
            residual_policy="allow_total_change",
            ratio_schedule={("A", next(iter(shares))): 0.8},
        )


def test_share_inputs_totals_and_policy_controls_are_not_mutated() -> None:
    shares = {epoch: dict(values) for epoch, values in SHARES.items()}
    totals = dict(TOTALS)
    weights = {"B": 0.25, "C": 0.75}
    _assert_equivalent(
        [_evidence()],
        shares=shares,
        totals=totals,
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy="custom_weights",
        ratio_schedule={("A", "E1"): 0.8, ("A", "E2"): 0.8},
        residual_weights=weights,
    )
    assert shares == SHARES
    assert totals == TOTALS
    assert weights == {"B": 0.25, "C": 0.75}
