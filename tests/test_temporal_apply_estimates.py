"""Public temporal estimates orchestration across ratio policies for CAL-065."""

import pytest

from do_attribution.calibration import (
    LiftTestEvidence,
    TemporalAttributionCalibration,
)


def _evidence(entity="A", test_id="a-test", ratio=0.8, coordinate=None):
    kwargs = {}
    if coordinate is not None:
        kwargs = {"coordinate_name": "index", "coordinate_value": coordinate}
    return LiftTestEvidence(
        tested_entity=entity,
        observed_lift=ratio * 100.0,
        matched_mta_estimate=100.0,
        test_id=test_id,
        **kwargs,
    )


def _estimates():
    return {
        "E1": {"A": 100.0, "B": 100.0},
        "E2": {"A": 100.0, "B": 100.0},
    }


def test_explicit_ratio_public_result_keeps_uncovered_epoch_wholly_unchanged() -> None:
    result = TemporalAttributionCalibration([_evidence()]).apply_estimates(
        _estimates(),
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        ratio_schedule={("A", "E1"): 0.8},
    )

    assert result.epochs == ("E1", "E2")
    assert result.calibrated_totals == {"E1": 180.0, "E2": 200.0}
    assert [(row.entity, row.calibrated_estimate) for row in result.rows["E1"]] == [
        ("A", 80.0),
        ("B", 100.0),
    ]
    assert [row.residual_policy for row in result.rows["E1"]] == [
        "allow_total_change",
        "allow_total_change",
    ]
    assert all(row.residual_policy is None for row in result.rows["E2"])
    assert all(row.role == "untouched" for row in result.rows["E2"])
    assert result.residual_provenance["E2"]["status"] == "not_calibrated"


def test_constant_policy_transports_only_declared_source_and_epochs() -> None:
    result = TemporalAttributionCalibration([_evidence()]).apply_estimates(
        _estimates(),
        ratio_policy="constant",
        epoch_ratio_policy="direct",
        residual_policy="proportional_untested",
        source_evidence={"A": "a-test"},
        tested_entity_epochs={"A": ["E1", "E2"]},
        eligible_entities=["B"],
    )

    assert [row.effective_ratio for row in result.epoch_ratio_schedule] == [0.8, 0.8]
    assert [row.source_evidence_ids for row in result.epoch_ratio_schedule] == [
        ("a-test",),
        ("a-test",),
    ]
    assert result.calibrated_totals == {"E1": 200.0, "E2": 200.0}
    assert [result.rows[epoch][1].calibrated_estimate for epoch in result.epochs] == [
        120.0,
        120.0,
    ]
    summary = result.summary()
    assert (
        "Transported under the practitioner-selected constant policy; this is "
        "a constructed schedule value, not a new observation."
    ) in summary
    assert (
        "transported constant schedule value from the declared source evidence; "
        "this is not observed evidence for the target epoch."
    ) in summary


def test_explicit_strength_public_path_retains_attenuation_provenance() -> None:
    result = TemporalAttributionCalibration([_evidence()]).apply_estimates(
        _estimates(),
        ratio_policy="explicit_strength",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        source_evidence={"A": "a-test"},
        evidence_strength={("A", "E1"): 0.5},
    )

    assert result.epoch_ratio_schedule[0].effective_ratio == 0.9
    assert result.epoch_ratio_schedule[0].ratio_source == "strength_derived"
    assert result.calibrated_totals == {"E1": 190.0, "E2": 200.0}
    provenance = result.epoch_reduction_provenance["by_pair"][("A", "E1")]
    assert provenance["evidence_strength"] == 0.5


def test_observed_only_conflict_is_explicitly_selected_and_auditable() -> None:
    evidence = [
        _evidence(test_id="a-first", ratio=0.8, coordinate=2.0),
        _evidence(test_id="a-second", ratio=1.0, coordinate=3.0),
    ]
    result = TemporalAttributionCalibration(evidence).apply_estimates(
        {"E1": {"A": 100.0, "B": 100.0}},
        ratio_policy="observed_only",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        epoch_windows={"E1": (0.0, 5.0)},
        evidence_conflict_policy="select_evidence",
        selected_evidence={("A", "E1"): "a-first"},
    )

    row = result.epoch_ratio_schedule[0]
    assert row.effective_ratio == 0.8
    assert row.evidence_conflict_policy == "select_evidence"
    assert row.source_evidence_ids == ("a-first",)
    assert tuple(point.evidence_id for point in result.evidence_points) == (
        "a-first",
        "a-second",
    )
    conflict = result.ratio_provenance["conflict_resolution_by_pair"][("A", "E1")]
    assert conflict["resolution"]["candidate_evidence_ids"] == (
        "a-first",
        "a-second",
    )


def test_observed_only_pooled_lift_public_path_uses_pooled_identity() -> None:
    evidence = [
        LiftTestEvidence(
            tested_entity="A",
            observed_lift=80.0,
            matched_mta_estimate=100.0,
            test_id="a-first",
            coordinate_name="index",
            coordinate_value=2.0,
        ),
        LiftTestEvidence(
            tested_entity="A",
            observed_lift=120.0,
            matched_mta_estimate=200.0,
            test_id="a-second",
            coordinate_name="index",
            coordinate_value=3.0,
        ),
    ]
    result = TemporalAttributionCalibration(evidence).apply_estimates(
        {"E1": {"A": 100.0, "B": 100.0}},
        ratio_policy="observed_only",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        epoch_windows={"E1": (0.0, 5.0)},
        evidence_conflict_policy="pooled_lift",
    )
    assert result.epoch_ratio_schedule[0].effective_ratio == pytest.approx(2.0 / 3.0)
    pooled = result.ratio_provenance["conflict_resolution_by_pair"][("A", "E1")]
    assert pooled["resolution"]["pooled_observed_lift"] == 200.0
    assert pooled["resolution"]["pooled_matched_mta_estimate"] == 300.0


def test_representative_coordinate_leaves_outside_support_epoch_unconfigured() -> None:
    calibrator = TemporalAttributionCalibration(
        [
            _evidence(test_id="left", ratio=0.8, coordinate=2.0),
            _evidence(test_id="right", ratio=1.2, coordinate=8.0),
        ]
    )
    result = calibrator.apply_estimates(
        _estimates(),
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="representative_coordinate",
        residual_policy={"E2": "allow_total_change"},
        epoch_windows={"E1": (0.0, 5.0), "E2": (5.0, 10.0)},
        epoch_coordinates={"E1": 1.0, "E2": 7.0},
        tested_entity_epochs={"A": ["E1", "E2"]},
    )

    assert [(row.tested_entity, row.epoch) for row in result.epoch_ratio_schedule] == [
        ("A", "E2")
    ]
    assert result.epoch_ratio_schedule[0].effective_ratio == pytest.approx(
        1.1333333333333333
    )
    assert result.epoch_reduction_provenance["uncalibrated_pairs"] == (
        ("A", "E1"),
    )
    assert all(row.residual_policy is None for row in result.rows["E1"])


def test_coordinate_average_public_path_uses_complete_interval() -> None:
    calibrator = TemporalAttributionCalibration(
        [
            _evidence(test_id="left", ratio=0.8, coordinate=0.0),
            _evidence(test_id="right", ratio=1.2, coordinate=10.0),
        ]
    )
    result = calibrator.apply_estimates(
        {"E1": {"A": 100.0, "B": 100.0}},
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="coordinate_average",
        residual_policy="allow_total_change",
        epoch_windows={"E1": (2.0, 8.0)},
        tested_entity_epochs={"A": ["E1"]},
    )

    assert result.epoch_ratio_schedule[0].effective_ratio == 1.0
    assert result.epoch_ratio_schedule[0].ratio_source == "coordinate_average"
    assert result.rows["E1"][0].role == "tested"
    assert result.rows["E1"][0].direct_test_adjustment == 0.0


def test_support_weighted_public_path_uses_declared_coordinate_weights() -> None:
    calibrator = TemporalAttributionCalibration(
        [
            _evidence(test_id="left", ratio=0.8, coordinate=2.0),
            _evidence(test_id="right", ratio=1.2, coordinate=8.0),
        ]
    )
    result = calibrator.apply_estimates(
        {"E1": {"A": 100.0, "B": 100.0}},
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="support_weighted_average",
        residual_policy="allow_total_change",
        epoch_windows={"E1": (0.0, 10.0)},
        epoch_coordinate_weights={"E1": [(2.0, 1.0), (8.0, 3.0)]},
        tested_entity_epochs={"A": ["E1"]},
    )

    assert result.epoch_ratio_schedule[0].effective_ratio == pytest.approx(1.1)
    assert result.calibrated_totals["E1"] == pytest.approx(210.0)
    provenance = result.epoch_reduction_provenance["by_pair"][("A", "E1")]
    assert provenance["total_coordinate_weight"] == 4.0


def test_final_ratio_bounds_raise_without_clipping_on_public_path() -> None:
    calibrator = TemporalAttributionCalibration([_evidence()])
    with pytest.raises(ValueError, match="outside declared raise-only bounds"):
        calibrator.apply_estimates(
            _estimates(),
            ratio_policy="explicit_ratio",
            epoch_ratio_policy="direct",
            residual_policy="allow_total_change",
            ratio_schedule={("A", "E1"): 0.8},
            ratio_bounds=(0.9, 1.1),
        )


def test_in_bound_ratio_is_retained_exactly_with_bound_metadata() -> None:
    result = TemporalAttributionCalibration([_evidence()]).apply_estimates(
        _estimates(),
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        ratio_schedule={("A", "E1"): 0.8},
        ratio_bounds=(0.8, 1.2),
    )
    row = result.epoch_ratio_schedule[0]
    assert row.effective_ratio == 0.8
    assert row.ratio_bound_lower == 0.8
    assert row.ratio_bound_upper == 1.2


def test_fully_uncovered_grid_requires_no_residual_arguments_or_policy_inference() -> None:
    result = TemporalAttributionCalibration(
        [_evidence(coordinate=20.0)]
    ).apply_estimates(
        _estimates(),
        ratio_policy="observed_only",
        epoch_ratio_policy="direct",
        residual_policy={},
        epoch_windows={"E1": (0.0, 5.0), "E2": (5.0, 10.0)},
    )
    assert result.epoch_ratio_schedule == ()
    assert result.residual_policy == {}
    assert result.calibrated_totals == result.original_totals
    assert all(
        row.residual_policy is None
        for epoch in result.epochs
        for row in result.rows[epoch]
    )


def test_conflict_inputs_are_rejected_for_nonobserved_ratio_policies() -> None:
    with pytest.raises(ValueError, match="does not accept conflict inputs"):
        TemporalAttributionCalibration([_evidence()]).apply_estimates(
            _estimates(),
            ratio_policy="explicit_ratio",
            epoch_ratio_policy="direct",
            residual_policy="allow_total_change",
            ratio_schedule={("A", "E1"): 0.8},
            evidence_conflict_policy="select_evidence",
        )


def test_multiple_tested_entities_apply_independently_before_one_residual() -> None:
    evidence = [_evidence(), _evidence("D", "d-test", 1.2)]
    result = TemporalAttributionCalibration(evidence).apply_estimates(
        {"E1": {"A": 100.0, "D": 50.0, "B": 50.0}},
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy="proportional_untested",
        ratio_schedule={("A", "E1"): 0.8, ("D", "E1"): 1.2},
        eligible_entities=["B"],
    )

    assert [(row.tested_entity, row.effective_ratio) for row in result.epoch_ratio_schedule] == [
        ("A", 0.8),
        ("D", 1.2),
    ]
    assert [(row.entity, row.calibrated_estimate, row.role) for row in result.rows["E1"]] == [
        ("A", 80.0, "tested"),
        ("D", 60.0, "tested"),
        ("B", 60.0, "residual_recipient"),
    ]
    assert result.required_residuals["E1"] == 10.0
    assert result.calibrated_totals["E1"] == 200.0


def test_equivalent_public_calls_are_deterministic_and_do_not_mutate_inputs() -> None:
    estimates = _estimates()
    schedule = {("A", "E1"): 0.8}
    calibrator = TemporalAttributionCalibration([_evidence()])
    first = calibrator.apply_estimates(
        estimates,
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        ratio_schedule=schedule,
    )
    second = calibrator.apply_estimates(
        estimates,
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        ratio_schedule=schedule,
    )

    assert first == second
    assert estimates == _estimates()
    assert schedule == {("A", "E1"): 0.8}
    with pytest.raises(TypeError):
        first.rows["E1"] = ()
