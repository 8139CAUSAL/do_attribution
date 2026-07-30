"""Policy-neutral static direct-adjustment tests for CAL-020 and CAL-021."""

from math import isclose

import pytest

from do_attribution.calibration import (
    AttributionCalibration,
    LiftTestEvidence,
    _static_direct_state,
)


def _evidence(
    entity: str,
    test_id: str,
    *,
    observed_lift: float,
    matched_estimate: float,
) -> LiftTestEvidence:
    return LiftTestEvidence(
        tested_entity=entity,
        observed_lift=observed_lift,
        matched_mta_estimate=matched_estimate,
        test_id=test_id,
    )


def _direct_state(
    calibrator: AttributionCalibration,
    estimates: object,
    *,
    allow_signed: bool = False,
):
    return _static_direct_state(
        estimates,
        evidence_by_entity=calibrator._evidence_by_entity,
        allow_signed=allow_signed,
        tolerance=1e-9,
    )


def test_canonical_direct_adjustment_changes_only_the_tested_entity() -> None:
    calibrator = AttributionCalibration(
        evidence=[
            _evidence(
                "A",
                "a-geo",
                observed_lift=360.0,
                matched_estimate=400.0,
            )
        ]
    )

    state = _direct_state(calibrator, {"A": 400.0, "B": 350.0, "C": 250.0})
    rows = {row.entity: row for row in state.rows}

    assert [row.entity for row in state.rows] == ["A", "B", "C"]
    assert rows["A"].direct_calibrated_estimate == 360.0
    assert rows["A"].direct_test_adjustment == -40.0
    assert rows["A"].evidence_ids == ("a-geo",)
    assert rows["B"].direct_calibrated_estimate == 350.0
    assert rows["B"].direct_test_adjustment == 0.0
    assert rows["B"].evidence_ids == ()
    assert rows["C"].direct_calibrated_estimate == 250.0
    assert rows["C"].direct_test_adjustment == 0.0
    assert rows["C"].evidence_ids == ()
    assert state.original_total == 1000.0
    assert state.raw_post_test_total == 960.0
    assert state.required_residual == 40.0


def test_each_tested_entity_uses_only_its_own_ratio() -> None:
    calibrator = AttributionCalibration(
        evidence=[
            _evidence("B", "b-test", observed_lift=80.0, matched_estimate=100.0),
            _evidence("A", "a-test", observed_lift=120.0, matched_estimate=100.0),
        ]
    )

    state = _direct_state(calibrator, [("A", 50.0), ("B", 75.0), ("C", 25.0)])
    rows = {row.entity: row for row in state.rows}

    assert isclose(rows["A"].direct_calibrated_estimate, 50.0 * 1.2)
    assert isclose(rows["B"].direct_calibrated_estimate, 75.0 * 0.8)
    assert rows["C"].direct_calibrated_estimate == 25.0
    assert rows["A"].evidence_ids == ("a-test",)
    assert rows["B"].evidence_ids == ("b-test",)
    assert state.required_residual == 5.0


def test_every_tested_entity_must_exist_in_estimates() -> None:
    calibrator = AttributionCalibration(
        evidence=[
            _evidence("A", "a-test", observed_lift=90.0, matched_estimate=100.0),
            _evidence("missing", "missing-test", observed_lift=1.0, matched_estimate=1.0),
        ]
    )

    with pytest.raises(ValueError, match=r"tested entity.*missing=\['missing'\]"):
        _direct_state(calibrator, {"A": 100.0, "B": 50.0})


def test_direct_adjustment_does_not_mutate_estimate_input() -> None:
    calibrator = AttributionCalibration(
        evidence=[
            _evidence("A", "a-test", observed_lift=90.0, matched_estimate=100.0)
        ]
    )
    estimates = [["A", 100.0], ["B", 50.0]]
    expected = [["A", 100.0], ["B", 50.0]]

    _direct_state(calibrator, estimates)

    assert estimates == expected


def test_zero_lift_sets_only_the_tested_entity_to_zero() -> None:
    calibrator = AttributionCalibration(
        evidence=[
            _evidence("A", "a-zero", observed_lift=0.0, matched_estimate=400.0)
        ]
    )

    state = _direct_state(calibrator, {"A": 400.0, "B": 600.0})
    rows = {row.entity: row for row in state.rows}

    assert rows["A"].direct_calibrated_estimate == 0.0
    assert rows["A"].direct_test_adjustment == -400.0
    assert rows["B"].direct_calibrated_estimate == 600.0
    assert rows["B"].direct_test_adjustment == 0.0
    assert state.raw_post_test_total == 600.0
    assert state.required_residual == 400.0


def test_signed_evidence_and_signed_outputs_require_explicit_permission() -> None:
    signed_evidence = AttributionCalibration(
        evidence=[
            _evidence(
                "A",
                "a-negative-lift",
                observed_lift=-25.0,
                matched_estimate=100.0,
            )
        ]
    )
    with pytest.raises(ValueError, match="Signed evidence requires allow_signed=True"):
        _direct_state(signed_evidence, {"A": 100.0})

    signed_evidence_state = _direct_state(
        signed_evidence, {"A": 100.0}, allow_signed=True
    )
    assert signed_evidence_state.rows[0].direct_calibrated_estimate == -25.0
    assert signed_evidence_state.rows[0].direct_test_adjustment == -125.0

    positive_evidence = AttributionCalibration(
        evidence=[
            _evidence("A", "a-half", observed_lift=50.0, matched_estimate=100.0)
        ]
    )
    with pytest.raises(ValueError, match=r"estimates\['A'\].*allow_signed=True"):
        _direct_state(positive_evidence, {"A": -100.0})

    signed_output_state = _direct_state(
        positive_evidence, {"A": -100.0}, allow_signed=True
    )
    assert signed_output_state.rows[0].direct_calibrated_estimate == -50.0
    assert signed_output_state.rows[0].direct_test_adjustment == 50.0


def test_mixed_directions_and_repeated_calls_preserve_all_declared_orders() -> None:
    calibrator = AttributionCalibration(
        evidence=[
            _evidence("B", "b-down", observed_lift=75.0, matched_estimate=100.0),
            _evidence("A", "a-up", observed_lift=125.0, matched_estimate=100.0),
        ]
    )
    estimates = [("C", 40.0), ("A", 80.0), ("B", 120.0)]

    first = _direct_state(calibrator, estimates)
    second = _direct_state(calibrator, estimates)

    assert first == second
    assert [item.test_id for item in calibrator._evidence] == ["b-down", "a-up"]
    assert [row.entity for row in first.rows] == ["C", "A", "B"]
    assert [row.evidence_ids for row in first.rows] == [(), ("a-up",), ("b-down",)]
    assert [row.direct_calibrated_estimate for row in first.rows] == [
        40.0,
        100.0,
        90.0,
    ]
    assert [row.direct_test_adjustment for row in first.rows] == [0.0, 20.0, -30.0]
    assert first.required_residual == 10.0
