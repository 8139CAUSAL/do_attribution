"""Static proportional residual-allocation tests for CAL-024."""

from math import isclose

import pytest

from do_attribution.calibration import AttributionCalibration, LiftTestEvidence


def _calibrator(observed_lift: float = 360.0) -> AttributionCalibration:
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


def _rows(result):
    return {row.entity: row for row in result.rows}


def test_canonical_proportional_allocation_matches_exact_plan_values() -> None:
    result = _calibrator().apply_estimates(
        {"A": 400.0, "B": 350.0, "C": 250.0},
        residual_policy="proportional_untested",
        eligible_entities=["B", "C"],
    )
    rows = _rows(result)

    assert rows["A"].calibrated_estimate == 360.0
    assert rows["A"].residual_adjustment == 0.0
    assert rows["A"].role == "tested"
    assert isclose(rows["B"].residual_adjustment, 40.0 * 350.0 / 600.0)
    assert isclose(rows["C"].residual_adjustment, 40.0 * 250.0 / 600.0)
    assert isclose(rows["B"].calibrated_estimate, 373.3333333333333)
    assert isclose(rows["C"].calibrated_estimate, 266.6666666666667)
    assert rows["B"].role == rows["C"].role == "residual_recipient"
    assert all(row.residual_policy == "proportional_untested" for row in result.rows)
    assert result.raw_post_test_total == 960.0
    assert result.required_residual == 40.0
    assert result.allocated_residual == 40.0
    assert result.calibrated_total == 1000.0
    assert isclose(
        rows["B"].calibrated_estimate / rows["C"].calibrated_estimate,
        350.0 / 250.0,
    )


def test_all_untested_is_an_explicit_opt_in_and_uses_estimate_order() -> None:
    result = _calibrator().apply_estimates(
        [("C", 250.0), ("A", 400.0), ("B", 350.0)],
        residual_policy="proportional_untested",
        all_untested=True,
    )

    assert [row.entity for row in result.rows] == ["C", "A", "B"]
    assert [row.role for row in result.rows] == [
        "residual_recipient",
        "tested",
        "residual_recipient",
    ]
    assert result.allocated_residual == result.required_residual


def test_only_selected_untested_entities_receive_the_residual() -> None:
    result = _calibrator().apply_estimates(
        {"A": 400.0, "B": 350.0, "C": 200.0, "D": 50.0},
        residual_policy="proportional_untested",
        eligible_entities=["B", "C"],
    )
    rows = _rows(result)

    assert rows["D"].calibrated_estimate == 50.0
    assert rows["D"].residual_adjustment == 0.0
    assert rows["D"].role == "untouched"
    assert rows["A"].calibrated_estimate == 360.0
    assert result.calibrated_total == result.original_total


def test_tested_entity_increase_allocates_a_feasible_negative_residual() -> None:
    result = _calibrator(500.0).apply_estimates(
        {"A": 400.0, "B": 350.0, "C": 250.0},
        residual_policy="proportional_untested",
        eligible_entities=["B", "C"],
    )
    rows = _rows(result)

    assert result.required_residual == -100.0
    assert result.allocated_residual == -100.0
    assert rows["A"].calibrated_estimate == 500.0
    assert isclose(rows["B"].calibrated_estimate, 350.0 * (500.0 / 600.0))
    assert isclose(rows["C"].calibrated_estimate, 250.0 * (500.0 / 600.0))
    # MODIFIED: exact float equality was over-rigid for standard IEEE-754
    # summation; the plan checks residual and allocated totals within the
    # frozen tolerance (1e-9) per CAL-010/CAL-028, so total preservation is
    # asserted within that tolerance instead of with `==`.
    assert isclose(result.calibrated_total, 1000.0, abs_tol=1e-9)


def test_proportional_rejects_unknown_and_zero_mass_recipients() -> None:
    with pytest.raises(ValueError, match=r"unknown=\['missing'\]"):
        _calibrator().apply_estimates(
            {"A": 400.0, "B": 600.0},
            residual_policy="proportional_untested",
            eligible_entities=["missing"],
        )

    with pytest.raises(ValueError, match="zero eligible mass"):
        _calibrator().apply_estimates(
            {"A": 400.0, "B": 0.0},
            residual_policy="proportional_untested",
            eligible_entities=["B"],
        )


def test_proportional_requires_explicit_recipients() -> None:
    with pytest.raises(ValueError, match="requires exactly one"):
        _calibrator().apply_estimates(
            {"A": 400.0, "B": 600.0},
            residual_policy="proportional_untested",
        )
