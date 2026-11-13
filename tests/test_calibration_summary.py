"""Deterministic static calibration reporting tests for CAL-067."""

from dataclasses import replace

from do_attribution.calibration import AttributionCalibration, LiftTestEvidence


def _calibrator() -> AttributionCalibration:
    return AttributionCalibration(
        evidence=[
            LiftTestEvidence(
                tested_entity="A",
                observed_lift=360.0,
                matched_mta_estimate=400.0,
                test_id="a-geo",
            )
        ]
    )


def test_canonical_summary_has_exact_evidence_allocation_and_total_layers() -> None:
    result = _calibrator().apply_estimates(
        {"A": 400.0, "B": 350.0, "C": 250.0},
        residual_policy="explicit_adjustments",
        residual_adjustments={"B": 25.0, "C": 15.0},
    )

    assert result.summary() == """Calibration Summary
Residual policy: explicit_adjustments

Observed evidence and direct test-supported changes
- A evidence a-geo: calibration ratio = 0.9000.
  Observed lift / matched attribution estimate: 360.00 / 400.00.
  Observed lift ± standard error: 360.00; no uncertainty supplied.
  Original estimate: 400.00
  Direct calibrated estimate: 360.00
  Direct test adjustment: -40.00

Residual allocation
Policy: explicit_adjustments
- Untested entity B received +25.00 through residual policy explicit_adjustments.
  This change was not independently measured by the test evidence for A (a-geo).
- Untested entity C received +15.00 through residual policy explicit_adjustments.
  This change was not independently measured by the test evidence for A (a-geo).

Total behavior
Original total: 1000.00
Raw post-test total: 960.00
Residual required to preserve total: +40.00
Residual allocated: +40.00
Calibrated total: 1000.00

Interpretation boundary
Changes to untested entities are residual-allocation adjustments, not experimentally measured effects.
do-attribution provides calibration mechanics; it does not select a calibration strategy."""


def test_summary_shows_lift_and_reported_standard_error_beside_the_evidence() -> None:
    evidence = LiftTestEvidence(
        tested_entity="A",
        observed_lift=360.0,
        matched_mta_estimate=400.0,
        test_id="a-geo",
        standard_error=25.0,
    )
    result = AttributionCalibration(evidence=[evidence]).apply_estimates(
        {"A": 400.0, "B": 350.0, "C": 250.0},
        residual_policy="explicit_adjustments",
        residual_adjustments={"B": 25.0, "C": 15.0},
    )

    lines = result.summary().splitlines()

    position = lines.index("- A evidence a-geo: calibration ratio = 0.9000.")
    assert lines[position + 1] == (
        "  Observed lift / matched attribution estimate: 360.00 / 400.00."
    )
    assert lines[position + 2] == (
        "  Observed lift ± standard error: 360.00 ± 25.00 (standard error "
        "reported by the experiment; calibration does not use it)."
    )
    assert lines[position + 3] == "  Original estimate: 400.00"
    assert "no uncertainty supplied" not in result.summary()


def test_summary_states_when_no_uncertainty_was_supplied() -> None:
    result = _calibrator().apply_estimates(
        {"A": 400.0, "B": 350.0, "C": 250.0},
        residual_policy="allow_total_change",
    )

    summary = result.summary()

    assert (
        "  Observed lift ± standard error: 360.00; no uncertainty supplied."
    ) in summary.splitlines()
    assert summary.count("no uncertainty supplied") == 1
    assert "±" not in summary.replace("Observed lift ± standard error:", "")


def test_a_reported_standard_error_of_zero_is_shown_not_treated_as_missing() -> None:
    evidence = LiftTestEvidence(
        tested_entity="A",
        observed_lift=360.0,
        matched_mta_estimate=400.0,
        test_id="a-geo",
        standard_error=0.0,
    )
    summary = (
        AttributionCalibration(evidence=[evidence])
        .apply_estimates(
            {"A": 400.0, "B": 600.0}, residual_policy="allow_total_change"
        )
        .summary()
    )

    assert "Observed lift ± standard error: 360.00 ± 0.00 (" in summary
    assert "no uncertainty supplied" not in summary


def test_allow_total_change_reports_no_recipient_and_the_changed_total() -> None:
    result = _calibrator().apply_estimates(
        {"A": 400.0, "B": 350.0, "C": 250.0},
        residual_policy="allow_total_change",
    )

    summary = result.summary()

    assert "Residual policy: allow_total_change" in summary
    assert (
        "No residual recipients; untested entities were unchanged by residual "
        "allocation."
    ) in summary
    assert "Residual required to preserve total: +40.00" in summary
    assert "Residual allocated: +0.00" in summary
    assert "Calibrated total: 960.00" in summary
    assert "received" not in summary


def test_bucket_role_and_warnings_are_reported_from_stored_state() -> None:
    result = _calibrator().apply_estimates(
        {"A": 400.0, "B": 600.0},
        residual_policy="residual_bucket",
        residual_entity="baseline",
    )
    with_warning = replace(
        result,
        warnings=("Practitioner review is required.",),
    )

    summary = with_warning.summary()

    assert (
        "Residual bucket baseline received +40.00 through residual policy "
        "residual_bucket."
    ) in summary
    assert summary.endswith("Warnings\n- Practitioner review is required.")


def test_summary_is_repeatable_and_avoids_prohibited_causal_claims() -> None:
    result = _calibrator().apply_estimates(
        {"C": 250.0, "A": 400.0, "B": 350.0},
        residual_policy="proportional_untested",
        eligible_entities=["C", "B"],
    )

    first = result.summary()
    second = result.summary()

    assert first == second
    assert first.index("Untested entity C") < first.index("Untested entity B")
    lowered = first.lower()
    assert "experimentally identified" not in lowered
    assert "causally identified" not in lowered
    assert "global normalization" not in lowered
    assert "global-normalization" not in lowered
    assert (
        "Changes to untested entities are residual-allocation adjustments, not "
        "experimentally measured effects."
    ) in first
