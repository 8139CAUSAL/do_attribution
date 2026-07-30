"""Deterministic temporal calibration audit-summary tests for CAL-072."""

from dataclasses import replace

import pytest

from do_attribution.calibration import (
    LiftTestEvidence,
    TemporalAttributionCalibration,
)


def _evidence(evidence_id: str, ratio: float, coordinate: float):
    return LiftTestEvidence(
        tested_entity="A",
        observed_lift=100.0 * ratio,
        matched_mta_estimate=100.0,
        test_id=evidence_id,
        coordinate_name="time",
        coordinate_value=coordinate,
    )


def _representative_result():
    return TemporalAttributionCalibration(
        [
            _evidence("a-left", 0.8, 2.0),
            _evidence("a-right", 1.0, 8.0),
        ]
    ).apply_estimates(
        {
            "E1": {"A": 100.0, "B": 100.0},
            "E2": {"A": 100.0, "B": 100.0},
        },
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="representative_coordinate",
        residual_policy={"E1": "proportional_untested"},
        epoch_windows={"E1": (0.0, 10.0), "E2": (10.0, 20.0)},
        epoch_coordinates={"E1": 5.0, "E2": 15.0},
        tested_entity_epochs={"A": ["E1", "E2"]},
        eligible_entities=["B"],
        ratio_bounds=(0.5, 1.5),
    )


def test_summary_exposes_every_temporal_audit_layer_and_uncovered_epoch() -> None:
    summary = _representative_result().summary()

    ordered_sections = [
        "Authoritative temporal design",
        "Observed evidence",
        "Coordinate schedule construction",
        "Final epoch-level ratios",
        "Direct test-supported changes",
        "Residual-allocation changes",
        "Total behavior",
        "Interpretation boundary",
    ]
    assert [summary.index(section) for section in ordered_sections] == sorted(
        summary.index(section) for section in ordered_sections
    )
    assert "Epoch E1 was supplied by the attribution result." in summary
    assert "Boundary source: outcome-signal epoch design." in summary
    assert "Calibration did not modify the epoch." in summary
    assert "A evidence a-left:" in summary
    assert "coordinate time = 2.0" in summary
    assert "observed calibration ratio = 0.8000" in summary
    assert "A interpolated schedule point at time = 5.0: ratio 0.9000." in summary
    assert "left anchor a-left; right anchor a-right." in summary
    assert "constructed schedule value, not a new observation." in summary
    assert (
        "A coordinate-linear schedule at time = 15.0: unsupported; no ratio "
        "was inferred and no extrapolation was performed."
    ) in summary
    assert "A / E1 effective ratio 0.9000:" in summary
    assert "schedule evaluated at representative coordinate 5.0." in summary
    assert "validated without clipping inside ratio bounds [0.5000, 1.5000]." in summary
    assert (
        "A / E2 was not calibrated; no effective ratio was supplied or inferred."
    ) in summary
    assert "A / E1 was adjusted from 100.00 to 90.00." in summary
    assert "Direct test adjustment: -10.00." in summary
    assert (
        "B / E1 received +10.00 through residual policy proportional_untested."
    ) in summary
    assert (
        "This change was not independently measured by the test evidence for "
        "A (a-left, a-right)."
    ) in summary
    assert "Epoch E2 was not calibrated; no residual policy was inferred." in summary
    assert "Residual required to preserve total: +10.00" in summary
    assert "Residual allocated: +10.00" in summary
    assert "Calibrated total: 200.00" in summary


def _conflict_summary(policy: str) -> str:
    evidence = [
        _evidence("a-low", 0.8, 2.0),
        _evidence("a-high", 1.2, 8.0),
    ]
    extra = {}
    if policy == "select_evidence":
        extra["selected_evidence"] = {("A", "E1"): "a-high"}
    elif policy == "weighted_ratio":
        extra["evidence_weights"] = {
            ("A", "E1"): {"a-low": 0.25, "a-high": 0.75}
        }
    result = TemporalAttributionCalibration(evidence).apply_estimates(
        {"E1": {"A": 100.0, "B": 100.0}},
        ratio_policy="observed_only",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        epoch_windows={"E1": (0.0, 10.0)},
        evidence_conflict_policy=policy,
        **extra,
    )
    return result.summary()


def test_conflict_summaries_report_selection_weights_and_pooled_components() -> None:
    selected = _conflict_summary("select_evidence")
    weighted = _conflict_summary("weighted_ratio")
    pooled = _conflict_summary("pooled_lift")

    assert "evidence conflict policy: select_evidence." in selected
    assert "selected evidence a-high from candidates a-low, a-high." in selected
    assert "evidence conflict policy: weighted_ratio." in weighted
    assert (
        "Evidence weights were supplied by the practitioner: a-low=0.2500, "
        "a-high=0.7500."
    ) in weighted
    assert "evidence conflict policy: pooled_lift." in pooled
    assert (
        "pooled observed lift / pooled matched attribution estimate: "
        "200.00 / 200.00."
    ) in pooled


@pytest.mark.parametrize(
    ("epoch_policy", "extra", "required_phrase"),
    [
        (
            "coordinate_average",
            {},
            "uniform average of the coordinate-linear schedule over the full "
            "authoritative epoch.",
        ),
        (
            "support_weighted_average",
            {"epoch_coordinate_weights": {"E1": [(3.0, 1.0), (7.0, 3.0)]}},
            "support-weighted average of the coordinate-linear schedule using "
            "practitioner-declared coordinate weights.",
        ),
    ],
)
def test_average_reduction_summaries_state_the_applied_rule(
    epoch_policy: str,
    extra: dict[str, object],
    required_phrase: str,
) -> None:
    result = TemporalAttributionCalibration(
        [_evidence("a-left", 0.8, 0.0), _evidence("a-right", 1.2, 10.0)]
    ).apply_estimates(
        {"E1": {"A": 100.0, "B": 100.0}},
        ratio_policy="coordinate_linear",
        epoch_ratio_policy=epoch_policy,
        residual_policy="allow_total_change",
        epoch_windows={"E1": (2.0, 8.0)},
        tested_entity_epochs={"A": ["E1"]},
        **extra,
    )

    summary = result.summary()

    assert required_phrase in summary
    if epoch_policy == "support_weighted_average":
        assert "declared coordinate weights: 3.0=1.0000, 7.0=3.0000." in summary


def test_mixed_policy_summary_reports_bucket_open_total_warnings_and_boundaries() -> None:
    result = TemporalAttributionCalibration(
        [
            LiftTestEvidence(
                tested_entity="A",
                observed_lift=80.0,
                matched_mta_estimate=100.0,
                test_id="a-test",
            )
        ]
    ).apply_estimates(
        {
            "E1": {"A": 100.0, "B": 100.0},
            "E2": {"A": 100.0, "B": 100.0},
        },
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy={
            "E1": "allow_total_change",
            "E2": "residual_bucket",
        },
        ratio_schedule={("A", "E1"): 0.8, ("A", "E2"): 0.8},
        residual_entity_by_epoch={"E2": "baseline"},
    )
    result = replace(result, warnings=("Review the declared strategy.",))

    first = result.summary()
    second = result.summary()

    assert first == second
    assert "No coordinate schedule rows; ratio policy explicit_ratio" in first
    assert "final epoch-level ratio was supplied explicitly by the practitioner." in first
    assert "Epoch E1 allocated no residual through policy allow_total_change." in first
    assert "baseline / E2 received +20.00 through residual policy residual_bucket." in first
    assert "Warnings\n- Review the declared strategy." in first
    assert (
        "Attribution epochs are authoritative outcome-driven or practitioner-supplied "
        "analysis periods; no tested entity owns them."
    ) in first
    assert (
        "Interpolation addresses missing evidence; it does not create experimental "
        "observations or entity-specific time partitions."
    ) in first
    assert (
        "Changes to untested entities are residual-allocation adjustments, not "
        "experimentally measured effects."
    ) in first
    assert (
        "do-attribution provides calibration mechanics; it does not select a "
        "calibration strategy."
    ) in first
    lowered = first.lower()
    assert "experimentally identified" not in lowered
    assert "global normalization" not in lowered
    assert "owns the epoch" not in lowered
