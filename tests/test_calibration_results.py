"""Deeply immutable static and temporal result tests for CAL-017."""

from dataclasses import FrozenInstanceError, fields

import pytest

from do_attribution.calibration import (
    CalibrationResult,
    CalibrationRow,
    EpochRatioRow,
    EvidencePointRecord,
    LiftTestEvidence,
    TemporalCalibrationResult,
)


STATIC_FIELDS = [
    "rows",
    "original_total",
    "raw_post_test_total",
    "required_residual",
    "allocated_residual",
    "calibrated_total",
    "residual_policy",
    "evidence",
    "warnings",
]

TEMPORAL_FIELDS = [
    "epochs",
    "epoch_windows",
    "epoch_coordinates",
    "epoch_support",
    "evidence_points",
    "coordinate_ratio_schedule",
    "epoch_ratio_schedule",
    "rows",
    "ratio_provenance",
    "epoch_reduction_provenance",
    "residual_provenance",
    "original_totals",
    "raw_post_test_totals",
    "required_residuals",
    "calibrated_totals",
    "ratio_policy",
    "epoch_ratio_policy",
    "residual_policy",
    "evidence",
    "warnings",
]


def _evidence(test_id: str = "a-test") -> LiftTestEvidence:
    return LiftTestEvidence(
        tested_entity="A",
        observed_lift=90.0,
        matched_mta_estimate=100.0,
        test_id=test_id,
        coordinate_name="time",
        coordinate_value=0.5,
        metadata={"source": {"ids": [test_id]}},
    )


def _tested_row(policy: str = "allow_total_change") -> CalibrationRow:
    return CalibrationRow(
        entity="A",
        original_estimate=100.0,
        direct_test_adjustment=-10.0,
        residual_adjustment=0.0,
        net_adjustment=-10.0,
        calibrated_estimate=90.0,
        original_share=0.5,
        calibrated_share=90.0 / 190.0,
        role="tested",
        evidence_ids=("a-test",),
        residual_policy=policy,
    )


def _untouched_row(policy: str = "allow_total_change") -> CalibrationRow:
    return CalibrationRow(
        entity="B",
        original_estimate=100.0,
        direct_test_adjustment=0.0,
        residual_adjustment=0.0,
        net_adjustment=0.0,
        calibrated_estimate=100.0,
        original_share=0.5,
        calibrated_share=100.0 / 190.0,
        role="untouched",
        evidence_ids=(),
        residual_policy=policy,
    )


def _static_result(**overrides) -> CalibrationResult:
    values = {
        "rows": [_tested_row(), _untouched_row()],
        "original_total": 200.0,
        "raw_post_test_total": 190.0,
        "required_residual": 10.0,
        "allocated_residual": 0.0,
        "calibrated_total": 190.0,
        "residual_policy": "allow_total_change",
        "evidence": [_evidence()],
        "warnings": ["Total changed under the declared residual policy."],
    }
    values.update(overrides)
    return CalibrationResult(**values)


def _epoch_ratio(epoch: str) -> EpochRatioRow:
    return EpochRatioRow(
        tested_entity="A",
        epoch=epoch,
        epoch_start=0.0 if epoch == "E1" else 1.0,
        epoch_end=1.0 if epoch == "E1" else 2.0,
        epoch_coordinate=None,
        epoch_support=10.0,
        effective_ratio=1.0,
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        ratio_source="explicit",
        evidence_conflict_policy=None,
        source_evidence_ids=(),
        evidence_weights=None,
        left_source_evidence_id=None,
        right_source_evidence_id=None,
        inside_observed_support=False,
        ratio_bound_lower=None,
        ratio_bound_upper=None,
    )


def _temporal_untouched_row() -> CalibrationRow:
    return CalibrationRow(
        entity="A",
        original_estimate=100.0,
        direct_test_adjustment=0.0,
        residual_adjustment=0.0,
        net_adjustment=0.0,
        calibrated_estimate=100.0,
        original_share=1.0,
        calibrated_share=1.0,
        role="untouched",
        evidence_ids=(),
        residual_policy="allow_total_change",
    )


def _temporal_inputs() -> dict:
    evidence = _evidence()
    return {
        "epochs": ["E1", "E2"],
        "epoch_windows": {"E1": (0.0, 1.0), "E2": (1.0, 2.0)},
        "epoch_coordinates": {"E1": 0.5, "E2": 1.5},
        "epoch_support": {"E1": 10.0, "E2": 20.0},
        "evidence_points": [
            EvidencePointRecord(
                tested_entity="A",
                evidence_id="a-test",
                observed_ratio=0.9,
                coordinate_name="time",
                coordinate_value=0.5,
                matched_epoch="E1",
                metadata={"source": {"ids": ["a-test"]}},
            )
        ],
        "coordinate_ratio_schedule": [],
        "epoch_ratio_schedule": [_epoch_ratio("E1"), _epoch_ratio("E2")],
        "rows": {
            "E1": [_temporal_untouched_row()],
            "E2": [_temporal_untouched_row()],
        },
        "ratio_provenance": {
            "pairs": {("A", "E1"): {"source_ids": ["a-test"]}}
        },
        "epoch_reduction_provenance": {"policy": {"name": "direct"}},
        "residual_provenance": {"E1": {"recipients": []}, "E2": {}},
        "original_totals": {"E1": 100.0, "E2": 100.0},
        "raw_post_test_totals": {"E1": 100.0, "E2": 100.0},
        "required_residuals": {"E1": 0.0, "E2": 0.0},
        "calibrated_totals": {"E1": 100.0, "E2": 100.0},
        "ratio_policy": "explicit_ratio",
        "epoch_ratio_policy": "direct",
        "residual_policy": "allow_total_change",
        "evidence": [evidence],
        "warnings": ["E2 has no direct experimental observation."],
    }


def test_static_fields_totals_warnings_and_rows_are_immutable() -> None:
    result = _static_result()
    assert [item.name for item in fields(result)] == STATIC_FIELDS
    assert result.rows == (_tested_row(), _untouched_row())
    assert result.required_residual == 10.0
    assert result.allocated_residual == 0.0
    assert result.warnings == (
        "Total changed under the declared residual policy.",
    )
    with pytest.raises(FrozenInstanceError):
        result.calibrated_total = 200.0


def test_static_caller_mutation_and_fresh_snapshots_cannot_change_result() -> None:
    rows = [_tested_row(), _untouched_row()]
    evidence = [_evidence()]
    warnings = ["Total changed under the declared residual policy."]
    result = _static_result(rows=rows, evidence=evidence, warnings=warnings)
    rows.reverse()
    evidence.append(_evidence("later"))
    warnings.append("later")
    snapshot = [row.entity for row in result.rows]
    snapshot.append("mutated export")
    assert tuple(row.entity for row in result.rows) == ("A", "B")
    assert tuple(item.test_id for item in result.evidence) == ("a-test",)
    assert len(result.warnings) == 1


def test_temporal_fields_ordering_nested_state_and_warnings_are_immutable() -> None:
    inputs = _temporal_inputs()
    result = TemporalCalibrationResult(**inputs)
    assert [item.name for item in fields(result)] == TEMPORAL_FIELDS
    assert result.epochs == ("E1", "E2")
    assert tuple(result.rows) == ("E1", "E2")
    assert tuple(result.original_totals) == ("E1", "E2")
    assert result.ratio_provenance["pairs"][("A", "E1")]["source_ids"] == (
        "a-test",
    )
    assert result.warnings == ("E2 has no direct experimental observation.",)
    with pytest.raises(TypeError):
        result.rows["E1"] = ()
    with pytest.raises(TypeError):
        result.ratio_provenance["new"] = "value"
    with pytest.raises(TypeError):
        result.ratio_provenance["pairs"][("A", "E1")]["new"] = "value"


def test_temporal_caller_mutation_and_snapshot_mutation_are_isolated() -> None:
    inputs = _temporal_inputs()
    result = TemporalCalibrationResult(**inputs)
    inputs["epochs"].reverse()
    inputs["epoch_windows"]["E1"] = (-10.0, -5.0)
    inputs["rows"]["E1"].append(_temporal_untouched_row())
    inputs["original_totals"]["E1"] = -1.0
    inputs["ratio_provenance"]["pairs"][("A", "E1")]["source_ids"].append(
        "mutated"
    )
    inputs["warnings"].append("mutated")

    row_snapshot = {epoch: list(rows) for epoch, rows in result.rows.items()}
    row_snapshot["E1"].clear()
    totals_snapshot = dict(result.original_totals)
    totals_snapshot["E1"] = -1.0
    assert result.epochs == ("E1", "E2")
    assert tuple(result.epoch_windows) == ("E1", "E2")
    assert result.epoch_windows["E1"] == (0.0, 1.0)
    assert len(result.rows["E1"]) == 1
    assert result.original_totals["E1"] == 100.0
    assert result.ratio_provenance["pairs"][("A", "E1")]["source_ids"] == (
        "a-test",
    )
    assert len(result.warnings) == 1


def test_equivalent_result_construction_compares_deterministically() -> None:
    assert _static_result() == _static_result()
    assert TemporalCalibrationResult(
        **_temporal_inputs()
    ) == TemporalCalibrationResult(**_temporal_inputs())


def test_required_reporting_methods_remain_explicit_later_item_boundaries() -> None:
    static = _static_result()
    temporal = TemporalCalibrationResult(**_temporal_inputs())
    assert static.to_dataframe()["entity"].tolist() == ["A", "B"]
    assert static.summary().startswith("Calibration Summary\n")
    assert callable(static.plot_calibration_effects)
    assert list(temporal.to_dataframes()) == [
        "evidence",
        "coordinate_schedule",
        "epoch_ratios",
        "allocations",
        "totals",
        "ratio_provenance",
        "epoch_reduction_provenance",
        "residual_provenance",
        "warnings",
    ]
    assert temporal.summary().startswith("Temporal Calibration Summary\n")
    assert callable(temporal.plot_epoch_alignment)
    assert callable(temporal.plot_ratio_schedule)
    assert callable(temporal.plot_calibration_effects)
    assert callable(temporal.plot_residual_allocation)
    assert callable(temporal.plot_entity_mix)
    assert callable(temporal.plot_evidence_strength)


def test_result_arithmetic_and_epoch_coverage_inconsistencies_raise() -> None:
    with pytest.raises(ValueError, match="raw_post_test_total"):
        _static_result(raw_post_test_total=191.0, required_residual=9.0)
    temporal = _temporal_inputs()
    temporal["original_totals"].pop("E2")
    with pytest.raises(ValueError, match="original_totals keys"):
        TemporalCalibrationResult(**temporal)


@pytest.mark.parametrize(
    "windows",
    [
        {"E1": (0.0, 1.5), "E2": (1.0, 2.0)},
        {"E1": (0.0, 1.0), "E2": (1.0, "2.0")},
        {"E1": (0.0, 1.0)},
    ],
)
def test_temporal_result_revalidates_authoritative_windows(windows) -> None:
    inputs = _temporal_inputs()
    inputs["epoch_windows"] = windows
    with pytest.raises((TypeError, ValueError), match="epoch_windows"):
        TemporalCalibrationResult(**inputs)


def test_temporal_result_rejects_evidence_matches_not_derived_from_windows() -> None:
    inputs = _temporal_inputs()
    original = inputs["evidence_points"][0]
    inputs["evidence_points"] = [
        EvidencePointRecord(
            tested_entity=original.tested_entity,
            evidence_id=original.evidence_id,
            observed_ratio=original.observed_ratio,
            coordinate_name=original.coordinate_name,
            coordinate_value=original.coordinate_value,
            window_start=original.window_start,
            window_end=original.window_end,
            matched_epoch="E2",
            conflict_group=original.conflict_group,
            metadata=original.metadata,
        )
    ]
    with pytest.raises(ValueError, match="authoritative epoch windows exactly"):
        TemporalCalibrationResult(**inputs)


@pytest.mark.parametrize(
    "coordinates",
    [
        {"E1": 1.0, "E2": 1.5},
        {"E1": 0.5, "E2": "1.5"},
        {"E3": 2.5},
    ],
)
def test_temporal_result_revalidates_representative_coordinates(coordinates) -> None:
    inputs = _temporal_inputs()
    inputs["epoch_coordinates"] = coordinates
    with pytest.raises((TypeError, ValueError), match="epoch_coordinates"):
        TemporalCalibrationResult(**inputs)


@pytest.mark.parametrize("support", [{"E1": -1.0}, {"E1": float("nan")}, {"E3": 1.0}])
def test_temporal_result_revalidates_scalar_epoch_support(support) -> None:
    inputs = _temporal_inputs()
    inputs["epoch_support"] = support
    with pytest.raises((TypeError, ValueError), match="epoch_support"):
        TemporalCalibrationResult(**inputs)
