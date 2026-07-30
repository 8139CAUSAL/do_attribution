"""Separate-module and skeleton contracts for CAL-007."""

from dataclasses import FrozenInstanceError, is_dataclass

import pytest

from do_attribution.base import BaseAttribution
from do_attribution.calibration import (
    AttributionCalibration,
    CalibrationResult,
    CalibrationRow,
    EpochRatioRow,
    EvidencePointRecord,
    LiftTestEvidence,
    RatioScheduleRow,
    TemporalAttributionCalibration,
    TemporalCalibrationResult,
)


RECORDS = (
    LiftTestEvidence,
    EvidencePointRecord,
    RatioScheduleRow,
    EpochRatioRow,
    CalibrationRow,
    CalibrationResult,
    TemporalCalibrationResult,
)


def _lift() -> LiftTestEvidence:
    return LiftTestEvidence(
        tested_entity="A",
        observed_lift=1.0,
        matched_mta_estimate=1.0,
        test_id="test-a",
    )


def _evidence_point() -> EvidencePointRecord:
    return EvidencePointRecord(
        tested_entity="A",
        evidence_id="test-a",
        observed_ratio=1.0,
    )


def _schedule_row() -> RatioScheduleRow:
    return RatioScheduleRow(
        tested_entity="A",
        coordinate_name="date_index",
        coordinate_value=1.0,
        schedule_ratio=1.0,
        schedule_source="observed",
        source_evidence_ids=("test-a",),
        left_source_evidence_id=None,
        right_source_evidence_id=None,
        inside_observed_support=True,
    )


def _epoch_ratio_row() -> EpochRatioRow:
    return EpochRatioRow(
        tested_entity="A",
        epoch="E1",
        epoch_start=None,
        epoch_end=None,
        epoch_coordinate=None,
        epoch_support=None,
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


def _calibration_row(residual_policy="allow_total_change") -> CalibrationRow:
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
        residual_policy=residual_policy,
    )


def _calibration_result() -> CalibrationResult:
    return CalibrationResult(
        rows=(_calibration_row(),),
        original_total=100.0,
        raw_post_test_total=100.0,
        required_residual=0.0,
        allocated_residual=0.0,
        calibrated_total=100.0,
        residual_policy="allow_total_change",
        evidence=(_lift(),),
        warnings=(),
    )


def _temporal_result() -> TemporalCalibrationResult:
    return TemporalCalibrationResult(
        epochs=("E1",),
        epoch_windows=None,
        epoch_coordinates=None,
        epoch_support=None,
        evidence_points=(_evidence_point(),),
        coordinate_ratio_schedule=(),
        epoch_ratio_schedule=(),
        rows={"E1": (_calibration_row(None),)},
        ratio_provenance={},
        epoch_reduction_provenance={},
        residual_provenance={},
        original_totals={"E1": 100.0},
        raw_post_test_totals={"E1": 100.0},
        required_residuals={"E1": 0.0},
        calibrated_totals={"E1": 100.0},
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        evidence=(_lift(),),
        warnings=(),
    )


def test_all_plan_classes_import_from_separate_module() -> None:
    assert len(RECORDS) == 7
    assert AttributionCalibration.__module__ == "do_attribution.calibration"
    assert TemporalAttributionCalibration.__module__ == "do_attribution.calibration"


def test_calibrators_do_not_inherit_from_base_attribution() -> None:
    assert not issubclass(AttributionCalibration, BaseAttribution)
    assert not issubclass(TemporalAttributionCalibration, BaseAttribution)


def test_record_and_result_skeletons_are_frozen_dataclasses() -> None:
    for record_type in RECORDS:
        assert is_dataclass(record_type)
        if record_type is LiftTestEvidence:
            record = _lift()
        elif record_type is EvidencePointRecord:
            record = _evidence_point()
        elif record_type is RatioScheduleRow:
            record = _schedule_row()
        elif record_type is EpochRatioRow:
            record = _epoch_ratio_row()
        elif record_type is CalibrationRow:
            record = _calibration_row()
        elif record_type is CalibrationResult:
            record = _calibration_result()
        elif record_type is TemporalCalibrationResult:
            record = _temporal_result()
        else:
            record = record_type()
        with pytest.raises(FrozenInstanceError):
            record.unplanned_state = "mutable"


def test_evidence_only_constructors_copy_the_outer_sequence() -> None:
    evidence = [_lift()]
    static = AttributionCalibration(evidence=evidence)
    temporal = TemporalAttributionCalibration(evidence=evidence)
    evidence.append(_lift())
    assert len(static._evidence) == 1
    assert len(temporal._evidence) == 1


def test_temporal_share_entry_point_and_evidence_export_are_live() -> None:
    temporal = TemporalAttributionCalibration(evidence=[_lift()])
    result = temporal.apply_shares(
        {"E1": {"A": 1.0}},
        totals_by_epoch={"E1": 1.0},
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        ratio_schedule={("A", "E1"): 1.0},
    )
    assert result.calibrated_totals == {"E1": 1.0}
    tables = result.to_dataframes()
    assert list(tables) == [
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
    evidence = tables["evidence"]
    assert evidence["evidence_id"].tolist() == ["test-a"]


# MODIFIED: CAL-089 executed the planned package-root exports, so the
# CAL-007 premature-export guard now asserts the post-CAL-089 state.
# Complete package-root export coverage lives in
# tests/test_calibration_package_exports.py.
def test_calibration_symbols_are_exported_at_package_root_after_cal_089() -> None:
    import do_attribution

    assert hasattr(do_attribution, "AttributionCalibration")
    assert hasattr(do_attribution, "LiftTestEvidence")
