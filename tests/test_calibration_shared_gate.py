"""Shared validation and public record-contract gate for CAL-018."""

import inspect
from dataclasses import fields

import do_attribution.calibration as calibration


EXPECTED_EXPORTS = [
    "Entity",
    "Epoch",
    "EvidenceId",
    "CoordinateValue",
    "ResidualPolicy",
    "RatioPolicy",
    "EpochRatioPolicy",
    "EvidenceConflictPolicy",
    "EstimateInput",
    "ShareInput",
    "EstimatesByEpochInput",
    "SharesByEpochInput",
    "TotalsByEpochInput",
    "EpochWindowsInput",
    "EpochCoordinatesInput",
    "EpochSupportInput",
    "EpochCoordinateWeightsInput",
    "RatioScheduleInput",
    "EvidenceStrengthInput",
    "SourceEvidenceInput",
    "SelectedEvidenceInput",
    "EvidenceWeightsInput",
    "LiftTestEvidence",
    "EvidencePointRecord",
    "RatioScheduleRow",
    "EpochRatioRow",
    "CalibrationRow",
    "CalibrationResult",
    "TemporalCalibrationResult",
    "AttributionCalibration",
    "TemporalAttributionCalibration",
]


EXPECTED_FIELDS = {
    calibration.LiftTestEvidence: [
        "tested_entity",
        "observed_lift",
        "matched_mta_estimate",
        "calibration_ratio",
        "test_id",
        "outcome_name",
        "window_start",
        "window_end",
        "population",
        "unit",
        "coordinate_name",
        "coordinate_value",
        "standard_error",
        "metadata",
    ],
    calibration.EvidencePointRecord: [
        "tested_entity",
        "evidence_id",
        "observed_ratio",
        "coordinate_name",
        "coordinate_value",
        "window_start",
        "window_end",
        "matched_epoch",
        "conflict_group",
        "metadata",
    ],
    calibration.RatioScheduleRow: [
        "tested_entity",
        "coordinate_name",
        "coordinate_value",
        "schedule_ratio",
        "schedule_source",
        "source_evidence_ids",
        "left_source_evidence_id",
        "right_source_evidence_id",
        "inside_observed_support",
    ],
    calibration.EpochRatioRow: [
        "tested_entity",
        "epoch",
        "epoch_start",
        "epoch_end",
        "epoch_coordinate",
        "epoch_support",
        "effective_ratio",
        "ratio_policy",
        "epoch_ratio_policy",
        "ratio_source",
        "evidence_conflict_policy",
        "source_evidence_ids",
        "evidence_weights",
        "left_source_evidence_id",
        "right_source_evidence_id",
        "inside_observed_support",
        "ratio_bound_lower",
        "ratio_bound_upper",
    ],
    calibration.CalibrationRow: [
        "entity",
        "original_estimate",
        "direct_test_adjustment",
        "residual_adjustment",
        "net_adjustment",
        "calibrated_estimate",
        "original_share",
        "calibrated_share",
        "role",
        "evidence_ids",
        "residual_policy",
    ],
    calibration.CalibrationResult: [
        "rows",
        "original_total",
        "raw_post_test_total",
        "required_residual",
        "allocated_residual",
        "calibrated_total",
        "residual_policy",
        "evidence",
        "warnings",
    ],
    calibration.TemporalCalibrationResult: [
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
    ],
}


EXPECTED_METHODS = {
    calibration.AttributionCalibration: {"apply_estimates", "apply_shares"},
    calibration.TemporalAttributionCalibration: {
        "apply_estimates",
        "apply_shares",
    },
    calibration.CalibrationResult: {
        "to_dataframe",
        "summary",
        "plot_calibration_effects",
    },
    calibration.TemporalCalibrationResult: {
        "to_dataframes",
        "summary",
        "plot_epoch_alignment",
        "plot_ratio_schedule",
        "plot_calibration_effects",
        "plot_residual_allocation",
        "plot_entity_mix",
        "plot_evidence_strength",
    },
}


def _public_methods(value: type) -> set[str]:
    return {
        name
        for name, member in value.__dict__.items()
        if not name.startswith("_") and callable(member)
    }


def test_module_exports_and_every_record_field_match_contract_exactly() -> None:
    assert calibration.__all__ == EXPECTED_EXPORTS
    for record_type, expected in EXPECTED_FIELDS.items():
        assert [item.name for item in fields(record_type)] == expected


def test_result_and_calibrator_public_methods_match_contract_exactly() -> None:
    for value, expected in EXPECTED_METHODS.items():
        assert _public_methods(value) == expected


def test_no_legacy_target_or_channel_specific_public_name_exists() -> None:
    forbidden = {
        "target",
        "target_epochs",
        "channel",
        "channels",
        "eligible_channels",
        "residual_channel",
    }
    public_names = set(calibration.__all__)
    for record_type, names in EXPECTED_FIELDS.items():
        public_names.update(names)
        public_names.update(_public_methods(record_type))
    for value, methods in EXPECTED_METHODS.items():
        public_names.update(methods)
        for method in methods:
            public_names.update(
                inspect.signature(getattr(value, method)).parameters
            )
    assert public_names.isdisjoint(forbidden)
