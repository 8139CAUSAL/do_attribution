"""Frozen public type-alias and calibrator-signature contract.

This file is parsed as source by ``tests/test_calibration_api_contract.py``.
It is deliberately not imported or executed.
"""

Entity: TypeAlias = Hashable
Epoch: TypeAlias = Hashable
EvidenceId: TypeAlias = str
CoordinateValue: TypeAlias = float | int | date | datetime

ResidualPolicy: TypeAlias = Literal[
    "allow_total_change",
    "proportional_untested",
    "custom_weights",
    "residual_bucket",
    "explicit_adjustments",
]

RatioPolicy: TypeAlias = Literal[
    "observed_only",
    "constant",
    "explicit_ratio",
    "explicit_strength",
    "coordinate_linear",
]

EpochRatioPolicy: TypeAlias = Literal[
    "direct",
    "representative_coordinate",
    "coordinate_average",
    "support_weighted_average",
]

EvidenceConflictPolicy: TypeAlias = Literal[
    "select_evidence",
    "weighted_ratio",
    "pooled_lift",
]

EstimateInput: TypeAlias = (
    Mapping[Entity, float]
    | Sequence[tuple[Entity, float]]
)

ShareInput: TypeAlias = (
    Mapping[Entity, float]
    | Sequence[tuple[Entity, float]]
)

EstimatesByEpochInput: TypeAlias = Mapping[
    Epoch,
    EstimateInput,
]

SharesByEpochInput: TypeAlias = Mapping[
    Epoch,
    ShareInput,
]

TotalsByEpochInput: TypeAlias = Mapping[Epoch, float]

EpochWindowsInput: TypeAlias = Mapping[
    Epoch,
    tuple[CoordinateValue, CoordinateValue],
]

EpochCoordinatesInput: TypeAlias = Mapping[
    Epoch,
    CoordinateValue,
]

EpochSupportInput: TypeAlias = Mapping[Epoch, float]

EpochCoordinateWeightsInput: TypeAlias = Mapping[
    Epoch,
    Sequence[tuple[CoordinateValue, float]],
]

RatioScheduleInput: TypeAlias = (
    Mapping[tuple[Entity, Epoch], float]
    | Sequence[tuple[tuple[Entity, Epoch], float]]
)

EvidenceStrengthInput: TypeAlias = (
    Mapping[tuple[Entity, Epoch], float]
    | Sequence[tuple[tuple[Entity, Epoch], float]]
)

SourceEvidenceInput: TypeAlias = Mapping[Entity, EvidenceId]

SelectedEvidenceInput: TypeAlias = Mapping[
    tuple[Entity, Epoch],
    EvidenceId,
]

EvidenceWeightsInput: TypeAlias = Mapping[
    tuple[Entity, Epoch],
    Mapping[EvidenceId, float],
]


class AttributionCalibration:
    def apply_estimates(
        self,
        estimates: EstimateInput,
        *,
        residual_policy: ResidualPolicy,
        eligible_entities: Sequence[Entity] | None = None,
        all_untested: bool | None = None,
        residual_weights: Mapping[Entity, float] | None = None,
        residual_entity: Entity | None = None,
        residual_adjustments: Mapping[Entity, float] | None = None,
        allow_bucket_collision: bool = False,
        allow_signed: bool = False,
        tolerance: float = 1e-9,
    ) -> CalibrationResult:
        ...

    def apply_shares(
        self,
        attribution: ShareInput,
        *,
        current_total: float,
        residual_policy: ResidualPolicy,
        eligible_entities: Sequence[Entity] | None = None,
        all_untested: bool | None = None,
        residual_weights: Mapping[Entity, float] | None = None,
        residual_entity: Entity | None = None,
        residual_adjustments: Mapping[Entity, float] | None = None,
        allow_bucket_collision: bool = False,
        allow_signed: bool = False,
        tolerance: float = 1e-9,
    ) -> CalibrationResult:
        ...


class TemporalAttributionCalibration:
    def apply_estimates(
        self,
        estimates_by_epoch: EstimatesByEpochInput,
        *,
        ratio_policy: RatioPolicy,
        epoch_ratio_policy: EpochRatioPolicy,
        residual_policy: ResidualPolicy | Mapping[Epoch, ResidualPolicy],
        epoch_windows: EpochWindowsInput | None = None,
        epoch_coordinates: EpochCoordinatesInput | None = None,
        epoch_support: EpochSupportInput | None = None,
        epoch_coordinate_weights: EpochCoordinateWeightsInput | None = None,
        source_evidence: SourceEvidenceInput | None = None,
        tested_entity_epochs: Mapping[Entity, Sequence[Epoch]] | None = None,
        ratio_schedule: RatioScheduleInput | None = None,
        evidence_strength: EvidenceStrengthInput | None = None,
        evidence_conflict_policy: (
            EvidenceConflictPolicy
            | Mapping[tuple[Entity, Epoch], EvidenceConflictPolicy]
            | None
        ) = None,
        selected_evidence: SelectedEvidenceInput | None = None,
        evidence_weights: EvidenceWeightsInput | None = None,
        eligible_entities: Sequence[Entity] | None = None,
        eligible_entities_by_epoch: Mapping[
            Epoch,
            Sequence[Entity],
        ] | None = None,
        all_untested: bool | None = None,
        residual_weights: Mapping[Entity, float] | None = None,
        residual_weights_by_epoch: Mapping[
            Epoch,
            Mapping[Entity, float],
        ] | None = None,
        residual_entity: Entity | None = None,
        residual_entity_by_epoch: Mapping[Epoch, Entity] | None = None,
        residual_adjustments_by_epoch: Mapping[
            Epoch,
            Mapping[Entity, float],
        ] | None = None,
        ratio_bounds: tuple[float, float] | None = None,
        allow_bucket_collision: bool = False,
        allow_signed: bool = False,
        tolerance: float = 1e-9,
    ) -> TemporalCalibrationResult:
        ...

    def apply_shares(
        self,
        attribution_by_epoch: SharesByEpochInput,
        *,
        totals_by_epoch: TotalsByEpochInput,
        ratio_policy: RatioPolicy,
        epoch_ratio_policy: EpochRatioPolicy,
        residual_policy: ResidualPolicy | Mapping[Epoch, ResidualPolicy],
        epoch_windows: EpochWindowsInput | None = None,
        epoch_coordinates: EpochCoordinatesInput | None = None,
        epoch_support: EpochSupportInput | None = None,
        epoch_coordinate_weights: EpochCoordinateWeightsInput | None = None,
        source_evidence: SourceEvidenceInput | None = None,
        tested_entity_epochs: Mapping[Entity, Sequence[Epoch]] | None = None,
        ratio_schedule: RatioScheduleInput | None = None,
        evidence_strength: EvidenceStrengthInput | None = None,
        evidence_conflict_policy: (
            EvidenceConflictPolicy
            | Mapping[tuple[Entity, Epoch], EvidenceConflictPolicy]
            | None
        ) = None,
        selected_evidence: SelectedEvidenceInput | None = None,
        evidence_weights: EvidenceWeightsInput | None = None,
        eligible_entities: Sequence[Entity] | None = None,
        eligible_entities_by_epoch: Mapping[
            Epoch,
            Sequence[Entity],
        ] | None = None,
        all_untested: bool | None = None,
        residual_weights: Mapping[Entity, float] | None = None,
        residual_weights_by_epoch: Mapping[
            Epoch,
            Mapping[Entity, float],
        ] | None = None,
        residual_entity: Entity | None = None,
        residual_entity_by_epoch: Mapping[Epoch, Entity] | None = None,
        residual_adjustments_by_epoch: Mapping[
            Epoch,
            Mapping[Entity, float],
        ] | None = None,
        ratio_bounds: tuple[float, float] | None = None,
        allow_bucket_collision: bool = False,
        allow_signed: bool = False,
        tolerance: float = 1e-9,
    ) -> TemporalCalibrationResult:
        ...
