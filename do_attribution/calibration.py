"""Post-attribution calibration terminology and causal boundaries.

Calibration is a separate post-attribution layer.  It consumes attribution
estimates and, for temporal analysis, an authoritative attribution epoch
partition without inheriting from or changing an attribution model.

The public contract uses these terms consistently:

``outcome``
    The quantity attribution is intended to explain, such as conversions or
    revenue.
``entity``
    An item receiving attribution.  A channel or campaign is an example of an
    entity, not a replacement for the general term.
``tested entity``
    The one entity measured by an experiment.  Evidence for a tested entity is
    never automatically applied to another entity.
``authoritative attribution epoch``
    An outcome-driven or practitioner-supplied analysis period determined
    before calibration.  An entity never owns an epoch, and calibration never
    creates, moves, splits, merges, relabels, or reorders epoch boundaries.
``evidence point``
    One retained experimental observation about one tested entity, identified
    by a unique evidence identifier and its experimental provenance.
``calibration ratio``
    The deterministic ratio of observed lift to the attribution estimate
    matched to the same population, outcome, and window.
``evidence coordinate``
    The explicit location of an evidence point on a named numeric, date, or
    datetime axis.  It is independent of the attribution epoch definition;
    several evidence points may occur inside one epoch.
``ratio schedule``
    A tested-entity schedule constructed from evidence by the practitioner's
    explicit ratio policy.  Constructed values are not new observations.
``epoch-ratio policy``
    The explicit rule reducing a coordinate schedule to attribution-epoch
    resolution.
``effective epoch ratio``
    The single ratio produced for one calibrated tested-entity--epoch pair.
``evidence-conflict policy``
    The explicit selection or combination rule used when several evidence
    points must be reduced directly to one tested-entity--epoch ratio.
``direct test adjustment``
    The evidence-backed change obtained by multiplying a tested entity's
    uncalibrated estimate by its effective ratio.
``residual``
    The difference between a declared total and the raw post-test total after
    direct adjustments.
``residual recipient``
    An explicitly eligible untested entity or named bucket that receives part
    of a residual.
``residual policy``
    The practitioner's explicit rule for allowing the total to change or for
    allocating a residual.  Residual allocation never changes a tested entity
    in this phase.

Interpolation is an explicit missing-evidence rule.  It neither discovers
epochs nor creates experimental observations.  Changes to untested entities
are residual-allocation adjustments, not effects identified by the experiment.

``do-attribution`` provides transparent calibration mechanics.  It does not
select a calibration strategy, evidence source, reduction rule, interpolation
rule, residual recipient, or residual-allocation rule for the practitioner.

Maintainability boundaries
--------------------------
``_calibration_validation`` owns generic input normalization, numeric
validation, recursive immutability, and export-copy mechanics.
This module remains the authority for public type aliases, evidence and
result records, ratio schedules, epoch reduction, residual-allocation
policies, reporting/serialization methods, and the two calibrator facades.
Those domain responsibilities remain colocated until their own behavioral
and serialization equivalence contracts justify another extraction.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from copy import deepcopy
from dataclasses import InitVar, dataclass, field, replace
from datetime import date, datetime, timezone
from math import isclose, isfinite
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, TypeVar

import pandas as pd

from ._calibration_validation import (
    coordinate_kind as _coordinate_kind,
    deep_export_copy as _deep_export_copy,
    deep_freeze as _deep_freeze,
    finite_float as _finite_float,
    normalize_estimates_by_epoch as _normalize_estimates_by_epoch,
    normalize_pairs as _normalize_pairs,
    normalize_unique_records as _normalize_unique_records,
    validate_numeric_pairs as _validate_numeric_pairs,
    validate_ratio_bounds as _validate_ratio_bounds,
    validate_shares as _validate_shares,
    validate_tolerance as _validate_tolerance,
    validate_total as _validate_total,
    validate_totals_by_epoch as _validate_totals_by_epoch,
    validate_unit_interval_pairs as _validate_unit_interval_pairs,
    validate_unit_sum_weights as _validate_unit_sum_weights,
)


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


_Value = TypeVar("_Value")


@dataclass(frozen=True)
class LiftTestEvidence:
    """One immutable, auditable lift-test observation for one tested entity."""

    tested_entity: Entity
    observed_lift: float
    matched_mta_estimate: float
    calibration_ratio: float = field(init=False)
    test_id: EvidenceId
    outcome_name: str | None = None
    window_start: Any | None = None
    window_end: Any | None = None
    population: str | None = None
    unit: str | None = None
    coordinate_name: str | None = None
    coordinate_value: CoordinateValue | None = None
    standard_error: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tested_entity is None:
            raise ValueError("tested_entity is required.")
        try:
            hash(self.tested_entity)
        except TypeError as exc:
            raise TypeError("tested_entity must be hashable.") from exc
        if not isinstance(self.test_id, str) or not self.test_id.strip():
            raise ValueError("test_id must be a nonempty string.")

        observed_lift = _finite_float(self.observed_lift, name="observed_lift")
        matched = _finite_float(
            self.matched_mta_estimate, name="matched_mta_estimate"
        )
        if matched == 0.0:
            raise ValueError("matched_mta_estimate must be nonzero.")
        ratio = observed_lift / matched
        if not isfinite(ratio):
            raise ValueError("calibration_ratio must be finite.")

        coordinate_name = self.coordinate_name
        coordinate_value = self.coordinate_value
        if (coordinate_name is None) != (coordinate_value is None):
            raise ValueError(
                "coordinate_name and coordinate_value must be supplied together."
            )
        if coordinate_name is not None:
            if not isinstance(coordinate_name, str) or not coordinate_name.strip():
                raise ValueError("coordinate_name must be a nonempty string.")
            _coordinate_kind(coordinate_value, name="coordinate_value")

        standard_error = self.standard_error
        if standard_error is not None:
            standard_error = _finite_float(standard_error, name="standard_error")
            if standard_error < 0.0:
                raise ValueError("standard_error must be nonnegative.")

        for name in ("outcome_name", "population", "unit"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty string or None.")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")

        object.__setattr__(self, "observed_lift", observed_lift)
        object.__setattr__(self, "matched_mta_estimate", matched)
        object.__setattr__(self, "calibration_ratio", ratio)
        object.__setattr__(self, "standard_error", standard_error)
        object.__setattr__(self, "metadata", _deep_freeze(self.metadata))


def _copy_lift_test_evidence(record: LiftTestEvidence) -> LiftTestEvidence:
    """Reconstruct one immutable evidence record with fresh nested state."""

    if not isinstance(record, LiftTestEvidence):
        raise TypeError("record must be a LiftTestEvidence instance.")
    return LiftTestEvidence(
        tested_entity=_deep_export_copy(record.tested_entity),
        observed_lift=record.observed_lift,
        matched_mta_estimate=record.matched_mta_estimate,
        test_id=_deep_export_copy(record.test_id),
        outcome_name=_deep_export_copy(record.outcome_name),
        window_start=_deep_export_copy(record.window_start),
        window_end=_deep_export_copy(record.window_end),
        population=_deep_export_copy(record.population),
        unit=_deep_export_copy(record.unit),
        coordinate_name=_deep_export_copy(record.coordinate_name),
        coordinate_value=_deep_export_copy(record.coordinate_value),
        standard_error=record.standard_error,
        metadata=_deep_export_copy(record.metadata),
    )


# Evidence-uncertainty annotations.  ``standard_error`` is the uncertainty the
# external experiment reported for its observed lift.  It is displayed beside
# the evidence it belongs to and nowhere else: it is never an interval on a
# calibrated value, and no calibrated number depends on it.
_NO_UNCERTAINTY_SUPPLIED = "no uncertainty supplied"
_UNCERTAINTY_REPORTED = "standard error reported by the experiment"


def _evidence_uncertainty_status(record: LiftTestEvidence) -> str:
    """State whether the experiment reported a standard error for its lift."""

    if record.standard_error is None:
        return _NO_UNCERTAINTY_SUPPLIED
    return _UNCERTAINTY_REPORTED


def _observed_lift_annotation(record: LiftTestEvidence) -> str:
    """Render ``observed lift ± standard error`` for one evidence record.

    A record without a standard error renders the explicit
    ``no uncertainty supplied`` state instead of an implied zero.
    """

    lift = 0.0 if record.observed_lift == 0.0 else record.observed_lift
    if record.standard_error is None:
        return f"{lift:.2f}; {_NO_UNCERTAINTY_SUPPLIED}"
    error = 0.0 if record.standard_error == 0.0 else record.standard_error
    return (
        f"{lift:.2f} ± {error:.2f} "
        f"({_UNCERTAINTY_REPORTED}; calibration does not use it)"
    )


@dataclass(frozen=True)
class _EvidenceCollection:
    records: tuple[LiftTestEvidence, ...]
    by_entity: Mapping[Entity, tuple[LiftTestEvidence, ...]]
    by_id: Mapping[EvidenceId, LiftTestEvidence]
    by_coordinate_name: Mapping[str | None, tuple[LiftTestEvidence, ...]]
    by_coordinate_value: Mapping[
        CoordinateValue | None,
        tuple[LiftTestEvidence, ...],
    ]


def _normalize_evidence_collection(
    evidence: Sequence[LiftTestEvidence],
    *,
    allow_multiple_per_entity: bool,
) -> _EvidenceCollection:
    if isinstance(evidence, (str, bytes, bytearray)):
        raise TypeError("evidence must be a sequence of LiftTestEvidence records.")
    try:
        records = tuple(evidence)
    except TypeError as exc:
        raise TypeError(
            "evidence must be a sequence of LiftTestEvidence records."
        ) from exc
    if not records:
        raise ValueError("evidence must not be empty.")

    by_id: dict[EvidenceId, LiftTestEvidence] = {}
    by_entity: dict[Entity, list[LiftTestEvidence]] = {}
    by_coordinate_name: dict[str | None, list[LiftTestEvidence]] = {}
    by_coordinate_value: dict[
        CoordinateValue | None,
        list[LiftTestEvidence],
    ] = {}
    for index, record in enumerate(records):
        if not isinstance(record, LiftTestEvidence):
            raise TypeError(
                f"evidence[{index}] must be a LiftTestEvidence record."
            )
        if record.test_id in by_id:
            raise ValueError(f"evidence contains duplicate test_id {record.test_id!r}.")
        by_id[record.test_id] = record
        by_entity.setdefault(record.tested_entity, []).append(record)
        by_coordinate_name.setdefault(record.coordinate_name, []).append(record)
        by_coordinate_value.setdefault(record.coordinate_value, []).append(record)

    if not allow_multiple_per_entity:
        ambiguous = [
            entity
            for entity, entity_records in by_entity.items()
            if len(entity_records) > 1
        ]
        if ambiguous:
            raise ValueError(
                "Static calibration requires exactly one evidence record per "
                f"tested entity; ambiguous entities={ambiguous!r}."
            )
    return _EvidenceCollection(
        records=records,
        by_entity=MappingProxyType(
            {
                entity: tuple(entity_records)
                for entity, entity_records in by_entity.items()
            }
        ),
        by_id=MappingProxyType(dict(by_id)),
        by_coordinate_name=MappingProxyType(
            {
                coordinate_name: tuple(coordinate_records)
                for coordinate_name, coordinate_records in by_coordinate_name.items()
            }
        ),
        by_coordinate_value=MappingProxyType(
            {
                coordinate_value: tuple(coordinate_records)
                for coordinate_value, coordinate_records in by_coordinate_value.items()
            }
        ),
    )


def _conflict_groups_by_epoch(
    records: Sequence[Any],
) -> Mapping[tuple[Entity, Epoch], tuple[Any, ...]]:
    """Identify matched same-entity/epoch groups without resolving them."""

    grouped: dict[tuple[Entity, Epoch], list[Any]] = {}
    for record in records:
        matched_epoch = getattr(record, "matched_epoch", None)
        if matched_epoch is None:
            continue
        key = (getattr(record, "tested_entity"), matched_epoch)
        grouped.setdefault(key, []).append(record)
    return MappingProxyType(
        {key: tuple(group) for key, group in grouped.items() if len(group) > 1}
    )


@dataclass(frozen=True)
class EvidencePointRecord:
    """One normalized evidence location aligned to, but not owned by, an epoch."""

    tested_entity: Entity
    evidence_id: EvidenceId
    observed_ratio: float
    coordinate_name: str | None = None
    coordinate_value: CoordinateValue | None = None
    window_start: Any | None = None
    window_end: Any | None = None
    matched_epoch: Epoch | None = None
    conflict_group: tuple[EvidenceId, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tested_entity is None:
            raise ValueError("tested_entity is required.")
        try:
            hash(self.tested_entity)
        except TypeError as exc:
            raise TypeError("tested_entity must be hashable.") from exc
        if not isinstance(self.evidence_id, str) or not self.evidence_id.strip():
            raise ValueError("evidence_id must be a nonempty string.")

        observed_ratio = _finite_float(
            self.observed_ratio, name="observed_ratio"
        )
        if (self.coordinate_name is None) != (self.coordinate_value is None):
            raise ValueError(
                "coordinate_name and coordinate_value must be supplied together."
            )
        if self.coordinate_name is not None:
            if (
                not isinstance(self.coordinate_name, str)
                or not self.coordinate_name.strip()
            ):
                raise ValueError("coordinate_name must be a nonempty string.")
            _coordinate_kind(self.coordinate_value, name="coordinate_value")

        matched_epoch = self.matched_epoch
        if matched_epoch is not None:
            try:
                hash(matched_epoch)
            except TypeError as exc:
                raise TypeError("matched_epoch must be hashable or None.") from exc

        conflict_group = self.conflict_group
        if conflict_group is not None:
            if isinstance(conflict_group, (str, bytes, bytearray)):
                raise TypeError(
                    "conflict_group must be a sequence of evidence IDs or None."
                )
            try:
                conflict_ids = tuple(conflict_group)
            except TypeError as exc:
                raise TypeError(
                    "conflict_group must be a sequence of evidence IDs or None."
                ) from exc
            if len(conflict_ids) < 2:
                raise ValueError(
                    "conflict_group must contain at least two evidence IDs."
                )
            if any(
                not isinstance(identifier, str) or not identifier.strip()
                for identifier in conflict_ids
            ):
                raise ValueError(
                    "conflict_group entries must be nonempty evidence ID strings."
                )
            if len(set(conflict_ids)) != len(conflict_ids):
                raise ValueError("conflict_group evidence IDs must be unique.")
            if self.evidence_id not in conflict_ids:
                raise ValueError(
                    "conflict_group must contain this record's evidence_id."
                )
            conflict_group = tuple(deepcopy(identifier) for identifier in conflict_ids)

        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")

        object.__setattr__(self, "tested_entity", deepcopy(self.tested_entity))
        object.__setattr__(self, "evidence_id", deepcopy(self.evidence_id))
        object.__setattr__(self, "observed_ratio", observed_ratio)
        object.__setattr__(self, "coordinate_name", deepcopy(self.coordinate_name))
        object.__setattr__(self, "coordinate_value", deepcopy(self.coordinate_value))
        object.__setattr__(self, "window_start", _deep_freeze(self.window_start))
        object.__setattr__(self, "window_end", _deep_freeze(self.window_end))
        object.__setattr__(self, "matched_epoch", deepcopy(matched_epoch))
        object.__setattr__(self, "conflict_group", conflict_group)
        object.__setattr__(self, "metadata", _deep_freeze(self.metadata))


def _normalize_epoch_windows_for_matching(
    epoch_windows: EpochWindowsInput,
) -> tuple[tuple[Epoch, CoordinateValue, CoordinateValue], ...]:
    """Copy authoritative windows and validate deterministic membership order."""

    if not isinstance(epoch_windows, Mapping):
        raise TypeError("epoch_windows must be a mapping.")
    if not epoch_windows:
        raise ValueError("epoch_windows must not be empty when supplied.")

    normalized: list[tuple[Epoch, CoordinateValue, CoordinateValue]] = []
    coordinate_kind: str | None = None
    previous_end: CoordinateValue | None = None
    for epoch, window in epoch_windows.items():
        if not isinstance(window, tuple) or len(window) != 2:
            raise TypeError(
                f"epoch_windows[{epoch!r}] must be a two-item tuple."
            )
        start, end = window
        start_kind = _coordinate_kind(
            start, name=f"epoch_windows[{epoch!r}] start"
        )
        end_kind = _coordinate_kind(end, name=f"epoch_windows[{epoch!r}] end")
        if start_kind != end_kind:
            raise TypeError(
                f"epoch_windows[{epoch!r}] endpoints must use compatible "
                "coordinate types."
            )
        if coordinate_kind is None:
            coordinate_kind = start_kind
        elif coordinate_kind != start_kind:
            raise TypeError(
                "epoch_windows must use one compatible coordinate type."
            )
        try:
            nonempty = start < end
            nonoverlapping = previous_end is None or start >= previous_end
        except TypeError as exc:
            raise TypeError(
                "epoch_windows coordinates must be mutually comparable."
            ) from exc
        if not nonempty:
            raise ValueError(
                f"epoch_windows[{epoch!r}] must be nonempty with start < end."
            )
        if not nonoverlapping:
            raise ValueError(
                "epoch_windows must be ordered and nonoverlapping in mapping order."
            )
        normalized.append((deepcopy(epoch), deepcopy(start), deepcopy(end)))
        previous_end = end
    return tuple(normalized)


def _normalize_authoritative_epoch_windows(
    epoch_windows: EpochWindowsInput | None,
    *,
    epochs: Sequence[Epoch],
) -> Mapping[Epoch, tuple[CoordinateValue, CoordinateValue]] | None:
    """Validate and freeze optional windows on one authoritative epoch grid."""

    authoritative_epochs = _epoch_tuple(epochs)
    frozen = _freeze_optional_epoch_mapping(
        epoch_windows,
        epochs=authoritative_epochs,
        name="epoch_windows",
        exact=True,
    )
    if frozen is None:
        return None
    normalized = _normalize_epoch_windows_for_matching(frozen)
    return MappingProxyType(
        {
            deepcopy(epoch): (deepcopy(start), deepcopy(end))
            for epoch, start, end in normalized
        }
    )


def _normalize_epoch_coordinates(
    epoch_coordinates: EpochCoordinatesInput | None,
    *,
    epochs: Sequence[Epoch],
    epoch_windows: EpochWindowsInput | None,
    required_epochs: Sequence[Epoch] | None = None,
) -> Mapping[Epoch, CoordinateValue] | None:
    """Freeze declared representative coordinates on authoritative windows."""

    authoritative_epochs = _epoch_tuple(epochs)
    required = (
        () if required_epochs is None else _epoch_tuple(required_epochs)
    )
    unknown_required = [
        epoch for epoch in required if epoch not in authoritative_epochs
    ]
    if unknown_required:
        raise ValueError(
            "required representative-coordinate epochs must belong to the "
            f"authoritative grid; unknown={unknown_required!r}."
        )

    coordinates = _freeze_optional_epoch_mapping(
        epoch_coordinates,
        epochs=authoritative_epochs,
        name="epoch_coordinates",
        exact=False,
    )
    if coordinates is None:
        if required:
            raise ValueError(
                "epoch_coordinates are required for affected epochs; "
                f"missing={list(required)!r}."
            )
        return None

    missing = [epoch for epoch in required if epoch not in coordinates]
    if missing:
        raise ValueError(
            "epoch_coordinates are required for affected epochs; "
            f"missing={missing!r}."
        )
    windows = _normalize_authoritative_epoch_windows(
        epoch_windows,
        epochs=authoritative_epochs,
    )
    if windows is None:
        raise ValueError(
            "epoch_windows are required when epoch_coordinates are supplied."
        )
    window_rows = tuple(
        (epoch, window[0], window[1]) for epoch, window in windows.items()
    )
    for epoch, coordinate in coordinates.items():
        coordinate_kind = _coordinate_kind(
            coordinate,
            name=f"epoch_coordinates[{epoch!r}]",
        )
        window_kind = _coordinate_kind(
            windows[epoch][0],
            name=f"epoch_windows[{epoch!r}] start",
        )
        if coordinate_kind != window_kind:
            raise TypeError(
                f"epoch_coordinates[{epoch!r}] must use the authoritative "
                "window coordinate type."
            )
        matched_epoch = _matched_epoch_for_coordinate(coordinate, window_rows)
        if matched_epoch != epoch:
            raise ValueError(
                f"epoch_coordinates[{epoch!r}] must lie inside that epoch's "
                "authoritative window."
            )
    return coordinates


def _normalize_epoch_support(
    epoch_support: EpochSupportInput | None,
    *,
    epochs: Sequence[Epoch],
) -> Mapping[Epoch, float] | None:
    """Freeze optional scalar outcome support as diagnostic metadata only."""

    authoritative_epochs = _epoch_tuple(epochs)
    support = _freeze_optional_epoch_mapping(
        epoch_support,
        epochs=authoritative_epochs,
        name="epoch_support",
        exact=False,
    )
    if support is None:
        return None

    checked: dict[Epoch, float] = {}
    for epoch, value in support.items():
        scalar = _finite_float(value, name=f"epoch_support[{epoch!r}]")
        if scalar < 0.0:
            raise ValueError(
                f"epoch_support[{epoch!r}] must be nonnegative."
            )
        checked[deepcopy(epoch)] = scalar
    return MappingProxyType(checked)


def _normalize_epoch_coordinate_weights(
    epoch_coordinate_weights: EpochCoordinateWeightsInput | None,
    *,
    epochs: Sequence[Epoch],
    epoch_windows: EpochWindowsInput | None,
    required_epochs: Sequence[Epoch] | None = None,
) -> Mapping[Epoch, tuple[tuple[CoordinateValue, float], ...]] | None:
    """Freeze explicit within-epoch coordinate weights without aggregation."""

    authoritative_epochs = _epoch_tuple(epochs)
    required = (
        () if required_epochs is None else _epoch_tuple(required_epochs)
    )
    unknown_required = [
        epoch for epoch in required if epoch not in authoritative_epochs
    ]
    if unknown_required:
        raise ValueError(
            "required coordinate-weight epochs must belong to the authoritative "
            f"grid; unknown={unknown_required!r}."
        )

    supplied = _freeze_optional_epoch_mapping(
        epoch_coordinate_weights,
        epochs=authoritative_epochs,
        name="epoch_coordinate_weights",
        exact=False,
    )
    if supplied is None:
        if required:
            raise ValueError(
                "epoch_coordinate_weights are required for affected epochs; "
                f"missing={list(required)!r}."
            )
        return None

    missing = [epoch for epoch in required if epoch not in supplied]
    if missing:
        raise ValueError(
            "epoch_coordinate_weights are required for affected epochs; "
            f"missing={missing!r}."
        )
    windows = _normalize_authoritative_epoch_windows(
        epoch_windows,
        epochs=authoritative_epochs,
    )
    if windows is None:
        raise ValueError(
            "epoch_windows are required when epoch_coordinate_weights are supplied."
        )
    window_rows = tuple(
        (epoch, window[0], window[1]) for epoch, window in windows.items()
    )

    normalized: dict[
        Epoch,
        tuple[tuple[CoordinateValue, float], ...],
    ] = {}
    for epoch, coordinate_weights in supplied.items():
        pairs = _normalize_pairs(
            coordinate_weights,
            name=f"epoch_coordinate_weights[{epoch!r}]",
        )
        checked_pairs: list[tuple[CoordinateValue, float]] = []
        window_kind = _coordinate_kind(
            windows[epoch][0],
            name=f"epoch_windows[{epoch!r}] start",
        )
        for coordinate, weight in pairs:
            coordinate_kind = _coordinate_kind(
                coordinate,
                name=f"epoch_coordinate_weights[{epoch!r}] coordinate",
            )
            if coordinate_kind != window_kind:
                raise TypeError(
                    f"epoch_coordinate_weights[{epoch!r}] coordinates must use "
                    "the authoritative window coordinate type."
                )
            matched_epoch = _matched_epoch_for_coordinate(
                coordinate,
                window_rows,
            )
            if matched_epoch != epoch:
                raise ValueError(
                    f"epoch_coordinate_weights[{epoch!r}] coordinates must lie "
                    "inside that epoch's authoritative window."
                )
            checked_weight = _finite_float(
                weight,
                name=(
                    f"epoch_coordinate_weights[{epoch!r}]"
                    f"[{coordinate!r}]"
                ),
            )
            if checked_weight < 0.0:
                raise ValueError(
                    f"epoch_coordinate_weights[{epoch!r}] weights must be "
                    "nonnegative."
                )
            checked_pairs.append((deepcopy(coordinate), checked_weight))

        total_weight = sum(weight for _, weight in checked_pairs)
        if not isfinite(total_weight) or total_weight <= 0.0:
            raise ValueError(
                f"epoch_coordinate_weights[{epoch!r}] must have finite positive "
                "total weight."
            )
        normalized[deepcopy(epoch)] = tuple(checked_pairs)
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class _RatioPolicyRequest:
    """Validated explicit schedule and epoch-reduction policy selection."""

    ratio_policy: RatioPolicy
    epoch_ratio_policy: EpochRatioPolicy


def _validate_ratio_policy_request(
    *,
    evidence: Sequence[LiftTestEvidence],
    ratio_policy: RatioPolicy,
    epoch_ratio_policy: EpochRatioPolicy,
    epoch_windows: EpochWindowsInput | None = None,
    epoch_coordinates: EpochCoordinatesInput | None = None,
    epoch_coordinate_weights: EpochCoordinateWeightsInput | None = None,
    epoch_support: EpochSupportInput | None = None,
    source_evidence: SourceEvidenceInput | None = None,
    tested_entity_epochs: Mapping[Entity, Sequence[Epoch]] | None = None,
    ratio_schedule: RatioScheduleInput | None = None,
    evidence_strength: EvidenceStrengthInput | None = None,
) -> _RatioPolicyRequest:
    """Validate policy compatibility without constructing any schedule."""

    if ratio_policy not in _RATIO_POLICIES:
        raise ValueError(
            f"ratio_policy must be one of {sorted(_RATIO_POLICIES)!r}."
        )
    if epoch_ratio_policy not in _EPOCH_RATIO_POLICIES:
        raise ValueError(
            "epoch_ratio_policy must be one of "
            f"{sorted(_EPOCH_RATIO_POLICIES)!r}."
        )
    direct_policies = {
        "observed_only",
        "constant",
        "explicit_ratio",
        "explicit_strength",
    }
    if ratio_policy in direct_policies and epoch_ratio_policy != "direct":
        raise ValueError(
            f"ratio_policy={ratio_policy!r} requires epoch_ratio_policy='direct'."
        )
    if ratio_policy == "coordinate_linear" and epoch_ratio_policy == "direct":
        raise ValueError(
            "ratio_policy='coordinate_linear' requires a coordinate epoch-ratio "
            "policy and is incompatible with 'direct'."
        )

    normalized_evidence = _normalize_evidence_collection(
        evidence,
        allow_multiple_per_entity=True,
    ).records
    if ratio_policy in {"observed_only", "coordinate_linear"}:
        missing_coordinates = [
            item.test_id
            for item in normalized_evidence
            if item.coordinate_name is None or item.coordinate_value is None
        ]
        if missing_coordinates:
            raise ValueError(
                f"ratio_policy={ratio_policy!r} requires explicit evidence "
                f"coordinates; missing={missing_coordinates!r}."
            )

    inputs = {
        "source_evidence": source_evidence,
        "tested_entity_epochs": tested_entity_epochs,
        "ratio_schedule": ratio_schedule,
        "evidence_strength": evidence_strength,
    }
    required_by_policy = {
        "observed_only": (),
        "constant": ("source_evidence", "tested_entity_epochs"),
        "explicit_ratio": ("ratio_schedule",),
        "explicit_strength": ("source_evidence", "evidence_strength"),
        "coordinate_linear": (),
    }
    allowed_by_policy = {
        "observed_only": frozenset(),
        "constant": frozenset({"source_evidence", "tested_entity_epochs"}),
        "explicit_ratio": frozenset({"ratio_schedule", "tested_entity_epochs"}),
        "explicit_strength": frozenset(
            {"source_evidence", "evidence_strength", "tested_entity_epochs"}
        ),
        "coordinate_linear": frozenset({"tested_entity_epochs"}),
    }
    missing = [
        name
        for name in required_by_policy[ratio_policy]
        if inputs[name] is None
    ]
    if missing:
        raise ValueError(
            f"ratio_policy={ratio_policy!r} requires inputs {missing!r}."
        )
    irrelevant = [
        name
        for name, value in inputs.items()
        if value is not None and name not in allowed_by_policy[ratio_policy]
    ]
    if irrelevant:
        raise ValueError(
            f"ratio_policy={ratio_policy!r} does not accept inputs {irrelevant!r}."
        )

    if ratio_policy == "observed_only" and epoch_windows is None:
        raise ValueError("ratio_policy='observed_only' requires epoch_windows.")
    if ratio_policy == "coordinate_linear" and epoch_windows is None:
        raise ValueError("ratio_policy='coordinate_linear' requires epoch_windows.")

    if epoch_ratio_policy == "representative_coordinate":
        if epoch_coordinates is None:
            raise ValueError(
                "epoch_ratio_policy='representative_coordinate' requires "
                "epoch_coordinates."
            )
        if epoch_coordinate_weights is not None:
            raise ValueError(
                "epoch_coordinate_weights are incompatible with "
                "epoch_ratio_policy='representative_coordinate'."
            )
    elif epoch_ratio_policy == "coordinate_average":
        incompatible = []
        if epoch_coordinates is not None:
            incompatible.append("epoch_coordinates")
        if epoch_coordinate_weights is not None:
            incompatible.append("epoch_coordinate_weights")
        if incompatible:
            raise ValueError(
                "epoch_ratio_policy='coordinate_average' does not accept "
                f"inputs {incompatible!r}."
            )
    elif epoch_ratio_policy == "support_weighted_average":
        if epoch_coordinate_weights is None:
            raise ValueError(
                "epoch_ratio_policy='support_weighted_average' requires "
                "epoch_coordinate_weights; scalar epoch_support is not a "
                "substitute."
            )
        if epoch_coordinates is not None:
            raise ValueError(
                "epoch_coordinates are incompatible with "
                "epoch_ratio_policy='support_weighted_average'."
            )
    else:
        incompatible = []
        if epoch_coordinates is not None:
            incompatible.append("epoch_coordinates")
        if epoch_coordinate_weights is not None:
            incompatible.append("epoch_coordinate_weights")
        if incompatible:
            raise ValueError(
                "epoch_ratio_policy='direct' does not accept inputs "
                f"{incompatible!r}."
            )

    # Scalar support is deliberately accepted only as diagnostic metadata;
    # referencing it here prevents accidental reinterpretation as a schedule.
    _ = epoch_support
    return _RatioPolicyRequest(
        ratio_policy=deepcopy(ratio_policy),
        epoch_ratio_policy=deepcopy(epoch_ratio_policy),
    )


@dataclass(frozen=True)
class _ObservedOnlySchedule:
    """Observed evidence aligned to epochs but not conflict-resolved."""

    evidence_by_id: Mapping[EvidenceId, LiftTestEvidence]
    evidence_points: tuple[EvidencePointRecord, ...]
    coordinate_schedule: tuple[RatioScheduleRow, ...]
    candidates: Mapping[
        tuple[Entity, Epoch],
        tuple[EvidencePointRecord, ...],
    ]
    status_by_pair: Mapping[tuple[Entity, Epoch], str]


def _build_observed_only_schedule(
    evidence: Sequence[LiftTestEvidence],
    *,
    epochs: Sequence[Epoch],
    epoch_windows: EpochWindowsInput | None,
) -> _ObservedOnlySchedule:
    """Retain direct observations without selecting or combining conflicts."""

    _validate_ratio_policy_request(
        evidence=evidence,
        ratio_policy="observed_only",
        epoch_ratio_policy="direct",
        epoch_windows=epoch_windows,
    )
    authoritative_epochs = _epoch_tuple(epochs)
    windows = _normalize_authoritative_epoch_windows(
        epoch_windows,
        epochs=authoritative_epochs,
    )
    if windows is None:  # pragma: no cover - guarded by request validation
        raise ValueError("observed_only requires epoch_windows.")

    collection = _normalize_evidence_collection(
        evidence,
        allow_multiple_per_entity=True,
    )
    evidence_points = _normalize_evidence_points(
        collection.records,
        epoch_windows=windows,
    )
    coordinate_schedule = tuple(
        RatioScheduleRow(
            tested_entity=point.tested_entity,
            coordinate_name=point.coordinate_name,
            coordinate_value=point.coordinate_value,
            schedule_ratio=point.observed_ratio,
            schedule_source="observed",
            source_evidence_ids=(point.evidence_id,),
            left_source_evidence_id=None,
            right_source_evidence_id=None,
            inside_observed_support=True,
        )
        for point in evidence_points
    )

    grouped: dict[
        tuple[Entity, Epoch],
        list[EvidencePointRecord],
    ] = {}
    for point in evidence_points:
        if point.matched_epoch is None:
            continue
        grouped.setdefault(
            (point.tested_entity, point.matched_epoch),
            [],
        ).append(point)

    entity_order = tuple(collection.by_entity)
    candidates: dict[
        tuple[Entity, Epoch],
        tuple[EvidencePointRecord, ...],
    ] = {}
    statuses: dict[tuple[Entity, Epoch], str] = {}
    for epoch in authoritative_epochs:
        for entity in entity_order:
            pair = (entity, epoch)
            group = grouped.get(pair, [])
            if group:
                candidates[deepcopy(pair)] = tuple(group)
                statuses[deepcopy(pair)] = (
                    "observed" if len(group) == 1 else "conflict_pending"
                )
            else:
                statuses[deepcopy(pair)] = "not_calibrated"

    return _ObservedOnlySchedule(
        evidence_by_id=collection.by_id,
        evidence_points=evidence_points,
        coordinate_schedule=coordinate_schedule,
        candidates=MappingProxyType(candidates),
        status_by_pair=MappingProxyType(statuses),
    )


@dataclass(frozen=True)
class _EvidenceConflictRequest:
    """Explicit policies for every direct same-entity/epoch conflict group."""

    evidence_by_id: Mapping[EvidenceId, LiftTestEvidence]
    evidence_points: tuple[EvidencePointRecord, ...]
    conflict_groups: Mapping[
        tuple[Entity, Epoch],
        tuple[EvidencePointRecord, ...],
    ]
    policy_by_pair: Mapping[
        tuple[Entity, Epoch],
        EvidenceConflictPolicy,
    ]


@dataclass(frozen=True)
class _EvidenceConflictResolution:
    """One explicit direct reduction with complete candidate provenance."""

    tested_entity: Entity
    epoch: Epoch
    effective_ratio: float
    evidence_conflict_policy: EvidenceConflictPolicy
    ratio_source: str
    source_evidence_ids: tuple[EvidenceId, ...]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("tested_entity", "epoch"):
            try:
                hash(getattr(self, name))
            except TypeError as exc:
                raise TypeError(f"{name} must be hashable.") from exc
        effective_ratio = _finite_float(
            self.effective_ratio,
            name="effective_ratio",
        )
        if self.evidence_conflict_policy not in _EVIDENCE_CONFLICT_POLICIES:
            raise ValueError(
                "evidence_conflict_policy must be one of "
                f"{sorted(_EVIDENCE_CONFLICT_POLICIES)!r}."
            )
        if not isinstance(self.ratio_source, str) or not self.ratio_source.strip():
            raise ValueError("ratio_source must be a nonempty string.")
        source_ids = _evidence_id_tuple(
            self.source_evidence_ids,
            name="source_evidence_ids",
        )
        if not isinstance(self.provenance, Mapping):
            raise TypeError("provenance must be a mapping.")

        object.__setattr__(self, "tested_entity", deepcopy(self.tested_entity))
        object.__setattr__(self, "epoch", deepcopy(self.epoch))
        object.__setattr__(self, "effective_ratio", effective_ratio)
        object.__setattr__(
            self,
            "evidence_conflict_policy",
            deepcopy(self.evidence_conflict_policy),
        )
        object.__setattr__(self, "ratio_source", deepcopy(self.ratio_source))
        object.__setattr__(self, "source_evidence_ids", source_ids)
        object.__setattr__(self, "provenance", _deep_freeze(self.provenance))


@dataclass(frozen=True)
class _EvidenceConflictDiagnostics:
    """Observational conflict alternatives kept separate from applied policy."""

    evidence_points: tuple[EvidencePointRecord, ...]
    conflict_groups: Mapping[
        tuple[Entity, Epoch],
        tuple[EvidencePointRecord, ...],
    ]
    applied_policy_by_pair: Mapping[
        tuple[Entity, Epoch],
        EvidenceConflictPolicy,
    ]
    alternatives_by_pair: Mapping[
        tuple[Entity, Epoch],
        Mapping[EvidenceConflictPolicy, _EvidenceConflictResolution],
    ]


def _validate_evidence_conflict_request(
    schedule: _ObservedOnlySchedule,
    *,
    evidence_conflict_policy: (
        EvidenceConflictPolicy
        | Mapping[tuple[Entity, Epoch], EvidenceConflictPolicy]
        | None
    ),
) -> _EvidenceConflictRequest:
    """Require one explicit applicable policy per unresolved direct group."""

    def checked_policy(value: Any, *, name: str) -> EvidenceConflictPolicy:
        if (
            not isinstance(value, str)
            or value not in _EVIDENCE_CONFLICT_POLICIES
        ):
            raise ValueError(
                f"{name} must be one of "
                f"{sorted(_EVIDENCE_CONFLICT_POLICIES)!r}."
            )
        return value

    if not isinstance(schedule, _ObservedOnlySchedule):
        raise TypeError("schedule must be an observed-only schedule.")

    conflict_groups = {
        deepcopy(pair): tuple(group)
        for pair, group in schedule.candidates.items()
        if len(group) > 1
    }
    conflict_pairs = tuple(conflict_groups)

    if evidence_conflict_policy is None:
        if conflict_pairs:
            raise ValueError(
                "evidence_conflict_policy is required for direct conflict "
                f"groups {list(conflict_pairs)!r}."
            )
        normalized_policies: dict[
            tuple[Entity, Epoch], EvidenceConflictPolicy
        ] = {}
    elif isinstance(evidence_conflict_policy, Mapping):
        supplied = dict(evidence_conflict_policy.items())
        missing = [pair for pair in conflict_pairs if pair not in supplied]
        extra = [pair for pair in supplied if pair not in conflict_groups]
        if missing or extra:
            raise ValueError(
                "evidence_conflict_policy mapping keys must exactly match "
                f"direct conflict groups; missing={missing!r}, extra={extra!r}."
            )
        normalized_policies = {}
        for pair in conflict_pairs:
            policy = checked_policy(
                supplied[pair],
                name=f"evidence_conflict_policy[{pair!r}]",
            )
            normalized_policies[deepcopy(pair)] = deepcopy(policy)
    else:
        policy = checked_policy(
            evidence_conflict_policy,
            name="evidence_conflict_policy",
        )
        if not conflict_pairs:
            raise ValueError(
                "evidence_conflict_policy is not applicable because no "
                "direct conflict groups require reduction."
            )
        normalized_policies = {
            deepcopy(pair): deepcopy(policy) for pair in conflict_pairs
        }

    return _EvidenceConflictRequest(
        evidence_by_id=schedule.evidence_by_id,
        evidence_points=tuple(schedule.evidence_points),
        conflict_groups=MappingProxyType(conflict_groups),
        policy_by_pair=MappingProxyType(normalized_policies),
    )


def _resolve_select_evidence(
    request: _EvidenceConflictRequest,
    *,
    selected_evidence: SelectedEvidenceInput | None,
) -> Mapping[tuple[Entity, Epoch], _EvidenceConflictResolution]:
    """Resolve only explicitly selected groups using one owned evidence ID."""

    if not isinstance(request, _EvidenceConflictRequest):
        raise TypeError("request must be a validated evidence-conflict request.")
    selected_pairs = tuple(
        pair
        for pair, policy in request.policy_by_pair.items()
        if policy == "select_evidence"
    )

    if selected_evidence is None:
        if selected_pairs:
            raise ValueError(
                "selected_evidence is required for select_evidence conflict "
                f"groups {list(selected_pairs)!r}."
            )
        return MappingProxyType({})
    if not isinstance(selected_evidence, Mapping):
        raise TypeError("selected_evidence must be a mapping.")

    supplied = dict(selected_evidence.items())
    missing = [pair for pair in selected_pairs if pair not in supplied]
    extra = [pair for pair in supplied if pair not in selected_pairs]
    if missing or extra:
        raise ValueError(
            "selected_evidence keys must exactly match select_evidence conflict "
            f"groups; missing={missing!r}, extra={extra!r}."
        )

    resolutions: dict[
        tuple[Entity, Epoch], _EvidenceConflictResolution
    ] = {}
    for pair in selected_pairs:
        selected_id = supplied[pair]
        if not isinstance(selected_id, str) or not selected_id.strip():
            raise ValueError(
                f"selected_evidence[{pair!r}] must be a nonempty evidence ID."
            )
        group = request.conflict_groups[pair]
        candidates_by_id = {point.evidence_id: point for point in group}
        if selected_id not in candidates_by_id:
            raise ValueError(
                f"selected_evidence[{pair!r}]={selected_id!r} must belong to "
                "that tested-entity/epoch conflict group."
            )
        selected = candidates_by_id[selected_id]
        candidate_ids = tuple(point.evidence_id for point in group)
        candidate_ratios = {
            point.evidence_id: point.observed_ratio for point in group
        }
        resolutions[deepcopy(pair)] = _EvidenceConflictResolution(
            tested_entity=pair[0],
            epoch=pair[1],
            effective_ratio=selected.observed_ratio,
            evidence_conflict_policy="select_evidence",
            ratio_source="selected_from_multiple",
            source_evidence_ids=(selected.evidence_id,),
            provenance={
                "selected_evidence_id": selected.evidence_id,
                "candidate_evidence_ids": candidate_ids,
                "candidate_observed_ratios": candidate_ratios,
            },
        )
    return MappingProxyType(resolutions)


def _resolve_weighted_ratio(
    request: _EvidenceConflictRequest,
    *,
    evidence_weights: EvidenceWeightsInput | None,
    tolerance: float = 1e-9,
) -> Mapping[tuple[Entity, Epoch], _EvidenceConflictResolution]:
    """Combine direct conflict ratios using explicit evidence-only weights."""

    if not isinstance(request, _EvidenceConflictRequest):
        raise TypeError("request must be a validated evidence-conflict request.")
    checked_tolerance = _validate_tolerance(tolerance)
    weighted_pairs = tuple(
        pair
        for pair, policy in request.policy_by_pair.items()
        if policy == "weighted_ratio"
    )

    if evidence_weights is None:
        if weighted_pairs:
            raise ValueError(
                "evidence_weights are required for weighted_ratio conflict "
                f"groups {list(weighted_pairs)!r}."
            )
        return MappingProxyType({})
    if not isinstance(evidence_weights, Mapping):
        raise TypeError("evidence_weights must be a mapping.")

    supplied = dict(evidence_weights.items())
    missing = [pair for pair in weighted_pairs if pair not in supplied]
    extra = [pair for pair in supplied if pair not in weighted_pairs]
    if missing or extra:
        raise ValueError(
            "evidence_weights keys must exactly match weighted_ratio conflict "
            f"groups; missing={missing!r}, extra={extra!r}."
        )

    resolutions: dict[
        tuple[Entity, Epoch], _EvidenceConflictResolution
    ] = {}
    for pair in weighted_pairs:
        group_weights = supplied[pair]
        if not isinstance(group_weights, Mapping):
            raise TypeError(f"evidence_weights[{pair!r}] must be a mapping.")
        supplied_group = dict(group_weights.items())
        group = request.conflict_groups[pair]
        candidate_ids = tuple(point.evidence_id for point in group)
        missing_ids = [
            identifier
            for identifier in candidate_ids
            if identifier not in supplied_group
        ]
        extra_ids = [
            identifier
            for identifier in supplied_group
            if identifier not in candidate_ids
        ]
        if missing_ids or extra_ids:
            raise ValueError(
                f"evidence_weights[{pair!r}] keys must exactly match retained "
                f"evidence IDs; missing={missing_ids!r}, extra={extra_ids!r}."
            )
        checked_weights = _validate_unit_sum_weights(
            tuple(
                (identifier, supplied_group[identifier])
                for identifier in candidate_ids
            ),
            tolerance=checked_tolerance,
            name=f"evidence_weights[{pair!r}] evidence-combination weights",
        )
        weight_by_id = dict(checked_weights)
        ratio_by_id = {
            point.evidence_id: point.observed_ratio for point in group
        }
        weighted_terms = {
            identifier: ratio_by_id[identifier] * weight_by_id[identifier]
            for identifier in candidate_ids
        }
        effective_ratio = sum(weighted_terms.values())
        resolutions[deepcopy(pair)] = _EvidenceConflictResolution(
            tested_entity=pair[0],
            epoch=pair[1],
            effective_ratio=effective_ratio,
            evidence_conflict_policy="weighted_ratio",
            ratio_source="weighted_from_multiple",
            source_evidence_ids=candidate_ids,
            provenance={
                "candidate_evidence_ids": candidate_ids,
                "candidate_observed_ratios": ratio_by_id,
                "evidence_combination_weights": weight_by_id,
                "weighted_ratio_terms": weighted_terms,
            },
        )
    return MappingProxyType(resolutions)


def _resolve_pooled_lift(
    request: _EvidenceConflictRequest,
) -> Mapping[tuple[Entity, Epoch], _EvidenceConflictResolution]:
    """Pool compatible lift numerators and matched attribution denominators."""

    if not isinstance(request, _EvidenceConflictRequest):
        raise TypeError("request must be a validated evidence-conflict request.")
    pooled_pairs = tuple(
        pair
        for pair, policy in request.policy_by_pair.items()
        if policy == "pooled_lift"
    )

    resolutions: dict[
        tuple[Entity, Epoch], _EvidenceConflictResolution
    ] = {}
    for pair in pooled_pairs:
        group = request.conflict_groups[pair]
        candidate_ids = tuple(point.evidence_id for point in group)
        evidence_records = tuple(
            request.evidence_by_id[identifier] for identifier in candidate_ids
        )
        compatibility_fields = (
            "tested_entity",
            "outcome_name",
            "population",
            "unit",
        )
        incompatible = [
            name
            for name in compatibility_fields
            if any(
                getattr(record, name) != getattr(evidence_records[0], name)
                for record in evidence_records[1:]
            )
        ]
        if incompatible:
            raise ValueError(
                f"pooled_lift conflict group {pair!r} requires compatible "
                f"tested entity, outcome, population, and unit metadata; "
                f"incompatible={incompatible!r}."
            )
        if evidence_records[0].tested_entity != pair[0]:
            raise ValueError(
                f"pooled_lift evidence for {pair!r} must belong to its tested "
                "entity."
            )

        observed_lifts = {
            record.test_id: record.observed_lift for record in evidence_records
        }
        matched_estimates = {
            record.test_id: record.matched_mta_estimate
            for record in evidence_records
        }
        observed_ratios = {
            record.test_id: record.calibration_ratio
            for record in evidence_records
        }
        pooled_observed_lift = sum(observed_lifts.values())
        pooled_matched_estimate = sum(matched_estimates.values())
        if pooled_matched_estimate == 0.0:
            raise ValueError(
                f"pooled_lift conflict group {pair!r} has a zero pooled "
                "matched_mta_estimate denominator."
            )
        effective_ratio = pooled_observed_lift / pooled_matched_estimate
        resolutions[deepcopy(pair)] = _EvidenceConflictResolution(
            tested_entity=pair[0],
            epoch=pair[1],
            effective_ratio=effective_ratio,
            evidence_conflict_policy="pooled_lift",
            ratio_source="pooled_from_multiple",
            source_evidence_ids=candidate_ids,
            provenance={
                "candidate_evidence_ids": candidate_ids,
                "candidate_observed_ratios": observed_ratios,
                "observed_lifts": observed_lifts,
                "matched_mta_estimates": matched_estimates,
                "pooled_observed_lift": pooled_observed_lift,
                "pooled_matched_mta_estimate": pooled_matched_estimate,
                "compatibility": {
                    name: getattr(evidence_records[0], name)
                    for name in compatibility_fields
                },
            },
        )
    return MappingProxyType(resolutions)


def _resolve_evidence_conflicts(
    request: _EvidenceConflictRequest,
    *,
    selected_evidence: SelectedEvidenceInput | None = None,
    evidence_weights: EvidenceWeightsInput | None = None,
    tolerance: float = 1e-9,
) -> Mapping[tuple[Entity, Epoch], _EvidenceConflictResolution]:
    """Dispatch every group to its one explicitly selected applied policy."""

    selected = _resolve_select_evidence(
        request,
        selected_evidence=selected_evidence,
    )
    weighted = _resolve_weighted_ratio(
        request,
        evidence_weights=evidence_weights,
        tolerance=tolerance,
    )
    pooled = _resolve_pooled_lift(request)
    by_policy = {
        "select_evidence": selected,
        "weighted_ratio": weighted,
        "pooled_lift": pooled,
    }
    resolutions = {
        deepcopy(pair): by_policy[policy][pair]
        for pair, policy in request.policy_by_pair.items()
    }
    return MappingProxyType(resolutions)


def _build_evidence_conflict_diagnostics(
    request: _EvidenceConflictRequest,
    *,
    diagnostic_policies: Mapping[
        tuple[Entity, Epoch],
        Sequence[EvidenceConflictPolicy],
    ],
    selected_evidence: SelectedEvidenceInput | None = None,
    evidence_weights: EvidenceWeightsInput | None = None,
    tolerance: float = 1e-9,
) -> _EvidenceConflictDiagnostics:
    """Evaluate explicitly requested alternatives without changing application."""

    if not isinstance(request, _EvidenceConflictRequest):
        raise TypeError("request must be a validated evidence-conflict request.")
    if not isinstance(diagnostic_policies, Mapping):
        raise TypeError("diagnostic_policies must be a mapping.")
    supplied = dict(diagnostic_policies.items())
    extra = [pair for pair in supplied if pair not in request.conflict_groups]
    if extra:
        raise ValueError(
            "diagnostic_policies keys must identify direct conflict groups; "
            f"extra={extra!r}."
        )

    requested_by_pair: dict[
        tuple[Entity, Epoch], tuple[EvidenceConflictPolicy, ...]
    ] = {}
    policy_order: tuple[EvidenceConflictPolicy, ...] = (
        "select_evidence",
        "weighted_ratio",
        "pooled_lift",
    )
    pairs_by_policy: dict[EvidenceConflictPolicy, list[tuple[Entity, Epoch]]] = {
        policy: [] for policy in policy_order
    }
    for pair in request.conflict_groups:
        if pair not in supplied:
            continue
        policies = supplied[pair]
        if isinstance(policies, (str, bytes, bytearray)):
            raise TypeError(
                f"diagnostic_policies[{pair!r}] must be a sequence of policies."
            )
        try:
            normalized = tuple(policies)
        except TypeError as exc:
            raise TypeError(
                f"diagnostic_policies[{pair!r}] must be a sequence of policies."
            ) from exc
        if not normalized:
            raise ValueError(
                f"diagnostic_policies[{pair!r}] must not be empty."
            )
        unknown = [
            policy
            for policy in normalized
            if not isinstance(policy, str)
            or policy not in _EVIDENCE_CONFLICT_POLICIES
        ]
        if unknown:
            raise ValueError(
                f"diagnostic_policies[{pair!r}] contains unknown policies "
                f"{unknown!r}."
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError(
                f"diagnostic_policies[{pair!r}] policies must be unique."
            )
        requested_by_pair[deepcopy(pair)] = normalized
        for policy in normalized:
            pairs_by_policy[policy].append(pair)

    resolutions_by_policy: dict[
        EvidenceConflictPolicy,
        Mapping[tuple[Entity, Epoch], _EvidenceConflictResolution],
    ] = {}
    for policy in policy_order:
        diagnostic_request = _EvidenceConflictRequest(
            evidence_by_id=request.evidence_by_id,
            evidence_points=request.evidence_points,
            conflict_groups=request.conflict_groups,
            policy_by_pair=MappingProxyType(
                {
                    deepcopy(pair): policy
                    for pair in pairs_by_policy[policy]
                }
            ),
        )
        if policy == "select_evidence":
            resolutions_by_policy[policy] = _resolve_select_evidence(
                diagnostic_request,
                selected_evidence=selected_evidence,
            )
        elif policy == "weighted_ratio":
            resolutions_by_policy[policy] = _resolve_weighted_ratio(
                diagnostic_request,
                evidence_weights=evidence_weights,
                tolerance=tolerance,
            )
        else:
            resolutions_by_policy[policy] = _resolve_pooled_lift(
                diagnostic_request
            )

    alternatives = {
        deepcopy(pair): MappingProxyType(
            {
                deepcopy(policy): resolutions_by_policy[policy][pair]
                for policy in policies
            }
        )
        for pair, policies in requested_by_pair.items()
    }
    return _EvidenceConflictDiagnostics(
        evidence_points=request.evidence_points,
        conflict_groups=request.conflict_groups,
        applied_policy_by_pair=request.policy_by_pair,
        alternatives_by_pair=MappingProxyType(alternatives),
    )


@dataclass(frozen=True)
class _ConstantSchedule:
    """Explicitly transported observed ratios on declared epoch pairs."""

    coordinate_schedule: tuple[RatioScheduleRow, ...]
    ratios_by_pair: Mapping[tuple[Entity, Epoch], float]
    source_evidence_ids_by_pair: Mapping[
        tuple[Entity, Epoch],
        tuple[EvidenceId, ...],
    ]
    status_by_pair: Mapping[tuple[Entity, Epoch], str]
    provenance_by_pair: Mapping[tuple[Entity, Epoch], Mapping[str, Any]]


def _build_constant_schedule(
    evidence: Sequence[LiftTestEvidence],
    *,
    epochs: Sequence[Epoch],
    source_evidence: SourceEvidenceInput,
    tested_entity_epochs: Mapping[Entity, Sequence[Epoch]],
) -> _ConstantSchedule:
    """Transport only explicitly selected evidence to explicitly named epochs."""

    _validate_ratio_policy_request(
        evidence=evidence,
        ratio_policy="constant",
        epoch_ratio_policy="direct",
        source_evidence=source_evidence,
        tested_entity_epochs=tested_entity_epochs,
    )
    authoritative_epochs = _epoch_tuple(epochs)
    collection = _normalize_evidence_collection(
        evidence,
        allow_multiple_per_entity=True,
    )
    if not isinstance(source_evidence, Mapping):
        raise TypeError("source_evidence must be a mapping.")
    if not isinstance(tested_entity_epochs, Mapping):
        raise TypeError("tested_entity_epochs must be a mapping.")
    if not source_evidence or not tested_entity_epochs:
        raise ValueError(
            "source_evidence and tested_entity_epochs must not be empty."
        )

    source_entities = tuple(source_evidence)
    declared_entities = tuple(tested_entity_epochs)
    missing_sources = [
        entity for entity in declared_entities if entity not in source_evidence
    ]
    extra_sources = [
        entity for entity in source_entities if entity not in tested_entity_epochs
    ]
    if missing_sources or extra_sources:
        raise ValueError(
            "source_evidence and tested_entity_epochs must have exact entity "
            f"coverage; missing_sources={missing_sources!r}, "
            f"extra_sources={extra_sources!r}."
        )

    known_entities = tuple(collection.by_entity)
    unknown_entities = [
        entity for entity in declared_entities if entity not in collection.by_entity
    ]
    if unknown_entities:
        raise ValueError(
            "tested_entity_epochs contains entities without evidence; "
            f"unknown={unknown_entities!r}."
        )
    entity_order = tuple(
        entity for entity in known_entities if entity in tested_entity_epochs
    )

    selected_sources: dict[Entity, LiftTestEvidence] = {}
    declared_epoch_sets: dict[Entity, frozenset[Epoch]] = {}
    for entity in entity_order:
        evidence_id = source_evidence[entity]
        try:
            source = collection.by_id[evidence_id]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"source_evidence[{entity!r}] refers to unknown evidence ID "
                f"{evidence_id!r}."
            ) from exc
        if source.tested_entity != entity:
            raise ValueError(
                f"source evidence {evidence_id!r} belongs to tested entity "
                f"{source.tested_entity!r}, not {entity!r}."
            )
        declared = _epoch_tuple(tested_entity_epochs[entity])
        unknown_epochs = [
            epoch for epoch in declared if epoch not in authoritative_epochs
        ]
        if unknown_epochs:
            raise ValueError(
                f"tested_entity_epochs[{entity!r}] contains unknown epochs "
                f"{unknown_epochs!r}."
            )
        selected_sources[deepcopy(entity)] = source
        declared_epoch_sets[deepcopy(entity)] = frozenset(declared)

    coordinate_schedule = tuple(
        RatioScheduleRow(
            tested_entity=entity,
            coordinate_name=None,
            coordinate_value=None,
            schedule_ratio=selected_sources[entity].calibration_ratio,
            schedule_source="constant",
            source_evidence_ids=(selected_sources[entity].test_id,),
            left_source_evidence_id=None,
            right_source_evidence_id=None,
            inside_observed_support=False,
        )
        for entity in entity_order
    )

    ratios: dict[tuple[Entity, Epoch], float] = {}
    source_ids: dict[tuple[Entity, Epoch], tuple[EvidenceId, ...]] = {}
    statuses: dict[tuple[Entity, Epoch], str] = {}
    provenance: dict[tuple[Entity, Epoch], Mapping[str, Any]] = {}
    for epoch in authoritative_epochs:
        for entity in known_entities:
            pair = (entity, epoch)
            if (
                entity in declared_epoch_sets
                and epoch in declared_epoch_sets[entity]
            ):
                source = selected_sources[entity]
                ratios[deepcopy(pair)] = source.calibration_ratio
                source_ids[deepcopy(pair)] = (source.test_id,)
                statuses[deepcopy(pair)] = "constant"
                provenance[deepcopy(pair)] = _deep_freeze(
                    {
                        "ratio_source": "constant",
                        "source_evidence_id": source.test_id,
                        "transport_assumption": True,
                    }
                )
            else:
                statuses[deepcopy(pair)] = "not_calibrated"
                provenance[deepcopy(pair)] = _deep_freeze(
                    {"ratio_source": "not_calibrated"}
                )

    return _ConstantSchedule(
        coordinate_schedule=coordinate_schedule,
        ratios_by_pair=MappingProxyType(ratios),
        source_evidence_ids_by_pair=MappingProxyType(source_ids),
        status_by_pair=MappingProxyType(statuses),
        provenance_by_pair=MappingProxyType(provenance),
    )


@dataclass(frozen=True)
class _ExplicitRatioSchedule:
    """Caller-supplied final ratios on explicit tested-entity epoch pairs."""

    ratios_by_pair: Mapping[tuple[Entity, Epoch], float]
    status_by_pair: Mapping[tuple[Entity, Epoch], str]
    provenance_by_pair: Mapping[tuple[Entity, Epoch], Mapping[str, Any]]


def _build_explicit_ratio_schedule(
    evidence: Sequence[LiftTestEvidence],
    *,
    epochs: Sequence[Epoch],
    ratio_schedule: RatioScheduleInput,
    tested_entity_epochs: Mapping[Entity, Sequence[Epoch]] | None = None,
    allow_signed: bool = False,
) -> _ExplicitRatioSchedule:
    """Validate and retain exact explicit epoch ratios without filling gaps."""

    if not isinstance(allow_signed, bool):
        raise TypeError("allow_signed must be a bool.")
    _validate_ratio_policy_request(
        evidence=evidence,
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        ratio_schedule=ratio_schedule,
        tested_entity_epochs=tested_entity_epochs,
    )
    authoritative_epochs = _epoch_tuple(epochs)
    collection = _normalize_evidence_collection(
        evidence,
        allow_multiple_per_entity=True,
    )
    raw_pairs = _normalize_pairs(ratio_schedule, name="ratio_schedule")
    shaped_pairs: list[tuple[tuple[Entity, Epoch], Any]] = []
    for key, value in raw_pairs:
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError(
                "ratio_schedule keys must be (tested_entity, epoch) tuples."
            )
        entity, epoch = key
        if entity not in collection.by_entity:
            raise ValueError(
                f"ratio_schedule refers to unknown tested entity {entity!r}."
            )
        if epoch not in authoritative_epochs:
            raise ValueError(
                f"ratio_schedule refers to unknown epoch {epoch!r}."
            )
        shaped_pairs.append(((entity, epoch), value))
    validated_pairs = _validate_numeric_pairs(
        shaped_pairs,
        name="ratio_schedule",
        allow_signed=allow_signed,
    )
    supplied_ratios = dict(validated_pairs)

    if tested_entity_epochs is not None:
        if not isinstance(tested_entity_epochs, Mapping):
            raise TypeError("tested_entity_epochs must be a mapping or None.")
        if not tested_entity_epochs:
            raise ValueError("tested_entity_epochs must not be empty when supplied.")
        declared_pairs: set[tuple[Entity, Epoch]] = set()
        for entity, declared_epochs in tested_entity_epochs.items():
            if entity not in collection.by_entity:
                raise ValueError(
                    "tested_entity_epochs refers to unknown tested entity "
                    f"{entity!r}."
                )
            for epoch in _epoch_tuple(declared_epochs):
                if epoch not in authoritative_epochs:
                    raise ValueError(
                        f"tested_entity_epochs[{entity!r}] contains unknown "
                        f"epoch {epoch!r}."
                    )
                declared_pairs.add((entity, epoch))
        supplied_pair_set = set(supplied_ratios)
        missing = [pair for pair in declared_pairs if pair not in supplied_pair_set]
        extra = [pair for pair in supplied_pair_set if pair not in declared_pairs]
        if missing or extra:
            raise ValueError(
                "ratio_schedule must exactly cover tested_entity_epochs; "
                f"missing={missing!r}, extra={extra!r}."
            )

    ratios: dict[tuple[Entity, Epoch], float] = {}
    statuses: dict[tuple[Entity, Epoch], str] = {}
    provenance: dict[tuple[Entity, Epoch], Mapping[str, Any]] = {}
    entity_order = tuple(collection.by_entity)
    for epoch in authoritative_epochs:
        for entity in entity_order:
            pair = (entity, epoch)
            if pair in supplied_ratios:
                ratios[deepcopy(pair)] = supplied_ratios[pair]
                statuses[deepcopy(pair)] = "explicit"
                provenance[deepcopy(pair)] = _deep_freeze(
                    {
                        "ratio_source": "explicit",
                        "source_evidence_ids": (),
                    }
                )
            else:
                statuses[deepcopy(pair)] = "not_calibrated"
                provenance[deepcopy(pair)] = _deep_freeze(
                    {"ratio_source": "not_calibrated"}
                )

    return _ExplicitRatioSchedule(
        ratios_by_pair=MappingProxyType(ratios),
        status_by_pair=MappingProxyType(statuses),
        provenance_by_pair=MappingProxyType(provenance),
    )


@dataclass(frozen=True)
class _ExplicitStrengthSchedule:
    """Explicit evidence-strength attenuation on declared epoch pairs."""

    ratios_by_pair: Mapping[tuple[Entity, Epoch], float]
    strengths_by_pair: Mapping[tuple[Entity, Epoch], float]
    source_evidence_ids_by_pair: Mapping[
        tuple[Entity, Epoch],
        tuple[EvidenceId, ...],
    ]
    status_by_pair: Mapping[tuple[Entity, Epoch], str]
    provenance_by_pair: Mapping[tuple[Entity, Epoch], Mapping[str, Any]]


def _build_explicit_strength_schedule(
    evidence: Sequence[LiftTestEvidence],
    *,
    epochs: Sequence[Epoch],
    source_evidence: SourceEvidenceInput,
    evidence_strength: EvidenceStrengthInput,
    tested_entity_epochs: Mapping[Entity, Sequence[Epoch]] | None = None,
    allow_signed: bool = False,
) -> _ExplicitStrengthSchedule:
    """Apply explicit attenuation strengths to explicitly selected evidence."""

    if not isinstance(allow_signed, bool):
        raise TypeError("allow_signed must be a bool.")
    _validate_ratio_policy_request(
        evidence=evidence,
        ratio_policy="explicit_strength",
        epoch_ratio_policy="direct",
        source_evidence=source_evidence,
        evidence_strength=evidence_strength,
        tested_entity_epochs=tested_entity_epochs,
    )
    authoritative_epochs = _epoch_tuple(epochs)
    collection = _normalize_evidence_collection(
        evidence,
        allow_multiple_per_entity=True,
    )
    if not isinstance(source_evidence, Mapping):
        raise TypeError("source_evidence must be a mapping.")
    if not source_evidence:
        raise ValueError("source_evidence must not be empty.")

    raw_pairs = _normalize_pairs(evidence_strength, name="evidence_strength")
    shaped_pairs: list[tuple[tuple[Entity, Epoch], Any]] = []
    for key, value in raw_pairs:
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError(
                "evidence_strength keys must be (tested_entity, epoch) tuples."
            )
        entity, epoch = key
        if entity not in collection.by_entity:
            raise ValueError(
                f"evidence_strength refers to unknown tested entity {entity!r}."
            )
        if epoch not in authoritative_epochs:
            raise ValueError(
                f"evidence_strength refers to unknown epoch {epoch!r}."
            )
        shaped_pairs.append(((entity, epoch), value))
    validated_strengths = dict(
        _validate_unit_interval_pairs(
            shaped_pairs,
            name="evidence_strength",
        )
    )

    selected_entities = {
        entity for entity, _ in validated_strengths
    }
    missing_sources = [
        entity for entity in selected_entities if entity not in source_evidence
    ]
    extra_sources = [
        entity for entity in source_evidence if entity not in selected_entities
    ]
    if missing_sources or extra_sources:
        raise ValueError(
            "source_evidence must exactly cover entities in evidence_strength; "
            f"missing={missing_sources!r}, extra={extra_sources!r}."
        )

    entity_order = tuple(
        entity for entity in collection.by_entity if entity in selected_entities
    )
    selected_sources: dict[Entity, LiftTestEvidence] = {}
    for entity in entity_order:
        evidence_id = source_evidence[entity]
        try:
            source = collection.by_id[evidence_id]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"source_evidence[{entity!r}] refers to unknown evidence ID "
                f"{evidence_id!r}."
            ) from exc
        if source.tested_entity != entity:
            raise ValueError(
                f"source evidence {evidence_id!r} belongs to tested entity "
                f"{source.tested_entity!r}, not {entity!r}."
            )
        if not allow_signed and source.calibration_ratio < 0.0:
            raise ValueError(
                f"source evidence {evidence_id!r} is signed; set "
                "allow_signed=True to use it."
            )
        selected_sources[deepcopy(entity)] = source

    if tested_entity_epochs is not None:
        if not isinstance(tested_entity_epochs, Mapping):
            raise TypeError("tested_entity_epochs must be a mapping or None.")
        if not tested_entity_epochs:
            raise ValueError("tested_entity_epochs must not be empty when supplied.")
        declared_pairs: set[tuple[Entity, Epoch]] = set()
        for entity, declared_epochs in tested_entity_epochs.items():
            if entity not in collection.by_entity:
                raise ValueError(
                    "tested_entity_epochs refers to unknown tested entity "
                    f"{entity!r}."
                )
            for epoch in _epoch_tuple(declared_epochs):
                if epoch not in authoritative_epochs:
                    raise ValueError(
                        f"tested_entity_epochs[{entity!r}] contains unknown "
                        f"epoch {epoch!r}."
                    )
                declared_pairs.add((entity, epoch))
        strength_pair_set = set(validated_strengths)
        missing = [pair for pair in declared_pairs if pair not in strength_pair_set]
        extra = [pair for pair in strength_pair_set if pair not in declared_pairs]
        if missing or extra:
            raise ValueError(
                "evidence_strength must exactly cover tested_entity_epochs; "
                f"missing={missing!r}, extra={extra!r}."
            )

    ratios: dict[tuple[Entity, Epoch], float] = {}
    strengths: dict[tuple[Entity, Epoch], float] = {}
    source_ids: dict[tuple[Entity, Epoch], tuple[EvidenceId, ...]] = {}
    statuses: dict[tuple[Entity, Epoch], str] = {}
    provenance: dict[tuple[Entity, Epoch], Mapping[str, Any]] = {}
    all_entity_order = tuple(collection.by_entity)
    for epoch in authoritative_epochs:
        for entity in all_entity_order:
            pair = (entity, epoch)
            if pair in validated_strengths:
                strength = validated_strengths[pair]
                source = selected_sources[entity]
                effective_ratio = 1.0 + strength * (
                    source.calibration_ratio - 1.0
                )
                ratios[deepcopy(pair)] = effective_ratio
                strengths[deepcopy(pair)] = strength
                source_ids[deepcopy(pair)] = (source.test_id,)
                statuses[deepcopy(pair)] = "strength_derived"
                provenance[deepcopy(pair)] = _deep_freeze(
                    {
                        "ratio_source": "strength_derived",
                        "source_evidence_id": source.test_id,
                        "observed_ratio": source.calibration_ratio,
                        "evidence_strength": strength,
                        "effective_ratio": effective_ratio,
                    }
                )
            else:
                statuses[deepcopy(pair)] = "not_calibrated"
                provenance[deepcopy(pair)] = _deep_freeze(
                    {"ratio_source": "not_calibrated"}
                )

    return _ExplicitStrengthSchedule(
        ratios_by_pair=MappingProxyType(ratios),
        strengths_by_pair=MappingProxyType(strengths),
        source_evidence_ids_by_pair=MappingProxyType(source_ids),
        status_by_pair=MappingProxyType(statuses),
        provenance_by_pair=MappingProxyType(provenance),
    )


@dataclass(frozen=True)
class _NormalizedCoordinateAnchor:
    """One evidence anchor with its original and arithmetic coordinates."""

    tested_entity: Entity
    evidence_id: EvidenceId
    coordinate_name: str
    coordinate_value: CoordinateValue
    normalized_coordinate: float
    coordinate_kind: str
    observed_ratio: float


@dataclass(frozen=True)
class _CoordinateAnchorSet:
    """Compatible unsorted anchors grouped in stable tested-entity order."""

    anchors: tuple[_NormalizedCoordinateAnchor, ...]
    by_entity: Mapping[Entity, tuple[_NormalizedCoordinateAnchor, ...]]


def _coordinate_numeric_value(value: CoordinateValue, *, kind: str) -> float:
    """Map a validated coordinate to deterministic interpolation arithmetic."""

    if kind == "numeric":
        return _finite_float(value, name="coordinate_value")
    if kind == "date":
        return float(value.toordinal())
    if kind == "datetime":
        if value.utcoffset() is None:
            origin = datetime(1970, 1, 1)
            return (value - origin).total_seconds()
        origin = datetime(1970, 1, 1, tzinfo=timezone.utc)
        return (value.astimezone(timezone.utc) - origin).total_seconds()
    raise ValueError(f"Unsupported coordinate kind {kind!r}.")


def _normalize_coordinate_linear_anchors(
    evidence: Sequence[LiftTestEvidence],
    *,
    selected_entities: Sequence[Entity] | None = None,
) -> _CoordinateAnchorSet:
    """Normalize compatible anchors without sorting or resolving duplicates."""

    collection = _normalize_evidence_collection(
        evidence,
        allow_multiple_per_entity=True,
    )
    known_entity_order = tuple(collection.by_entity)
    if selected_entities is None:
        entity_order = known_entity_order
    else:
        selected = _entity_tuple(selected_entities, name="selected_entities")
        unknown = [
            entity for entity in selected if entity not in collection.by_entity
        ]
        if unknown:
            raise ValueError(
                f"selected_entities contains entities without evidence {unknown!r}."
            )
        entity_order = tuple(
            entity for entity in known_entity_order if entity in selected
        )

    anchors: list[_NormalizedCoordinateAnchor] = []
    grouped: dict[Entity, tuple[_NormalizedCoordinateAnchor, ...]] = {}
    for entity in entity_order:
        entity_evidence = collection.by_entity[entity]
        if len(entity_evidence) < 2:
            raise ValueError(
                "coordinate_linear requires at least two retained evidence "
                f"points for tested entity {entity!r}."
            )
        coordinate_names = {item.coordinate_name for item in entity_evidence}
        if None in coordinate_names:
            missing = [
                item.test_id
                for item in entity_evidence
                if item.coordinate_name is None or item.coordinate_value is None
            ]
            raise ValueError(
                f"coordinate_linear requires explicit coordinates for {entity!r}; "
                f"missing={missing!r}."
            )
        if len(coordinate_names) != 1:
            raise ValueError(
                f"coordinate_linear evidence for {entity!r} must use one common "
                f"coordinate_name; observed={sorted(coordinate_names)!r}."
            )
        coordinate_name = next(iter(coordinate_names))

        kinds = {
            _coordinate_kind(
                item.coordinate_value,
                name=f"evidence {item.test_id!r} coordinate_value",
            )
            for item in entity_evidence
        }
        if len(kinds) != 1:
            raise TypeError(
                f"coordinate_linear evidence for {entity!r} must use one "
                f"compatible coordinate type; observed={sorted(kinds)!r}."
            )
        kind = next(iter(kinds))
        if kind == "datetime":
            awareness = {
                item.coordinate_value.utcoffset() is not None
                for item in entity_evidence
            }
            if len(awareness) != 1:
                raise TypeError(
                    f"coordinate_linear datetime evidence for {entity!r} cannot "
                    "mix timezone-aware and timezone-naive values."
                )

        entity_anchors: list[_NormalizedCoordinateAnchor] = []
        seen_coordinates: dict[float, EvidenceId] = {}
        for item in entity_evidence:
            normalized = _coordinate_numeric_value(
                item.coordinate_value,
                kind=kind,
            )
            if normalized in seen_coordinates:
                raise ValueError(
                    f"coordinate_linear evidence for {entity!r} has duplicate "
                    f"coordinate {item.coordinate_value!r} in evidence "
                    f"{seen_coordinates[normalized]!r} and {item.test_id!r}; "
                    "resolve it through the explicit conflict layer."
                )
            seen_coordinates[normalized] = item.test_id
            anchor = _NormalizedCoordinateAnchor(
                tested_entity=deepcopy(entity),
                evidence_id=deepcopy(item.test_id),
                coordinate_name=deepcopy(coordinate_name),
                coordinate_value=deepcopy(item.coordinate_value),
                normalized_coordinate=normalized,
                coordinate_kind=kind,
                observed_ratio=item.calibration_ratio,
            )
            entity_anchors.append(anchor)
            anchors.append(anchor)
        grouped[deepcopy(entity)] = tuple(entity_anchors)

    return _CoordinateAnchorSet(
        anchors=tuple(anchors),
        by_entity=MappingProxyType(grouped),
    )


@dataclass(frozen=True)
class _CoordinateLinearSchedule:
    """Continuous piecewise-linear schedule stored as observed anchors."""

    anchor_rows: tuple[RatioScheduleRow, ...]
    normalized_anchors_by_entity: Mapping[
        Entity,
        tuple[_NormalizedCoordinateAnchor, ...],
    ]


def _build_coordinate_linear_schedule(
    evidence: Sequence[LiftTestEvidence],
    *,
    selected_entities: Sequence[Entity] | None = None,
    allow_signed: bool = False,
) -> _CoordinateLinearSchedule:
    """Sort compatible evidence anchors without expanding a dense grid."""

    if not isinstance(allow_signed, bool):
        raise TypeError("allow_signed must be a bool.")
    normalized = _normalize_coordinate_linear_anchors(
        evidence,
        selected_entities=selected_entities,
    )
    sorted_by_entity: dict[
        Entity,
        tuple[_NormalizedCoordinateAnchor, ...],
    ] = {}
    rows: list[RatioScheduleRow] = []
    for entity, entity_anchors in normalized.by_entity.items():
        ordered = tuple(
            sorted(
                entity_anchors,
                key=lambda anchor: anchor.normalized_coordinate,
            )
        )
        signed_ids = [
            anchor.evidence_id
            for anchor in ordered
            if anchor.observed_ratio < 0.0
        ]
        if signed_ids and not allow_signed:
            raise ValueError(
                "coordinate_linear signed evidence requires allow_signed=True; "
                f"evidence_ids={signed_ids!r}."
            )
        sorted_by_entity[deepcopy(entity)] = ordered
        rows.extend(
            RatioScheduleRow(
                tested_entity=anchor.tested_entity,
                coordinate_name=anchor.coordinate_name,
                coordinate_value=anchor.coordinate_value,
                schedule_ratio=anchor.observed_ratio,
                schedule_source="observed",
                source_evidence_ids=(anchor.evidence_id,),
                left_source_evidence_id=None,
                right_source_evidence_id=None,
                inside_observed_support=True,
            )
            for anchor in ordered
        )
    return _CoordinateLinearSchedule(
        anchor_rows=tuple(rows),
        normalized_anchors_by_entity=MappingProxyType(sorted_by_entity),
    )


def _evaluate_coordinate_linear_schedule(
    schedule: _CoordinateLinearSchedule,
    *,
    tested_entity: Entity,
    coordinate_value: CoordinateValue,
) -> RatioScheduleRow:
    """Evaluate at an anchor/interior coordinate or emit unsupported."""

    if not isinstance(schedule, _CoordinateLinearSchedule):
        raise TypeError("schedule must be a _CoordinateLinearSchedule.")
    try:
        anchors = schedule.normalized_anchors_by_entity[tested_entity]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"coordinate_linear schedule has no tested entity {tested_entity!r}."
        ) from exc
    first = anchors[0]
    coordinate_kind = _coordinate_kind(
        coordinate_value,
        name="coordinate_value",
    )
    if coordinate_kind != first.coordinate_kind:
        raise TypeError(
            f"coordinate_value for {tested_entity!r} must use coordinate type "
            f"{first.coordinate_kind!r}."
        )
    if coordinate_kind == "datetime":
        expected_aware = first.coordinate_value.utcoffset() is not None
        observed_aware = coordinate_value.utcoffset() is not None
        if expected_aware != observed_aware:
            raise TypeError(
                "coordinate_value cannot mix timezone-aware and timezone-naive "
                "datetimes with the schedule."
            )
    normalized = _coordinate_numeric_value(
        coordinate_value,
        kind=coordinate_kind,
    )

    for anchor in anchors:
        if normalized == anchor.normalized_coordinate:
            return RatioScheduleRow(
                tested_entity=anchor.tested_entity,
                coordinate_name=anchor.coordinate_name,
                coordinate_value=deepcopy(coordinate_value),
                schedule_ratio=anchor.observed_ratio,
                schedule_source="observed",
                source_evidence_ids=(anchor.evidence_id,),
                left_source_evidence_id=None,
                right_source_evidence_id=None,
                inside_observed_support=True,
            )

    if (
        normalized < anchors[0].normalized_coordinate
        or normalized > anchors[-1].normalized_coordinate
    ):
        return RatioScheduleRow(
            tested_entity=deepcopy(tested_entity),
            coordinate_name=first.coordinate_name,
            coordinate_value=deepcopy(coordinate_value),
            schedule_ratio=None,
            schedule_source="unsupported",
            source_evidence_ids=(),
            left_source_evidence_id=None,
            right_source_evidence_id=None,
            inside_observed_support=False,
        )

    for left, right in zip(anchors, anchors[1:]):
        if left.normalized_coordinate < normalized < right.normalized_coordinate:
            offset = normalized - left.normalized_coordinate
            span = right.normalized_coordinate - left.normalized_coordinate
            if isfinite(offset) and isfinite(span):
                position = offset / span
            else:
                scale = max(
                    abs(left.normalized_coordinate),
                    abs(normalized),
                    abs(right.normalized_coordinate),
                    1.0,
                )
                scaled_left = left.normalized_coordinate / scale
                scaled_coordinate = normalized / scale
                scaled_right = right.normalized_coordinate / scale
                position = (scaled_coordinate - scaled_left) / (
                    scaled_right - scaled_left
                )
            ratio = (
                (1.0 - position) * left.observed_ratio
                + position * right.observed_ratio
            )
            if not isfinite(ratio):
                raise ValueError(
                    "coordinate_linear interpolation produced a nonfinite ratio."
                )
            lower = min(left.observed_ratio, right.observed_ratio)
            upper = max(left.observed_ratio, right.observed_ratio)
            if ratio < lower or ratio > upper:
                raise ValueError(
                    "coordinate_linear interpolation violated endpoint containment."
                )
            return RatioScheduleRow(
                tested_entity=deepcopy(tested_entity),
                coordinate_name=first.coordinate_name,
                coordinate_value=deepcopy(coordinate_value),
                schedule_ratio=ratio,
                schedule_source="interpolated",
                source_evidence_ids=(left.evidence_id, right.evidence_id),
                left_source_evidence_id=left.evidence_id,
                right_source_evidence_id=right.evidence_id,
                inside_observed_support=True,
            )

    raise RuntimeError("coordinate_linear evaluation could not bracket coordinate.")


def _matched_epoch_for_coordinate(
    coordinate: CoordinateValue | None,
    windows: Sequence[tuple[Epoch, CoordinateValue, CoordinateValue]],
) -> Epoch | None:
    """Locate a coordinate using [start, end), closing only the final endpoint."""

    if coordinate is None:
        return None
    coordinate_kind = _coordinate_kind(coordinate, name="coordinate_value")
    window_kind = _coordinate_kind(windows[0][1], name="epoch_windows start")
    if coordinate_kind != window_kind:
        raise TypeError(
            "coordinate_value and epoch_windows must use compatible coordinate types."
        )

    final_index = len(windows) - 1
    for index, (epoch, start, end) in enumerate(windows):
        try:
            inside = start <= coordinate < end
            on_final_endpoint = index == final_index and coordinate == end
        except TypeError as exc:
            raise TypeError(
                "coordinate_value and epoch_windows must be mutually comparable."
            ) from exc
        if inside or on_final_endpoint:
            return deepcopy(epoch)
    return None


def _normalize_evidence_points(
    evidence: Sequence[LiftTestEvidence],
    *,
    epoch_windows: EpochWindowsInput | None = None,
) -> tuple[EvidencePointRecord, ...]:
    """Normalize every observation without selecting or combining evidence."""

    normalized_evidence = _normalize_evidence_collection(
        evidence, allow_multiple_per_entity=True
    ).records
    windows = (
        None
        if epoch_windows is None
        else _normalize_epoch_windows_for_matching(epoch_windows)
    )

    records = tuple(
        EvidencePointRecord(
            tested_entity=item.tested_entity,
            evidence_id=item.test_id,
            observed_ratio=item.calibration_ratio,
            coordinate_name=item.coordinate_name,
            coordinate_value=item.coordinate_value,
            window_start=item.window_start,
            window_end=item.window_end,
            matched_epoch=(
                None
                if windows is None
                else _matched_epoch_for_coordinate(item.coordinate_value, windows)
            ),
            conflict_group=None,
            metadata=item.metadata,
        )
        for item in normalized_evidence
    )

    groups = _conflict_groups_by_epoch(records)
    conflict_ids_by_key = {
        key: tuple(record.evidence_id for record in group)
        for key, group in groups.items()
    }
    return tuple(
        replace(
            record,
            conflict_group=conflict_ids_by_key.get(
                (record.tested_entity, record.matched_epoch)
            ),
        )
        if record.matched_epoch is not None
        else record
        for record in records
    )


_SCHEDULE_SOURCES = frozenset(
    {
        "observed",
        "explicit",
        "strength_derived",
        "constant",
        "interpolated",
        "unsupported",
    }
)

_RATIO_POLICIES = frozenset(
    {
        "observed_only",
        "constant",
        "explicit_ratio",
        "explicit_strength",
        "coordinate_linear",
    }
)

_EPOCH_RATIO_POLICIES = frozenset(
    {
        "direct",
        "representative_coordinate",
        "coordinate_average",
        "support_weighted_average",
    }
)

_EVIDENCE_CONFLICT_POLICIES = frozenset(
    {"select_evidence", "weighted_ratio", "pooled_lift"}
)

_RESIDUAL_POLICIES = frozenset(
    {
        "allow_total_change",
        "proportional_untested",
        "custom_weights",
        "residual_bucket",
        "explicit_adjustments",
    }
)

_CALIBRATION_ROLES = frozenset(
    {
        "tested",
        "residual_recipient",
        "untouched",
        "residual_bucket",
        "tested_and_residual_recipient",
    }
)


def _evidence_id_tuple(value: Any, *, name: str) -> tuple[EvidenceId, ...]:
    """Copy a sequence of unique nonempty evidence identifiers."""

    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence of evidence IDs.")
    try:
        identifiers = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of evidence IDs.") from exc
    if any(
        not isinstance(identifier, str) or not identifier.strip()
        for identifier in identifiers
    ):
        raise ValueError(f"{name} entries must be nonempty strings.")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{name} entries must be unique.")
    return tuple(deepcopy(identifier) for identifier in identifiers)


@dataclass(frozen=True)
class RatioScheduleRow:
    """One immutable schedule value before attribution-epoch reduction."""

    tested_entity: Entity
    coordinate_name: str | None
    coordinate_value: CoordinateValue | None
    schedule_ratio: float | None
    schedule_source: str
    source_evidence_ids: tuple[EvidenceId, ...]
    left_source_evidence_id: EvidenceId | None
    right_source_evidence_id: EvidenceId | None
    inside_observed_support: bool

    def __post_init__(self) -> None:
        if self.tested_entity is None:
            raise ValueError("tested_entity is required.")
        try:
            hash(self.tested_entity)
        except TypeError as exc:
            raise TypeError("tested_entity must be hashable.") from exc

        if (self.coordinate_name is None) != (self.coordinate_value is None):
            raise ValueError(
                "coordinate_name and coordinate_value must be supplied together."
            )
        if self.coordinate_name is not None:
            if (
                not isinstance(self.coordinate_name, str)
                or not self.coordinate_name.strip()
            ):
                raise ValueError("coordinate_name must be a nonempty string.")
            _coordinate_kind(self.coordinate_value, name="coordinate_value")

        if not isinstance(self.schedule_source, str):
            raise TypeError("schedule_source must be a string.")
        if self.schedule_source not in _SCHEDULE_SOURCES:
            raise ValueError(
                "schedule_source must be one of "
                f"{sorted(_SCHEDULE_SOURCES)!r}."
            )
        if not isinstance(self.inside_observed_support, bool):
            raise TypeError("inside_observed_support must be bool.")

        source_ids = _evidence_id_tuple(
            self.source_evidence_ids, name="source_evidence_ids"
        )
        for name in ("left_source_evidence_id", "right_source_evidence_id"):
            identifier = getattr(self, name)
            if identifier is not None and (
                not isinstance(identifier, str) or not identifier.strip()
            ):
                raise ValueError(f"{name} must be a nonempty string or None.")

        source = self.schedule_source
        ratio = self.schedule_ratio
        if source == "unsupported":
            if ratio is not None:
                raise ValueError("unsupported schedule rows cannot receive a ratio.")
            if self.inside_observed_support:
                raise ValueError(
                    "unsupported schedule rows must be outside observed support."
                )
            if self.coordinate_value is None:
                raise ValueError(
                    "unsupported schedule rows require an explicit coordinate."
                )
        else:
            if ratio is None:
                raise ValueError(
                    f"{source} schedule rows require a schedule_ratio."
                )
            ratio = _finite_float(ratio, name="schedule_ratio")

        expected_source_count = {
            "observed": 1,
            "explicit": 0,
            "strength_derived": 1,
            "constant": 1,
            "interpolated": 2,
            "unsupported": 0,
        }[source]
        if len(source_ids) != expected_source_count:
            raise ValueError(
                f"{source} schedule rows require exactly "
                f"{expected_source_count} source_evidence_ids."
            )

        left_id = self.left_source_evidence_id
        right_id = self.right_source_evidence_id
        if source == "interpolated":
            if self.coordinate_value is None:
                raise ValueError(
                    "interpolated schedule rows require an explicit coordinate."
                )
            if not self.inside_observed_support:
                raise ValueError(
                    "interpolated schedule rows must be inside observed support."
                )
            if left_id is None or right_id is None:
                raise ValueError(
                    "interpolated schedule rows require both source anchor IDs."
                )
            if left_id == right_id:
                raise ValueError("interpolation anchor IDs must be distinct.")
            if source_ids != (left_id, right_id):
                raise ValueError(
                    "interpolated source_evidence_ids must equal the ordered "
                    "left and right anchor IDs."
                )
        elif left_id is not None or right_id is not None:
            raise ValueError(
                "source anchor IDs are valid only for interpolated schedule rows."
            )

        if source == "observed":
            if self.coordinate_value is None:
                raise ValueError(
                    "observed schedule rows require an explicit coordinate."
                )
            if not self.inside_observed_support:
                raise ValueError(
                    "observed schedule rows must be inside observed support."
                )

        object.__setattr__(self, "tested_entity", deepcopy(self.tested_entity))
        object.__setattr__(self, "coordinate_name", deepcopy(self.coordinate_name))
        object.__setattr__(self, "coordinate_value", deepcopy(self.coordinate_value))
        object.__setattr__(self, "schedule_ratio", ratio)
        object.__setattr__(self, "schedule_source", deepcopy(source))
        object.__setattr__(self, "source_evidence_ids", source_ids)
        object.__setattr__(self, "left_source_evidence_id", deepcopy(left_id))
        object.__setattr__(self, "right_source_evidence_id", deepcopy(right_id))


@dataclass(frozen=True)
class EpochRatioRow:
    """One immutable final ratio for a tested-entity--epoch pair."""

    tested_entity: Entity
    epoch: Epoch
    epoch_start: CoordinateValue | None
    epoch_end: CoordinateValue | None
    epoch_coordinate: CoordinateValue | None
    epoch_support: float | None
    effective_ratio: float
    ratio_policy: RatioPolicy
    epoch_ratio_policy: EpochRatioPolicy
    ratio_source: str
    evidence_conflict_policy: EvidenceConflictPolicy | None
    source_evidence_ids: tuple[EvidenceId, ...]
    evidence_weights: Mapping[EvidenceId, float] | None
    left_source_evidence_id: EvidenceId | None
    right_source_evidence_id: EvidenceId | None
    inside_observed_support: bool
    ratio_bound_lower: float | None
    ratio_bound_upper: float | None

    def __post_init__(self) -> None:
        if self.tested_entity is None:
            raise ValueError("tested_entity is required.")
        if self.epoch is None:
            raise ValueError("epoch is required.")
        for name in ("tested_entity", "epoch"):
            try:
                hash(getattr(self, name))
            except TypeError as exc:
                raise TypeError(f"{name} must be hashable.") from exc

        if (self.epoch_start is None) != (self.epoch_end is None):
            raise ValueError(
                "epoch_start and epoch_end must be supplied together."
            )
        boundary_kind: str | None = None
        if self.epoch_start is not None:
            start_kind = _coordinate_kind(self.epoch_start, name="epoch_start")
            end_kind = _coordinate_kind(self.epoch_end, name="epoch_end")
            if start_kind != end_kind:
                raise TypeError(
                    "epoch_start and epoch_end must use compatible coordinate types."
                )
            try:
                nonempty = self.epoch_start < self.epoch_end
            except TypeError as exc:
                raise TypeError(
                    "epoch_start and epoch_end must be mutually comparable."
                ) from exc
            if not nonempty:
                raise ValueError("epoch boundaries must satisfy start < end.")
            boundary_kind = start_kind

        if self.epoch_coordinate is not None:
            coordinate_kind = _coordinate_kind(
                self.epoch_coordinate, name="epoch_coordinate"
            )
            if boundary_kind is not None and coordinate_kind != boundary_kind:
                raise TypeError(
                    "epoch_coordinate and epoch boundaries must use compatible "
                    "coordinate types."
                )

        epoch_support = self.epoch_support
        if epoch_support is not None:
            epoch_support = _finite_float(epoch_support, name="epoch_support")
            if epoch_support < 0.0:
                raise ValueError("epoch_support must be nonnegative.")
        effective_ratio = _finite_float(
            self.effective_ratio, name="effective_ratio"
        )

        if self.ratio_policy not in _RATIO_POLICIES:
            raise ValueError(
                f"ratio_policy must be one of {sorted(_RATIO_POLICIES)!r}."
            )
        if self.epoch_ratio_policy not in _EPOCH_RATIO_POLICIES:
            raise ValueError(
                "epoch_ratio_policy must be one of "
                f"{sorted(_EPOCH_RATIO_POLICIES)!r}."
            )
        if not isinstance(self.ratio_source, str) or not self.ratio_source.strip():
            raise ValueError("ratio_source must be a nonempty string.")
        conflict_policy = self.evidence_conflict_policy
        if (
            conflict_policy is not None
            and conflict_policy not in _EVIDENCE_CONFLICT_POLICIES
        ):
            raise ValueError(
                "evidence_conflict_policy must be None or one of "
                f"{sorted(_EVIDENCE_CONFLICT_POLICIES)!r}."
            )

        source_ids = _evidence_id_tuple(
            self.source_evidence_ids, name="source_evidence_ids"
        )
        evidence_weights = self.evidence_weights
        if conflict_policy == "weighted_ratio":
            if not isinstance(evidence_weights, Mapping):
                raise ValueError(
                    "weighted_ratio provenance requires evidence_weights."
                )
            weight_pairs = _validate_unit_sum_weights(
                tuple(evidence_weights.items()),
                tolerance=1e-9,
                name="evidence_weights",
            )
            weight_keys = tuple(key for key, _ in weight_pairs)
            if set(weight_keys) != set(source_ids):
                raise ValueError(
                    "evidence_weights keys must exactly match source_evidence_ids."
                )
            evidence_weights = MappingProxyType(dict(weight_pairs))
        elif evidence_weights is not None:
            raise ValueError(
                "evidence_weights are valid only with weighted_ratio provenance."
            )

        left_id = self.left_source_evidence_id
        right_id = self.right_source_evidence_id
        for name, identifier in (
            ("left_source_evidence_id", left_id),
            ("right_source_evidence_id", right_id),
        ):
            if identifier is not None and (
                not isinstance(identifier, str) or not identifier.strip()
            ):
                raise ValueError(f"{name} must be a nonempty string or None.")
        if (left_id is None) != (right_id is None):
            raise ValueError(
                "left and right source evidence IDs must be supplied together."
            )
        if left_id is not None:
            if left_id == right_id:
                raise ValueError("left and right source evidence IDs must differ.")
            if left_id not in source_ids or right_id not in source_ids:
                raise ValueError(
                    "source anchor IDs must be present in source_evidence_ids."
                )

        if not isinstance(self.inside_observed_support, bool):
            raise TypeError("inside_observed_support must be bool.")

        if (self.ratio_bound_lower is None) != (self.ratio_bound_upper is None):
            raise ValueError(
                "ratio_bound_lower and ratio_bound_upper must be supplied together."
            )
        bounds = (
            None
            if self.ratio_bound_lower is None
            else _validate_ratio_bounds(
                (self.ratio_bound_lower, self.ratio_bound_upper)
            )
        )
        if bounds is not None:
            lower, upper = bounds
            if effective_ratio < lower or effective_ratio > upper:
                raise ValueError(
                    f"effective_ratio {effective_ratio} is outside ratio bounds "
                    f"[{lower}, {upper}]; bounds raise and never clip."
                )
        else:
            lower = upper = None

        object.__setattr__(self, "tested_entity", deepcopy(self.tested_entity))
        object.__setattr__(self, "epoch", deepcopy(self.epoch))
        object.__setattr__(self, "epoch_start", deepcopy(self.epoch_start))
        object.__setattr__(self, "epoch_end", deepcopy(self.epoch_end))
        object.__setattr__(self, "epoch_coordinate", deepcopy(self.epoch_coordinate))
        object.__setattr__(self, "epoch_support", epoch_support)
        object.__setattr__(self, "effective_ratio", effective_ratio)
        object.__setattr__(self, "ratio_policy", deepcopy(self.ratio_policy))
        object.__setattr__(
            self, "epoch_ratio_policy", deepcopy(self.epoch_ratio_policy)
        )
        object.__setattr__(self, "ratio_source", deepcopy(self.ratio_source))
        object.__setattr__(
            self, "evidence_conflict_policy", deepcopy(conflict_policy)
        )
        object.__setattr__(self, "source_evidence_ids", source_ids)
        object.__setattr__(self, "evidence_weights", evidence_weights)
        object.__setattr__(self, "left_source_evidence_id", deepcopy(left_id))
        object.__setattr__(self, "right_source_evidence_id", deepcopy(right_id))
        object.__setattr__(self, "ratio_bound_lower", lower)
        object.__setattr__(self, "ratio_bound_upper", upper)


def _validate_epoch_ratio_rows(
    rows: Sequence[EpochRatioRow],
    *,
    expected_pairs: Sequence[tuple[Entity, Epoch]] | None = None,
) -> tuple[EpochRatioRow, ...]:
    """Require one and only one final row for every calibrated pair."""

    if isinstance(rows, (str, bytes, bytearray)):
        raise TypeError("epoch ratio rows must be a sequence.")
    try:
        normalized = tuple(rows)
    except TypeError as exc:
        raise TypeError("epoch ratio rows must be a sequence.") from exc

    observed: dict[tuple[Entity, Epoch], EpochRatioRow] = {}
    for index, row in enumerate(normalized):
        if not isinstance(row, EpochRatioRow):
            raise TypeError(
                f"epoch ratio rows[{index}] must be an EpochRatioRow."
            )
        key = (row.tested_entity, row.epoch)
        if key in observed:
            raise ValueError(
                "epoch ratio rows contain multiple rows for calibrated pair "
                f"{key!r}."
            )
        observed[key] = row

    if expected_pairs is not None:
        if isinstance(expected_pairs, (str, bytes, bytearray)):
            raise TypeError("expected_pairs must be a sequence of pairs.")
        expected = tuple(expected_pairs)
        if len(set(expected)) != len(expected):
            raise ValueError("expected_pairs must be unique.")
        missing = [pair for pair in expected if pair not in observed]
        extra = [pair for pair in observed if pair not in expected]
        if missing or extra:
            raise ValueError(
                "epoch ratio rows must cover every calibrated pair exactly once; "
                f"missing={missing!r}, extra={extra!r}."
            )
    return normalized


@dataclass(frozen=True)
class _EpochRatioReductionRequest:
    """Validated authoritative inputs for one explicit epoch reduction rule."""

    ratio_policy: RatioPolicy
    epoch_ratio_policy: EpochRatioPolicy
    epochs: tuple[Epoch, ...]
    calibrated_pairs: tuple[tuple[Entity, Epoch], ...]
    epoch_windows: Mapping[
        Epoch, tuple[CoordinateValue, CoordinateValue]
    ] | None
    epoch_coordinates: Mapping[Epoch, CoordinateValue] | None
    epoch_support: Mapping[Epoch, float] | None
    epoch_coordinate_weights: Mapping[
        Epoch, tuple[tuple[CoordinateValue, float], ...]
    ] | None


def _normalize_calibrated_pairs(
    value: Sequence[tuple[Entity, Epoch]],
    *,
    epochs: Sequence[Epoch],
) -> tuple[tuple[Entity, Epoch], ...]:
    """Copy unique pairs and require membership in the authoritative epochs."""

    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("calibrated_pairs must be a sequence of entity/epoch pairs.")
    try:
        pairs = tuple(value)
    except TypeError as exc:
        raise TypeError(
            "calibrated_pairs must be a sequence of entity/epoch pairs."
        ) from exc
    authoritative_epochs = tuple(epochs)
    seen: set[tuple[Entity, Epoch]] = set()
    copied: list[tuple[Entity, Epoch]] = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise TypeError(
                f"calibrated_pairs[{index}] must be a two-item entity/epoch pair."
            )
        entity, epoch = pair
        try:
            key = (entity, epoch)
            duplicate = key in seen
        except TypeError as exc:
            raise TypeError(
                f"calibrated_pairs[{index}] entity and epoch must be hashable."
            ) from exc
        if epoch not in authoritative_epochs:
            raise ValueError(
                f"calibrated_pairs[{index}] uses unknown epoch {epoch!r}."
            )
        if duplicate:
            raise ValueError(f"calibrated_pairs contains duplicate pair {key!r}.")
        seen.add(key)
        copied.append((deepcopy(entity), deepcopy(epoch)))
    return tuple(copied)


def _validate_epoch_ratio_reduction_request(
    *,
    ratio_policy: RatioPolicy,
    epoch_ratio_policy: EpochRatioPolicy,
    epochs: Sequence[Epoch],
    calibrated_pairs: Sequence[tuple[Entity, Epoch]],
    epoch_windows: EpochWindowsInput | None = None,
    epoch_coordinates: EpochCoordinatesInput | None = None,
    epoch_support: EpochSupportInput | None = None,
    epoch_coordinate_weights: EpochCoordinateWeightsInput | None = None,
) -> _EpochRatioReductionRequest:
    """Validate dispatch compatibility and all reduction metadata up front."""

    if ratio_policy not in _RATIO_POLICIES:
        raise ValueError(
            f"ratio_policy must be one of {sorted(_RATIO_POLICIES)!r}."
        )
    if epoch_ratio_policy not in _EPOCH_RATIO_POLICIES:
        raise ValueError(
            "epoch_ratio_policy must be one of "
            f"{sorted(_EPOCH_RATIO_POLICIES)!r}."
        )
    direct_policies = {
        "observed_only",
        "constant",
        "explicit_ratio",
        "explicit_strength",
    }
    if ratio_policy in direct_policies and epoch_ratio_policy != "direct":
        raise ValueError(
            f"ratio_policy={ratio_policy!r} requires epoch_ratio_policy='direct'."
        )
    if ratio_policy == "coordinate_linear" and epoch_ratio_policy == "direct":
        raise ValueError(
            "ratio_policy='coordinate_linear' is incompatible with "
            "epoch_ratio_policy='direct'."
        )

    authoritative_epochs = _epoch_tuple(epochs)
    normalized_pairs = _normalize_calibrated_pairs(
        calibrated_pairs,
        epochs=authoritative_epochs,
    )
    affected_epoch_set = {epoch for _, epoch in normalized_pairs}
    affected_epochs = tuple(
        epoch for epoch in authoritative_epochs if epoch in affected_epoch_set
    )

    if ratio_policy in {"observed_only", "coordinate_linear"}:
        if epoch_windows is None:
            raise ValueError(f"ratio_policy={ratio_policy!r} requires epoch_windows.")
    windows = _normalize_authoritative_epoch_windows(
        epoch_windows,
        epochs=authoritative_epochs,
    )
    support = _normalize_epoch_support(
        epoch_support,
        epochs=authoritative_epochs,
    )

    coordinates: Mapping[Epoch, CoordinateValue] | None = None
    coordinate_weights: Mapping[
        Epoch, tuple[tuple[CoordinateValue, float], ...]
    ] | None = None
    if epoch_ratio_policy == "direct":
        incompatible = []
        if epoch_coordinates is not None:
            incompatible.append("epoch_coordinates")
        if epoch_coordinate_weights is not None:
            incompatible.append("epoch_coordinate_weights")
        if incompatible:
            raise ValueError(
                "epoch_ratio_policy='direct' does not accept inputs "
                f"{incompatible!r}."
            )
    elif epoch_ratio_policy == "representative_coordinate":
        if epoch_coordinates is None:
            raise ValueError(
                "epoch_ratio_policy='representative_coordinate' requires "
                "epoch_coordinates."
            )
        if epoch_coordinate_weights is not None:
            raise ValueError(
                "epoch_coordinate_weights are incompatible with "
                "epoch_ratio_policy='representative_coordinate'."
            )
        coordinates = _normalize_epoch_coordinates(
            epoch_coordinates,
            epochs=authoritative_epochs,
            epoch_windows=windows,
            required_epochs=affected_epochs,
        )
    elif epoch_ratio_policy == "coordinate_average":
        incompatible = []
        if epoch_coordinates is not None:
            incompatible.append("epoch_coordinates")
        if epoch_coordinate_weights is not None:
            incompatible.append("epoch_coordinate_weights")
        if incompatible:
            raise ValueError(
                "epoch_ratio_policy='coordinate_average' does not accept "
                f"inputs {incompatible!r}."
            )
    else:
        if epoch_coordinate_weights is None:
            raise ValueError(
                "epoch_ratio_policy='support_weighted_average' requires "
                "epoch_coordinate_weights; scalar epoch_support is not a "
                "substitute."
            )
        if epoch_coordinates is not None:
            raise ValueError(
                "epoch_coordinates are incompatible with "
                "epoch_ratio_policy='support_weighted_average'."
            )
        coordinate_weights = _normalize_epoch_coordinate_weights(
            epoch_coordinate_weights,
            epochs=authoritative_epochs,
            epoch_windows=windows,
            required_epochs=affected_epochs,
        )

    return _EpochRatioReductionRequest(
        ratio_policy=deepcopy(ratio_policy),
        epoch_ratio_policy=deepcopy(epoch_ratio_policy),
        epochs=authoritative_epochs,
        calibrated_pairs=normalized_pairs,
        epoch_windows=windows,
        epoch_coordinates=coordinates,
        epoch_support=support,
        epoch_coordinate_weights=coordinate_weights,
    )


def _validate_epoch_ratio_reduction_output(
    request: _EpochRatioReductionRequest,
    rows: Sequence[EpochRatioRow],
) -> tuple[EpochRatioRow, ...]:
    """Require exact request coverage and matching selected policy fields."""

    if not isinstance(request, _EpochRatioReductionRequest):
        raise TypeError("request must be an epoch-ratio reduction request.")
    normalized = _validate_epoch_ratio_rows(
        rows,
        expected_pairs=request.calibrated_pairs,
    )
    for row in normalized:
        if row.ratio_policy != request.ratio_policy:
            raise ValueError(
                "epoch ratio row ratio_policy must match the reduction request."
            )
        if row.epoch_ratio_policy != request.epoch_ratio_policy:
            raise ValueError(
                "epoch ratio row epoch_ratio_policy must match the reduction "
                "request."
            )
    return normalized


@dataclass(frozen=True)
class _EpochRatioReductionResult:
    """Final immutable rows plus stored reduction and uncovered-pair state."""

    rows: tuple[EpochRatioRow, ...]
    provenance_by_pair: Mapping[tuple[Entity, Epoch], Mapping[str, Any]]
    uncalibrated_pairs: tuple[tuple[Entity, Epoch], ...]
    coordinate_evaluations: tuple[RatioScheduleRow, ...] = ()


def _reduce_direct_epoch_ratios(
    request: _EpochRatioReductionRequest,
    schedule: (
        _ObservedOnlySchedule
        | _ConstantSchedule
        | _ExplicitRatioSchedule
        | _ExplicitStrengthSchedule
    ),
    *,
    conflict_resolutions: (
        Mapping[tuple[Entity, Epoch], _EvidenceConflictResolution] | None
    ) = None,
) -> _EpochRatioReductionResult:
    """Emit unchanged direct ratios and omit every explicitly uncovered pair."""

    if not isinstance(request, _EpochRatioReductionRequest):
        raise TypeError("request must be an epoch-ratio reduction request.")
    if request.epoch_ratio_policy != "direct":
        raise ValueError("direct reduction requires epoch_ratio_policy='direct'.")
    expected_schedule_type = {
        "observed_only": _ObservedOnlySchedule,
        "constant": _ConstantSchedule,
        "explicit_ratio": _ExplicitRatioSchedule,
        "explicit_strength": _ExplicitStrengthSchedule,
    }.get(request.ratio_policy)
    if expected_schedule_type is None or not isinstance(
        schedule, expected_schedule_type
    ):
        raise TypeError(
            f"ratio_policy={request.ratio_policy!r} requires its matching "
            "direct schedule type."
        )

    if isinstance(schedule, _ObservedOnlySchedule):
        actual_pairs = tuple(schedule.candidates)
        conflict_pairs = tuple(
            pair
            for pair, group in schedule.candidates.items()
            if len(group) > 1
        )
    else:
        actual_pairs = tuple(schedule.ratios_by_pair)
        conflict_pairs = ()
    requested_pair_set = set(request.calibrated_pairs)
    actual_pair_set = set(actual_pairs)
    missing = [pair for pair in actual_pairs if pair not in requested_pair_set]
    extra = [
        pair for pair in request.calibrated_pairs if pair not in actual_pair_set
    ]
    if missing or extra:
        raise ValueError(
            "calibrated_pairs must exactly match populated direct schedule pairs; "
            f"missing={missing!r}, extra={extra!r}."
        )

    if conflict_resolutions is None:
        supplied_conflicts: dict[
            tuple[Entity, Epoch], _EvidenceConflictResolution
        ] = {}
    elif isinstance(conflict_resolutions, Mapping):
        supplied_conflicts = dict(conflict_resolutions.items())
    else:
        raise TypeError("conflict_resolutions must be a mapping or None.")
    missing_conflicts = [
        pair for pair in conflict_pairs if pair not in supplied_conflicts
    ]
    extra_conflicts = [
        pair for pair in supplied_conflicts if pair not in conflict_pairs
    ]
    if missing_conflicts or extra_conflicts:
        raise ValueError(
            "conflict_resolutions must exactly cover observed conflict pairs; "
            f"missing={missing_conflicts!r}, extra={extra_conflicts!r}."
        )

    rows: list[EpochRatioRow] = []
    provenance_by_pair: dict[
        tuple[Entity, Epoch], Mapping[str, Any]
    ] = {}
    for pair in actual_pairs:
        entity, epoch = pair
        conflict_policy: EvidenceConflictPolicy | None = None
        evidence_weights: Mapping[EvidenceId, float] | None = None
        inside_observed_support = False
        if isinstance(schedule, _ObservedOnlySchedule):
            group = schedule.candidates[pair]
            inside_observed_support = True
            if len(group) == 1:
                point = group[0]
                effective_ratio = point.observed_ratio
                ratio_source = "observed"
                source_ids = (point.evidence_id,)
                provenance = {
                    "ratio_source": ratio_source,
                    "source_evidence_ids": source_ids,
                    "observed_ratio": effective_ratio,
                    "evidence_conflict_policy": None,
                }
            else:
                resolution = supplied_conflicts[pair]
                if not isinstance(resolution, _EvidenceConflictResolution):
                    raise TypeError(
                        f"conflict_resolutions[{pair!r}] must be an evidence "
                        "conflict resolution."
                    )
                if (
                    resolution.tested_entity != entity
                    or resolution.epoch != epoch
                ):
                    raise ValueError(
                        f"conflict_resolutions[{pair!r}] must match its pair."
                    )
                effective_ratio = resolution.effective_ratio
                ratio_source = resolution.ratio_source
                source_ids = resolution.source_evidence_ids
                conflict_policy = resolution.evidence_conflict_policy
                if conflict_policy == "weighted_ratio":
                    evidence_weights = resolution.provenance[
                        "evidence_combination_weights"
                    ]
                provenance = {
                    "ratio_source": ratio_source,
                    "source_evidence_ids": source_ids,
                    "evidence_conflict_policy": conflict_policy,
                    "conflict_resolution": resolution.provenance,
                }
        else:
            effective_ratio = schedule.ratios_by_pair[pair]
            provenance = dict(schedule.provenance_by_pair[pair])
            ratio_source = provenance["ratio_source"]
            if isinstance(schedule, _ExplicitRatioSchedule):
                source_ids = ()
            else:
                source_ids = schedule.source_evidence_ids_by_pair[pair]
            provenance.update(
                {
                    "source_evidence_ids": source_ids,
                    "evidence_conflict_policy": None,
                }
            )

        window = (
            None
            if request.epoch_windows is None
            else request.epoch_windows[epoch]
        )
        row = EpochRatioRow(
            tested_entity=entity,
            epoch=epoch,
            epoch_start=None if window is None else window[0],
            epoch_end=None if window is None else window[1],
            epoch_coordinate=None,
            epoch_support=(
                None
                if request.epoch_support is None
                else request.epoch_support.get(epoch)
            ),
            effective_ratio=effective_ratio,
            ratio_policy=request.ratio_policy,
            epoch_ratio_policy="direct",
            ratio_source=ratio_source,
            evidence_conflict_policy=conflict_policy,
            source_evidence_ids=source_ids,
            evidence_weights=evidence_weights,
            left_source_evidence_id=None,
            right_source_evidence_id=None,
            inside_observed_support=inside_observed_support,
            ratio_bound_lower=None,
            ratio_bound_upper=None,
        )
        rows.append(row)
        provenance_by_pair[deepcopy(pair)] = _deep_freeze(
            {
                **provenance,
                "epoch_ratio_policy": "direct",
                "epoch_support": row.epoch_support,
                "effective_ratio": row.effective_ratio,
            }
        )

    normalized_rows = _validate_epoch_ratio_reduction_output(request, rows)
    uncalibrated_pairs = tuple(
        deepcopy(pair)
        for pair, status in schedule.status_by_pair.items()
        if status == "not_calibrated"
    )
    return _EpochRatioReductionResult(
        rows=normalized_rows,
        provenance_by_pair=MappingProxyType(provenance_by_pair),
        uncalibrated_pairs=uncalibrated_pairs,
    )


def _reduce_representative_coordinate(
    request: _EpochRatioReductionRequest,
    schedule: _CoordinateLinearSchedule,
) -> _EpochRatioReductionResult:
    """Evaluate one explicit representative coordinate for each affected pair."""

    if not isinstance(request, _EpochRatioReductionRequest):
        raise TypeError("request must be an epoch-ratio reduction request.")
    if (
        request.ratio_policy != "coordinate_linear"
        or request.epoch_ratio_policy != "representative_coordinate"
    ):
        raise ValueError(
            "representative reduction requires ratio_policy='coordinate_linear' "
            "and epoch_ratio_policy='representative_coordinate'."
        )
    if not isinstance(schedule, _CoordinateLinearSchedule):
        raise TypeError("schedule must be a coordinate-linear schedule.")
    if request.epoch_windows is None or request.epoch_coordinates is None:
        raise ValueError(
            "representative reduction requires epoch windows and coordinates."
        )

    entity_order = tuple(schedule.normalized_anchors_by_entity)
    unknown_entities = [
        entity
        for entity, _ in request.calibrated_pairs
        if entity not in schedule.normalized_anchors_by_entity
    ]
    if unknown_entities:
        raise ValueError(
            "representative calibrated pairs contain entities without a "
            f"coordinate schedule; unknown={unknown_entities!r}."
        )
    requested_pair_set = set(request.calibrated_pairs)
    ordered_pairs = tuple(
        (entity, epoch)
        for epoch in request.epochs
        for entity in entity_order
        if (entity, epoch) in requested_pair_set
    )

    rows: list[EpochRatioRow] = []
    coordinate_evaluations: list[RatioScheduleRow] = []
    supported_pairs: list[tuple[Entity, Epoch]] = []
    uncalibrated_pairs: list[tuple[Entity, Epoch]] = []
    provenance_by_pair: dict[
        tuple[Entity, Epoch], Mapping[str, Any]
    ] = {}
    for pair in ordered_pairs:
        entity, epoch = pair
        coordinate = request.epoch_coordinates[epoch]
        evaluated = _evaluate_coordinate_linear_schedule(
            schedule,
            tested_entity=entity,
            coordinate_value=coordinate,
        )
        coordinate_evaluations.append(evaluated)
        provenance = {
            "epoch_ratio_policy": "representative_coordinate",
            "evaluation_rule": "schedule_at_declared_epoch_coordinate",
            "epoch_coordinate": coordinate,
            "ratio_source": evaluated.schedule_source,
            "source_evidence_ids": evaluated.source_evidence_ids,
            "left_source_evidence_id": evaluated.left_source_evidence_id,
            "right_source_evidence_id": evaluated.right_source_evidence_id,
            "inside_observed_support": evaluated.inside_observed_support,
            "epoch_support": (
                None
                if request.epoch_support is None
                else request.epoch_support.get(epoch)
            ),
        }
        if evaluated.schedule_ratio is None:
            uncalibrated_pairs.append(deepcopy(pair))
            provenance_by_pair[deepcopy(pair)] = _deep_freeze(
                {**provenance, "effective_ratio": None}
            )
            continue

        window = request.epoch_windows[epoch]
        row = EpochRatioRow(
            tested_entity=entity,
            epoch=epoch,
            epoch_start=window[0],
            epoch_end=window[1],
            epoch_coordinate=coordinate,
            epoch_support=provenance["epoch_support"],
            effective_ratio=evaluated.schedule_ratio,
            ratio_policy="coordinate_linear",
            epoch_ratio_policy="representative_coordinate",
            ratio_source=evaluated.schedule_source,
            evidence_conflict_policy=None,
            source_evidence_ids=evaluated.source_evidence_ids,
            evidence_weights=None,
            left_source_evidence_id=evaluated.left_source_evidence_id,
            right_source_evidence_id=evaluated.right_source_evidence_id,
            inside_observed_support=evaluated.inside_observed_support,
            ratio_bound_lower=None,
            ratio_bound_upper=None,
        )
        rows.append(row)
        supported_pairs.append(deepcopy(pair))
        provenance_by_pair[deepcopy(pair)] = _deep_freeze(
            {**provenance, "effective_ratio": row.effective_ratio}
        )

    normalized_rows = _validate_epoch_ratio_rows(
        rows,
        expected_pairs=supported_pairs,
    )
    return _EpochRatioReductionResult(
        rows=normalized_rows,
        provenance_by_pair=MappingProxyType(provenance_by_pair),
        uncalibrated_pairs=tuple(uncalibrated_pairs),
        coordinate_evaluations=tuple(coordinate_evaluations),
    )


def _reduce_coordinate_average(
    request: _EpochRatioReductionRequest,
    schedule: _CoordinateLinearSchedule,
) -> _EpochRatioReductionResult:
    """Integrate a piecewise-linear schedule over each complete epoch window."""

    if not isinstance(request, _EpochRatioReductionRequest):
        raise TypeError("request must be an epoch-ratio reduction request.")
    if (
        request.ratio_policy != "coordinate_linear"
        or request.epoch_ratio_policy != "coordinate_average"
    ):
        raise ValueError(
            "coordinate-average reduction requires "
            "ratio_policy='coordinate_linear' and "
            "epoch_ratio_policy='coordinate_average'."
        )
    if not isinstance(schedule, _CoordinateLinearSchedule):
        raise TypeError("schedule must be a coordinate-linear schedule.")
    if request.epoch_windows is None:
        raise ValueError("coordinate-average reduction requires epoch_windows.")

    entity_order = tuple(schedule.normalized_anchors_by_entity)
    unknown_entities = [
        entity
        for entity, _ in request.calibrated_pairs
        if entity not in schedule.normalized_anchors_by_entity
    ]
    if unknown_entities:
        raise ValueError(
            "coordinate-average calibrated pairs contain entities without a "
            f"coordinate schedule; unknown={unknown_entities!r}."
        )
    requested_pair_set = set(request.calibrated_pairs)
    ordered_pairs = tuple(
        (entity, epoch)
        for epoch in request.epochs
        for entity in entity_order
        if (entity, epoch) in requested_pair_set
    )

    rows: list[EpochRatioRow] = []
    coordinate_evaluations: list[RatioScheduleRow] = []
    provenance_by_pair: dict[
        tuple[Entity, Epoch], Mapping[str, Any]
    ] = {}
    for pair in ordered_pairs:
        entity, epoch = pair
        anchors = schedule.normalized_anchors_by_entity[entity]
        kind = anchors[0].coordinate_kind
        start, end = request.epoch_windows[epoch]
        start_kind = _coordinate_kind(start, name="epoch_start")
        end_kind = _coordinate_kind(end, name="epoch_end")
        if start_kind != kind or end_kind != kind:
            raise TypeError(
                f"epoch window for {pair!r} must use schedule coordinate "
                f"type {kind!r}."
            )
        if kind == "datetime":
            expected_aware = anchors[0].coordinate_value.utcoffset() is not None
            if (
                (start.utcoffset() is not None) != expected_aware
                or (end.utcoffset() is not None) != expected_aware
            ):
                raise TypeError(
                    "epoch window cannot mix timezone-aware and timezone-naive "
                    "datetimes with the schedule."
                )
        start_numeric = _coordinate_numeric_value(start, kind=kind)
        end_numeric = _coordinate_numeric_value(end, kind=kind)
        if (
            start_numeric < anchors[0].normalized_coordinate
            or end_numeric > anchors[-1].normalized_coordinate
        ):
            raise ValueError(
                f"coordinate_average for {pair!r} requires the complete epoch "
                "interval inside observed evidence support; partial support is "
                "not averaged."
            )

        breakpoints: list[tuple[float, CoordinateValue]] = [
            (start_numeric, start)
        ]
        breakpoints.extend(
            (anchor.normalized_coordinate, anchor.coordinate_value)
            for anchor in anchors
            if start_numeric
            < anchor.normalized_coordinate
            < end_numeric
        )
        breakpoints.append((end_numeric, end))
        evaluations = [
            _evaluate_coordinate_linear_schedule(
                schedule,
                tested_entity=entity,
                coordinate_value=coordinate,
            )
            for _, coordinate in breakpoints
        ]
        coordinate_evaluations.extend(evaluations)
        if any(row.schedule_ratio is None for row in evaluations):
            raise ValueError(
                f"coordinate_average for {pair!r} encountered incomplete "
                "observed support."
            )

        numeric_points = [numeric for numeric, _ in breakpoints]
        width = end_numeric - start_numeric
        if isfinite(width):
            integration_points = [
                numeric - start_numeric for numeric in numeric_points
            ]
        else:
            scale = max(
                *(abs(numeric) for numeric in numeric_points),
                1.0,
            )
            scaled_start = start_numeric / scale
            integration_points = [
                numeric / scale - scaled_start for numeric in numeric_points
            ]
            width = end_numeric / scale - scaled_start
        if not isfinite(width) or width <= 0.0:
            raise ValueError(
                f"coordinate_average for {pair!r} requires a finite positive "
                "normalized interval width."
            )

        area = 0.0
        segments: list[Mapping[str, Any]] = []
        source_ids: list[EvidenceId] = []
        for evaluation in evaluations:
            for identifier in evaluation.source_evidence_ids:
                if identifier not in source_ids:
                    source_ids.append(identifier)
        for index, (left_row, right_row) in enumerate(
            zip(evaluations, evaluations[1:])
        ):
            segment_width = (
                integration_points[index + 1] - integration_points[index]
            )
            segment_area = segment_width * (
                left_row.schedule_ratio + right_row.schedule_ratio
            ) / 2.0
            area += segment_area
            segments.append(
                {
                    "left_coordinate": breakpoints[index][1],
                    "right_coordinate": breakpoints[index + 1][1],
                    "left_ratio": left_row.schedule_ratio,
                    "right_ratio": right_row.schedule_ratio,
                    "normalized_width": segment_width,
                    "normalized_area": segment_area,
                    "source_evidence_ids": tuple(
                        dict.fromkeys(
                            left_row.source_evidence_ids
                            + right_row.source_evidence_ids
                        )
                    ),
                }
            )
        effective_ratio = area / width
        if not isfinite(effective_ratio):
            raise ValueError(
                f"coordinate_average for {pair!r} produced a nonfinite ratio."
            )
        source_id_tuple = tuple(source_ids)
        left_id = source_id_tuple[0]
        right_id = source_id_tuple[-1]
        if left_id == right_id:
            raise ValueError(
                f"coordinate_average for {pair!r} requires distinct support "
                "anchors."
            )
        epoch_support = (
            None
            if request.epoch_support is None
            else request.epoch_support.get(epoch)
        )
        row = EpochRatioRow(
            tested_entity=entity,
            epoch=epoch,
            epoch_start=start,
            epoch_end=end,
            epoch_coordinate=None,
            epoch_support=epoch_support,
            effective_ratio=effective_ratio,
            ratio_policy="coordinate_linear",
            epoch_ratio_policy="coordinate_average",
            ratio_source="coordinate_average",
            evidence_conflict_policy=None,
            source_evidence_ids=source_id_tuple,
            evidence_weights=None,
            left_source_evidence_id=left_id,
            right_source_evidence_id=right_id,
            inside_observed_support=True,
            ratio_bound_lower=None,
            ratio_bound_upper=None,
        )
        rows.append(row)
        provenance_by_pair[deepcopy(pair)] = _deep_freeze(
            {
                "epoch_ratio_policy": "coordinate_average",
                "evaluation_rule": "uniform_piecewise_linear_interval_average",
                "interval_start": start,
                "interval_end": end,
                "coordinate_kind": kind,
                "normalized_interval_width": width,
                "contributing_segments": tuple(segments),
                "source_evidence_ids": source_id_tuple,
                "left_source_evidence_id": left_id,
                "right_source_evidence_id": right_id,
                "inside_observed_support": True,
                "epoch_support": epoch_support,
                "effective_ratio": effective_ratio,
            }
        )

    normalized_rows = _validate_epoch_ratio_rows(
        rows,
        expected_pairs=ordered_pairs,
    )
    return _EpochRatioReductionResult(
        rows=normalized_rows,
        provenance_by_pair=MappingProxyType(provenance_by_pair),
        uncalibrated_pairs=(),
        coordinate_evaluations=tuple(coordinate_evaluations),
    )


def _reduce_support_weighted_average(
    request: _EpochRatioReductionRequest,
    schedule: _CoordinateLinearSchedule,
) -> _EpochRatioReductionResult:
    """Average schedule evaluations using explicit within-epoch weights."""

    if not isinstance(request, _EpochRatioReductionRequest):
        raise TypeError("request must be an epoch-ratio reduction request.")
    if (
        request.ratio_policy != "coordinate_linear"
        or request.epoch_ratio_policy != "support_weighted_average"
    ):
        raise ValueError(
            "support-weighted reduction requires "
            "ratio_policy='coordinate_linear' and "
            "epoch_ratio_policy='support_weighted_average'."
        )
    if not isinstance(schedule, _CoordinateLinearSchedule):
        raise TypeError("schedule must be a coordinate-linear schedule.")
    if (
        request.epoch_windows is None
        or request.epoch_coordinate_weights is None
    ):
        raise ValueError(
            "support-weighted reduction requires epoch windows and explicit "
            "within-epoch coordinate weights."
        )

    entity_order = tuple(schedule.normalized_anchors_by_entity)
    unknown_entities = [
        entity
        for entity, _ in request.calibrated_pairs
        if entity not in schedule.normalized_anchors_by_entity
    ]
    if unknown_entities:
        raise ValueError(
            "support-weighted calibrated pairs contain entities without a "
            f"coordinate schedule; unknown={unknown_entities!r}."
        )
    requested_pair_set = set(request.calibrated_pairs)
    ordered_pairs = tuple(
        (entity, epoch)
        for epoch in request.epochs
        for entity in entity_order
        if (entity, epoch) in requested_pair_set
    )

    rows: list[EpochRatioRow] = []
    coordinate_evaluations: list[RatioScheduleRow] = []
    provenance_by_pair: dict[
        tuple[Entity, Epoch], Mapping[str, Any]
    ] = {}
    for pair in ordered_pairs:
        entity, epoch = pair
        coordinate_weights = request.epoch_coordinate_weights[epoch]
        evaluated_points: list[Mapping[str, Any]] = []
        source_ids: list[EvidenceId] = []
        weighted_sum = 0.0
        total_weight = 0.0
        for coordinate, weight in coordinate_weights:
            evaluated = _evaluate_coordinate_linear_schedule(
                schedule,
                tested_entity=entity,
                coordinate_value=coordinate,
            )
            coordinate_evaluations.append(evaluated)
            if evaluated.schedule_ratio is None:
                raise ValueError(
                    f"support_weighted_average for {pair!r} requires every "
                    "coordinate inside observed evidence support."
                )
            weighted_term = weight * evaluated.schedule_ratio
            weighted_sum += weighted_term
            total_weight += weight
            for identifier in evaluated.source_evidence_ids:
                if identifier not in source_ids:
                    source_ids.append(identifier)
            evaluated_points.append(
                {
                    "coordinate": coordinate,
                    "weight": weight,
                    "evaluated_ratio": evaluated.schedule_ratio,
                    "weighted_term": weighted_term,
                    "schedule_source": evaluated.schedule_source,
                    "source_evidence_ids": evaluated.source_evidence_ids,
                    "left_source_evidence_id": (
                        evaluated.left_source_evidence_id
                    ),
                    "right_source_evidence_id": (
                        evaluated.right_source_evidence_id
                    ),
                    "inside_observed_support": (
                        evaluated.inside_observed_support
                    ),
                }
            )
        if not isfinite(total_weight) or total_weight <= 0.0:
            raise ValueError(
                f"support_weighted_average for {pair!r} requires finite "
                "positive total coordinate weight."
            )
        effective_ratio = weighted_sum / total_weight
        if not isfinite(effective_ratio):
            raise ValueError(
                f"support_weighted_average for {pair!r} produced a nonfinite "
                "ratio."
            )
        source_id_tuple = tuple(source_ids)
        if len(source_id_tuple) >= 2:
            left_id = source_id_tuple[0]
            right_id = source_id_tuple[-1]
        else:
            left_id = right_id = None
        epoch_support = (
            None
            if request.epoch_support is None
            else request.epoch_support.get(epoch)
        )
        window = request.epoch_windows[epoch]
        row = EpochRatioRow(
            tested_entity=entity,
            epoch=epoch,
            epoch_start=window[0],
            epoch_end=window[1],
            epoch_coordinate=None,
            epoch_support=epoch_support,
            effective_ratio=effective_ratio,
            ratio_policy="coordinate_linear",
            epoch_ratio_policy="support_weighted_average",
            ratio_source="support_weighted_average",
            evidence_conflict_policy=None,
            source_evidence_ids=source_id_tuple,
            evidence_weights=None,
            left_source_evidence_id=left_id,
            right_source_evidence_id=right_id,
            inside_observed_support=True,
            ratio_bound_lower=None,
            ratio_bound_upper=None,
        )
        rows.append(row)
        provenance_by_pair[deepcopy(pair)] = _deep_freeze(
            {
                "epoch_ratio_policy": "support_weighted_average",
                "evaluation_rule": "declared_coordinate_weighted_average",
                "coordinate_evaluations": tuple(evaluated_points),
                "total_coordinate_weight": total_weight,
                "weighted_ratio_sum": weighted_sum,
                "source_evidence_ids": source_id_tuple,
                "left_source_evidence_id": left_id,
                "right_source_evidence_id": right_id,
                "inside_observed_support": True,
                "epoch_support": epoch_support,
                "effective_ratio": effective_ratio,
            }
        )

    normalized_rows = _validate_epoch_ratio_rows(
        rows,
        expected_pairs=ordered_pairs,
    )
    return _EpochRatioReductionResult(
        rows=normalized_rows,
        provenance_by_pair=MappingProxyType(provenance_by_pair),
        uncalibrated_pairs=(),
        coordinate_evaluations=tuple(coordinate_evaluations),
    )


def _apply_epoch_ratio_bounds(
    result: _EpochRatioReductionResult,
    *,
    ratio_bounds: tuple[float, float] | None,
) -> _EpochRatioReductionResult:
    """Apply inclusive final bounds by raising only; effective ratios never clip."""

    if not isinstance(result, _EpochRatioReductionResult):
        raise TypeError("result must be an epoch-ratio reduction result.")
    bounds = _validate_ratio_bounds(ratio_bounds)
    if bounds is None:
        return result
    lower, upper = bounds

    bounded_rows: list[EpochRatioRow] = []
    for row in result.rows:
        if row.effective_ratio < lower or row.effective_ratio > upper:
            raise ValueError(
                "final effective ratio is outside declared raise-only bounds: "
                f"tested_entity={row.tested_entity!r}, epoch={row.epoch!r}, "
                f"effective_ratio={row.effective_ratio!r}, lower={lower!r}, "
                f"upper={upper!r}, ratio_policy={row.ratio_policy!r}, "
                f"epoch_ratio_policy={row.epoch_ratio_policy!r}, "
                f"ratio_source={row.ratio_source!r}, "
                f"source_evidence_ids={row.source_evidence_ids!r}."
            )
        bounded_rows.append(
            replace(
                row,
                ratio_bound_lower=lower,
                ratio_bound_upper=upper,
            )
        )
    normalized_rows = _validate_epoch_ratio_rows(bounded_rows)
    rows_by_pair = {
        (row.tested_entity, row.epoch): row for row in normalized_rows
    }
    provenance_by_pair: dict[
        tuple[Entity, Epoch], Mapping[str, Any]
    ] = {}
    for pair, provenance in result.provenance_by_pair.items():
        if pair in rows_by_pair:
            provenance_by_pair[deepcopy(pair)] = _deep_freeze(
                {
                    **dict(provenance),
                    "ratio_bound_lower": lower,
                    "ratio_bound_upper": upper,
                    "bounds_action": "validated_without_clipping",
                }
            )
        else:
            provenance_by_pair[deepcopy(pair)] = _deep_freeze(provenance)
    return _EpochRatioReductionResult(
        rows=normalized_rows,
        provenance_by_pair=MappingProxyType(provenance_by_pair),
        uncalibrated_pairs=tuple(result.uncalibrated_pairs),
        coordinate_evaluations=tuple(result.coordinate_evaluations),
    )


@dataclass(frozen=True)
class _TemporalRatioPipelineResult:
    """Resolved temporal schedule, final ratios, and immutable provenance."""

    evidence_points: tuple[EvidencePointRecord, ...]
    coordinate_schedule: tuple[RatioScheduleRow, ...]
    reduction: _EpochRatioReductionResult
    ratio_provenance: Mapping[str, Any]


def _complete_coordinate_schedule(
    anchors: Sequence[RatioScheduleRow],
    evaluations: Sequence[RatioScheduleRow],
) -> tuple[RatioScheduleRow, ...]:
    """Merge compact coordinate anchors and requested evaluations in order."""

    rows = _record_tuple(
        tuple(anchors) + tuple(evaluations),
        record_type=RatioScheduleRow,
        name="coordinate schedule rows",
    )
    entity_order: list[Entity] = []
    grouped: dict[Entity, dict[tuple[Any, Any], RatioScheduleRow]] = {}
    for row in rows:
        if row.coordinate_value is None:
            raise ValueError(
                "coordinate-linear schedule rows require explicit coordinates."
            )
        if row.tested_entity not in grouped:
            entity_order.append(row.tested_entity)
            grouped[row.tested_entity] = {}
        key = (row.coordinate_name, row.coordinate_value)
        previous = grouped[row.tested_entity].get(key)
        if previous is not None and previous != row:
            raise ValueError(
                "coordinate schedule contains inconsistent duplicate "
                f"evaluation for entity={row.tested_entity!r}, "
                f"coordinate={row.coordinate_value!r}."
            )
        grouped[row.tested_entity][key] = row if previous is None else previous

    completed: list[RatioScheduleRow] = []
    for entity in entity_order:
        entity_rows = tuple(grouped[entity].values())
        kind = _coordinate_kind(
            entity_rows[0].coordinate_value,
            name=f"coordinate schedule for {entity!r}",
        )
        if any(
            _coordinate_kind(
                row.coordinate_value,
                name=f"coordinate schedule for {entity!r}",
            )
            != kind
            for row in entity_rows
        ):
            raise TypeError(
                f"coordinate schedule for {entity!r} mixes coordinate types."
            )
        completed.extend(
            sorted(
                entity_rows,
                key=lambda row: _coordinate_numeric_value(
                    row.coordinate_value,
                    kind=kind,
                ),
            )
        )
    return tuple(completed)


def _coordinate_candidate_pairs(
    schedule: _CoordinateLinearSchedule,
    *,
    epochs: Sequence[Epoch],
    tested_entity_epochs: Mapping[Entity, Sequence[Epoch]] | None,
) -> tuple[tuple[Entity, Epoch], ...]:
    """Resolve optional coordinate-schedule candidate restrictions explicitly."""

    authoritative_epochs = _epoch_tuple(epochs)
    entity_order = tuple(schedule.normalized_anchors_by_entity)
    if tested_entity_epochs is None:
        return tuple(
            (deepcopy(entity), deepcopy(epoch))
            for epoch in authoritative_epochs
            for entity in entity_order
        )
    if not isinstance(tested_entity_epochs, Mapping):
        raise TypeError("tested_entity_epochs must be a mapping or None.")
    if not tested_entity_epochs:
        raise ValueError("tested_entity_epochs must not be empty when supplied.")
    unknown_entities = [
        entity
        for entity in tested_entity_epochs
        if entity not in schedule.normalized_anchors_by_entity
    ]
    if unknown_entities:
        raise ValueError(
            "tested_entity_epochs contains entities without a coordinate "
            f"schedule; unknown={unknown_entities!r}."
        )
    declared: dict[Entity, frozenset[Epoch]] = {}
    for entity in entity_order:
        if entity not in tested_entity_epochs:
            continue
        entity_epochs = _epoch_tuple(tested_entity_epochs[entity])
        unknown_epochs = [
            epoch for epoch in entity_epochs if epoch not in authoritative_epochs
        ]
        if unknown_epochs:
            raise ValueError(
                f"tested_entity_epochs[{entity!r}] contains unknown epochs "
                f"{unknown_epochs!r}."
            )
        declared[deepcopy(entity)] = frozenset(entity_epochs)
    return tuple(
        (deepcopy(entity), deepcopy(epoch))
        for epoch in authoritative_epochs
        for entity in entity_order
        if entity in declared and epoch in declared[entity]
    )


def _resolve_temporal_ratio_pipeline(
    evidence: Sequence[LiftTestEvidence],
    *,
    epochs: Sequence[Epoch],
    ratio_policy: RatioPolicy,
    epoch_ratio_policy: EpochRatioPolicy,
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
    ratio_bounds: tuple[float, float] | None = None,
    allow_signed: bool = False,
    tolerance: float = 1e-9,
) -> _TemporalRatioPipelineResult:
    """Construct, reduce, and bound one explicitly selected ratio pipeline."""

    if not isinstance(allow_signed, bool):
        raise TypeError("allow_signed must be a bool.")
    _validate_tolerance(tolerance)
    authoritative_epochs = _epoch_tuple(epochs)
    _validate_ratio_policy_request(
        evidence=evidence,
        ratio_policy=ratio_policy,
        epoch_ratio_policy=epoch_ratio_policy,
        epoch_windows=epoch_windows,
        epoch_coordinates=epoch_coordinates,
        epoch_coordinate_weights=epoch_coordinate_weights,
        epoch_support=epoch_support,
        source_evidence=source_evidence,
        tested_entity_epochs=tested_entity_epochs,
        ratio_schedule=ratio_schedule,
        evidence_strength=evidence_strength,
    )
    if ratio_policy != "observed_only":
        irrelevant_conflict = [
            name
            for name, value in (
                ("evidence_conflict_policy", evidence_conflict_policy),
                ("selected_evidence", selected_evidence),
                ("evidence_weights", evidence_weights),
            )
            if value is not None
        ]
        if irrelevant_conflict:
            raise ValueError(
                f"ratio_policy={ratio_policy!r} does not accept conflict "
                f"inputs {irrelevant_conflict!r}."
            )

    conflict_request: _EvidenceConflictRequest | None = None
    conflict_resolutions: Mapping[
        tuple[Entity, Epoch], _EvidenceConflictResolution
    ] | None = None
    if ratio_policy == "observed_only":
        schedule: Any = _build_observed_only_schedule(
            evidence,
            epochs=authoritative_epochs,
            epoch_windows=epoch_windows,
        )
        conflict_request = _validate_evidence_conflict_request(
            schedule,
            evidence_conflict_policy=evidence_conflict_policy,
        )
        conflict_resolutions = _resolve_evidence_conflicts(
            conflict_request,
            selected_evidence=selected_evidence,
            evidence_weights=evidence_weights,
            tolerance=tolerance,
        )
        calibrated_pairs = tuple(schedule.candidates)
        coordinate_schedule = schedule.coordinate_schedule
    elif ratio_policy == "constant":
        schedule = _build_constant_schedule(
            evidence,
            epochs=authoritative_epochs,
            source_evidence=source_evidence,
            tested_entity_epochs=tested_entity_epochs,
        )
        calibrated_pairs = tuple(schedule.ratios_by_pair)
        coordinate_schedule = schedule.coordinate_schedule
    elif ratio_policy == "explicit_ratio":
        schedule = _build_explicit_ratio_schedule(
            evidence,
            epochs=authoritative_epochs,
            ratio_schedule=ratio_schedule,
            tested_entity_epochs=tested_entity_epochs,
            allow_signed=allow_signed,
        )
        calibrated_pairs = tuple(schedule.ratios_by_pair)
        coordinate_schedule = ()
    elif ratio_policy == "explicit_strength":
        schedule = _build_explicit_strength_schedule(
            evidence,
            epochs=authoritative_epochs,
            source_evidence=source_evidence,
            evidence_strength=evidence_strength,
            tested_entity_epochs=tested_entity_epochs,
            allow_signed=allow_signed,
        )
        calibrated_pairs = tuple(schedule.ratios_by_pair)
        coordinate_schedule = ()
    else:
        selected_entities: Sequence[Entity] | None = None
        if tested_entity_epochs is not None:
            if not isinstance(tested_entity_epochs, Mapping):
                raise TypeError("tested_entity_epochs must be a mapping or None.")
            if not tested_entity_epochs:
                raise ValueError(
                    "tested_entity_epochs must not be empty when supplied."
                )
            selected_entities = tuple(tested_entity_epochs)
        schedule = _build_coordinate_linear_schedule(
            evidence,
            selected_entities=selected_entities,
            allow_signed=allow_signed,
        )
        calibrated_pairs = _coordinate_candidate_pairs(
            schedule,
            epochs=authoritative_epochs,
            tested_entity_epochs=tested_entity_epochs,
        )
        coordinate_schedule = schedule.anchor_rows

    reduction_request = _validate_epoch_ratio_reduction_request(
        ratio_policy=ratio_policy,
        epoch_ratio_policy=epoch_ratio_policy,
        epochs=authoritative_epochs,
        calibrated_pairs=calibrated_pairs,
        epoch_windows=epoch_windows,
        epoch_coordinates=epoch_coordinates,
        epoch_support=epoch_support,
        epoch_coordinate_weights=epoch_coordinate_weights,
    )
    if epoch_ratio_policy == "direct":
        reduction = _reduce_direct_epoch_ratios(
            reduction_request,
            schedule,
            conflict_resolutions=conflict_resolutions,
        )
    elif epoch_ratio_policy == "representative_coordinate":
        reduction = _reduce_representative_coordinate(
            reduction_request,
            schedule,
        )
    elif epoch_ratio_policy == "coordinate_average":
        reduction = _reduce_coordinate_average(
            reduction_request,
            schedule,
        )
    else:
        reduction = _reduce_support_weighted_average(
            reduction_request,
            schedule,
        )
    reduction = _apply_epoch_ratio_bounds(
        reduction,
        ratio_bounds=ratio_bounds,
    )
    if ratio_policy == "coordinate_linear":
        coordinate_schedule = _complete_coordinate_schedule(
            coordinate_schedule,
            reduction.coordinate_evaluations,
        )

    evidence_points = _normalize_evidence_points(
        evidence,
        epoch_windows=epoch_windows,
    )
    schedule_status = getattr(schedule, "status_by_pair", MappingProxyType({}))
    schedule_provenance = getattr(
        schedule,
        "provenance_by_pair",
        MappingProxyType({}),
    )
    conflict_policy_by_pair = (
        MappingProxyType({})
        if conflict_request is None
        else conflict_request.policy_by_pair
    )
    conflict_provenance = MappingProxyType(
        {
            deepcopy(pair): _deep_freeze(
                {
                    "evidence_conflict_policy": resolution.evidence_conflict_policy,
                    "effective_ratio": resolution.effective_ratio,
                    "ratio_source": resolution.ratio_source,
                    "source_evidence_ids": resolution.source_evidence_ids,
                    "resolution": resolution.provenance,
                }
            )
            for pair, resolution in (conflict_resolutions or {}).items()
        }
    )
    ratio_provenance = _deep_freeze(
        {
            "ratio_policy": ratio_policy,
            "candidate_pairs": tuple(calibrated_pairs),
            "schedule_status_by_pair": schedule_status,
            "schedule_provenance_by_pair": schedule_provenance,
            "conflict_policy_by_pair": conflict_policy_by_pair,
            "conflict_resolution_by_pair": conflict_provenance,
            "uncalibrated_pairs": reduction.uncalibrated_pairs,
        }
    )
    return _TemporalRatioPipelineResult(
        evidence_points=evidence_points,
        coordinate_schedule=tuple(coordinate_schedule),
        reduction=reduction,
        ratio_provenance=ratio_provenance,
    )


def _record_tuple(
    value: Any,
    *,
    record_type: type[_Value],
    name: str,
) -> tuple[_Value, ...]:
    """Copy a record sequence into stable tuple storage with type checks."""

    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence of {record_type.__name__}.")
    try:
        records = tuple(value)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be a sequence of {record_type.__name__}."
        ) from exc
    for index, record in enumerate(records):
        if not isinstance(record, record_type):
            raise TypeError(
                f"{name}[{index}] must be a {record_type.__name__}."
            )
    return records


def _warning_tuple(value: Any) -> tuple[str, ...]:
    """Store caller-created notable conditions without inventing warnings."""

    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("warnings must be a sequence of strings.")
    try:
        warnings = tuple(value)
    except TypeError as exc:
        raise TypeError("warnings must be a sequence of strings.") from exc
    if any(not isinstance(item, str) or not item.strip() for item in warnings):
        raise ValueError("warnings entries must be nonempty strings.")
    return tuple(deepcopy(item) for item in warnings)


def _epoch_tuple(value: Any) -> tuple[Epoch, ...]:
    """Copy unique authoritative epoch keys in caller order."""

    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("epochs must be a sequence of hashable keys.")
    try:
        epochs = tuple(value)
    except TypeError as exc:
        raise TypeError("epochs must be a sequence of hashable keys.") from exc
    if not epochs:
        raise ValueError("epochs must not be empty.")
    seen: set[Epoch] = set()
    copied: list[Epoch] = []
    for epoch in epochs:
        try:
            duplicate = epoch in seen
        except TypeError as exc:
            raise TypeError("epoch keys must be hashable.") from exc
        if duplicate:
            raise ValueError(f"epochs contains duplicate key {epoch!r}.")
        seen.add(epoch)
        copied.append(deepcopy(epoch))
    return tuple(copied)


def _finite_mapping_by_epoch(
    value: Any,
    *,
    epochs: Sequence[Epoch],
    name: str,
) -> Mapping[Epoch, float]:
    """Validate an exact finite total mapping and store it in epoch order."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    missing = [epoch for epoch in epochs if epoch not in value]
    extra = [epoch for epoch in value if epoch not in epochs]
    if missing or extra:
        raise ValueError(
            f"{name} keys must exactly match epochs; "
            f"missing={missing!r}, extra={extra!r}."
        )
    return MappingProxyType(
        {
            deepcopy(epoch): _finite_float(value[epoch], name=f"{name}[{epoch!r}]")
            for epoch in epochs
        }
    )


def _freeze_optional_epoch_mapping(
    value: Any,
    *,
    epochs: Sequence[Epoch],
    name: str,
    exact: bool,
) -> Mapping[Epoch, Any] | None:
    """Copy optional epoch metadata in authoritative order."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping or None.")
    missing = [epoch for epoch in epochs if exact and epoch not in value]
    extra = [epoch for epoch in value if epoch not in epochs]
    if missing or extra:
        raise ValueError(
            f"{name} has invalid epoch coverage; "
            f"missing={missing!r}, extra={extra!r}."
        )
    return MappingProxyType(
        {
            deepcopy(epoch): _deep_freeze(value[epoch])
            for epoch in epochs
            if epoch in value
        }
    )


def _residual_policy_by_epoch(
    value: Any,
    *,
    epochs: Sequence[Epoch],
) -> tuple[ResidualPolicy | Mapping[Epoch, ResidualPolicy], Mapping[Epoch, str]]:
    """Freeze scalar or exact per-epoch residual policy state."""

    if isinstance(value, Mapping):
        missing = [epoch for epoch in epochs if epoch not in value]
        extra = [epoch for epoch in value if epoch not in epochs]
        if missing or extra:
            raise ValueError(
                "residual_policy mapping keys must exactly match epochs; "
                f"missing={missing!r}, extra={extra!r}."
            )
        policies: dict[Epoch, str] = {}
        for epoch in epochs:
            policy = value[epoch]
            if policy not in _RESIDUAL_POLICIES:
                raise ValueError(
                    f"residual_policy[{epoch!r}] must be one of "
                    f"{sorted(_RESIDUAL_POLICIES)!r}."
                )
            policies[deepcopy(epoch)] = deepcopy(policy)
        frozen = MappingProxyType(policies)
        return frozen, frozen

    if value not in _RESIDUAL_POLICIES:
        raise ValueError(
            f"residual_policy must be one of {sorted(_RESIDUAL_POLICIES)!r}."
        )
    scalar = deepcopy(value)
    return scalar, MappingProxyType({deepcopy(epoch): scalar for epoch in epochs})


@dataclass(frozen=True)
class _DirectCalibrationRow:
    """Policy-neutral static value after direct tested-entity calibration."""

    entity: Entity
    original_estimate: float
    direct_calibrated_estimate: float
    direct_test_adjustment: float
    evidence_ids: tuple[EvidenceId, ...]


@dataclass(frozen=True)
class _DirectCalibrationState:
    """Fixed direct values and totals consumed by residual allocation."""

    rows: tuple[_DirectCalibrationRow, ...]
    original_total: float
    raw_post_test_total: float
    required_residual: float


@dataclass(frozen=True)
class _TemporalDirectCalibrationState:
    """Epoch-local direct values produced from final effective ratio rows."""

    epochs: tuple[Epoch, ...]
    calibrated_epochs: tuple[Epoch, ...]
    uncalibrated_epochs: tuple[Epoch, ...]
    states_by_epoch: Mapping[Epoch, _DirectCalibrationState]
    epoch_ratio_rows: tuple[EpochRatioRow, ...]


@dataclass(frozen=True)
class _StaticResidualRequest:
    """Validated policy selection and only its compatible caller controls."""

    policy: ResidualPolicy
    eligible_entities: tuple[Entity, ...] | None
    all_untested: bool | None
    residual_weights: Mapping[Entity, float] | None
    residual_entity: Entity | None
    residual_adjustments: Mapping[Entity, float] | None
    allow_bucket_collision: bool


@dataclass(frozen=True)
class _TemporalResidualRequest:
    """Normalized residual controls for exactly the calibrated epochs."""

    epochs: tuple[Epoch, ...]
    calibrated_epochs: tuple[Epoch, ...]
    uncalibrated_epochs: tuple[Epoch, ...]
    residual_policy: ResidualPolicy | Mapping[Epoch, ResidualPolicy]
    policies_by_epoch: Mapping[Epoch, ResidualPolicy]
    requests_by_epoch: Mapping[Epoch, _StaticResidualRequest]


@dataclass(frozen=True)
class _StaticResidualAllocation:
    """Explicit residual adjustments and roles returned by one policy handler."""

    adjustments: tuple[tuple[Entity, float], ...]
    roles: Mapping[Entity, str]
    new_entities: tuple[Entity, ...] = ()


@dataclass(frozen=True)
class _TemporalResidualAllocationResult:
    """Immutable epoch-local allocations and totals before public finalization."""

    epochs: tuple[Epoch, ...]
    epoch_ratio_rows: tuple[EpochRatioRow, ...]
    direct_states_by_epoch: Mapping[Epoch, _DirectCalibrationState]
    allocations_by_epoch: Mapping[Epoch, _StaticResidualAllocation]
    calibrated_values_by_epoch: Mapping[
        Epoch, tuple[tuple[Entity, float], ...]
    ]
    original_totals: Mapping[Epoch, float]
    raw_post_test_totals: Mapping[Epoch, float]
    required_residuals: Mapping[Epoch, float]
    allocated_residuals: Mapping[Epoch, float]
    calibrated_totals: Mapping[Epoch, float]
    residual_provenance: Mapping[Epoch, Mapping[str, Any]]


def _entity_tuple(value: Any, *, name: str) -> tuple[Entity, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence of entity keys.")
    try:
        entities = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of entity keys.") from exc
    if not entities:
        raise ValueError(f"{name} must not be empty.")
    seen: set[Entity] = set()
    copied: list[Entity] = []
    for entity in entities:
        try:
            duplicate = entity in seen
        except TypeError as exc:
            raise TypeError(f"{name} entries must be hashable.") from exc
        if duplicate:
            raise ValueError(f"{name} contains duplicate entity {entity!r}.")
        seen.add(entity)
        copied.append(deepcopy(entity))
    return tuple(copied)


def _optional_entity_mapping(value: Any, *, name: str) -> Mapping[Entity, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping or None.")
    pairs = _normalize_pairs(value, name=name)
    return MappingProxyType({key: item for key, item in pairs})


def _static_residual_request(
    *,
    residual_policy: Any,
    tested_entities: Sequence[Entity],
    eligible_entities: Sequence[Entity] | None,
    all_untested: bool | None,
    residual_weights: Mapping[Entity, float] | None,
    residual_entity: Entity | None,
    residual_adjustments: Mapping[Entity, float] | None,
    allow_bucket_collision: bool,
) -> _StaticResidualRequest:
    """Validate exact static policy selection without performing allocation."""

    if residual_policy not in _RESIDUAL_POLICIES:
        raise ValueError(
            f"residual_policy must be one of {sorted(_RESIDUAL_POLICIES)!r}."
        )
    if not isinstance(allow_bucket_collision, bool):
        raise TypeError("allow_bucket_collision must be a bool.")

    eligible = (
        None
        if eligible_entities is None
        else _entity_tuple(eligible_entities, name="eligible_entities")
    )
    weights = _optional_entity_mapping(residual_weights, name="residual_weights")
    adjustments = _optional_entity_mapping(
        residual_adjustments, name="residual_adjustments"
    )
    tested = set(tested_entities)

    if residual_policy == "allow_total_change":
        incompatible = {
            "eligible_entities": eligible,
            "all_untested": all_untested,
            "residual_weights": weights,
            "residual_entity": residual_entity,
            "residual_adjustments": adjustments,
        }
        supplied = [name for name, value in incompatible.items() if value is not None]
        if allow_bucket_collision:
            supplied.append("allow_bucket_collision")
        if supplied:
            raise ValueError(
                "allow_total_change does not accept residual-allocation inputs; "
                f"supplied={supplied!r}."
            )
    elif residual_policy == "proportional_untested":
        if eligible is None:
            if all_untested is not True:
                raise ValueError(
                    "proportional_untested requires exactly one of "
                    "eligible_entities or all_untested=True."
                )
        elif all_untested is not None:
            raise ValueError(
                "proportional_untested requires exactly one of eligible_entities "
                "or all_untested=True."
            )
        if weights is not None or residual_entity is not None or adjustments is not None:
            raise ValueError(
                "proportional_untested does not accept residual weights, a residual "
                "entity, or explicit adjustments."
            )
        if allow_bucket_collision:
            raise ValueError(
                "allow_bucket_collision is relevant only to residual_bucket."
            )
    elif residual_policy == "custom_weights":
        if weights is None:
            raise ValueError("custom_weights requires residual_weights.")
        if (
            eligible is not None
            or all_untested is not None
            or residual_entity is not None
            or adjustments is not None
        ):
            raise ValueError(
                "custom_weights accepts only residual_weights as its allocation input."
            )
        if allow_bucket_collision:
            raise ValueError(
                "allow_bucket_collision is relevant only to residual_bucket."
            )
    elif residual_policy == "residual_bucket":
        if residual_entity is None or (
            isinstance(residual_entity, str) and not residual_entity.strip()
        ):
            raise ValueError("residual_bucket requires a nonempty residual_entity.")
        try:
            hash(residual_entity)
        except TypeError as exc:
            raise TypeError("residual_entity must be hashable.") from exc
        if (
            eligible is not None
            or all_untested is not None
            or weights is not None
            or adjustments is not None
        ):
            raise ValueError(
                "residual_bucket accepts only residual_entity and optional "
                "allow_bucket_collision."
            )
    else:
        if adjustments is None:
            raise ValueError("explicit_adjustments requires residual_adjustments.")
        if (
            eligible is not None
            or all_untested is not None
            or weights is not None
            or residual_entity is not None
        ):
            raise ValueError(
                "explicit_adjustments accepts only residual_adjustments as its "
                "allocation input."
            )
        if allow_bucket_collision:
            raise ValueError(
                "allow_bucket_collision is relevant only to residual_bucket."
            )

    recipients: tuple[Entity, ...]
    if eligible is not None:
        recipients = eligible
    elif weights is not None:
        recipients = tuple(weights)
    elif adjustments is not None:
        recipients = tuple(adjustments)
    elif residual_policy == "residual_bucket":
        recipients = (residual_entity,)
    else:
        recipients = ()
    invalid_tested = [entity for entity in recipients if entity in tested]
    if invalid_tested:
        raise ValueError(
            "Tested entities cannot receive residual allocation in this phase; "
            f"tested recipients={invalid_tested!r}."
        )

    return _StaticResidualRequest(
        policy=deepcopy(residual_policy),
        eligible_entities=eligible,
        all_untested=all_untested,
        residual_weights=weights,
        residual_entity=deepcopy(residual_entity),
        residual_adjustments=adjustments,
        allow_bucket_collision=allow_bucket_collision,
    )


def _ordered_epoch_subset(
    value: Any,
    *,
    epochs: Sequence[Epoch],
    name: str,
) -> tuple[Epoch, ...]:
    """Validate a possibly empty epoch subset and return authoritative order."""

    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence of epoch keys.")
    try:
        supplied = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of epoch keys.") from exc

    seen: set[Epoch] = set()
    for epoch in supplied:
        try:
            duplicate = epoch in seen
        except TypeError as exc:
            raise TypeError(f"{name} entries must be hashable.") from exc
        if duplicate:
            raise ValueError(f"{name} contains duplicate epoch {epoch!r}.")
        seen.add(epoch)

    unknown = [epoch for epoch in supplied if epoch not in epochs]
    if unknown:
        raise ValueError(
            f"{name} must contain only authoritative epochs; unknown={unknown!r}."
        )
    return tuple(deepcopy(epoch) for epoch in epochs if epoch in seen)


def _exact_policy_epoch_mapping(
    value: Any,
    *,
    required_epochs: Sequence[Epoch],
    name: str,
) -> Mapping[Epoch, Any] | None:
    """Freeze one optional control with exact policy-specific epoch coverage."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping or None.")
    missing = [epoch for epoch in required_epochs if epoch not in value]
    extra = [epoch for epoch in value if epoch not in required_epochs]
    if missing or extra:
        raise ValueError(
            f"{name} must exactly cover its policy epochs; "
            f"missing={missing!r}, extra={extra!r}."
        )
    return MappingProxyType(
        {
            deepcopy(epoch): deepcopy(value[epoch])
            for epoch in required_epochs
        }
    )


def _normalize_temporal_residual_request(
    *,
    epochs: Sequence[Epoch],
    calibrated_epochs: Sequence[Epoch],
    residual_policy: ResidualPolicy | Mapping[Epoch, ResidualPolicy],
    eligible_entities: Sequence[Entity] | None = None,
    eligible_entities_by_epoch: Mapping[
        Epoch, Sequence[Entity]
    ] | None = None,
    all_untested: bool | None = None,
    residual_weights: Mapping[Entity, float] | None = None,
    residual_weights_by_epoch: Mapping[
        Epoch, Mapping[Entity, float]
    ] | None = None,
    residual_entity: Entity | None = None,
    residual_entity_by_epoch: Mapping[Epoch, Entity] | None = None,
    residual_adjustments_by_epoch: Mapping[
        Epoch, Mapping[Entity, float]
    ] | None = None,
    allow_bucket_collision: bool = False,
) -> _TemporalResidualRequest:
    """Resolve temporal residual policy inputs without performing arithmetic."""

    authoritative_epochs = _epoch_tuple(epochs)
    calibrated = _ordered_epoch_subset(
        calibrated_epochs,
        epochs=authoritative_epochs,
        name="calibrated_epochs",
    )
    calibrated_set = set(calibrated)
    uncalibrated = tuple(
        deepcopy(epoch)
        for epoch in authoritative_epochs
        if epoch not in calibrated_set
    )
    normalized_policy, policies = _residual_policy_by_epoch(
        residual_policy,
        epochs=calibrated,
    )

    if not isinstance(allow_bucket_collision, bool):
        raise TypeError("allow_bucket_collision must be a bool.")
    if eligible_entities is not None and eligible_entities_by_epoch is not None:
        raise ValueError(
            "eligible_entities and eligible_entities_by_epoch cannot both be "
            "supplied."
        )
    if residual_weights is not None and residual_weights_by_epoch is not None:
        raise ValueError(
            "residual_weights and residual_weights_by_epoch cannot both be "
            "supplied."
        )
    if residual_entity is not None and residual_entity_by_epoch is not None:
        raise ValueError(
            "residual_entity and residual_entity_by_epoch cannot both be supplied."
        )

    policy_epochs = {
        policy: tuple(
            epoch for epoch in calibrated if policies[epoch] == policy
        )
        for policy in _RESIDUAL_POLICIES
    }

    global_controls = (
        ("eligible_entities", eligible_entities, "proportional_untested"),
        ("residual_weights", residual_weights, "custom_weights"),
        ("residual_entity", residual_entity, "residual_bucket"),
    )
    for name, value, relevant_policy in global_controls:
        if value is not None and policy_epochs[relevant_policy] != calibrated:
            raise ValueError(
                f"Global {name} is valid only when every calibrated epoch uses "
                f"residual_policy={relevant_policy!r}."
            )

    proportional_epochs = policy_epochs["proportional_untested"]
    if all_untested is not None and all_untested is not True:
        raise ValueError("all_untested must be True when supplied.")
    if all_untested is True and not proportional_epochs:
        raise ValueError(
            "all_untested is relevant only to proportional_untested epochs."
        )

    if allow_bucket_collision and not policy_epochs["residual_bucket"]:
        raise ValueError(
            "allow_bucket_collision is relevant only to residual_bucket epochs."
        )

    eligible_by_epoch = _exact_policy_epoch_mapping(
        eligible_entities_by_epoch,
        required_epochs=proportional_epochs,
        name="eligible_entities_by_epoch",
    )
    weights_by_epoch = _exact_policy_epoch_mapping(
        residual_weights_by_epoch,
        required_epochs=policy_epochs["custom_weights"],
        name="residual_weights_by_epoch",
    )
    entities_by_epoch = _exact_policy_epoch_mapping(
        residual_entity_by_epoch,
        required_epochs=policy_epochs["residual_bucket"],
        name="residual_entity_by_epoch",
    )
    adjustments_by_epoch = _exact_policy_epoch_mapping(
        residual_adjustments_by_epoch,
        required_epochs=policy_epochs["explicit_adjustments"],
        name="residual_adjustments_by_epoch",
    )

    if proportional_epochs:
        selectors = sum(
            (
                eligible_entities is not None,
                eligible_by_epoch is not None,
                all_untested is True,
            )
        )
        if selectors != 1:
            raise ValueError(
                "proportional_untested epochs require exactly one of global "
                "eligible_entities, eligible_entities_by_epoch, or "
                "all_untested=True."
            )
    elif eligible_by_epoch is not None:
        raise ValueError(
            "eligible_entities_by_epoch is relevant only to "
            "proportional_untested epochs."
        )

    custom_epochs = policy_epochs["custom_weights"]
    if custom_epochs and residual_weights is None and weights_by_epoch is None:
        raise ValueError(
            "custom_weights epochs require residual_weights or "
            "residual_weights_by_epoch."
        )
    if not custom_epochs and weights_by_epoch is not None:
        raise ValueError(
            "residual_weights_by_epoch is relevant only to custom_weights epochs."
        )

    bucket_epochs = policy_epochs["residual_bucket"]
    if bucket_epochs and residual_entity is None and entities_by_epoch is None:
        raise ValueError(
            "residual_bucket epochs require residual_entity or "
            "residual_entity_by_epoch."
        )
    if not bucket_epochs and entities_by_epoch is not None:
        raise ValueError(
            "residual_entity_by_epoch is relevant only to residual_bucket epochs."
        )

    explicit_epochs = policy_epochs["explicit_adjustments"]
    if explicit_epochs and adjustments_by_epoch is None:
        raise ValueError(
            "explicit_adjustments epochs require "
            "residual_adjustments_by_epoch."
        )
    if not explicit_epochs and adjustments_by_epoch is not None:
        raise ValueError(
            "residual_adjustments_by_epoch is relevant only to "
            "explicit_adjustments epochs."
        )

    requests: dict[Epoch, _StaticResidualRequest] = {}
    for epoch in calibrated:
        policy = policies[epoch]
        request = _static_residual_request(
            residual_policy=policy,
            tested_entities=(),
            eligible_entities=(
                eligible_entities
                if eligible_entities is not None
                else (
                    eligible_by_epoch[epoch]
                    if eligible_by_epoch is not None
                    and policy == "proportional_untested"
                    else None
                )
            ),
            all_untested=(
                True
                if all_untested is True and policy == "proportional_untested"
                else None
            ),
            residual_weights=(
                residual_weights
                if residual_weights is not None
                else (
                    weights_by_epoch[epoch]
                    if weights_by_epoch is not None and policy == "custom_weights"
                    else None
                )
            ),
            residual_entity=(
                residual_entity
                if residual_entity is not None
                else (
                    entities_by_epoch[epoch]
                    if entities_by_epoch is not None and policy == "residual_bucket"
                    else None
                )
            ),
            residual_adjustments=(
                adjustments_by_epoch[epoch]
                if adjustments_by_epoch is not None
                and policy == "explicit_adjustments"
                else None
            ),
            allow_bucket_collision=(
                allow_bucket_collision and policy == "residual_bucket"
            ),
        )
        requests[deepcopy(epoch)] = request

    return _TemporalResidualRequest(
        epochs=authoritative_epochs,
        calibrated_epochs=calibrated,
        uncalibrated_epochs=uncalibrated,
        residual_policy=normalized_policy,
        policies_by_epoch=policies,
        requests_by_epoch=MappingProxyType(requests),
    )


def _allocate_allow_total_change(
    state: _DirectCalibrationState,
    request: _StaticResidualRequest,
    *,
    allow_signed: bool,
    tolerance: float,
) -> _StaticResidualAllocation:
    del state, allow_signed, tolerance
    if request.policy != "allow_total_change":
        raise ValueError("allow-total-change handler received the wrong policy.")
    return _StaticResidualAllocation(
        adjustments=(), roles=MappingProxyType({}), new_entities=()
    )


def _bind_temporal_epoch_residual_request(
    state: _DirectCalibrationState,
    request: _StaticResidualRequest,
) -> _StaticResidualRequest:
    """Revalidate one normalized request against its actual tested entities."""

    if not isinstance(state, _DirectCalibrationState):
        raise TypeError("state must be a direct calibration state.")
    if not isinstance(request, _StaticResidualRequest):
        raise TypeError("request must be a static residual request.")
    tested_entities = tuple(
        row.entity for row in state.rows if row.evidence_ids
    )
    return _static_residual_request(
        residual_policy=request.policy,
        tested_entities=tested_entities,
        eligible_entities=request.eligible_entities,
        all_untested=request.all_untested,
        residual_weights=request.residual_weights,
        residual_entity=request.residual_entity,
        residual_adjustments=request.residual_adjustments,
        allow_bucket_collision=request.allow_bucket_collision,
    )


def _allocate_temporal_allow_total_change(
    state: _TemporalDirectCalibrationState,
    request: _TemporalResidualRequest,
    *,
    allow_signed: bool,
    tolerance: float,
) -> Mapping[Epoch, _StaticResidualAllocation]:
    """Allocate nothing in each epoch selecting ``allow_total_change``."""

    if not isinstance(state, _TemporalDirectCalibrationState):
        raise TypeError("state must be a temporal direct calibration state.")
    if not isinstance(request, _TemporalResidualRequest):
        raise TypeError("request must be a temporal residual request.")
    if state.epochs != request.epochs:
        raise ValueError("Temporal direct and residual epoch grids must match.")
    if state.calibrated_epochs != request.calibrated_epochs:
        raise ValueError(
            "Temporal direct and residual calibrated epochs must match."
        )

    allocations: dict[Epoch, _StaticResidualAllocation] = {}
    for epoch in request.calibrated_epochs:
        epoch_request = request.requests_by_epoch[epoch]
        if epoch_request.policy != "allow_total_change":
            continue
        epoch_request = _bind_temporal_epoch_residual_request(
            state.states_by_epoch[epoch], epoch_request
        )
        allocations[deepcopy(epoch)] = _allocate_allow_total_change(
            state.states_by_epoch[epoch],
            epoch_request,
            allow_signed=allow_signed,
            tolerance=tolerance,
        )
    return MappingProxyType(allocations)


def _allocate_temporal_proportional_untested(
    state: _TemporalDirectCalibrationState,
    request: _TemporalResidualRequest,
    *,
    allow_signed: bool,
    tolerance: float,
) -> Mapping[Epoch, _StaticResidualAllocation]:
    """Allocate each matching epoch residual from that epoch's current mix."""

    if not isinstance(state, _TemporalDirectCalibrationState):
        raise TypeError("state must be a temporal direct calibration state.")
    if not isinstance(request, _TemporalResidualRequest):
        raise TypeError("request must be a temporal residual request.")
    if state.epochs != request.epochs:
        raise ValueError("Temporal direct and residual epoch grids must match.")
    if state.calibrated_epochs != request.calibrated_epochs:
        raise ValueError(
            "Temporal direct and residual calibrated epochs must match."
        )

    allocations: dict[Epoch, _StaticResidualAllocation] = {}
    for epoch in request.calibrated_epochs:
        epoch_request = request.requests_by_epoch[epoch]
        if epoch_request.policy != "proportional_untested":
            continue
        epoch_request = _bind_temporal_epoch_residual_request(
            state.states_by_epoch[epoch], epoch_request
        )
        allocations[deepcopy(epoch)] = _allocate_proportional_untested(
            state.states_by_epoch[epoch],
            epoch_request,
            allow_signed=allow_signed,
            tolerance=tolerance,
        )
    return MappingProxyType(allocations)


def _allocate_temporal_custom_weights(
    state: _TemporalDirectCalibrationState,
    request: _TemporalResidualRequest,
    *,
    allow_signed: bool,
    tolerance: float,
) -> Mapping[Epoch, _StaticResidualAllocation]:
    """Allocate each matching epoch residual by its exact declared weights."""

    if not isinstance(state, _TemporalDirectCalibrationState):
        raise TypeError("state must be a temporal direct calibration state.")
    if not isinstance(request, _TemporalResidualRequest):
        raise TypeError("request must be a temporal residual request.")
    if state.epochs != request.epochs:
        raise ValueError("Temporal direct and residual epoch grids must match.")
    if state.calibrated_epochs != request.calibrated_epochs:
        raise ValueError(
            "Temporal direct and residual calibrated epochs must match."
        )

    allocations: dict[Epoch, _StaticResidualAllocation] = {}
    for epoch in request.calibrated_epochs:
        epoch_request = request.requests_by_epoch[epoch]
        if epoch_request.policy != "custom_weights":
            continue
        epoch_request = _bind_temporal_epoch_residual_request(
            state.states_by_epoch[epoch], epoch_request
        )
        allocations[deepcopy(epoch)] = _allocate_custom_weights(
            state.states_by_epoch[epoch],
            epoch_request,
            allow_signed=allow_signed,
            tolerance=tolerance,
        )
    return MappingProxyType(allocations)


def _allocate_temporal_residual_bucket(
    state: _TemporalDirectCalibrationState,
    request: _TemporalResidualRequest,
    *,
    allow_signed: bool,
    tolerance: float,
) -> Mapping[Epoch, _StaticResidualAllocation]:
    """Assign each matching epoch residual to its explicit bucket entity."""

    if not isinstance(state, _TemporalDirectCalibrationState):
        raise TypeError("state must be a temporal direct calibration state.")
    if not isinstance(request, _TemporalResidualRequest):
        raise TypeError("request must be a temporal residual request.")
    if state.epochs != request.epochs:
        raise ValueError("Temporal direct and residual epoch grids must match.")
    if state.calibrated_epochs != request.calibrated_epochs:
        raise ValueError(
            "Temporal direct and residual calibrated epochs must match."
        )

    allocations: dict[Epoch, _StaticResidualAllocation] = {}
    for epoch in request.calibrated_epochs:
        epoch_request = request.requests_by_epoch[epoch]
        if epoch_request.policy != "residual_bucket":
            continue
        epoch_request = _bind_temporal_epoch_residual_request(
            state.states_by_epoch[epoch], epoch_request
        )
        allocations[deepcopy(epoch)] = _allocate_residual_bucket(
            state.states_by_epoch[epoch],
            epoch_request,
            allow_signed=allow_signed,
            tolerance=tolerance,
        )
    return MappingProxyType(allocations)


def _allocate_temporal_explicit_adjustments(
    state: _TemporalDirectCalibrationState,
    request: _TemporalResidualRequest,
    *,
    allow_signed: bool,
    tolerance: float,
) -> Mapping[Epoch, _StaticResidualAllocation]:
    """Apply each matching epoch's exact caller-supplied adjustments."""

    if not isinstance(state, _TemporalDirectCalibrationState):
        raise TypeError("state must be a temporal direct calibration state.")
    if not isinstance(request, _TemporalResidualRequest):
        raise TypeError("request must be a temporal residual request.")
    if state.epochs != request.epochs:
        raise ValueError("Temporal direct and residual epoch grids must match.")
    if state.calibrated_epochs != request.calibrated_epochs:
        raise ValueError(
            "Temporal direct and residual calibrated epochs must match."
        )

    allocations: dict[Epoch, _StaticResidualAllocation] = {}
    for epoch in request.calibrated_epochs:
        epoch_request = request.requests_by_epoch[epoch]
        if epoch_request.policy != "explicit_adjustments":
            continue
        epoch_request = _bind_temporal_epoch_residual_request(
            state.states_by_epoch[epoch], epoch_request
        )
        allocations[deepcopy(epoch)] = _allocate_explicit_adjustments(
            state.states_by_epoch[epoch],
            epoch_request,
            allow_signed=allow_signed,
            tolerance=tolerance,
        )
    return MappingProxyType(allocations)


def _dispatch_temporal_residual_allocations(
    state: _TemporalDirectCalibrationState,
    request: _TemporalResidualRequest,
    *,
    allow_signed: bool,
    tolerance: float,
) -> Mapping[Epoch, _StaticResidualAllocation]:
    """Route every calibrated epoch to exactly one selected policy handler."""

    handlers = (
        _allocate_temporal_allow_total_change,
        _allocate_temporal_proportional_untested,
        _allocate_temporal_custom_weights,
        _allocate_temporal_residual_bucket,
        _allocate_temporal_explicit_adjustments,
    )
    allocations: dict[Epoch, _StaticResidualAllocation] = {}
    for handler in handlers:
        selected = handler(
            state,
            request,
            allow_signed=allow_signed,
            tolerance=tolerance,
        )
        duplicates = [epoch for epoch in selected if epoch in allocations]
        if duplicates:
            raise ValueError(
                "Temporal residual dispatch produced multiple allocations for "
                f"epochs {duplicates!r}."
            )
        allocations.update(selected)

    missing = [
        epoch
        for epoch in request.calibrated_epochs
        if epoch not in allocations
    ]
    extra = [
        epoch for epoch in allocations if epoch not in request.calibrated_epochs
    ]
    if missing or extra:
        raise ValueError(
            "Temporal residual dispatch must produce exactly one allocation per "
            f"calibrated epoch; missing={missing!r}, extra={extra!r}."
        )
    return MappingProxyType(
        {
            deepcopy(epoch): allocations[epoch]
            for epoch in request.calibrated_epochs
        }
    )


def _apply_temporal_residual_request(
    state: _TemporalDirectCalibrationState,
    request: _TemporalResidualRequest,
    *,
    allow_signed: bool,
    tolerance: float,
) -> _TemporalResidualAllocationResult:
    """Dispatch and finalize one complete temporal residual request."""

    allocations = _dispatch_temporal_residual_allocations(
        state,
        request,
        allow_signed=allow_signed,
        tolerance=tolerance,
    )
    return _finalize_temporal_residual_allocations(
        state,
        request,
        allocations,
        allow_signed=allow_signed,
        tolerance=tolerance,
    )


def _finalize_temporal_residual_allocations(
    state: _TemporalDirectCalibrationState,
    request: _TemporalResidualRequest,
    allocations_by_epoch: Mapping[Epoch, _StaticResidualAllocation],
    *,
    allow_signed: bool,
    tolerance: float,
) -> _TemporalResidualAllocationResult:
    """Validate complete epoch-local allocations and record resulting totals."""

    if not isinstance(state, _TemporalDirectCalibrationState):
        raise TypeError("state must be a temporal direct calibration state.")
    if not isinstance(request, _TemporalResidualRequest):
        raise TypeError("request must be a temporal residual request.")
    if not isinstance(allocations_by_epoch, Mapping):
        raise TypeError("allocations_by_epoch must be a mapping.")
    if not isinstance(allow_signed, bool):
        raise TypeError("allow_signed must be a bool.")
    checked_tolerance = _validate_tolerance(tolerance)
    if state.epochs != request.epochs:
        raise ValueError("Temporal direct and residual epoch grids must match.")
    if state.calibrated_epochs != request.calibrated_epochs:
        raise ValueError(
            "Temporal direct and residual calibrated epochs must match."
        )

    missing = [
        epoch
        for epoch in request.calibrated_epochs
        if epoch not in allocations_by_epoch
    ]
    extra = [
        epoch
        for epoch in allocations_by_epoch
        if epoch not in request.calibrated_epochs
    ]
    if missing or extra:
        raise ValueError(
            "allocations_by_epoch must exactly cover calibrated epochs; "
            f"missing={missing!r}, extra={extra!r}."
        )

    allocations: dict[Epoch, _StaticResidualAllocation] = {}
    calibrated_values: dict[Epoch, tuple[tuple[Entity, float], ...]] = {}
    original_totals: dict[Epoch, float] = {}
    raw_totals: dict[Epoch, float] = {}
    required_residuals: dict[Epoch, float] = {}
    allocated_residuals: dict[Epoch, float] = {}
    calibrated_totals: dict[Epoch, float] = {}
    provenance: dict[Epoch, Mapping[str, Any]] = {}

    calibrated_set = set(request.calibrated_epochs)
    for epoch in state.epochs:
        direct_state = state.states_by_epoch[epoch]
        original_totals[deepcopy(epoch)] = direct_state.original_total
        raw_totals[deepcopy(epoch)] = direct_state.raw_post_test_total
        required_residuals[deepcopy(epoch)] = direct_state.required_residual

        if epoch not in calibrated_set:
            if any(
                row.direct_test_adjustment != 0.0 or row.evidence_ids
                for row in direct_state.rows
            ):
                raise ValueError(
                    f"Uncalibrated epoch {epoch!r} cannot contain direct changes."
                )
            values = tuple(
                (deepcopy(row.entity), row.original_estimate)
                for row in direct_state.rows
            )
            calibrated_values[deepcopy(epoch)] = values
            allocated_residuals[deepcopy(epoch)] = 0.0
            calibrated_totals[deepcopy(epoch)] = direct_state.original_total
            provenance[deepcopy(epoch)] = _deep_freeze(
                {
                    "status": "not_calibrated",
                    "residual_policy": None,
                    "recipients": (),
                    "adjustments": (),
                    "allocated_residual": 0.0,
                }
            )
            continue

        allocation = allocations_by_epoch[epoch]
        if not isinstance(allocation, _StaticResidualAllocation):
            raise TypeError(
                f"allocations_by_epoch[{epoch!r}] must be a static residual "
                "allocation."
            )
        epoch_request = request.requests_by_epoch[epoch]
        adjustment_pairs = _validate_numeric_pairs(
            allocation.adjustments,
            name=f"residual adjustments[{epoch!r}]",
            allow_signed=True,
        )
        adjustments = dict(adjustment_pairs)
        if len(adjustments) != len(adjustment_pairs):
            raise ValueError(
                f"Residual adjustments[{epoch!r}] contain duplicate recipients."
            )
        if not isinstance(allocation.roles, Mapping):
            raise TypeError(
                f"residual allocation roles[{epoch!r}] must be a mapping."
            )

        direct_entities = tuple(row.entity for row in direct_state.rows)
        tested_entities = {
            row.entity for row in direct_state.rows if row.evidence_ids
        }
        invalid_tested = [
            entity for entity in adjustments if entity in tested_entities
        ]
        if invalid_tested:
            raise ValueError(
                "Tested entities cannot receive temporal residual allocation; "
                f"epoch={epoch!r}, tested recipients={invalid_tested!r}."
            )
        new_entities = (
            _entity_tuple(
                allocation.new_entities,
                name=f"new residual entities[{epoch!r}]",
            )
            if allocation.new_entities
            else ()
        )
        unknown = [
            entity
            for entity in adjustments
            if entity not in direct_entities and entity not in new_entities
        ]
        if unknown:
            raise ValueError(
                "Temporal residual recipients must exist in epoch estimates or "
                f"be declared new; epoch={epoch!r}, unknown={unknown!r}."
            )

        raw_rows = list(direct_state.rows)
        raw_rows.extend(
            _DirectCalibrationRow(
                entity=deepcopy(entity),
                original_estimate=0.0,
                direct_calibrated_estimate=0.0,
                direct_test_adjustment=0.0,
                evidence_ids=(),
            )
            for entity in new_entities
        )
        values_list: list[tuple[Entity, float]] = []
        for row in raw_rows:
            value = _finite_float(
                row.direct_calibrated_estimate
                + adjustments.get(row.entity, 0.0),
                name=f"calibrated estimate[{epoch!r}, {row.entity!r}]",
            )
            if not allow_signed and value < -checked_tolerance:
                raise ValueError(
                    "Evidence and the selected epoch total cannot be satisfied "
                    "with nonnegative calibrated estimates; "
                    f"epoch={epoch!r}, entity={row.entity!r}, value={value}."
                )
            values_list.append((deepcopy(row.entity), value))

        allocated = _finite_float(
            sum(adjustments.values()),
            name=f"allocated_residual[{epoch!r}]",
        )
        calibrated_total = _finite_float(
            sum(value for _, value in values_list),
            name=f"calibrated_total[{epoch!r}]",
        )
        if epoch_request.policy == "allow_total_change":
            if allocated != 0.0:
                raise ValueError(
                    f"allow_total_change cannot allocate residual in {epoch!r}."
                )
        elif not isclose(
            allocated,
            direct_state.required_residual,
            rel_tol=1e-12,
            abs_tol=checked_tolerance,
        ):
            raise ValueError(
                "Closed temporal residual policies must allocate the required "
                f"residual independently; epoch={epoch!r}."
            )

        allocations[deepcopy(epoch)] = allocation
        calibrated_values[deepcopy(epoch)] = tuple(values_list)
        allocated_residuals[deepcopy(epoch)] = allocated
        calibrated_totals[deepcopy(epoch)] = calibrated_total
        provenance[deepcopy(epoch)] = _deep_freeze(
            {
                "status": "calibrated",
                "residual_policy": epoch_request.policy,
                "eligible_entities": epoch_request.eligible_entities,
                "all_untested": epoch_request.all_untested,
                "residual_weights": epoch_request.residual_weights,
                "residual_entity": epoch_request.residual_entity,
                "recipients": tuple(adjustments),
                "adjustments": tuple(adjustment_pairs),
                "required_residual": direct_state.required_residual,
                "allocated_residual": allocated,
                "allow_bucket_collision": epoch_request.allow_bucket_collision,
            }
        )

    return _TemporalResidualAllocationResult(
        epochs=state.epochs,
        epoch_ratio_rows=state.epoch_ratio_rows,
        direct_states_by_epoch=state.states_by_epoch,
        allocations_by_epoch=MappingProxyType(allocations),
        calibrated_values_by_epoch=MappingProxyType(calibrated_values),
        original_totals=MappingProxyType(original_totals),
        raw_post_test_totals=MappingProxyType(raw_totals),
        required_residuals=MappingProxyType(required_residuals),
        allocated_residuals=MappingProxyType(allocated_residuals),
        calibrated_totals=MappingProxyType(calibrated_totals),
        residual_provenance=MappingProxyType(provenance),
    )


def _allocate_proportional_untested(
    state: _DirectCalibrationState,
    request: _StaticResidualRequest,
    *,
    allow_signed: bool,
    tolerance: float,
) -> _StaticResidualAllocation:
    del allow_signed
    if request.policy != "proportional_untested":
        raise ValueError("proportional handler received the wrong policy.")
    checked_tolerance = _validate_tolerance(tolerance)
    rows_by_entity = {row.entity: row for row in state.rows}
    tested = {row.entity for row in state.rows if row.evidence_ids}
    if request.all_untested is True:
        recipients = tuple(
            row.entity for row in state.rows if row.entity not in tested
        )
    else:
        recipients = request.eligible_entities or ()
    if not recipients:
        raise ValueError(
            "proportional_untested requires at least one untested residual recipient."
        )
    unknown = [entity for entity in recipients if entity not in rows_by_entity]
    if unknown:
        raise ValueError(
            "eligible_entities must refer to entities in estimates; "
            f"unknown={unknown!r}."
        )

    originals = [rows_by_entity[entity].original_estimate for entity in recipients]
    eligible_mass = _finite_float(sum(originals), name="eligible recipient mass")
    if eligible_mass <= checked_tolerance:
        raise ValueError(
            "Evidence and the selected total constraint cannot be satisfied "
            "simultaneously because proportional recipients have zero eligible mass."
        )
    adjustments: list[tuple[Entity, float]] = []
    allocated = 0.0
    for index, (entity, original) in enumerate(zip(recipients, originals)):
        if index == len(recipients) - 1:
            adjustment = state.required_residual - allocated
        else:
            adjustment = state.required_residual * original / eligible_mass
            allocated += adjustment
        adjustments.append((deepcopy(entity), adjustment))
    return _StaticResidualAllocation(
        adjustments=tuple(adjustments),
        roles=MappingProxyType(
            {deepcopy(entity): "residual_recipient" for entity in recipients}
        ),
        new_entities=(),
    )


def _allocate_custom_weights(
    state: _DirectCalibrationState,
    request: _StaticResidualRequest,
    *,
    allow_signed: bool,
    tolerance: float,
) -> _StaticResidualAllocation:
    del allow_signed
    if request.policy != "custom_weights" or request.residual_weights is None:
        raise ValueError("custom-weights handler received an invalid request.")
    weights = _validate_unit_sum_weights(
        tuple(request.residual_weights.items()),
        tolerance=tolerance,
        name="residual_weights",
    )
    known = {row.entity for row in state.rows}
    unknown = [entity for entity, _ in weights if entity not in known]
    if unknown:
        raise ValueError(
            "residual_weights keys must refer to entities in estimates; "
            f"unknown={unknown!r}."
        )

    adjustments: list[tuple[Entity, float]] = []
    allocated = 0.0
    for index, (entity, weight) in enumerate(weights):
        if index == len(weights) - 1:
            adjustment = state.required_residual - allocated
        else:
            adjustment = state.required_residual * weight
            allocated += adjustment
        adjustments.append((deepcopy(entity), adjustment))
    recipients = tuple(entity for entity, _ in weights)
    return _StaticResidualAllocation(
        adjustments=tuple(adjustments),
        roles=MappingProxyType(
            {deepcopy(entity): "residual_recipient" for entity in recipients}
        ),
        new_entities=(),
    )


def _allocate_residual_bucket(
    state: _DirectCalibrationState,
    request: _StaticResidualRequest,
    *,
    allow_signed: bool,
    tolerance: float,
) -> _StaticResidualAllocation:
    del allow_signed, tolerance
    if request.policy != "residual_bucket" or request.residual_entity is None:
        raise ValueError("residual-bucket handler received an invalid request.")
    bucket = request.residual_entity
    existing = {row.entity for row in state.rows}
    if bucket in existing and not request.allow_bucket_collision:
        raise ValueError(
            "residual_entity collides with an existing entity; set "
            "allow_bucket_collision=True to reuse an existing untested entity."
        )
    return _StaticResidualAllocation(
        adjustments=((deepcopy(bucket), state.required_residual),),
        roles=MappingProxyType({deepcopy(bucket): "residual_bucket"}),
        new_entities=() if bucket in existing else (deepcopy(bucket),),
    )


def _allocate_explicit_adjustments(
    state: _DirectCalibrationState,
    request: _StaticResidualRequest,
    *,
    allow_signed: bool,
    tolerance: float,
) -> _StaticResidualAllocation:
    del allow_signed
    if request.policy != "explicit_adjustments" or request.residual_adjustments is None:
        raise ValueError("explicit-adjustments handler received an invalid request.")
    checked_tolerance = _validate_tolerance(tolerance)
    adjustments = _validate_numeric_pairs(
        tuple(request.residual_adjustments.items()),
        name="residual_adjustments",
        allow_signed=True,
    )
    allocated = _finite_float(
        sum(value for _, value in adjustments), name="allocated residual"
    )
    if abs(allocated - state.required_residual) > checked_tolerance:
        raise ValueError(
            "residual_adjustments must sum to the required residual within "
            f"tolerance {checked_tolerance}; required={state.required_residual}, "
            f"observed={allocated}."
        )
    known = {row.entity for row in state.rows}
    unknown = [entity for entity, _ in adjustments if entity not in known]
    if unknown:
        raise ValueError(
            "residual_adjustments keys must refer to entities in estimates; "
            f"unknown={unknown!r}."
        )
    recipients = tuple(entity for entity, _ in adjustments)
    return _StaticResidualAllocation(
        adjustments=adjustments,
        roles=MappingProxyType(
            {deepcopy(entity): "residual_recipient" for entity in recipients}
        ),
        new_entities=(),
    )


def _dispatch_static_residual(
    state: _DirectCalibrationState,
    request: _StaticResidualRequest,
    *,
    allow_signed: bool,
    tolerance: float,
) -> Any:
    """Route one validated request to exactly one explicit policy handler."""

    handlers = {
        "allow_total_change": _allocate_allow_total_change,
        "proportional_untested": _allocate_proportional_untested,
        "custom_weights": _allocate_custom_weights,
        "residual_bucket": _allocate_residual_bucket,
        "explicit_adjustments": _allocate_explicit_adjustments,
    }
    return handlers[request.policy](
        state,
        request,
        allow_signed=allow_signed,
        tolerance=tolerance,
    )


def _static_calibration_result(
    state: _DirectCalibrationState,
    request: _StaticResidualRequest,
    allocation: _StaticResidualAllocation,
    *,
    evidence: Sequence[LiftTestEvidence],
    allow_signed: bool,
    tolerance: float,
) -> CalibrationResult:
    """Build immutable public rows from fixed direct values plus residuals."""

    checked_tolerance = _validate_tolerance(tolerance)
    adjustment_pairs = _validate_numeric_pairs(
        allocation.adjustments,
        name="residual adjustments",
        allow_signed=True,
    )
    adjustments = dict(adjustment_pairs)
    if len(adjustments) != len(adjustment_pairs):
        raise ValueError("residual adjustments contain duplicate recipients.")
    if not isinstance(allocation.roles, Mapping):
        raise TypeError("residual allocation roles must be a mapping.")

    direct_entities = tuple(row.entity for row in state.rows)
    tested_entities = {
        row.entity for row in state.rows if row.evidence_ids
    }
    invalid_tested = [entity for entity in adjustments if entity in tested_entities]
    if invalid_tested:
        raise ValueError(
            "Tested entities cannot receive residual allocation in this phase; "
            f"tested recipients={invalid_tested!r}."
        )
    new_entities = _entity_tuple(
        allocation.new_entities, name="new residual entities"
    ) if allocation.new_entities else ()
    missing_adjustments = [
        entity
        for entity in adjustments
        if entity not in direct_entities and entity not in new_entities
    ]
    if missing_adjustments:
        raise ValueError(
            "Residual adjustment recipients must exist in estimates or be declared "
            f"new residual entities; unknown={missing_adjustments!r}."
        )

    raw_rows = list(state.rows)
    raw_rows.extend(
        _DirectCalibrationRow(
            entity=deepcopy(entity),
            original_estimate=0.0,
            direct_calibrated_estimate=0.0,
            direct_test_adjustment=0.0,
            evidence_ids=(),
        )
        for entity in new_entities
    )
    calibrated_values = [
        row.direct_calibrated_estimate + adjustments.get(row.entity, 0.0)
        for row in raw_rows
    ]
    for row, calibrated in zip(raw_rows, calibrated_values):
        _finite_float(calibrated, name=f"calibrated estimate for {row.entity!r}")
        if not allow_signed and calibrated < -checked_tolerance:
            raise ValueError(
                "Evidence and the selected total constraint cannot be satisfied "
                "simultaneously with nonnegative calibrated estimates; "
                f"entity={row.entity!r}, calibrated_estimate={calibrated}."
            )

    calibrated_total = _finite_float(
        sum(calibrated_values), name="calibrated_total"
    )
    original_total = state.original_total
    rows: list[CalibrationRow] = []
    for raw, calibrated in zip(raw_rows, calibrated_values):
        residual = adjustments.get(raw.entity, 0.0)
        if raw.evidence_ids:
            role = "tested"
        else:
            role = allocation.roles.get(raw.entity, "untouched")
        rows.append(
            CalibrationRow(
                entity=raw.entity,
                original_estimate=raw.original_estimate,
                direct_test_adjustment=raw.direct_test_adjustment,
                residual_adjustment=residual,
                net_adjustment=raw.direct_test_adjustment + residual,
                calibrated_estimate=calibrated,
                original_share=(
                    None
                    if original_total == 0.0
                    else raw.original_estimate / original_total
                ),
                calibrated_share=(
                    None if calibrated_total == 0.0 else calibrated / calibrated_total
                ),
                role=role,
                evidence_ids=raw.evidence_ids,
                residual_policy=request.policy,
            )
        )

    allocated_residual = _finite_float(
        sum(adjustments.values()), name="allocated_residual"
    )
    return CalibrationResult(
        rows=tuple(rows),
        original_total=state.original_total,
        raw_post_test_total=state.raw_post_test_total,
        required_residual=state.required_residual,
        allocated_residual=allocated_residual,
        calibrated_total=calibrated_total,
        residual_policy=request.policy,
        evidence=tuple(evidence),
        warnings=(),
        _tolerance=checked_tolerance,
    )


def _static_direct_state(
    estimates: EstimateInput,
    *,
    evidence_by_entity: Mapping[Entity, tuple[LiftTestEvidence, ...]],
    allow_signed: bool,
    tolerance: Any,
) -> _DirectCalibrationState:
    """Apply each static evidence ratio only to its declared tested entity."""

    _validate_tolerance(tolerance)
    estimate_pairs = _validate_numeric_pairs(
        _normalize_pairs(estimates, name="estimates"),
        name="estimates",
        allow_signed=allow_signed,
    )
    estimate_entities = {entity for entity, _ in estimate_pairs}
    missing = [entity for entity in evidence_by_entity if entity not in estimate_entities]
    if missing:
        raise ValueError(
            "Every tested entity must exist in estimates; "
            f"missing={missing!r}."
        )

    evidence_by_entity_copy = dict(evidence_by_entity)
    rows: list[_DirectCalibrationRow] = []
    for entity, original in estimate_pairs:
        entity_evidence = evidence_by_entity_copy.get(entity, ())
        if not entity_evidence:
            direct_value = original
            evidence_ids: tuple[EvidenceId, ...] = ()
        else:
            if len(entity_evidence) != 1:
                raise ValueError(
                    "Static calibration requires exactly one evidence record per "
                    f"tested entity; entity={entity!r}."
                )
            evidence = entity_evidence[0]
            if not allow_signed and (
                evidence.observed_lift < 0.0
                or evidence.matched_mta_estimate < 0.0
                or evidence.calibration_ratio < 0.0
            ):
                raise ValueError(
                    "Signed evidence requires allow_signed=True; "
                    f"tested_entity={entity!r}, evidence_id={evidence.test_id!r}."
                )
            direct_value = _finite_float(
                original * evidence.calibration_ratio,
                name=f"direct calibrated estimate for {entity!r}",
            )
            if not allow_signed and direct_value < 0.0:
                raise ValueError(
                    "Direct calibrated estimates must be nonnegative unless "
                    f"allow_signed=True; tested_entity={entity!r}."
                )
            evidence_ids = (evidence.test_id,)

        rows.append(
            _DirectCalibrationRow(
                entity=deepcopy(entity),
                original_estimate=original,
                direct_calibrated_estimate=direct_value,
                direct_test_adjustment=direct_value - original,
                evidence_ids=evidence_ids,
            )
        )

    original_total = _finite_float(
        sum(row.original_estimate for row in rows), name="original_total"
    )
    raw_post_test_total = _finite_float(
        sum(row.direct_calibrated_estimate for row in rows),
        name="raw_post_test_total",
    )
    required_residual = _finite_float(
        original_total - raw_post_test_total, name="required_residual"
    )
    return _DirectCalibrationState(
        rows=tuple(rows),
        original_total=original_total,
        raw_post_test_total=raw_post_test_total,
        required_residual=required_residual,
    )


def _temporal_direct_state(
    estimates_by_epoch: EstimatesByEpochInput,
    *,
    epoch_ratio_rows: Sequence[EpochRatioRow],
    evidence_by_entity: Mapping[Entity, tuple[LiftTestEvidence, ...]],
    allow_signed: bool,
    tolerance: Any,
) -> _TemporalDirectCalibrationState:
    """Apply each final effective ratio only to its declared entity and epoch."""

    if not isinstance(allow_signed, bool):
        raise TypeError("allow_signed must be a bool.")
    _validate_tolerance(tolerance)
    normalized_estimates = _normalize_estimates_by_epoch(estimates_by_epoch)
    epochs = _epoch_tuple(epoch for epoch, _ in normalized_estimates)
    rows = _validate_epoch_ratio_rows(epoch_ratio_rows)
    if not isinstance(evidence_by_entity, Mapping):
        raise TypeError("evidence_by_entity must be a mapping.")

    estimates: dict[Epoch, tuple[tuple[Entity, float], ...]] = {}
    entity_sets: dict[Epoch, set[Entity]] = {}
    for epoch, estimate_pairs in normalized_estimates:
        checked = _validate_numeric_pairs(
            estimate_pairs,
            name=f"estimates_by_epoch[{epoch!r}]",
            allow_signed=allow_signed,
        )
        estimates[deepcopy(epoch)] = checked
        entity_sets[deepcopy(epoch)] = {entity for entity, _ in checked}

    evidence_ids_by_entity: dict[Entity, tuple[EvidenceId, ...]] = {}
    evidence_entity_by_id: dict[EvidenceId, Entity] = {}
    for entity, records in evidence_by_entity.items():
        if isinstance(records, (str, bytes, bytearray)):
            raise TypeError(
                f"evidence_by_entity[{entity!r}] must be a sequence of evidence."
            )
        try:
            normalized_records = tuple(records)
        except TypeError as exc:
            raise TypeError(
                f"evidence_by_entity[{entity!r}] must be a sequence of evidence."
            ) from exc
        if not normalized_records:
            raise ValueError(
                f"evidence_by_entity[{entity!r}] must not be empty."
            )
        identifiers: list[EvidenceId] = []
        for record in normalized_records:
            if not isinstance(record, LiftTestEvidence):
                raise TypeError(
                    f"evidence_by_entity[{entity!r}] entries must be "
                    "LiftTestEvidence records."
                )
            if record.tested_entity != entity:
                raise ValueError(
                    "evidence_by_entity keys must match each record's tested "
                    f"entity; key={entity!r}, record={record.tested_entity!r}."
                )
            if record.test_id in evidence_entity_by_id:
                raise ValueError(
                    f"evidence ID {record.test_id!r} appears more than once."
                )
            evidence_entity_by_id[record.test_id] = deepcopy(entity)
            identifiers.append(deepcopy(record.test_id))
        evidence_ids_by_entity[deepcopy(entity)] = tuple(identifiers)

    ratios_by_epoch: dict[Epoch, dict[Entity, EpochRatioRow]] = {
        deepcopy(epoch): {} for epoch in epochs
    }
    for row in rows:
        if row.epoch not in ratios_by_epoch:
            raise ValueError(
                f"epoch ratio row refers to unknown epoch {row.epoch!r}."
            )
        if row.tested_entity not in entity_sets[row.epoch]:
            raise ValueError(
                "Every calibrated tested entity must exist in its epoch "
                f"estimates; pair={(row.tested_entity, row.epoch)!r}."
            )
        if row.tested_entity not in evidence_ids_by_entity:
            raise ValueError(
                "Every calibrated tested entity must have constructor evidence; "
                f"tested_entity={row.tested_entity!r}."
            )
        for identifier in row.source_evidence_ids:
            if evidence_entity_by_id.get(identifier) != row.tested_entity:
                raise ValueError(
                    "epoch ratio source evidence must belong to its tested "
                    f"entity; pair={(row.tested_entity, row.epoch)!r}, "
                    f"evidence_id={identifier!r}."
                )
        if not allow_signed and row.effective_ratio < 0.0:
            raise ValueError(
                "Signed effective ratios require allow_signed=True; "
                f"pair={(row.tested_entity, row.epoch)!r}."
            )
        ratios_by_epoch[row.epoch][row.tested_entity] = row

    states: dict[Epoch, _DirectCalibrationState] = {}
    calibrated_epochs: list[Epoch] = []
    for epoch in epochs:
        epoch_ratios = ratios_by_epoch[epoch]
        if epoch_ratios:
            calibrated_epochs.append(deepcopy(epoch))
        direct_rows: list[_DirectCalibrationRow] = []
        for entity, original in estimates[epoch]:
            ratio_row = epoch_ratios.get(entity)
            if ratio_row is None:
                direct_value = original
                source_ids: tuple[EvidenceId, ...] = ()
            else:
                direct_value = _finite_float(
                    original * ratio_row.effective_ratio,
                    name=(
                        "direct calibrated estimate for "
                        f"{(entity, epoch)!r}"
                    ),
                )
                if not allow_signed and direct_value < 0.0:
                    raise ValueError(
                        "Direct calibrated estimates must be nonnegative unless "
                        f"allow_signed=True; pair={(entity, epoch)!r}."
                    )
                source_ids = (
                    ratio_row.source_evidence_ids
                    if ratio_row.source_evidence_ids
                    else evidence_ids_by_entity[entity]
                )
            direct_rows.append(
                _DirectCalibrationRow(
                    entity=deepcopy(entity),
                    original_estimate=original,
                    direct_calibrated_estimate=direct_value,
                    direct_test_adjustment=direct_value - original,
                    evidence_ids=tuple(deepcopy(source_ids)),
                )
            )

        original_total = _finite_float(
            sum(row.original_estimate for row in direct_rows),
            name=f"original_total[{epoch!r}]",
        )
        raw_total = _finite_float(
            sum(row.direct_calibrated_estimate for row in direct_rows),
            name=f"raw_post_test_total[{epoch!r}]",
        )
        states[deepcopy(epoch)] = _DirectCalibrationState(
            rows=tuple(direct_rows),
            original_total=original_total,
            raw_post_test_total=raw_total,
            required_residual=_finite_float(
                original_total - raw_total,
                name=f"required_residual[{epoch!r}]",
            ),
        )

    calibrated = tuple(calibrated_epochs)
    calibrated_set = set(calibrated)
    return _TemporalDirectCalibrationState(
        epochs=epochs,
        calibrated_epochs=calibrated,
        uncalibrated_epochs=tuple(
            deepcopy(epoch) for epoch in epochs if epoch not in calibrated_set
        ),
        states_by_epoch=MappingProxyType(states),
        epoch_ratio_rows=rows,
    )


@dataclass(frozen=True)
class CalibrationRow:
    """One immutable entity row separating direct and residual effects."""

    entity: Entity
    original_estimate: float
    direct_test_adjustment: float
    residual_adjustment: float
    net_adjustment: float
    calibrated_estimate: float
    original_share: float | None
    calibrated_share: float | None
    role: str
    evidence_ids: tuple[EvidenceId, ...]
    residual_policy: ResidualPolicy | None

    def __post_init__(self) -> None:
        if self.entity is None:
            raise ValueError("entity is required.")
        try:
            hash(self.entity)
        except TypeError as exc:
            raise TypeError("entity must be hashable.") from exc

        numeric_names = (
            "original_estimate",
            "direct_test_adjustment",
            "residual_adjustment",
            "net_adjustment",
            "calibrated_estimate",
        )
        values = {
            name: _finite_float(getattr(self, name), name=name)
            for name in numeric_names
        }
        shares: dict[str, float | None] = {}
        for name in ("original_share", "calibrated_share"):
            value = getattr(self, name)
            shares[name] = (
                None if value is None else _finite_float(value, name=name)
            )

        direct_plus_residual = (
            values["direct_test_adjustment"] + values["residual_adjustment"]
        )
        if not isclose(
            values["net_adjustment"],
            direct_plus_residual,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "net_adjustment must equal direct_test_adjustment + "
                "residual_adjustment."
            )
        expected_calibrated = (
            values["original_estimate"] + values["net_adjustment"]
        )
        if not isclose(
            values["calibrated_estimate"],
            expected_calibrated,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "calibrated_estimate must equal original_estimate + "
                "net_adjustment."
            )

        if self.role not in _CALIBRATION_ROLES:
            raise ValueError(f"role must be one of {sorted(_CALIBRATION_ROLES)!r}.")
        if self.role == "tested_and_residual_recipient":
            raise ValueError(
                "tested_and_residual_recipient is reserved and disallowed in "
                "this phase."
            )
        if (
            self.residual_policy is not None
            and self.residual_policy not in _RESIDUAL_POLICIES
        ):
            raise ValueError(
                "residual_policy must be one of "
                f"{sorted(_RESIDUAL_POLICIES)!r}, or None only for a wholly "
                "uncalibrated untouched row."
            )

        evidence_ids = _evidence_id_tuple(
            self.evidence_ids, name="evidence_ids"
        )
        direct = values["direct_test_adjustment"]
        residual = values["residual_adjustment"]
        if self.role == "tested":
            if not evidence_ids:
                raise ValueError("tested rows require at least one evidence ID.")
            if residual != 0.0:
                raise ValueError(
                    "tested entities cannot receive residual allocation."
                )
        else:
            if evidence_ids:
                raise ValueError(
                    "only tested rows may contain evidence IDs."
                )
            if direct != 0.0:
                raise ValueError(
                    "only tested rows may have a direct test adjustment."
                )

        if self.residual_policy == "allow_total_change" and residual != 0.0:
            raise ValueError(
                "allow_total_change rows cannot have residual adjustments."
            )
        if self.role == "untouched" and residual != 0.0:
            raise ValueError("untouched rows cannot have a residual adjustment.")
        if self.residual_policy is None:
            if self.role != "untouched":
                raise ValueError(
                    "Only wholly uncalibrated untouched rows may omit "
                    "residual_policy."
                )
        else:
            if self.role == "residual_bucket":
                if self.residual_policy != "residual_bucket":
                    raise ValueError(
                        "residual_bucket role requires residual_bucket policy."
                    )
            elif self.residual_policy == "residual_bucket" and residual != 0.0:
                raise ValueError(
                    "residual_bucket policy adjustments require residual_bucket "
                    "role."
                )
            if self.role == "residual_recipient" and self.residual_policy in {
                "allow_total_change",
                "residual_bucket",
            }:
                raise ValueError(
                    "residual_recipient role requires a recipient allocation "
                    "policy."
                )
        object.__setattr__(self, "entity", deepcopy(self.entity))
        for name, value in values.items():
            object.__setattr__(self, name, value)
        for name, value in shares.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "role", deepcopy(self.role))
        object.__setattr__(self, "evidence_ids", evidence_ids)
        object.__setattr__(
            self, "residual_policy", deepcopy(self.residual_policy)
        )


def _temporal_calibration_rows(
    result: _TemporalResidualAllocationResult,
    request: _TemporalResidualRequest,
) -> Mapping[Epoch, tuple[CalibrationRow, ...]]:
    """Convert finalized epoch arithmetic into immutable public row records."""

    if not isinstance(result, _TemporalResidualAllocationResult):
        raise TypeError("result must be a temporal residual allocation result.")
    if not isinstance(request, _TemporalResidualRequest):
        raise TypeError("request must be a temporal residual request.")
    if result.epochs != request.epochs:
        raise ValueError("Temporal allocation and residual request epochs must match.")

    rows_by_epoch: dict[Epoch, tuple[CalibrationRow, ...]] = {}
    for epoch in result.epochs:
        direct_state = result.direct_states_by_epoch[epoch]
        allocation = result.allocations_by_epoch.get(epoch)
        if allocation is None:
            adjustment_pairs: tuple[tuple[Entity, float], ...] = ()
            roles: Mapping[Entity, str] = MappingProxyType({})
            new_entities: tuple[Entity, ...] = ()
            policy: ResidualPolicy | None = None
        else:
            adjustment_pairs = tuple(allocation.adjustments)
            roles = allocation.roles
            new_entities = tuple(allocation.new_entities)
            policy = request.policies_by_epoch[epoch]
        adjustments = dict(adjustment_pairs)
        if len(adjustments) != len(adjustment_pairs):
            raise ValueError(
                f"Residual adjustments[{epoch!r}] contain duplicate entities."
            )

        raw_rows = list(direct_state.rows)
        raw_rows.extend(
            _DirectCalibrationRow(
                entity=deepcopy(entity),
                original_estimate=0.0,
                direct_calibrated_estimate=0.0,
                direct_test_adjustment=0.0,
                evidence_ids=(),
            )
            for entity in new_entities
        )
        calibrated_pairs = result.calibrated_values_by_epoch[epoch]
        calibrated_values = dict(calibrated_pairs)
        if len(calibrated_values) != len(calibrated_pairs):
            raise ValueError(
                f"calibrated_values_by_epoch[{epoch!r}] contains duplicates."
            )
        expected_entities = tuple(row.entity for row in raw_rows)
        if tuple(calibrated_values) != expected_entities:
            raise ValueError(
                f"calibrated_values_by_epoch[{epoch!r}] must preserve exact "
                "entity order."
            )

        original_total = result.original_totals[epoch]
        calibrated_total = result.calibrated_totals[epoch]
        epoch_rows: list[CalibrationRow] = []
        for raw in raw_rows:
            residual = adjustments.get(raw.entity, 0.0)
            if raw.evidence_ids:
                role = "tested"
            else:
                role = roles.get(raw.entity, "untouched")
            calibrated = calibrated_values[raw.entity]
            epoch_rows.append(
                CalibrationRow(
                    entity=raw.entity,
                    original_estimate=raw.original_estimate,
                    direct_test_adjustment=raw.direct_test_adjustment,
                    residual_adjustment=residual,
                    net_adjustment=raw.direct_test_adjustment + residual,
                    calibrated_estimate=calibrated,
                    original_share=(
                        None
                        if original_total == 0.0
                        else raw.original_estimate / original_total
                    ),
                    calibrated_share=(
                        None
                        if calibrated_total == 0.0
                        else calibrated / calibrated_total
                    ),
                    role=role,
                    evidence_ids=raw.evidence_ids,
                    residual_policy=policy,
                )
            )
        rows_by_epoch[deepcopy(epoch)] = tuple(epoch_rows)
    return MappingProxyType(rows_by_epoch)


@dataclass(frozen=True)
class CalibrationResult:
    """Immutable static calibration result state used by later exporters."""

    rows: tuple[CalibrationRow, ...]
    original_total: float
    raw_post_test_total: float
    required_residual: float
    allocated_residual: float
    calibrated_total: float
    residual_policy: ResidualPolicy
    evidence: tuple[LiftTestEvidence, ...]
    warnings: tuple[str, ...]
    _tolerance: InitVar[float] = 1e-9

    def __post_init__(self, _tolerance: float) -> None:
        checked_tolerance = _validate_tolerance(_tolerance)
        rows = _record_tuple(
            self.rows, record_type=CalibrationRow, name="rows"
        )
        if not rows:
            raise ValueError("rows must not be empty.")
        entities = [row.entity for row in rows]
        if len(set(entities)) != len(entities):
            raise ValueError("rows must contain each entity exactly once.")

        totals = {
            name: _finite_float(getattr(self, name), name=name)
            for name in (
                "original_total",
                "raw_post_test_total",
                "required_residual",
                "allocated_residual",
                "calibrated_total",
            )
        }
        if self.residual_policy not in _RESIDUAL_POLICIES:
            raise ValueError(
                "residual_policy must be one of "
                f"{sorted(_RESIDUAL_POLICIES)!r}."
            )
        for row in rows:
            if row.residual_policy != self.residual_policy:
                raise ValueError(
                    "every row residual_policy must match the result policy."
                )

        arithmetic = {
            "original_total": sum(row.original_estimate for row in rows),
            "raw_post_test_total": sum(
                row.original_estimate + row.direct_test_adjustment for row in rows
            ),
            "allocated_residual": sum(row.residual_adjustment for row in rows),
            "calibrated_total": sum(row.calibrated_estimate for row in rows),
        }
        arithmetic["required_residual"] = (
            totals["original_total"] - totals["raw_post_test_total"]
        )
        for name, expected in arithmetic.items():
            if not isclose(
                totals[name],
                expected,
                rel_tol=1e-12,
                abs_tol=checked_tolerance,
            ):
                raise ValueError(
                    f"{name} is inconsistent with immutable allocation rows."
                )
        if not isclose(
            totals["calibrated_total"],
            totals["raw_post_test_total"] + totals["allocated_residual"],
            rel_tol=1e-12,
            abs_tol=checked_tolerance,
        ):
            raise ValueError(
                "calibrated_total must equal raw_post_test_total + "
                "allocated_residual."
            )
        if self.residual_policy == "allow_total_change":
            if totals["allocated_residual"] != 0.0:
                raise ValueError(
                    "allow_total_change cannot allocate a residual."
                )
        elif not isclose(
            totals["allocated_residual"],
            totals["required_residual"],
            rel_tol=1e-12,
            abs_tol=checked_tolerance,
        ):
            raise ValueError(
                "closed residual policies must allocate the required residual."
            )

        normalized_evidence = _normalize_evidence_collection(
            self.evidence, allow_multiple_per_entity=False
        ).records
        known_ids = {item.test_id for item in normalized_evidence}
        row_ids = {identifier for row in rows for identifier in row.evidence_ids}
        if not row_ids <= known_ids:
            raise ValueError(
                "row evidence_ids must refer to result evidence records."
            )
        warnings = _warning_tuple(self.warnings)

        object.__setattr__(self, "rows", rows)
        for name, value in totals.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self, "residual_policy", deepcopy(self.residual_policy)
        )
        object.__setattr__(self, "evidence", normalized_evidence)
        object.__setattr__(self, "warnings", warnings)

    def to_dataframe(self) -> Any:
        """Return the stored allocation rows as a defensive flat table.

        Columns follow :class:`CalibrationRow` field order exactly.  Row order
        is the immutable result order, including an appended residual bucket.
        The projection reads stored rows only and deep-copies cell values so
        callers may mutate the returned DataFrame without changing this result
        or a later export.
        """

        columns = (
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
        )
        records = [
            {column: deepcopy(getattr(row, column)) for column in columns}
            for row in self.rows
        ]
        return pd.DataFrame.from_records(records, columns=columns).copy(deep=True)

    def plot_calibration_effects(self) -> Any:
        """Plot the additive calibration bridge for every static result row.

        Each entity is rendered as
        ``original + direct test adjustment + residual adjustment = calibrated``.
        Direct and residual adjustments use distinct artists so a policy
        allocation cannot be mistaken for an experimentally supported change.
        """

        import matplotlib.pyplot as plt  # lazy optional dependency

        figure_height = max(4.5, 0.72 * len(self.rows) + 2.4)
        figure, axis = plt.subplots(figsize=(12, figure_height))
        positions = list(range(len(self.rows)))
        original_values = [row.original_estimate for row in self.rows]
        direct_values = [row.direct_test_adjustment for row in self.rows]
        direct_totals = [
            row.original_estimate + row.direct_test_adjustment
            for row in self.rows
        ]
        residual_values = [row.residual_adjustment for row in self.rows]
        calibrated_values = [row.calibrated_estimate for row in self.rows]

        axis.barh(
            positions,
            original_values,
            left=0.0,
            height=0.56,
            color="0.72",
            edgecolor="0.25",
            linewidth=0.6,
            label="Original estimate",
        )
        axis.barh(
            positions,
            direct_values,
            left=original_values,
            height=0.56,
            color="tab:orange",
            edgecolor="0.25",
            linewidth=0.6,
            label="Direct test adjustment",
        )
        axis.barh(
            positions,
            residual_values,
            left=direct_totals,
            height=0.56,
            color="tab:blue",
            edgecolor="0.25",
            linewidth=0.6,
            label="Residual adjustment",
        )
        axis.scatter(
            calibrated_values,
            positions,
            marker="D",
            s=58,
            color="black",
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
            label="Calibrated estimate",
        )

        for position, row in zip(positions, self.rows):
            axis.annotate(
                f"{row.calibrated_estimate:.2f}",
                (row.calibrated_estimate, position),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
            )

        axis.axvline(0.0, color="0.2", linewidth=0.8)
        axis.set_yticks(positions)
        axis.set_yticklabels([str(row.entity) for row in self.rows])
        axis.invert_yaxis()
        axis.set_xlabel("Attribution estimate")
        axis.set_ylabel("Entity")
        axis.set_title(
            "Static Calibration Effects: "
            "Original + Direct + Residual = Calibrated"
        )
        axis.grid(axis="x", alpha=0.2)
        axis.legend(loc="best")
        figure.tight_layout()
        return figure

    def summary(self) -> str:
        """Render a deterministic audit summary from stored result state."""

        def amount(value: float, *, signed: bool = False) -> str:
            normalized = 0.0 if value == 0.0 else value
            return f"{normalized:+.2f}" if signed else f"{normalized:.2f}"

        evidence_by_id = {item.test_id: item for item in self.evidence}
        tested_rows = tuple(row for row in self.rows if row.role == "tested")
        tested_sources = ", ".join(
            f"{row.entity} ({', '.join(row.evidence_ids)})"
            for row in tested_rows
        )

        lines = [
            "Calibration Summary",
            f"Residual policy: {self.residual_policy}",
            "",
            "Observed evidence and direct test-supported changes",
        ]
        for row in tested_rows:
            for evidence_id in row.evidence_ids:
                evidence = evidence_by_id[evidence_id]
                lines.append(
                    f"- {row.entity} evidence {evidence_id}: calibration ratio = "
                    f"{evidence.calibration_ratio:.4f}."
                )
                lines.append(
                    "  Observed lift / matched attribution estimate: "
                    f"{amount(evidence.observed_lift)} / "
                    f"{amount(evidence.matched_mta_estimate)}."
                )
                lines.append(
                    "  Observed lift ± standard error: "
                    f"{_observed_lift_annotation(evidence)}."
                )
            lines.extend(
                [
                    f"  Original estimate: {amount(row.original_estimate)}",
                    "  Direct calibrated estimate: "
                    f"{amount(row.original_estimate + row.direct_test_adjustment)}",
                    "  Direct test adjustment: "
                    f"{amount(row.direct_test_adjustment, signed=True)}",
                ]
            )

        lines.extend(["", "Residual allocation", f"Policy: {self.residual_policy}"])
        recipients = tuple(
            row
            for row in self.rows
            if row.role in {"residual_recipient", "residual_bucket"}
        )
        if recipients:
            for row in recipients:
                recipient_kind = (
                    "Residual bucket" if row.role == "residual_bucket" else "Untested entity"
                )
                lines.append(
                    f"- {recipient_kind} {row.entity} received "
                    f"{amount(row.residual_adjustment, signed=True)} through "
                    f"residual policy {self.residual_policy}."
                )
                lines.append(
                    "  This change was not independently measured by the test "
                    f"evidence for {tested_sources}."
                )
        else:
            lines.append(
                "- No residual recipients; untested entities were unchanged by "
                "residual allocation."
            )

        lines.extend(
            [
                "",
                "Total behavior",
                f"Original total: {amount(self.original_total)}",
                f"Raw post-test total: {amount(self.raw_post_test_total)}",
                "Residual required to preserve total: "
                f"{amount(self.required_residual, signed=True)}",
                f"Residual allocated: {amount(self.allocated_residual, signed=True)}",
                f"Calibrated total: {amount(self.calibrated_total)}",
                "",
                "Interpretation boundary",
                "Changes to untested entities are residual-allocation adjustments, "
                "not experimentally measured effects.",
                "do-attribution provides calibration mechanics; it does not "
                "select a calibration strategy.",
            ]
        )
        if self.warnings:
            lines.extend(["", "Warnings"])
            lines.extend(f"- {warning}" for warning in self.warnings)
        return "\n".join(lines)


@dataclass(frozen=True)
class TemporalCalibrationResult:
    """Immutable temporal calibration state used by later reporting layers."""

    epochs: tuple[Epoch, ...]
    epoch_windows: Mapping[Epoch, tuple[CoordinateValue, CoordinateValue]] | None
    epoch_coordinates: Mapping[Epoch, CoordinateValue] | None
    epoch_support: Mapping[Epoch, float] | None
    evidence_points: tuple[EvidencePointRecord, ...]
    coordinate_ratio_schedule: tuple[RatioScheduleRow, ...]
    epoch_ratio_schedule: tuple[EpochRatioRow, ...]
    rows: Mapping[Epoch, tuple[CalibrationRow, ...]]
    ratio_provenance: Mapping[Any, Any]
    epoch_reduction_provenance: Mapping[Any, Any]
    residual_provenance: Mapping[Any, Any]
    original_totals: Mapping[Epoch, float]
    raw_post_test_totals: Mapping[Epoch, float]
    required_residuals: Mapping[Epoch, float]
    calibrated_totals: Mapping[Epoch, float]
    ratio_policy: RatioPolicy
    epoch_ratio_policy: EpochRatioPolicy
    residual_policy: ResidualPolicy | Mapping[Epoch, ResidualPolicy]
    evidence: tuple[LiftTestEvidence, ...]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        epochs = _epoch_tuple(self.epochs)
        if self.ratio_policy not in _RATIO_POLICIES:
            raise ValueError(
                f"ratio_policy must be one of {sorted(_RATIO_POLICIES)!r}."
            )
        if self.epoch_ratio_policy not in _EPOCH_RATIO_POLICIES:
            raise ValueError(
                "epoch_ratio_policy must be one of "
                f"{sorted(_EPOCH_RATIO_POLICIES)!r}."
            )
        epoch_windows = _normalize_authoritative_epoch_windows(
            self.epoch_windows,
            epochs=epochs,
        )
        epoch_coordinates = _normalize_epoch_coordinates(
            self.epoch_coordinates,
            epochs=epochs,
            epoch_windows=epoch_windows,
        )
        epoch_support = _normalize_epoch_support(
            self.epoch_support,
            epochs=epochs,
        )

        evidence_points = _record_tuple(
            self.evidence_points,
            record_type=EvidencePointRecord,
            name="evidence_points",
        )
        coordinate_schedule = _record_tuple(
            self.coordinate_ratio_schedule,
            record_type=RatioScheduleRow,
            name="coordinate_ratio_schedule",
        )
        epoch_schedule = _validate_epoch_ratio_rows(
            _record_tuple(
                self.epoch_ratio_schedule,
                record_type=EpochRatioRow,
                name="epoch_ratio_schedule",
            )
        )
        for row in epoch_schedule:
            if row.epoch not in epochs:
                raise ValueError(
                    f"epoch ratio row refers to unknown epoch {row.epoch!r}."
                )
            if row.ratio_policy != self.ratio_policy:
                raise ValueError(
                    "epoch ratio row ratio_policy must match the result."
                )
            if row.epoch_ratio_policy != self.epoch_ratio_policy:
                raise ValueError(
                    "epoch ratio row epoch_ratio_policy must match the result."
                )
        calibrated_epoch_set = {row.epoch for row in epoch_schedule}
        calibrated_epochs = tuple(
            epoch for epoch in epochs if epoch in calibrated_epoch_set
        )
        residual_policy, policies_by_epoch = _residual_policy_by_epoch(
            self.residual_policy,
            epochs=calibrated_epochs,
        )

        if not isinstance(self.rows, Mapping):
            raise TypeError("rows must be a mapping of epochs to calibration rows.")
        missing_rows = [epoch for epoch in epochs if epoch not in self.rows]
        extra_rows = [epoch for epoch in self.rows if epoch not in epochs]
        if missing_rows or extra_rows:
            raise ValueError(
                "rows keys must exactly match epochs; "
                f"missing={missing_rows!r}, extra={extra_rows!r}."
            )
        rows_by_epoch: dict[Epoch, tuple[CalibrationRow, ...]] = {}
        for epoch in epochs:
            epoch_rows = _record_tuple(
                self.rows[epoch],
                record_type=CalibrationRow,
                name=f"rows[{epoch!r}]",
            )
            if not epoch_rows:
                raise ValueError(f"rows[{epoch!r}] must not be empty.")
            entities = [row.entity for row in epoch_rows]
            if len(set(entities)) != len(entities):
                raise ValueError(
                    f"rows[{epoch!r}] must contain each entity exactly once."
                )
            if epoch in calibrated_epoch_set:
                expected_policy = policies_by_epoch[epoch]
                if any(
                    row.residual_policy != expected_policy for row in epoch_rows
                ):
                    raise ValueError(
                        f"rows[{epoch!r}] residual policies must match the "
                        "result."
                    )
            elif any(row.residual_policy is not None for row in epoch_rows):
                raise ValueError(
                    f"uncalibrated rows[{epoch!r}] must not infer a residual "
                    "policy."
                )
            rows_by_epoch[deepcopy(epoch)] = epoch_rows
        frozen_rows = MappingProxyType(rows_by_epoch)

        original_totals = _finite_mapping_by_epoch(
            self.original_totals, epochs=epochs, name="original_totals"
        )
        raw_totals = _finite_mapping_by_epoch(
            self.raw_post_test_totals,
            epochs=epochs,
            name="raw_post_test_totals",
        )
        required_residuals = _finite_mapping_by_epoch(
            self.required_residuals,
            epochs=epochs,
            name="required_residuals",
        )
        calibrated_totals = _finite_mapping_by_epoch(
            self.calibrated_totals,
            epochs=epochs,
            name="calibrated_totals",
        )
        for epoch in epochs:
            epoch_rows = frozen_rows[epoch]
            expected_original = sum(row.original_estimate for row in epoch_rows)
            expected_raw = sum(
                row.original_estimate + row.direct_test_adjustment
                for row in epoch_rows
            )
            expected_required = expected_original - expected_raw
            allocated = sum(row.residual_adjustment for row in epoch_rows)
            expected_calibrated = sum(
                row.calibrated_estimate for row in epoch_rows
            )
            expectations = (
                ("original_totals", original_totals[epoch], expected_original),
                ("raw_post_test_totals", raw_totals[epoch], expected_raw),
                (
                    "required_residuals",
                    required_residuals[epoch],
                    expected_required,
                ),
                (
                    "calibrated_totals",
                    calibrated_totals[epoch],
                    expected_calibrated,
                ),
            )
            for name, observed, expected in expectations:
                if not isclose(
                    observed, expected, rel_tol=1e-12, abs_tol=1e-9
                ):
                    raise ValueError(
                        f"{name}[{epoch!r}] is inconsistent with allocation rows."
                    )
            if not isclose(
                calibrated_totals[epoch],
                raw_totals[epoch] + allocated,
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"calibrated_totals[{epoch!r}] is inconsistent with residuals."
                )
            if epoch not in calibrated_epoch_set:
                if (
                    allocated != 0.0
                    or not isclose(
                        raw_totals[epoch],
                        original_totals[epoch],
                        rel_tol=1e-12,
                        abs_tol=1e-9,
                    )
                    or not isclose(
                        calibrated_totals[epoch],
                        original_totals[epoch],
                        rel_tol=1e-12,
                        abs_tol=1e-9,
                    )
                ):
                    raise ValueError(
                        f"uncalibrated epoch {epoch!r} must remain wholly "
                        "unchanged."
                    )
            elif policies_by_epoch[epoch] == "allow_total_change":
                if allocated != 0.0:
                    raise ValueError(
                        f"allow_total_change cannot allocate residual in {epoch!r}."
                    )
            elif not isclose(
                allocated,
                required_residuals[epoch],
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"closed residual policy in {epoch!r} must allocate the "
                    "required residual."
                )

        normalized_evidence = _normalize_evidence_collection(
            self.evidence, allow_multiple_per_entity=True
        ).records
        evidence_ids = tuple(item.test_id for item in normalized_evidence)
        known_evidence_ids = set(evidence_ids)
        if tuple(point.evidence_id for point in evidence_points) != evidence_ids:
            raise ValueError(
                "evidence_points must preserve every evidence record in order."
            )
        expected_evidence_points = _normalize_evidence_points(
            normalized_evidence,
            epoch_windows=epoch_windows,
        )
        if evidence_points != expected_evidence_points:
            raise ValueError(
                "evidence_points must match evidence coordinates against the "
                "authoritative epoch windows exactly."
            )
        referenced_evidence_ids = {
            identifier
            for row in coordinate_schedule
            for identifier in row.source_evidence_ids
        }
        referenced_evidence_ids.update(
            identifier
            for row in epoch_schedule
            for identifier in row.source_evidence_ids
        )
        referenced_evidence_ids.update(
            identifier
            for epoch_rows in frozen_rows.values()
            for row in epoch_rows
            for identifier in row.evidence_ids
        )
        unknown_evidence_ids = [
            identifier
            for identifier in referenced_evidence_ids
            if identifier not in known_evidence_ids
        ]
        if unknown_evidence_ids:
            raise ValueError(
                "Temporal schedules and allocation rows may reference only "
                f"result evidence IDs; unknown={unknown_evidence_ids!r}."
            )

        provenance = {}
        for name in (
            "ratio_provenance",
            "epoch_reduction_provenance",
            "residual_provenance",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping.")
            provenance[name] = _deep_freeze(value)
        warnings = _warning_tuple(self.warnings)

        object.__setattr__(self, "epochs", epochs)
        object.__setattr__(self, "epoch_windows", epoch_windows)
        object.__setattr__(self, "epoch_coordinates", epoch_coordinates)
        object.__setattr__(self, "epoch_support", epoch_support)
        object.__setattr__(self, "evidence_points", evidence_points)
        object.__setattr__(
            self, "coordinate_ratio_schedule", coordinate_schedule
        )
        object.__setattr__(self, "epoch_ratio_schedule", epoch_schedule)
        object.__setattr__(self, "rows", frozen_rows)
        for name, value in provenance.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "original_totals", original_totals)
        object.__setattr__(self, "raw_post_test_totals", raw_totals)
        object.__setattr__(self, "required_residuals", required_residuals)
        object.__setattr__(self, "calibrated_totals", calibrated_totals)
        object.__setattr__(self, "ratio_policy", deepcopy(self.ratio_policy))
        object.__setattr__(
            self, "epoch_ratio_policy", deepcopy(self.epoch_ratio_policy)
        )
        object.__setattr__(self, "residual_policy", residual_policy)
        object.__setattr__(self, "evidence", normalized_evidence)
        object.__setattr__(self, "warnings", warnings)

    def to_dataframes(self) -> Any:
        """Return defensive flat tables projected from stored temporal state.

        Reporting tables are added in their documented contract order.  The
        evidence table retains every normalized observation, including points
        unmatched to an authoritative epoch and members of conflict groups.
        Beside each observation it carries the experiment's ``observed_lift``
        and reported ``standard_error``, with ``uncertainty_status`` stating
        explicitly when no uncertainty was supplied.  Those three columns
        annotate the evidence only; no other table depends on them.
        """

        point_columns = (
            "tested_entity",
            "evidence_id",
            "observed_ratio",
            "coordinate_name",
            "coordinate_value",
            "window_start",
            "window_end",
            "matched_epoch",
            "conflict_group",
        )
        evidence_columns = point_columns + (
            "observed_lift",
            "standard_error",
            "uncertainty_status",
        )
        evidence_by_id = {item.test_id: item for item in self.evidence}
        evidence_records = []
        for point in self.evidence_points:
            record = {
                column: deepcopy(getattr(point, column))
                for column in point_columns
            }
            source = evidence_by_id[point.evidence_id]
            record["observed_lift"] = source.observed_lift
            record["standard_error"] = source.standard_error
            record["uncertainty_status"] = _evidence_uncertainty_status(source)
            evidence_records.append(record)
        tables = {
            "evidence": pd.DataFrame.from_records(
                evidence_records,
                columns=evidence_columns,
            ).copy(deep=True)
        }

        schedule_columns = (
            "tested_entity",
            "coordinate_name",
            "coordinate_value",
            "schedule_ratio",
            "schedule_source",
            "source_evidence_ids",
            "left_source_evidence_id",
            "right_source_evidence_id",
            "inside_observed_support",
        )
        schedule_records = [
            {
                column: deepcopy(getattr(row, column))
                for column in schedule_columns
            }
            for row in self.coordinate_ratio_schedule
        ]
        tables["coordinate_schedule"] = pd.DataFrame.from_records(
            schedule_records,
            columns=schedule_columns,
        ).copy(deep=True)

        epoch_ratio_columns = (
            "tested_entity",
            "epoch",
            "epoch_start",
            "epoch_end",
            "epoch_coordinate",
            "epoch_support",
            "observed_ratio_count",
            "effective_ratio",
            "ratio_policy",
            "epoch_ratio_policy",
            "ratio_source",
            "evidence_conflict_policy",
            "source_evidence_ids",
            "evidence_weights",
            "left_source_evidence_id",
            "right_source_evidence_id",
            "ratio_bound_lower",
            "ratio_bound_upper",
        )
        epoch_ratio_records: list[dict[str, Any]] = []
        for row in self.epoch_ratio_schedule:
            if row.ratio_policy == "observed_only":
                observed_ratio_count = sum(
                    point.tested_entity == row.tested_entity
                    and point.matched_epoch == row.epoch
                    for point in self.evidence_points
                )
            elif row.ratio_policy == "explicit_ratio":
                observed_ratio_count = 0
            else:
                observed_ratio_count = len(row.source_evidence_ids)
            epoch_ratio_records.append(
                {
                    column: (
                        observed_ratio_count
                        if column == "observed_ratio_count"
                        else _deep_export_copy(getattr(row, column))
                    )
                    for column in epoch_ratio_columns
                }
            )
        tables["epoch_ratios"] = pd.DataFrame.from_records(
            epoch_ratio_records,
            columns=epoch_ratio_columns,
        ).copy(deep=True)

        allocation_columns = (
            "epoch",
            "entity",
            "original_estimate",
            "direct_test_adjustment",
            "residual_adjustment",
            "net_adjustment",
            "calibrated_estimate",
            "original_share",
            "calibrated_share",
            "role",
        )
        allocation_records = [
            {
                column: (
                    _deep_export_copy(epoch)
                    if column == "epoch"
                    else _deep_export_copy(getattr(row, column))
                )
                for column in allocation_columns
            }
            for epoch in self.epochs
            for row in self.rows[epoch]
        ]
        tables["allocations"] = pd.DataFrame.from_records(
            allocation_records,
            columns=allocation_columns,
        ).copy(deep=True)

        policy_by_epoch: dict[Epoch, ResidualPolicy | None] = {}
        allocated_by_epoch: dict[Epoch, float] = {}
        for epoch in self.epochs:
            policies = {row.residual_policy for row in self.rows[epoch]}
            if len(policies) != 1:  # pragma: no cover - guarded by result validation
                raise ValueError(
                    f"rows[{epoch!r}] must have one residual policy for export."
                )
            policy_by_epoch[epoch] = next(iter(policies))
            allocated_by_epoch[epoch] = sum(
                row.residual_adjustment for row in self.rows[epoch]
            )
        total_columns = (
            "epoch",
            "original_total",
            "raw_post_test_total",
            "required_residual",
            "allocated_residual",
            "calibrated_total",
            "residual_policy",
        )
        total_records = [
            {
                "epoch": _deep_export_copy(epoch),
                "original_total": self.original_totals[epoch],
                "raw_post_test_total": self.raw_post_test_totals[epoch],
                "required_residual": self.required_residuals[epoch],
                "allocated_residual": allocated_by_epoch[epoch],
                "calibrated_total": self.calibrated_totals[epoch],
                "residual_policy": _deep_export_copy(policy_by_epoch[epoch]),
            }
            for epoch in self.epochs
        ]
        tables["totals"] = pd.DataFrame.from_records(
            total_records,
            columns=total_columns,
        ).copy(deep=True)

        def ordered_pairs(*collections: Sequence[Any]) -> tuple[tuple[Any, Any], ...]:
            seen: set[tuple[Any, Any]] = set()
            supplied: list[tuple[Any, Any]] = []
            for collection in collections:
                for pair in collection:
                    if pair not in seen:
                        seen.add(pair)
                        supplied.append(pair)
            ordered = [
                pair
                for epoch in self.epochs
                for pair in supplied
                if pair[1] == epoch
            ]
            ordered.extend(pair for pair in supplied if pair not in ordered)
            return tuple(ordered)

        ratio_provenance = self.ratio_provenance
        candidate_pairs = tuple(ratio_provenance.get("candidate_pairs", ()))
        schedule_status = ratio_provenance.get("schedule_status_by_pair", {})
        schedule_details = ratio_provenance.get("schedule_provenance_by_pair", {})
        conflict_policies = ratio_provenance.get("conflict_policy_by_pair", {})
        conflict_details = ratio_provenance.get(
            "conflict_resolution_by_pair", {}
        )
        uncalibrated_ratio_pairs = tuple(
            ratio_provenance.get("uncalibrated_pairs", ())
        )
        ratio_pairs = ordered_pairs(
            candidate_pairs,
            tuple(schedule_status),
            tuple(schedule_details),
            tuple(conflict_policies),
            tuple(conflict_details),
            uncalibrated_ratio_pairs,
        )
        candidate_pair_set = set(candidate_pairs)
        uncalibrated_ratio_set = set(uncalibrated_ratio_pairs)
        ratio_provenance_columns = (
            "tested_entity",
            "epoch",
            "ratio_policy",
            "candidate",
            "schedule_status",
            "schedule_provenance",
            "evidence_conflict_policy",
            "conflict_resolution",
            "uncalibrated",
        )
        ratio_provenance_records = [
            {
                "tested_entity": _deep_export_copy(pair[0]),
                "epoch": _deep_export_copy(pair[1]),
                "ratio_policy": _deep_export_copy(
                    ratio_provenance.get("ratio_policy", self.ratio_policy)
                ),
                "candidate": pair in candidate_pair_set,
                "schedule_status": _deep_export_copy(
                    schedule_status.get(pair)
                ),
                "schedule_provenance": _deep_export_copy(
                    schedule_details.get(pair)
                ),
                "evidence_conflict_policy": _deep_export_copy(
                    conflict_policies.get(pair)
                ),
                "conflict_resolution": _deep_export_copy(
                    conflict_details.get(pair)
                ),
                "uncalibrated": pair in uncalibrated_ratio_set,
            }
            for pair in ratio_pairs
        ]
        tables["ratio_provenance"] = pd.DataFrame.from_records(
            ratio_provenance_records,
            columns=ratio_provenance_columns,
        ).copy(deep=True)

        reduction_by_pair = self.epoch_reduction_provenance.get("by_pair", {})
        uncalibrated_reduction_pairs = tuple(
            self.epoch_reduction_provenance.get("uncalibrated_pairs", ())
        )
        reduction_pairs = ordered_pairs(
            tuple(reduction_by_pair),
            uncalibrated_reduction_pairs,
        )
        calibrated_pair_set = {
            (row.tested_entity, row.epoch) for row in self.epoch_ratio_schedule
        }
        reduction_provenance_columns = (
            "tested_entity",
            "epoch",
            "epoch_ratio_policy",
            "calibrated",
            "provenance",
        )
        reduction_provenance_records = [
            {
                "tested_entity": _deep_export_copy(pair[0]),
                "epoch": _deep_export_copy(pair[1]),
                "epoch_ratio_policy": _deep_export_copy(
                    self.epoch_ratio_policy
                ),
                "calibrated": pair in calibrated_pair_set,
                "provenance": _deep_export_copy(reduction_by_pair.get(pair)),
            }
            for pair in reduction_pairs
        ]
        tables["epoch_reduction_provenance"] = pd.DataFrame.from_records(
            reduction_provenance_records,
            columns=reduction_provenance_columns,
        ).copy(deep=True)

        residual_provenance_columns = (
            "epoch",
            "residual_policy",
            "allocated_residual",
            "provenance",
        )
        residual_provenance_records = [
            {
                "epoch": _deep_export_copy(epoch),
                "residual_policy": _deep_export_copy(policy_by_epoch[epoch]),
                "allocated_residual": allocated_by_epoch[epoch],
                "provenance": _deep_export_copy(
                    self.residual_provenance.get(epoch)
                ),
            }
            for epoch in self.epochs
        ]
        tables["residual_provenance"] = pd.DataFrame.from_records(
            residual_provenance_records,
            columns=residual_provenance_columns,
        ).copy(deep=True)

        tables["warnings"] = pd.DataFrame.from_records(
            [{"warning": deepcopy(warning)} for warning in self.warnings],
            columns=("warning",),
        ).copy(deep=True)
        return tables

    def summary(self) -> str:
        """Render deterministic temporal calibration audit text from stored state."""

        def amount(value: float, *, signed: bool = False) -> str:
            normalized = 0.0 if value == 0.0 else value
            return f"{normalized:+.2f}" if signed else f"{normalized:.2f}"

        def ratio(value: float) -> str:
            normalized = 0.0 if value == 0.0 else value
            return f"{normalized:.4f}"

        def label(value: Any) -> str:
            if isinstance(value, (date, datetime)):
                return value.isoformat()
            return str(value)

        if isinstance(self.residual_policy, Mapping):
            residual_policy_label = "by epoch (" + ", ".join(
                f"{label(epoch)}={policy}"
                for epoch, policy in self.residual_policy.items()
            ) + ")"
        else:
            residual_policy_label = self.residual_policy

        lines = [
            "Temporal Calibration Summary",
            f"Ratio policy: {self.ratio_policy}",
            f"Epoch-ratio policy: {self.epoch_ratio_policy}",
            f"Residual policy: {residual_policy_label}",
            "",
            "Authoritative temporal design",
        ]
        for epoch in self.epochs:
            lines.append(
                f"- Epoch {label(epoch)} was supplied by the attribution result."
            )
            if self.epoch_windows is not None:
                start, end = self.epoch_windows[epoch]
                lines.append(
                    f"  Boundary: {label(start)} to {label(end)}."
                )
            lines.extend(
                [
                    "  Boundary source: outcome-signal epoch design.",
                    "  Calibration did not modify the epoch.",
                ]
            )

        evidence_by_id = {item.test_id: item for item in self.evidence}
        lines.extend(["", "Observed evidence"])
        for point in self.evidence_points:
            lines.append(
                f"- {label(point.tested_entity)} evidence {point.evidence_id}:"
            )
            if point.coordinate_name is None:
                lines.append("  coordinate: not supplied")
            else:
                lines.append(
                    f"  coordinate {point.coordinate_name} = "
                    f"{label(point.coordinate_value)}"
                )
            lines.append(
                "  observed calibration ratio = "
                f"{ratio(point.observed_ratio)}"
            )
            lines.append(
                "  observed lift ± standard error = "
                f"{_observed_lift_annotation(evidence_by_id[point.evidence_id])}"
            )
            if point.matched_epoch is None:
                lines.append("  matched epoch: none (observation retained)")
            else:
                lines.append(
                    f"  matched epoch = {label(point.matched_epoch)}"
                )
            if point.conflict_group is not None:
                lines.append(
                    "  conflict group = " + ", ".join(point.conflict_group)
                )

        lines.extend(["", "Coordinate schedule construction"])
        if not self.coordinate_ratio_schedule:
            lines.append(
                f"- No coordinate schedule rows; ratio policy "
                f"{self.ratio_policy} supplies epoch-level values directly."
            )
        for row in self.coordinate_ratio_schedule:
            entity = label(row.tested_entity)
            if row.coordinate_name is None:
                coordinate_text = "without a coordinate"
            else:
                coordinate_text = (
                    f"at {row.coordinate_name} = {label(row.coordinate_value)}"
                )
            if row.schedule_source == "unsupported":
                lines.append(
                    f"- {entity} coordinate-linear schedule {coordinate_text}: "
                    "unsupported; no ratio was inferred and no extrapolation "
                    "was performed."
                )
                continue
            lines.append(
                f"- {entity} {row.schedule_source} schedule point "
                f"{coordinate_text}: ratio {ratio(row.schedule_ratio)}."
            )
            if row.schedule_source == "interpolated":
                lines.append(
                    f"  left anchor {row.left_source_evidence_id}; "
                    f"right anchor {row.right_source_evidence_id}."
                )
                lines.append(
                    "  Interpolated inside observed support; this is a "
                    "constructed schedule value, not a new observation."
                )
            elif row.source_evidence_ids:
                lines.append(
                    "  source evidence: " + ", ".join(row.source_evidence_ids)
                )
                if row.schedule_source == "constant":
                    lines.append(
                        "  Transported under the practitioner-selected constant "
                        "policy; this is a constructed schedule value, not a "
                        "new observation."
                    )

        lines.extend(["", "Final epoch-level ratios"])
        conflict_by_pair = self.ratio_provenance.get(
            "conflict_resolution_by_pair", {}
        )
        for row in self.epoch_ratio_schedule:
            pair = (row.tested_entity, row.epoch)
            lines.append(
                f"- {label(row.tested_entity)} / {label(row.epoch)} effective "
                f"ratio {ratio(row.effective_ratio)}:"
            )
            if row.epoch_ratio_policy == "representative_coordinate":
                lines.append(
                    "  schedule evaluated at representative coordinate "
                    f"{label(row.epoch_coordinate)}."
                )
            elif row.epoch_ratio_policy == "coordinate_average":
                lines.append(
                    "  uniform average of the coordinate-linear schedule over "
                    "the full authoritative epoch."
                )
            elif row.epoch_ratio_policy == "support_weighted_average":
                lines.append(
                    "  support-weighted average of the coordinate-linear "
                    "schedule using practitioner-declared coordinate weights."
                )
                provenance = self.epoch_reduction_provenance.get(
                    "by_pair", {}
                ).get(pair, {})
                evaluations = provenance.get("coordinate_evaluations", ())
                if evaluations:
                    lines.append(
                        "  declared coordinate weights: "
                        + ", ".join(
                            f"{label(item['coordinate'])}={item['weight']:.4f}"
                            for item in evaluations
                        )
                        + "."
                    )
            elif row.ratio_source == "explicit":
                lines.append(
                    "  final epoch-level ratio was supplied explicitly by the "
                    "practitioner."
                )
            elif row.ratio_source == "constant":
                lines.append(
                    "  transported constant schedule value from the declared "
                    "source evidence; this is not observed evidence for the "
                    "target epoch."
                )
            elif row.evidence_conflict_policy is None:
                lines.append("  direct observed evidence was used.")

            if row.evidence_conflict_policy is not None:
                lines.append(
                    "  evidence conflict policy: "
                    f"{row.evidence_conflict_policy}."
                )
                conflict = conflict_by_pair.get(pair, {})
                resolution = conflict.get("resolution", {})
                if row.evidence_conflict_policy == "select_evidence":
                    selected = resolution.get("selected_evidence_id")
                    candidates = resolution.get("candidate_evidence_ids", ())
                    lines.append(
                        f"  selected evidence {selected} from candidates "
                        f"{', '.join(candidates)}."
                    )
                elif row.evidence_conflict_policy == "weighted_ratio":
                    weights = resolution.get(
                        "evidence_combination_weights", {}
                    )
                    lines.append(
                        "  Evidence weights were supplied by the practitioner: "
                        + ", ".join(
                            f"{identifier}={weight:.4f}"
                            for identifier, weight in weights.items()
                        )
                        + "."
                    )
                else:
                    lines.append(
                        "  pooled observed lift / pooled matched attribution "
                        "estimate: "
                        f"{amount(resolution['pooled_observed_lift'])} / "
                        f"{amount(resolution['pooled_matched_mta_estimate'])}."
                    )
            if row.source_evidence_ids:
                lines.append(
                    "  source evidence IDs: "
                    + ", ".join(row.source_evidence_ids)
                    + "."
                )
            if row.ratio_bound_lower is not None:
                lines.append(
                    "  validated without clipping inside ratio bounds "
                    f"[{ratio(row.ratio_bound_lower)}, "
                    f"{ratio(row.ratio_bound_upper)}]."
                )

        uncalibrated_pairs = self.epoch_reduction_provenance.get(
            "uncalibrated_pairs", ()
        )
        for entity, epoch in uncalibrated_pairs:
            lines.append(
                f"- {label(entity)} / {label(epoch)} was not calibrated; no "
                "effective ratio was supplied or inferred."
            )

        lines.extend(["", "Direct test-supported changes"])
        for epoch in self.epochs:
            for row in self.rows[epoch]:
                if row.role != "tested":
                    continue
                direct_value = row.original_estimate + row.direct_test_adjustment
                lines.append(
                    f"- {label(row.entity)} / {label(epoch)} was adjusted from "
                    f"{amount(row.original_estimate)} to {amount(direct_value)}."
                )
                lines.append(
                    "  Direct test adjustment: "
                    f"{amount(row.direct_test_adjustment, signed=True)}."
                )
                lines.append(
                    "  evidence IDs: " + ", ".join(row.evidence_ids) + "."
                )

        lines.extend(["", "Residual-allocation changes"])
        for epoch in self.epochs:
            epoch_rows = self.rows[epoch]
            recipients = tuple(
                row
                for row in epoch_rows
                if row.role in {"residual_recipient", "residual_bucket"}
            )
            if recipients:
                tested_sources = ", ".join(
                    f"{label(row.entity)} ({', '.join(row.evidence_ids)})"
                    for row in epoch_rows
                    if row.role == "tested"
                )
                for row in recipients:
                    lines.append(
                        f"- {label(row.entity)} / {label(epoch)} received "
                        f"{amount(row.residual_adjustment, signed=True)} through "
                        f"residual policy {row.residual_policy}."
                    )
                    lines.append(
                        "  This change was not independently measured by the "
                        f"test evidence for {tested_sources}."
                    )
            elif all(row.residual_policy is None for row in epoch_rows):
                lines.append(
                    f"- Epoch {label(epoch)} was not calibrated; no residual "
                    "policy was inferred."
                )
            else:
                policy = epoch_rows[0].residual_policy
                lines.append(
                    f"- Epoch {label(epoch)} allocated no residual through "
                    f"policy {policy}."
                )

        lines.extend(["", "Total behavior"])
        for epoch in self.epochs:
            allocated = sum(
                row.residual_adjustment for row in self.rows[epoch]
            )
            lines.extend(
                [
                    f"- Epoch {label(epoch)}:",
                    f"  Original total: {amount(self.original_totals[epoch])}",
                    "  Raw post-test total: "
                    f"{amount(self.raw_post_test_totals[epoch])}",
                    "  Residual required to preserve total: "
                    f"{amount(self.required_residuals[epoch], signed=True)}",
                    f"  Residual allocated: {amount(allocated, signed=True)}",
                    f"  Calibrated total: {amount(self.calibrated_totals[epoch])}",
                ]
            )

        lines.extend(
            [
                "",
                "Interpretation boundary",
                "Attribution epochs are authoritative outcome-driven or "
                "practitioner-supplied analysis periods; no tested entity owns "
                "them.",
                "Calibration did not create, move, split, merge, relabel, or "
                "reorder any epoch.",
                "Interpolation addresses missing evidence; it does not create "
                "experimental observations or entity-specific time partitions.",
                "Changes to untested entities are residual-allocation adjustments, "
                "not experimentally measured effects.",
                "do-attribution provides calibration mechanics; it does not "
                "select a calibration strategy.",
            ]
        )
        if self.warnings:
            lines.extend(["", "Warnings"])
            lines.extend(f"- {warning}" for warning in self.warnings)
        return "\n".join(lines)

    def plot_epoch_alignment(self) -> Any:
        """Plot authoritative epoch boundaries, evidence, and stored support."""

        import matplotlib.pyplot as plt  # lazy optional dependency

        show_support = self.epoch_support is not None and bool(self.epoch_support)
        if show_support:
            fig, (ax, support_ax) = plt.subplots(
                2,
                1,
                figsize=(12, 7),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
            )
        else:
            fig, ax = plt.subplots(figsize=(12, 5))
            support_ax = None

        if self.epoch_windows is None:
            windows = tuple(
                (epoch, float(index), float(index + 1))
                for index, epoch in enumerate(self.epochs)
            )
            categorical = True
        else:
            windows = tuple(
                (epoch, self.epoch_windows[epoch][0], self.epoch_windows[epoch][1])
                for epoch in self.epochs
            )
            categorical = False

        boundary_values: list[Any] = []
        for _, start, end in windows:
            for boundary in (start, end):
                if boundary not in boundary_values:
                    boundary_values.append(boundary)
        for index, boundary in enumerate(boundary_values):
            ax.axvline(
                boundary,
                color="tab:gray",
                linestyle="--",
                linewidth=1.2,
                label=(
                    "Authoritative epoch boundary" if index == 0 else None
                ),
            )
            if support_ax is not None:
                support_ax.axvline(
                    boundary,
                    color="tab:gray",
                    linestyle="--",
                    linewidth=0.8,
                )

        for index, (epoch, start, end) in enumerate(windows):
            color = "tab:blue" if index % 2 == 0 else "tab:orange"
            ax.axvspan(start, end, color=color, alpha=0.06)
            midpoint = start + (end - start) / 2
            ax.text(
                midpoint,
                0.98,
                f"Epoch {epoch}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=9,
            )

        entity_order = tuple(
            dict.fromkeys(point.tested_entity for point in self.evidence_points)
        )
        entity_position = {
            entity: float(index + 1) for index, entity in enumerate(entity_order)
        }
        colors = plt.cm.tab10.colors
        for entity_index, entity in enumerate(entity_order):
            entity_points = tuple(
                point
                for point in self.evidence_points
                if point.tested_entity == entity
            )
            for matched, marker, status in (
                (True, "o", "matched"),
                (False, "X", "unmatched / unpositioned"),
            ):
                selected = tuple(
                    point
                    for point in entity_points
                    if (point.matched_epoch is not None) == matched
                )
                if not selected:
                    continue
                x_values: list[Any] = []
                for point in selected:
                    if not categorical and point.coordinate_value is not None:
                        x_values.append(point.coordinate_value)
                    elif point.matched_epoch in self.epochs:
                        epoch_index = self.epochs.index(point.matched_epoch)
                        x_values.append(float(epoch_index) + 0.5)
                    else:
                        x_values.append(float(len(self.epochs)) + 0.25)
                y_values = [entity_position[entity]] * len(selected)
                ax.scatter(
                    x_values,
                    y_values,
                    marker=marker,
                    s=65,
                    color=colors[entity_index % len(colors)],
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=3,
                    label=f"Evidence: {entity} ({status})",
                )
                for x_value, y_value, point in zip(
                    x_values, y_values, selected
                ):
                    ax.annotate(
                        point.evidence_id,
                        (x_value, y_value),
                        xytext=(4, 5),
                        textcoords="offset points",
                        fontsize=8,
                    )

        ax.set_yticks([entity_position[entity] for entity in entity_order])
        ax.set_yticklabels([str(entity) for entity in entity_order])
        ax.set_ylabel("Tested entity evidence")
        ax.set_title("Authoritative Attribution Epoch Alignment")
        ax.grid(axis="y", alpha=0.2)
        handles, legend_labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, legend_labels, loc="best")

        if support_ax is not None:
            centers: list[Any] = []
            widths: list[Any] = []
            supports: list[float] = []
            for epoch, start, end in windows:
                if epoch not in self.epoch_support:
                    continue
                centers.append(start + (end - start) / 2)
                widths.append((end - start) * 0.8)
                supports.append(self.epoch_support[epoch])
            support_ax.bar(
                centers,
                supports,
                width=widths,
                color="tab:green",
                alpha=0.55,
                label="Stored epoch support (diagnostic)",
            )
            support_ax.set_ylabel("Epoch support")
            support_ax.grid(axis="y", alpha=0.2)
            support_ax.legend(loc="best")

        if categorical:
            ax.set_xticks(
                [float(index) + 0.5 for index in range(len(self.epochs))]
            )
            ax.set_xticklabels([str(epoch) for epoch in self.epochs])
        ax.set_xlabel("Evidence coordinate / authoritative epoch")
        if support_ax is not None:
            support_ax.set_xlabel("Evidence coordinate / authoritative epoch")
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig

    def plot_ratio_schedule(self) -> Any:
        """Plot coordinate schedule layers separately from final epoch ratios."""

        import matplotlib.pyplot as plt  # lazy optional dependency

        fig, (schedule_ax, epoch_ax) = plt.subplots(
            2,
            1,
            figsize=(12, 8),
            gridspec_kw={"height_ratios": [3, 2]},
        )
        colors = plt.cm.tab10.colors
        entity_order = tuple(
            dict.fromkeys(
                [row.tested_entity for row in self.coordinate_ratio_schedule]
                + [row.tested_entity for row in self.epoch_ratio_schedule]
                + [
                    pair[0]
                    for pair in self.epoch_reduction_provenance.get(
                        "uncalibrated_pairs", ()
                    )
                ]
            )
        )
        color_by_entity = {
            entity: colors[index % len(colors)]
            for index, entity in enumerate(entity_order)
        }
        evidence_by_id = {item.test_id: item for item in self.evidence}

        coordinate_keys: list[tuple[Any, Any]] = []
        for row in self.coordinate_ratio_schedule:
            key = (row.coordinate_name, row.coordinate_value)
            if key not in coordinate_keys:
                coordinate_keys.append(key)
        coordinate_position = {
            key: float(index) for index, key in enumerate(coordinate_keys)
        }

        if not self.coordinate_ratio_schedule:
            schedule_ax.text(
                0.5,
                0.5,
                f"No coordinate schedule: {self.ratio_policy} supplies "
                "epoch-level values directly.",
                transform=schedule_ax.transAxes,
                ha="center",
                va="center",
            )
        for entity in entity_order:
            rows = tuple(
                row
                for row in self.coordinate_ratio_schedule
                if row.tested_entity == entity
            )
            if not rows:
                continue
            supported = tuple(row for row in rows if row.schedule_ratio is not None)
            if supported:
                schedule_ax.plot(
                    [
                        coordinate_position[
                            (row.coordinate_name, row.coordinate_value)
                        ]
                        for row in supported
                    ],
                    [row.schedule_ratio for row in supported],
                    color=color_by_entity[entity],
                    linewidth=1.5,
                    alpha=0.75,
                    label=f"Schedule path — {entity}",
                )
            for source, marker, size, legend_prefix in (
                ("observed", "o", 65, "Observed anchor"),
                ("interpolated", "D", 55, "Constructed schedule value"),
                ("constant", "s", 55, "Constant schedule value"),
            ):
                selected = tuple(row for row in rows if row.schedule_source == source)
                if not selected:
                    continue
                schedule_ax.scatter(
                    [
                        coordinate_position[
                            (row.coordinate_name, row.coordinate_value)
                        ]
                        for row in selected
                    ],
                    [row.schedule_ratio for row in selected],
                    marker=marker,
                    s=size,
                    color=color_by_entity[entity],
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=3,
                    label=f"{legend_prefix} — {entity}",
                )
                if source != "observed":
                    continue
                # An observed anchor is the experiment's lift divided by its
                # matched attribution estimate, so the experiment's reported
                # standard error of that lift sits on this axis after the
                # same division.  The bar annotates the evidence only; it is
                # not an interval on any constructed or calibrated value.
                reported = tuple(
                    (row, evidence_by_id[row.source_evidence_ids[0]])
                    for row in selected
                    if evidence_by_id[row.source_evidence_ids[0]].standard_error
                    is not None
                )
                if reported:
                    schedule_ax.errorbar(
                        [
                            coordinate_position[
                                (row.coordinate_name, row.coordinate_value)
                            ]
                            for row, _ in reported
                        ],
                        [row.schedule_ratio for row, _ in reported],
                        yerr=[
                            item.standard_error / abs(item.matched_mta_estimate)
                            for _, item in reported
                        ],
                        fmt="none",
                        ecolor=color_by_entity[entity],
                        elinewidth=1.3,
                        capsize=4,
                        zorder=2,
                        label=(
                            "Experiment-reported ±1 SE of observed lift "
                            f"(ratio scale) — {entity}"
                        ),
                    )
            unsupported = tuple(
                row for row in rows if row.schedule_source == "unsupported"
            )
            if unsupported:
                schedule_ax.scatter(
                    [
                        coordinate_position[
                            (row.coordinate_name, row.coordinate_value)
                        ]
                        for row in unsupported
                    ],
                    [0.03] * len(unsupported),
                    transform=schedule_ax.get_xaxis_transform(),
                    marker="X",
                    s=65,
                    color=color_by_entity[entity],
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=3,
                    label=f"Outside observed support — {entity}",
                )
                for row in unsupported:
                    schedule_ax.annotate(
                        "not calibrated / no extrapolation",
                        (
                            coordinate_position[
                                (row.coordinate_name, row.coordinate_value)
                            ],
                            0.03,
                        ),
                        xycoords=schedule_ax.get_xaxis_transform(),
                        xytext=(3, 5),
                        textcoords="offset points",
                        fontsize=7,
                    )

        schedule_ax.set_xticks(range(len(coordinate_keys)))
        schedule_ax.set_xticklabels(
            [
                (
                    "no coordinate"
                    if name is None
                    else f"{name}={value}"
                )
                for name, value in coordinate_keys
            ],
            rotation=25,
            ha="right",
        )
        schedule_ax.set_ylabel("Schedule ratio")
        schedule_ax.set_title(
            "Coordinate Schedule: Observed vs Constructed Values"
        )
        schedule_ax.grid(axis="y", alpha=0.2)
        schedule_handles, schedule_labels = schedule_ax.get_legend_handles_labels()
        if schedule_handles:
            schedule_ax.legend(schedule_handles, schedule_labels, loc="best")

        epoch_positions = {
            epoch: float(index) for index, epoch in enumerate(self.epochs)
        }
        for entity in entity_order:
            rows = tuple(
                row
                for row in self.epoch_ratio_schedule
                if row.tested_entity == entity
            )
            if rows:
                epoch_ax.scatter(
                    [epoch_positions[row.epoch] for row in rows],
                    [row.effective_ratio for row in rows],
                    marker="s",
                    s=65,
                    color=color_by_entity[entity],
                    edgecolor="black",
                    linewidth=0.5,
                    label=f"Final epoch ratio — {entity}",
                )
        uncalibrated_pairs = tuple(
            self.epoch_reduction_provenance.get("uncalibrated_pairs", ())
        )
        for entity in entity_order:
            pairs = tuple(
                pair for pair in uncalibrated_pairs if pair[0] == entity
            )
            if not pairs:
                continue
            epoch_ax.scatter(
                [epoch_positions[pair[1]] for pair in pairs],
                [0.03] * len(pairs),
                transform=epoch_ax.get_xaxis_transform(),
                marker="X",
                s=65,
                color=color_by_entity[entity],
                edgecolor="black",
                linewidth=0.5,
                label=f"Not calibrated — {entity}",
            )

        bounds = tuple(
            dict.fromkeys(
                (row.ratio_bound_lower, row.ratio_bound_upper)
                for row in self.epoch_ratio_schedule
                if row.ratio_bound_lower is not None
            )
        )
        for lower, upper in bounds:
            epoch_ax.axhline(
                lower,
                color="tab:red",
                linestyle="--",
                linewidth=1.0,
                label=f"Lower ratio bound ({lower:.4f})",
            )
            epoch_ax.axhline(
                upper,
                color="tab:red",
                linestyle=":",
                linewidth=1.0,
                label=f"Upper ratio bound ({upper:.4f})",
            )
        epoch_ax.set_xticks(range(len(self.epochs)))
        epoch_ax.set_xticklabels([str(epoch) for epoch in self.epochs])
        epoch_ax.set_xlabel("Authoritative attribution epoch")
        epoch_ax.set_ylabel("Effective ratio")
        epoch_ax.set_title("Final Epoch-Level Ratios and Coverage")
        epoch_ax.grid(axis="y", alpha=0.2)
        epoch_handles, epoch_labels = epoch_ax.get_legend_handles_labels()
        if epoch_handles:
            epoch_ax.legend(epoch_handles, epoch_labels, loc="best")
        fig.tight_layout()
        return fig

    def plot_evidence_uncertainty(self) -> Any:
        """Plot each evidence record's observed lift with its reported uncertainty.

        One row per stored evidence record, in evidence order, on an axis in
        the experiment's own lift units.  The point is the observed lift.  A
        horizontal bar spans ``observed lift ± standard_error`` when the
        experiment reported a standard error; a record without one is drawn
        as a point with no bar and labelled ``no uncertainty supplied``.

        The bars are the experiment's reported standard error of its observed
        lift.  They are not intervals on calibrated output, attribution
        estimates, calibration ratios, or schedule values, and nothing in the
        calibration arithmetic reads them.
        """

        import matplotlib.pyplot as plt  # lazy optional dependency

        records = self.evidence
        figure_height = max(3.6, 0.7 * len(records) + 2.4)
        fig, ax = plt.subplots(figsize=(12, figure_height))
        positions = [float(index) for index in range(len(records))]
        colors = plt.cm.tab10.colors
        entity_order = tuple(
            dict.fromkeys(record.tested_entity for record in records)
        )

        for entity_index, entity in enumerate(entity_order):
            color = colors[entity_index % len(colors)]
            placed = tuple(
                (position, record)
                for position, record in zip(positions, records)
                if record.tested_entity == entity
            )
            reported = tuple(
                (position, record)
                for position, record in placed
                if record.standard_error is not None
            )
            if reported:
                ax.errorbar(
                    [record.observed_lift for _, record in reported],
                    [position for position, _ in reported],
                    xerr=[record.standard_error for _, record in reported],
                    fmt="none",
                    ecolor=color,
                    elinewidth=1.6,
                    capsize=5,
                    zorder=2,
                    label=f"Experiment-reported ±1 standard error — {entity}",
                )
            ax.scatter(
                [record.observed_lift for _, record in placed],
                [position for position, _ in placed],
                marker="o",
                s=65,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
                label=f"Observed experimental lift — {entity}",
            )

        for position, record in zip(positions, records):
            if record.standard_error is None:
                text = f"{record.observed_lift:.2f} ({_NO_UNCERTAINTY_SUPPLIED})"
            else:
                text = (
                    f"{record.observed_lift:.2f} ± {record.standard_error:.2f}"
                )
            ax.annotate(
                text,
                (record.observed_lift, position),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

        ax.axvline(0.0, color="0.2", linewidth=0.8)
        ax.margins(x=0.1)
        ax.set_yticks(positions)
        ax.set_yticklabels(
            [f"{record.test_id} ({record.tested_entity})" for record in records]
        )
        ax.set_ylim(len(records) - 0.4, -0.6)
        ax.set_xlabel("Observed experimental lift")
        ax.set_ylabel("Evidence record")
        ax.set_title(
            "Experimental Evidence: Observed Lift ± Experiment-Reported "
            "Standard Error"
        )
        ax.grid(axis="x", alpha=0.2)
        # The bars are explained where they are decoded, in the legend, so
        # the axis label stays one line under any figure layout.
        any_reported = any(
            record.standard_error is not None for record in records
        )
        ax.legend(
            loc="best",
            title=(
                "Bars: ±1 standard error reported by the experiment for its "
                "observed lift —\nnot a confidence interval on calibrated "
                "output or attribution estimates"
                if any_reported
                else None
            ),
            title_fontsize=8,
        )
        fig.tight_layout()
        return fig

    def plot_calibration_effects(self) -> Any:
        """Plot original versus direct-calibrated tested-entity values by epoch."""

        import matplotlib.pyplot as plt  # lazy optional dependency

        fig, (value_ax, adjustment_ax) = plt.subplots(
            2,
            1,
            figsize=(12, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 2]},
        )
        tested_entities = tuple(
            dict.fromkeys(item.tested_entity for item in self.evidence)
        )
        colors = plt.cm.tab10.colors
        epoch_position = {
            epoch: float(index) for index, epoch in enumerate(self.epochs)
        }
        entity_span = 0.8 / max(1, len(tested_entities))
        bar_width = entity_span * 0.38
        calibrated_pairs = {
            (row.tested_entity, row.epoch) for row in self.epoch_ratio_schedule
        }

        for entity_index, entity in enumerate(tested_entities):
            center_offset = (
                entity_index - (len(tested_entities) - 1) / 2.0
            ) * entity_span
            present: list[tuple[Epoch, CalibrationRow]] = []
            for epoch in self.epochs:
                row = next(
                    (
                        item
                        for item in self.rows[epoch]
                        if item.entity == entity
                    ),
                    None,
                )
                if row is not None:
                    present.append((epoch, row))
            if not present:
                continue
            centers = [
                epoch_position[epoch] + center_offset
                for epoch, _ in present
            ]
            original_values = [row.original_estimate for _, row in present]
            direct_values = [
                row.original_estimate + row.direct_test_adjustment
                for _, row in present
            ]
            adjustments = [row.direct_test_adjustment for _, row in present]
            color = colors[entity_index % len(colors)]
            value_ax.bar(
                [center - bar_width / 2 for center in centers],
                original_values,
                width=bar_width,
                color=color,
                alpha=0.32,
                edgecolor="black",
                linewidth=0.4,
                label=f"Original — {entity}",
            )
            value_ax.bar(
                [center + bar_width / 2 for center in centers],
                direct_values,
                width=bar_width,
                color=color,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.4,
                label=f"Direct calibrated — {entity}",
            )
            adjustment_ax.bar(
                centers,
                adjustments,
                width=bar_width * 1.6,
                color=color,
                alpha=0.75,
                edgecolor="black",
                linewidth=0.4,
                label=f"Direct test adjustment — {entity}",
            )

            uncalibrated = tuple(
                (epoch, row, center)
                for (epoch, row), center in zip(present, centers)
                if (entity, epoch) not in calibrated_pairs
            )
            if uncalibrated:
                value_ax.scatter(
                    [center + bar_width / 2 for _, _, center in uncalibrated],
                    [row.original_estimate for _, row, _ in uncalibrated],
                    marker="X",
                    s=65,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=1.5,
                    zorder=4,
                    label=f"Uncalibrated; original retained — {entity}",
                )
                adjustment_ax.scatter(
                    [center for _, _, center in uncalibrated],
                    [0.0] * len(uncalibrated),
                    marker="X",
                    s=55,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=1.4,
                    zorder=4,
                    label=f"No direct adjustment — {entity}",
                )

        value_ax.set_ylabel("Estimate")
        value_ax.set_title(
            "Tested-Entity Original vs Evidence-Backed Direct Values"
        )
        value_ax.grid(axis="y", alpha=0.2)
        value_handles, value_labels = value_ax.get_legend_handles_labels()
        if value_handles:
            value_ax.legend(value_handles, value_labels, loc="best")

        adjustment_ax.axhline(0.0, color="black", linewidth=0.8)
        adjustment_ax.set_xticks(range(len(self.epochs)))
        adjustment_ax.set_xticklabels([str(epoch) for epoch in self.epochs])
        adjustment_ax.set_xlabel("Authoritative attribution epoch")
        adjustment_ax.set_ylabel("Direct test adjustment")
        adjustment_ax.set_title(
            "Direct Test Adjustments (Residual Adjustments Excluded)"
        )
        adjustment_ax.grid(axis="y", alpha=0.2)
        adjustment_handles, adjustment_labels = (
            adjustment_ax.get_legend_handles_labels()
        )
        if adjustment_handles:
            adjustment_ax.legend(
                adjustment_handles,
                adjustment_labels,
                loc="best",
            )
        fig.tight_layout()
        return fig

    def plot_residual_allocation(self) -> Any:
        """Plot required/allocated residuals and recipient-level adjustments."""

        import matplotlib.pyplot as plt  # lazy optional dependency

        fig, (total_ax, recipient_ax) = plt.subplots(
            2,
            1,
            figsize=(12, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 3]},
        )
        epoch_positions = [float(index) for index in range(len(self.epochs))]
        required = [self.required_residuals[epoch] for epoch in self.epochs]
        allocated = [
            sum(row.residual_adjustment for row in self.rows[epoch])
            for epoch in self.epochs
        ]
        policy_by_epoch: dict[Epoch, ResidualPolicy | None] = {}
        for epoch in self.epochs:
            policies = {row.residual_policy for row in self.rows[epoch]}
            policy_by_epoch[epoch] = next(iter(policies))

        width = 0.36
        total_ax.bar(
            [position - width / 2 for position in epoch_positions],
            required,
            width=width,
            color="tab:orange",
            alpha=0.75,
            edgecolor="black",
            linewidth=0.4,
            label="Required residual to preserve total",
        )
        total_ax.bar(
            [position + width / 2 for position in epoch_positions],
            allocated,
            width=width,
            color="tab:blue",
            alpha=0.75,
            edgecolor="black",
            linewidth=0.4,
            label="Allocated residual",
        )
        open_positions: list[float] = []
        uncalibrated_positions: list[float] = []
        for position, epoch in zip(epoch_positions, self.epochs):
            policy = policy_by_epoch[epoch]
            if policy is None:
                status = "not calibrated"
                uncalibrated_positions.append(position)
            elif policy == "allow_total_change":
                status = "allow_total_change (total may change)"
                open_positions.append(position)
            else:
                status = f"closed: {policy}"
            total_ax.annotate(
                status,
                (position, 0.98),
                xycoords=("data", "axes fraction"),
                xytext=(0, -2),
                textcoords="offset points",
                ha="center",
                va="top",
                rotation=90,
                fontsize=7,
            )
        if open_positions:
            total_ax.scatter(
                open_positions,
                [0.03] * len(open_positions),
                transform=total_ax.get_xaxis_transform(),
                marker="X",
                s=55,
                color="tab:red",
                label="Open allocation / total may change",
            )
        if uncalibrated_positions:
            total_ax.scatter(
                uncalibrated_positions,
                [0.03] * len(uncalibrated_positions),
                transform=total_ax.get_xaxis_transform(),
                marker="o",
                s=50,
                facecolor="white",
                edgecolor="tab:gray",
                linewidth=1.3,
                label="Not calibrated / no residual policy",
            )
        total_ax.axhline(0.0, color="black", linewidth=0.8)
        total_ax.set_ylabel("Residual")
        total_ax.set_title("Required vs Allocated Residual by Epoch")
        total_ax.grid(axis="y", alpha=0.2)
        total_ax.legend(loc="best")

        recipient_order: list[Entity] = []
        role_by_entity: dict[Entity, str] = {}
        for epoch in self.epochs:
            for row in self.rows[epoch]:
                if row.role not in {"residual_recipient", "residual_bucket"}:
                    continue
                if row.entity not in recipient_order:
                    recipient_order.append(row.entity)
                    role_by_entity[row.entity] = row.role

        colors = plt.cm.tab10.colors
        if recipient_order:
            entity_width = 0.8 / len(recipient_order)
            for entity_index, entity in enumerate(recipient_order):
                offset = (
                    entity_index - (len(recipient_order) - 1) / 2.0
                ) * entity_width
                values: list[float] = []
                for epoch in self.epochs:
                    row = next(
                        (
                            item
                            for item in self.rows[epoch]
                            if item.entity == entity
                            and item.role
                            in {"residual_recipient", "residual_bucket"}
                        ),
                        None,
                    )
                    values.append(0.0 if row is None else row.residual_adjustment)
                role_label = (
                    "Residual bucket"
                    if role_by_entity[entity] == "residual_bucket"
                    else "Residual recipient"
                )
                recipient_ax.bar(
                    [position + offset for position in epoch_positions],
                    values,
                    width=entity_width * 0.8,
                    color=colors[entity_index % len(colors)],
                    alpha=0.75,
                    edgecolor="black",
                    linewidth=0.4,
                    label=f"{role_label} — {entity}",
                )
        else:
            recipient_ax.text(
                0.5,
                0.5,
                "No recipient or residual-bucket adjustments were allocated.",
                transform=recipient_ax.transAxes,
                ha="center",
                va="center",
            )
        recipient_ax.axhline(0.0, color="black", linewidth=0.8)
        recipient_ax.set_xticks(range(len(self.epochs)))
        recipient_ax.set_xticklabels([str(epoch) for epoch in self.epochs])
        recipient_ax.set_xlabel("Authoritative attribution epoch")
        recipient_ax.set_ylabel("Residual adjustment")
        recipient_ax.set_title(
            "Residual Recipients and Buckets (Direct Adjustments Excluded)"
        )
        recipient_ax.grid(axis="y", alpha=0.2)
        if recipient_order:
            recipient_ax.legend(loc="best")
        fig.tight_layout()
        return fig

    def plot_entity_mix(self) -> Any:
        """Plot original and calibrated entity shares by authoritative epoch."""

        import matplotlib.pyplot as plt  # lazy optional dependency

        fig, (original_ax, calibrated_ax) = plt.subplots(
            2,
            1,
            figsize=(12, 8),
            sharex=True,
        )
        epoch_positions = [float(index) for index in range(len(self.epochs))]

        entity_order: list[Entity] = []
        tested_entities: set[Entity] = set()
        residual_buckets: set[Entity] = set()
        rows_by_pair: dict[tuple[Entity, Epoch], CalibrationRow] = {}
        for epoch in self.epochs:
            for row in self.rows[epoch]:
                rows_by_pair[(row.entity, epoch)] = row
                if row.entity not in entity_order:
                    entity_order.append(row.entity)
                if row.evidence_ids:
                    tested_entities.add(row.entity)
                if row.role == "residual_bucket":
                    residual_buckets.add(row.entity)

        colors = plt.cm.tab20.colors

        def draw_mix(ax: Any, *, calibrated: bool) -> None:
            positive_bottom = [0.0] * len(self.epochs)
            negative_bottom = [0.0] * len(self.epochs)
            undefined_epochs: list[int] = []
            share_name = "calibrated_share" if calibrated else "original_share"
            for entity_index, entity in enumerate(entity_order):
                values: list[float] = []
                bottoms: list[float] = []
                for epoch_index, epoch in enumerate(self.epochs):
                    row = rows_by_pair.get((entity, epoch))
                    share = None if row is None else getattr(row, share_name)
                    value = 0.0 if share is None else share
                    values.append(value)
                    if value >= 0.0:
                        bottoms.append(positive_bottom[epoch_index])
                        positive_bottom[epoch_index] += value
                    else:
                        bottoms.append(negative_bottom[epoch_index])
                        negative_bottom[epoch_index] += value
                    if row is not None and share is None:
                        undefined_epochs.append(epoch_index)

                if entity in residual_buckets:
                    label = f"Residual bucket — {entity}"
                elif entity in tested_entities:
                    label = f"Tested entity — {entity}"
                else:
                    label = f"Entity — {entity}"
                ax.bar(
                    epoch_positions,
                    values,
                    width=0.72,
                    bottom=bottoms,
                    color=colors[entity_index % len(colors)],
                    alpha=0.82,
                    edgecolor="black",
                    linewidth=0.4,
                    label=label,
                )

            undefined_positions = sorted(set(undefined_epochs))
            if undefined_positions:
                ax.scatter(
                    undefined_positions,
                    [0.0] * len(undefined_positions),
                    marker="X",
                    s=65,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=1.4,
                    zorder=5,
                    label="Share undefined (epoch total is zero)",
                )
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_ylabel("Share of epoch total")
            ax.grid(axis="y", alpha=0.2)

        draw_mix(original_ax, calibrated=False)
        draw_mix(calibrated_ax, calibrated=True)
        original_ax.set_title("Original Entity Mix by Authoritative Epoch")
        calibrated_ax.set_title("Calibrated Entity Mix by Authoritative Epoch")
        calibrated_ax.set_xticks(range(len(self.epochs)))
        calibrated_ax.set_xticklabels([str(epoch) for epoch in self.epochs])
        calibrated_ax.set_xlabel("Authoritative attribution epoch")

        # The stacked bars fill each panel from edge to edge, so a legend
        # placed inside an axes always hides shares.  Both panels draw every
        # entity in the same color, so one key serves both; it sits outside
        # the plot area, to the right of the top panel.
        legend_entries: dict[str, Any] = {}
        for ax in (original_ax, calibrated_ax):
            for handle, label in zip(*ax.get_legend_handles_labels()):
                legend_entries.setdefault(label, handle)
        if legend_entries:
            original_ax.legend(
                list(legend_entries.values()),
                list(legend_entries),
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0.0,
            )
        fig.tight_layout()
        return fig

    def plot_evidence_strength(
        self,
        *,
        diagnostic_conflict_policies: Mapping[
            tuple[Entity, Epoch],
            Sequence[EvidenceConflictPolicy],
        ]
        | None = None,
        diagnostic_selected_evidence: SelectedEvidenceInput | None = None,
        diagnostic_evidence_weights: EvidenceWeightsInput | None = None,
        diagnostic_epoch_ratio_policies: Mapping[
            tuple[Entity, Epoch],
            Sequence[EpochRatioPolicy],
        ]
        | None = None,
        diagnostic_epoch_coordinates: EpochCoordinatesInput | None = None,
    ) -> Any:
        """Plot strength schedules and explicitly requested nonapplied ratios.

        Diagnostic requests are evaluated from defensive copies of evidence and
        authoritative metadata.  They are displayed beside stored applied
        ratios and never replace result rows, policies, or provenance.
        """

        irrelevant_conflict_inputs = []
        if diagnostic_conflict_policies is None:
            if diagnostic_selected_evidence is not None:
                irrelevant_conflict_inputs.append(
                    "diagnostic_selected_evidence"
                )
            if diagnostic_evidence_weights is not None:
                irrelevant_conflict_inputs.append(
                    "diagnostic_evidence_weights"
                )
        if irrelevant_conflict_inputs:
            raise ValueError(
                "Diagnostic conflict inputs require "
                "diagnostic_conflict_policies; irrelevant="
                f"{irrelevant_conflict_inputs!r}."
            )
        if (
            diagnostic_epoch_ratio_policies is None
            and diagnostic_epoch_coordinates is not None
        ):
            raise ValueError(
                "diagnostic_epoch_coordinates requires "
                "diagnostic_epoch_ratio_policies."
            )
        if (
            isinstance(diagnostic_conflict_policies, Mapping)
            and not diagnostic_conflict_policies
        ):
            raise ValueError(
                "diagnostic_conflict_policies must not be empty when supplied."
            )

        import matplotlib.pyplot as plt  # lazy optional dependency

        fig, (strength_ax, diagnostic_ax) = plt.subplots(
            2,
            1,
            figsize=(12, 8),
        )

        if self.ratio_policy == "explicit_strength":
            schedule_provenance = self.ratio_provenance.get(
                "schedule_provenance_by_pair", {}
            )
            strength_rows = tuple(
                row
                for row in self.epoch_ratio_schedule
                if (row.tested_entity, row.epoch) in schedule_provenance
            )
            positions = [float(index) for index in range(len(strength_rows))]
            strengths = [
                schedule_provenance[(row.tested_entity, row.epoch)][
                    "evidence_strength"
                ]
                for row in strength_rows
            ]
            observed_ratios = [
                schedule_provenance[(row.tested_entity, row.epoch)][
                    "observed_ratio"
                ]
                for row in strength_rows
            ]
            effective_ratios = [row.effective_ratio for row in strength_rows]
            width = 0.24
            strength_ax.bar(
                [position - width for position in positions],
                strengths,
                width=width,
                color="tab:purple",
                alpha=0.75,
                edgecolor="black",
                linewidth=0.4,
                label="Explicit evidence strength",
            )
            strength_ax.bar(
                positions,
                observed_ratios,
                width=width,
                color="tab:orange",
                alpha=0.75,
                edgecolor="black",
                linewidth=0.4,
                label="Source observed calibration ratio",
            )
            strength_ax.bar(
                [position + width for position in positions],
                effective_ratios,
                width=width,
                color="tab:blue",
                alpha=0.75,
                edgecolor="black",
                linewidth=0.4,
                label="Applied effective ratio",
            )
            strength_ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.7)
            strength_ax.set_xticks(positions)
            strength_ax.set_xticklabels(
                [
                    f"{row.tested_entity} / {row.epoch}"
                    for row in strength_rows
                ]
            )
            strength_ax.set_title("Explicit Evidence Strength Schedule")
            strength_ax.legend(loc="best")
            strength_ax.set_ylabel("Strength or ratio")
            strength_ax.set_xlabel("Tested entity / authoritative epoch")
            strength_ax.grid(axis="y", alpha=0.2)
        else:
            strength_ax.text(
                0.5,
                0.5,
                "Evidence strength is not applicable to the applied "
                f"ratio policy: {self.ratio_policy}.",
                transform=strength_ax.transAxes,
                ha="center",
                va="center",
            )
            strength_ax.set_title("Evidence Strength Not Applicable")
            # The panel states a fact and plots nothing.  A value scale, axis
            # labels or a grid would present it as an empty data plot.
            strength_ax.set_xticks([])
            strength_ax.set_yticks([])

        diagnostic_values: list[
            tuple[tuple[Entity, Epoch], str, float | None]
        ] = []
        if diagnostic_conflict_policies is not None:
            if self.ratio_policy != "observed_only":
                raise ValueError(
                    "Conflict-policy diagnostics are applicable only when the "
                    "result ratio_policy is 'observed_only'."
                )
            observed_schedule = _build_observed_only_schedule(
                tuple(_copy_lift_test_evidence(item) for item in self.evidence),
                epochs=tuple(deepcopy(epoch) for epoch in self.epochs),
                epoch_windows=_deep_export_copy(self.epoch_windows),
            )
            applied_policy = _deep_export_copy(
                self.ratio_provenance.get("conflict_policy_by_pair", {})
            )
            conflict_request = _validate_evidence_conflict_request(
                observed_schedule,
                evidence_conflict_policy=applied_policy,
            )
            conflict_diagnostics = _build_evidence_conflict_diagnostics(
                conflict_request,
                diagnostic_policies=_deep_export_copy(
                    diagnostic_conflict_policies
                ),
                selected_evidence=_deep_export_copy(
                    diagnostic_selected_evidence
                ),
                evidence_weights=_deep_export_copy(
                    diagnostic_evidence_weights
                ),
            )
            for pair, alternatives in (
                conflict_diagnostics.alternatives_by_pair.items()
            ):
                for policy, resolution in alternatives.items():
                    diagnostic_values.append(
                        (deepcopy(pair), deepcopy(policy), resolution.effective_ratio)
                    )

        if diagnostic_epoch_ratio_policies is not None:
            if self.ratio_policy != "coordinate_linear":
                raise ValueError(
                    "Epoch-reduction diagnostics are applicable only when the "
                    "result ratio_policy is 'coordinate_linear'."
                )
            if not isinstance(diagnostic_epoch_ratio_policies, Mapping):
                raise TypeError(
                    "diagnostic_epoch_ratio_policies must be a mapping."
                )
            supplied_reductions = _deep_export_copy(
                diagnostic_epoch_ratio_policies
            )
            if not supplied_reductions:
                raise ValueError(
                    "diagnostic_epoch_ratio_policies must not be empty when "
                    "supplied."
                )
            allowed_reductions = {
                "representative_coordinate",
                "coordinate_average",
            }
            normalized_reductions: dict[
                tuple[Entity, Epoch], tuple[EpochRatioPolicy, ...]
            ] = {}
            known_entity_order: list[Entity] = []
            for row in self.coordinate_ratio_schedule:
                if row.tested_entity not in known_entity_order:
                    known_entity_order.append(row.tested_entity)
            known_entities = set(known_entity_order)
            applied_pairs = {
                (row.tested_entity, row.epoch)
                for row in self.epoch_ratio_schedule
            }
            for pair, policies in supplied_reductions.items():
                if not isinstance(pair, tuple) or len(pair) != 2:
                    raise TypeError(
                        "diagnostic_epoch_ratio_policies keys must be "
                        "(tested_entity, epoch) pairs."
                    )
                entity, epoch = pair
                if entity not in known_entities or epoch not in self.epochs:
                    raise ValueError(
                        "diagnostic_epoch_ratio_policies contains an unknown "
                        f"tested-entity/epoch pair {pair!r}."
                    )
                if pair not in applied_pairs:
                    raise ValueError(
                        "diagnostic_epoch_ratio_policies pairs must identify "
                        f"stored applied ratios; not_applied={pair!r}."
                    )
                if isinstance(policies, (str, bytes, bytearray)):
                    raise TypeError(
                        f"diagnostic_epoch_ratio_policies[{pair!r}] must be a "
                        "sequence."
                    )
                normalized = tuple(policies)
                if not normalized:
                    raise ValueError(
                        f"diagnostic_epoch_ratio_policies[{pair!r}] must not "
                        "be empty."
                    )
                unknown = [
                    policy
                    for policy in normalized
                    if policy not in allowed_reductions
                ]
                if unknown:
                    raise ValueError(
                        "Diagnostic epoch-ratio policies support only "
                        "representative_coordinate and coordinate_average; "
                        f"unknown={unknown!r}."
                    )
                if len(set(normalized)) != len(normalized):
                    raise ValueError(
                        f"diagnostic_epoch_ratio_policies[{pair!r}] must be "
                        "unique."
                    )
                normalized_reductions[deepcopy(pair)] = normalized

            representative_epochs = tuple(
                epoch
                for epoch in self.epochs
                if any(
                    pair[1] == epoch
                    and "representative_coordinate" in policies
                    for pair, policies in normalized_reductions.items()
                )
            )
            supplied_coordinates = _deep_export_copy(
                diagnostic_epoch_coordinates
            )
            if supplied_coordinates is not None:
                if not isinstance(supplied_coordinates, Mapping):
                    raise TypeError(
                        "diagnostic_epoch_coordinates must be a mapping."
                    )
                missing_coordinates = [
                    epoch
                    for epoch in representative_epochs
                    if epoch not in supplied_coordinates
                ]
                extra_coordinates = [
                    epoch
                    for epoch in supplied_coordinates
                    if epoch not in representative_epochs
                ]
                if missing_coordinates or extra_coordinates:
                    raise ValueError(
                        "diagnostic_epoch_coordinates keys must exactly match "
                        "requested representative-coordinate epochs; "
                        f"missing={missing_coordinates!r}, "
                        f"extra={extra_coordinates!r}."
                    )
            coordinate_schedule = _build_coordinate_linear_schedule(
                tuple(_copy_lift_test_evidence(item) for item in self.evidence),
                selected_entities=tuple(known_entity_order),
                allow_signed=True,
            )
            for pair, policies in normalized_reductions.items():
                for policy in policies:
                    if policy == "representative_coordinate":
                        entity, epoch = pair
                        coordinate = None
                        if supplied_coordinates is not None:
                            coordinate = supplied_coordinates.get(epoch)
                        if coordinate is None and self.epoch_coordinates is not None:
                            coordinate = self.epoch_coordinates.get(epoch)
                        if coordinate is None:
                            raise ValueError(
                                "Representative-coordinate diagnostics require "
                                f"diagnostic_epoch_coordinates for epoch {epoch!r}."
                            )
                        coordinates = {deepcopy(epoch): deepcopy(coordinate)}
                    else:
                        coordinates = None
                    request = _validate_epoch_ratio_reduction_request(
                        ratio_policy="coordinate_linear",
                        epoch_ratio_policy=policy,
                        epochs=tuple(deepcopy(epoch) for epoch in self.epochs),
                        calibrated_pairs=(deepcopy(pair),),
                        epoch_windows=_deep_export_copy(self.epoch_windows),
                        epoch_coordinates=coordinates,
                        epoch_support=_deep_export_copy(self.epoch_support),
                    )
                    reduction = (
                        _reduce_representative_coordinate(
                            request,
                            coordinate_schedule,
                        )
                        if policy == "representative_coordinate"
                        else _reduce_coordinate_average(
                            request,
                            coordinate_schedule,
                        )
                    )
                    value_by_pair = {
                        (row.tested_entity, row.epoch): row.effective_ratio
                        for row in reduction.rows
                    }
                    diagnostic_values.append(
                        (deepcopy(pair), deepcopy(policy), value_by_pair.get(pair))
                    )

        if diagnostic_values:
            requested_pairs: list[tuple[Entity, Epoch]] = []
            for pair, _, _ in diagnostic_values:
                if pair not in requested_pairs:
                    requested_pairs.append(pair)
            pair_position = {
                pair: float(index) for index, pair in enumerate(requested_pairs)
            }
            applied_by_pair = {
                (row.tested_entity, row.epoch): row.effective_ratio
                for row in self.epoch_ratio_schedule
            }
            diagnostic_ax.scatter(
                [pair_position[pair] for pair in requested_pairs],
                [applied_by_pair[pair] for pair in requested_pairs],
                marker="D",
                s=70,
                color="black",
                label="Stored applied effective ratio",
                zorder=4,
            )
            policy_order: list[str] = []
            for _, policy, _ in diagnostic_values:
                if policy not in policy_order:
                    policy_order.append(policy)
            colors = plt.cm.tab10.colors
            for policy_index, policy in enumerate(policy_order):
                policy_values = tuple(
                    item for item in diagnostic_values if item[1] == policy
                )
                finite_values = tuple(
                    item for item in policy_values if item[2] is not None
                )
                if finite_values:
                    diagnostic_ax.scatter(
                        [
                            pair_position[pair]
                            + 0.08 * (policy_index + 1)
                            for pair, _, _ in finite_values
                        ],
                        [value for _, _, value in finite_values],
                        marker="o",
                        s=65,
                        color=colors[policy_index % len(colors)],
                        label=f"Diagnostic only — not applied: {policy}",
                        zorder=3,
                    )
                unsupported = tuple(
                    item for item in policy_values if item[2] is None
                )
                for pair, _, _ in unsupported:
                    diagnostic_ax.annotate(
                        f"{policy}: unsupported (not applied)",
                        (pair_position[pair], 0.02),
                        xycoords=("data", "axes fraction"),
                        rotation=90,
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )
            diagnostic_ax.set_xticks(range(len(requested_pairs)))
            diagnostic_ax.set_xticklabels(
                [f"{entity} / {epoch}" for entity, epoch in requested_pairs]
            )
            # One band per requested pair, centered on its tick.  Left to
            # autoscale, the limits shrink to the markers themselves: with a
            # single pair the applied ratio and its alternatives land on
            # opposite edges of the panel and read as separate categories.
            diagnostic_ax.set_xlim(-0.5, len(requested_pairs) - 0.5)
            diagnostic_ax.legend(loc="best")
            diagnostic_ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.7)
            diagnostic_ax.set_xlabel("Tested entity / authoritative epoch")
            diagnostic_ax.set_ylabel("Ratio")
            diagnostic_ax.grid(axis="y", alpha=0.2)
        else:
            diagnostic_ax.text(
                0.5,
                0.5,
                "No diagnostic alternatives were requested; stored applied "
                "policies and ratios are unchanged.",
                transform=diagnostic_ax.transAxes,
                ha="center",
                va="center",
            )
            # As above: a statement, not an empty data plot.
            diagnostic_ax.set_xticks([])
            diagnostic_ax.set_yticks([])
        diagnostic_ax.set_title(
            "Applied Ratios and Explicitly Requested Diagnostic Alternatives"
        )
        # A panel that only carries a statement is a short strip; the room
        # goes to the panel that plots values.  Two panels of the same kind
        # share the figure equally.
        strength_ax.get_gridspec().set_height_ratios(
            [
                3 if self.ratio_policy == "explicit_strength" else 1,
                3 if diagnostic_values else 1,
            ]
        )
        fig.tight_layout()
        return fig


class AttributionCalibration:
    """Static calibrator signature shell; implementation follows in later items."""

    def __init__(self, evidence: Sequence[LiftTestEvidence]) -> None:
        normalized = _normalize_evidence_collection(
            evidence, allow_multiple_per_entity=False
        )
        self._evidence = normalized.records
        self._evidence_by_entity = normalized.by_entity
        self._evidence_by_id = normalized.by_id

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
        if not isinstance(allow_signed, bool):
            raise TypeError("allow_signed must be a bool.")
        direct_state = _static_direct_state(
            estimates,
            evidence_by_entity=self._evidence_by_entity,
            allow_signed=allow_signed,
            tolerance=tolerance,
        )
        request = _static_residual_request(
            residual_policy=residual_policy,
            tested_entities=tuple(self._evidence_by_entity),
            eligible_entities=eligible_entities,
            all_untested=all_untested,
            residual_weights=residual_weights,
            residual_entity=residual_entity,
            residual_adjustments=residual_adjustments,
            allow_bucket_collision=allow_bucket_collision,
        )
        allocation = _dispatch_static_residual(
            direct_state,
            request,
            allow_signed=allow_signed,
            tolerance=tolerance,
        )
        return _static_calibration_result(
            direct_state,
            request,
            allocation,
            evidence=self._evidence,
            allow_signed=allow_signed,
            tolerance=tolerance,
        )

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
        if not isinstance(allow_signed, bool):
            raise TypeError("allow_signed must be a bool.")
        if attribution is None:
            raise ValueError(
                "attribution is unavailable (None); calibration requires "
                "normalized attribution shares. Raw signed diagnostics are "
                "not shares and must remain separate."
            )
        shares = _validate_shares(
            _normalize_pairs(attribution, name="attribution"),
            tolerance=tolerance,
            name="attribution",
        )
        total = _validate_total(
            current_total,
            name="current_total",
            allow_signed=allow_signed,
        )
        estimates = tuple(
            (deepcopy(entity), share * total) for entity, share in shares
        )
        return self.apply_estimates(
            estimates,
            residual_policy=residual_policy,
            eligible_entities=eligible_entities,
            all_untested=all_untested,
            residual_weights=residual_weights,
            residual_entity=residual_entity,
            residual_adjustments=residual_adjustments,
            allow_bucket_collision=allow_bucket_collision,
            allow_signed=allow_signed,
            tolerance=tolerance,
        )


class TemporalAttributionCalibration:
    """Temporal calibrator signature shell; implementation follows later."""

    def __init__(self, evidence: Sequence[LiftTestEvidence]) -> None:
        normalized = _normalize_evidence_collection(
            evidence, allow_multiple_per_entity=True
        )
        self._evidence = normalized.records
        self._evidence_by_entity = normalized.by_entity
        self._evidence_by_id = normalized.by_id
        self._evidence_by_coordinate_name = normalized.by_coordinate_name
        self._evidence_by_coordinate_value = normalized.by_coordinate_value
        self._evidence_points = _normalize_evidence_points(normalized.records)

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
        if not isinstance(allow_signed, bool):
            raise TypeError("allow_signed must be a bool.")
        _validate_tolerance(tolerance)
        normalized_estimates = _normalize_estimates_by_epoch(
            estimates_by_epoch
        )
        epochs = _epoch_tuple(epoch for epoch, _ in normalized_estimates)
        ratio_pipeline = _resolve_temporal_ratio_pipeline(
            self._evidence,
            epochs=epochs,
            ratio_policy=ratio_policy,
            epoch_ratio_policy=epoch_ratio_policy,
            epoch_windows=epoch_windows,
            epoch_coordinates=epoch_coordinates,
            epoch_support=epoch_support,
            epoch_coordinate_weights=epoch_coordinate_weights,
            source_evidence=source_evidence,
            tested_entity_epochs=tested_entity_epochs,
            ratio_schedule=ratio_schedule,
            evidence_strength=evidence_strength,
            evidence_conflict_policy=evidence_conflict_policy,
            selected_evidence=selected_evidence,
            evidence_weights=evidence_weights,
            ratio_bounds=ratio_bounds,
            allow_signed=allow_signed,
            tolerance=tolerance,
        )
        direct_state = _temporal_direct_state(
            estimates_by_epoch,
            epoch_ratio_rows=ratio_pipeline.reduction.rows,
            evidence_by_entity=self._evidence_by_entity,
            allow_signed=allow_signed,
            tolerance=tolerance,
        )
        residual_request = _normalize_temporal_residual_request(
            epochs=epochs,
            calibrated_epochs=direct_state.calibrated_epochs,
            residual_policy=residual_policy,
            eligible_entities=eligible_entities,
            eligible_entities_by_epoch=eligible_entities_by_epoch,
            all_untested=all_untested,
            residual_weights=residual_weights,
            residual_weights_by_epoch=residual_weights_by_epoch,
            residual_entity=residual_entity,
            residual_entity_by_epoch=residual_entity_by_epoch,
            residual_adjustments_by_epoch=residual_adjustments_by_epoch,
            allow_bucket_collision=allow_bucket_collision,
        )
        allocation = _apply_temporal_residual_request(
            direct_state,
            residual_request,
            allow_signed=allow_signed,
            tolerance=tolerance,
        )
        rows = _temporal_calibration_rows(allocation, residual_request)
        return TemporalCalibrationResult(
            epochs=epochs,
            epoch_windows=epoch_windows,
            epoch_coordinates=epoch_coordinates,
            epoch_support=epoch_support,
            evidence_points=ratio_pipeline.evidence_points,
            coordinate_ratio_schedule=ratio_pipeline.coordinate_schedule,
            epoch_ratio_schedule=ratio_pipeline.reduction.rows,
            rows=rows,
            ratio_provenance=ratio_pipeline.ratio_provenance,
            epoch_reduction_provenance={
                "by_pair": ratio_pipeline.reduction.provenance_by_pair,
                "uncalibrated_pairs": (
                    ratio_pipeline.reduction.uncalibrated_pairs
                ),
            },
            residual_provenance=allocation.residual_provenance,
            original_totals=allocation.original_totals,
            raw_post_test_totals=allocation.raw_post_test_totals,
            required_residuals=allocation.required_residuals,
            calibrated_totals=allocation.calibrated_totals,
            ratio_policy=ratio_policy,
            epoch_ratio_policy=epoch_ratio_policy,
            residual_policy=residual_request.residual_policy,
            evidence=self._evidence,
            warnings=(),
        )

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
        if not isinstance(allow_signed, bool):
            raise TypeError("allow_signed must be a bool.")
        _validate_tolerance(tolerance)
        if isinstance(attribution_by_epoch, Mapping):
            for epoch, attribution in attribution_by_epoch.items():
                if attribution is None:
                    raise ValueError(
                        f"attribution_by_epoch[{epoch!r}] is unavailable "
                        "(None); calibration requires normalized attribution "
                        "shares for every epoch. Raw signed diagnostics are "
                        "not shares and must remain separate."
                    )
        normalized_shares = _normalize_estimates_by_epoch(
            attribution_by_epoch,
            name="attribution_by_epoch",
        )
        epochs = _epoch_tuple(epoch for epoch, _ in normalized_shares)
        shares_by_epoch = {
            deepcopy(epoch): _validate_shares(
                pairs,
                tolerance=tolerance,
                name=f"attribution_by_epoch[{epoch!r}]",
            )
            for epoch, pairs in normalized_shares
        }
        totals = dict(
            _validate_totals_by_epoch(
                epochs,
                totals_by_epoch,
                allow_signed=allow_signed,
            )
        )
        estimates_by_epoch = MappingProxyType(
            {
                deepcopy(epoch): tuple(
                    (deepcopy(entity), share * totals[epoch])
                    for entity, share in shares_by_epoch[epoch]
                )
                for epoch in epochs
            }
        )
        return self.apply_estimates(
            estimates_by_epoch,
            ratio_policy=ratio_policy,
            epoch_ratio_policy=epoch_ratio_policy,
            residual_policy=residual_policy,
            epoch_windows=epoch_windows,
            epoch_coordinates=epoch_coordinates,
            epoch_support=epoch_support,
            epoch_coordinate_weights=epoch_coordinate_weights,
            source_evidence=source_evidence,
            tested_entity_epochs=tested_entity_epochs,
            ratio_schedule=ratio_schedule,
            evidence_strength=evidence_strength,
            evidence_conflict_policy=evidence_conflict_policy,
            selected_evidence=selected_evidence,
            evidence_weights=evidence_weights,
            eligible_entities=eligible_entities,
            eligible_entities_by_epoch=eligible_entities_by_epoch,
            all_untested=all_untested,
            residual_weights=residual_weights,
            residual_weights_by_epoch=residual_weights_by_epoch,
            residual_entity=residual_entity,
            residual_entity_by_epoch=residual_entity_by_epoch,
            residual_adjustments_by_epoch=residual_adjustments_by_epoch,
            ratio_bounds=ratio_bounds,
            allow_bucket_collision=allow_bucket_collision,
            allow_signed=allow_signed,
            tolerance=tolerance,
        )


__all__ = [
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
