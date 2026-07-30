"""Validation and defensive-copy helpers for calibration.

The calibration domain model, policies, and public records remain in
``do_attribution.calibration``.  This private module owns the reusable input
normalization and immutability boundary shared by those responsibilities.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from copy import deepcopy
from datetime import date, datetime
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Any, TypeAlias, TypeVar


Entity: TypeAlias = Hashable
Epoch: TypeAlias = Hashable
EstimateInput: TypeAlias = (
    Mapping[Entity, float]
    | Sequence[tuple[Entity, float]]
)

_Key = TypeVar("_Key", bound=Hashable)
_Value = TypeVar("_Value")


def normalize_pairs(
    value: Mapping[_Key, _Value] | Sequence[tuple[_Key, _Value]],
    *,
    name: str,
) -> tuple[tuple[_Key, _Value], ...]:
    """Copy a mapping or pair sequence into deterministic tuple storage.

    Sequence inputs are checked for duplicates before any mapping conversion.
    A mapping can expose only its surviving keys; Python may already have
    discarded duplicate literal or constructor keys before this function is
    called, and this function deliberately makes no recovery claim.
    """
    if isinstance(value, Mapping):
        pairs = list(value.items())
    else:
        if isinstance(value, (str, bytes, bytearray)):
            raise TypeError(f"{name} must be a mapping or a sequence of pairs.")
        try:
            raw_pairs = list(value)
        except TypeError as exc:
            raise TypeError(
                f"{name} must be a mapping or a sequence of pairs."
            ) from exc
        pairs = []
        for index, pair in enumerate(raw_pairs):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise TypeError(f"{name}[{index}] must be a two-item pair.")
            pairs.append((pair[0], pair[1]))

    if not pairs:
        raise ValueError(f"{name} must not be empty.")

    seen: set[_Key] = set()
    copied: list[tuple[_Key, _Value]] = []
    for key, item_value in pairs:
        try:
            duplicate = key in seen
        except TypeError as exc:
            raise TypeError(f"{name} key {key!r} must be hashable.") from exc
        if duplicate:
            raise ValueError(f"{name} contains duplicate key {key!r}.")
        seen.add(key)
        copied.append((deepcopy(key), deepcopy(item_value)))
    return tuple(copied)


def normalize_estimates_by_epoch(
    value: Mapping[Epoch, EstimateInput],
    *,
    name: str = "estimates_by_epoch",
) -> tuple[tuple[Epoch, tuple[tuple[Entity, float], ...]], ...]:
    """Defensively copy nested epoch/entity estimates in mapping order."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping of epochs to estimates.")
    if not value:
        raise ValueError(f"{name} must not be empty.")
    return tuple(
        (
            deepcopy(epoch),
            normalize_pairs(estimates, name=f"{name}[{epoch!r}]"),
        )
        for epoch, estimates in value.items()
    )


def normalize_unique_records(
    records: Sequence[_Value],
    *,
    id_attribute: str,
    name: str,
) -> tuple[_Value, ...]:
    """Copy records in caller order while rejecting duplicate record IDs."""
    if isinstance(records, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence of records.")
    try:
        materialized = list(records)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of records.") from exc
    if not materialized:
        raise ValueError(f"{name} must not be empty.")

    seen: set[Any] = set()
    copied: list[_Value] = []
    for index, record in enumerate(materialized):
        if not hasattr(record, id_attribute):
            raise TypeError(
                f"{name}[{index}] must expose {id_attribute!r}."
            )
        identifier = getattr(record, id_attribute)
        try:
            duplicate = identifier in seen
        except TypeError as exc:
            raise TypeError(
                f"{name}[{index}] {id_attribute} must be hashable."
            ) from exc
        if duplicate:
            raise ValueError(
                f"{name} contains duplicate {id_attribute} {identifier!r}."
            )
        seen.add(identifier)
        copied.append(deepcopy(record))
    return tuple(copied)


def finite_float(value: Any, *, name: str) -> float:
    """Return one finite real value, rejecting bool and coercive strings."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not {value!r}.")
    converted = float(value)
    if not isfinite(converted):
        raise ValueError(f"{name} must be finite, not {value!r}.")
    return converted


def validate_tolerance(value: Any) -> float:
    tolerance = finite_float(value, name="tolerance")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be greater than zero.")
    return tolerance


def validate_numeric_pairs(
    pairs: Sequence[tuple[_Key, Any]],
    *,
    name: str,
    allow_signed: bool,
) -> tuple[tuple[_Key, float], ...]:
    """Validate finite estimates, schedules, or adjustments in stable order."""
    validated: list[tuple[_Key, float]] = []
    for key, value in pairs:
        number = finite_float(value, name=f"{name}[{key!r}]")
        if not allow_signed and number < 0.0:
            raise ValueError(
                f"{name}[{key!r}] must be nonnegative unless allow_signed=True."
            )
        validated.append((key, number))
    return tuple(validated)


def validate_shares(
    pairs: Sequence[tuple[_Key, Any]],
    *,
    tolerance: Any,
    name: str,
) -> tuple[tuple[_Key, float], ...]:
    checked_tolerance = validate_tolerance(tolerance)
    shares = validate_numeric_pairs(
        pairs,
        name=name,
        allow_signed=False,
    )
    total = sum(value for _, value in shares)
    if abs(total - 1.0) > checked_tolerance:
        raise ValueError(
            f"{name} must sum to one within tolerance {checked_tolerance}; "
            f"observed {total}."
        )
    return shares


def validate_total(value: Any, *, name: str, allow_signed: bool) -> float:
    total = finite_float(value, name=name)
    if not allow_signed and total < 0.0:
        raise ValueError(f"{name} must be nonnegative unless allow_signed=True.")
    return total


def validate_totals_by_epoch(
    epochs: Sequence[Epoch],
    totals: Mapping[Epoch, Any],
    *,
    allow_signed: bool,
    name: str = "totals_by_epoch",
) -> tuple[tuple[Epoch, float], ...]:
    if not isinstance(totals, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    expected = tuple(epochs)
    missing = [epoch for epoch in expected if epoch not in totals]
    extra = [epoch for epoch in totals if epoch not in expected]
    if missing or extra:
        raise ValueError(
            f"{name} keys must exactly match authoritative epochs; "
            f"missing={missing!r}, extra={extra!r}."
        )
    return tuple(
        (
            deepcopy(epoch),
            validate_total(
                totals[epoch],
                name=f"{name}[{epoch!r}]",
                allow_signed=allow_signed,
            ),
        )
        for epoch in expected
    )


def validate_unit_interval_pairs(
    pairs: Sequence[tuple[_Key, Any]],
    *,
    name: str,
) -> tuple[tuple[_Key, float], ...]:
    values = validate_numeric_pairs(
        pairs,
        name=name,
        allow_signed=False,
    )
    for key, value in values:
        if value > 1.0:
            raise ValueError(f"{name}[{key!r}] must be in [0, 1].")
    return values


def validate_unit_sum_weights(
    pairs: Sequence[tuple[_Key, Any]],
    *,
    tolerance: Any,
    name: str,
) -> tuple[tuple[_Key, float], ...]:
    checked_tolerance = validate_tolerance(tolerance)
    weights = validate_numeric_pairs(
        pairs,
        name=name,
        allow_signed=False,
    )
    total = sum(value for _, value in weights)
    if abs(total - 1.0) > checked_tolerance:
        raise ValueError(
            f"{name} must sum to one within tolerance {checked_tolerance}; "
            f"observed {total}."
        )
    return weights


def validate_ratio_bounds(
    bounds: tuple[Any, Any] | None,
) -> tuple[float, float] | None:
    if bounds is None:
        return None
    if not isinstance(bounds, tuple) or len(bounds) != 2:
        raise TypeError("ratio_bounds must be a two-item tuple or None.")
    lower = finite_float(bounds[0], name="ratio_bounds lower")
    upper = finite_float(bounds[1], name="ratio_bounds upper")
    if lower > upper:
        raise ValueError("ratio_bounds must satisfy lower <= upper.")
    return lower, upper


def deep_freeze(value: Any) -> Any:
    """Defensively copy common containers into recursively immutable state."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                deepcopy(key): deep_freeze(item_value)
                for key, item_value in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(deep_freeze(item) for item in value)
    return deepcopy(value)


def deep_export_copy(value: Any) -> Any:
    """Copy immutable nested result state into caller-owned export cells."""
    if isinstance(value, Mapping):
        return {
            deepcopy(key): deep_export_copy(item_value)
            for key, item_value in value.items()
        }
    if isinstance(value, tuple):
        return tuple(deep_export_copy(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(deep_export_copy(item) for item in value)
    return deepcopy(value)


def coordinate_kind(value: Any, *, name: str) -> str:
    """Validate one coordinate and return its comparison family."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number, date, or datetime.")
    if isinstance(value, datetime):
        return "datetime"
    if isinstance(value, date):
        return "date"
    if isinstance(value, Real):
        finite_float(value, name=name)
        return "numeric"
    raise TypeError(f"{name} must be a finite number, date, or datetime.")
