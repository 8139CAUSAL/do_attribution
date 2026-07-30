# Temporal calibration with dated evidence coordinates

Authoritative attribution epochs are outcome-driven or supplied by the user.
Print
does not own the epochs merely because Print has experimental evidence.
The calibration layer consumes the declared epoch partition; calibration never creates,
moves, splits, merges, or reassigns epoch boundaries.

Interpolation is a
missing-evidence rule between observed evidence coordinates. It is not entity-specific time
partitioning and does not create a new lift-test observation.

## Dated evidence

Each retained experiment declares its location on the same explicit date axis.

```python
from datetime import date

from do_attribution.calibration import (
    LiftTestEvidence,
    TemporalAttributionCalibration,
)

may_print = LiftTestEvidence(
    tested_entity="Print",
    observed_lift=90.0,
    matched_mta_estimate=100.0,
    test_id="print-may",
    coordinate_name="date",
    coordinate_value=date(2023, 5, 16),
)
dec_print = LiftTestEvidence(
    tested_entity="Print",
    observed_lift=110.0,
    matched_mta_estimate=100.0,
    test_id="print-dec",
    coordinate_name="date",
    coordinate_value=date(2023, 12, 16),
)
temporal_calibration = TemporalAttributionCalibration(
    evidence=[may_print, dec_print]
)
```

## Interpolate at declared representative coordinates

All three policy layers are explicit. Print receives the schedule ratio at the
declared representative date, while the residual is allocated only to the
named untested entity.

```python
result = temporal_calibration.apply_estimates(
    {
        "early": {"Print": 100.0, "Search": 50.0},
        "middle": {"Print": 100.0, "Search": 50.0},
        "late": {"Print": 100.0, "Search": 50.0},
    },
    ratio_policy="coordinate_linear",
    epoch_ratio_policy="representative_coordinate",
    residual_policy="proportional_untested",
    epoch_windows={
        "early": (date(2023, 5, 16), date(2023, 7, 1)),
        "middle": (date(2023, 7, 1), date(2023, 10, 1)),
        "late": (date(2023, 10, 1), date(2023, 12, 16)),
    },
    epoch_coordinates={
        "early": date(2023, 5, 16),
        "middle": date(2023, 8, 31),
        "late": date(2023, 12, 16),
    },
    eligible_entities=["Search"],
)

assert [row.ratio_source for row in result.epoch_ratio_schedule] == [
    "observed",
    "interpolated",
    "observed",
]
assert result.epochs == ("early", "middle", "late")
```

## Refuse a representative coordinate outside observed support

The inside epoch is calibrated, while the outside epoch remains unchanged and
is reported as uncalibrated. There is no extrapolation or carry-forward.

```python
result_outside = temporal_calibration.apply_estimates(
    {
        "inside": {"Print": 100.0, "Search": 50.0},
        "outside": {"Print": 100.0, "Search": 50.0},
    },
    ratio_policy="coordinate_linear",
    epoch_ratio_policy="representative_coordinate",
    residual_policy="proportional_untested",
    epoch_windows={
        "inside": (date(2023, 5, 16), date(2023, 6, 16)),
        "outside": (date(2023, 12, 16), date(2024, 1, 16)),
    },
    epoch_coordinates={
        "inside": date(2023, 5, 16),
        "outside": date(2024, 1, 1),
    },
    eligible_entities=["Search"],
)

assert [row.epoch for row in result_outside.epoch_ratio_schedule] == ["inside"]
assert result_outside.ratio_provenance["uncalibrated_pairs"] == (
    ("Print", "outside"),
)
assert all(row.role == "untouched" for row in result_outside.rows["outside"])
```

## Coordinate boundaries

The API never silently converts an epoch-label string or an interval midpoint
into an evidence coordinate. Labels such as `"2023-Q3"` identify epochs; they
cannot serve as interpolation coordinates. Numeric, date, and datetime
coordinates must be supplied explicitly and use a compatible coordinate type.
