# Calibration conflict and epoch-reduction examples

Temporal calibration keeps three decisions explicit: how evidence becomes a
ratio schedule, how that schedule is reduced to an attribution epoch, and how
any residual is handled. The package validates the choices you declare;
it does not select a conflict policy, an epoch-ratio policy, or a residual
policy for you.

## Resolving multiple observations in one epoch

Both observations below concern the same tested entity and fall within the
same authoritative attribution epoch. Their coordinates belong to the
evidence; the epoch does not belong to either observation.

```python
import pytest

from do_attribution.calibration import (
    LiftTestEvidence,
    TemporalAttributionCalibration,
)

print_geo_a = LiftTestEvidence(
    tested_entity="Print",
    observed_lift=72.0,
    matched_mta_estimate=80.0,
    test_id="print-geo-a",
    outcome_name="orders",
    population="all customers",
    unit="orders",
    coordinate_name="week",
    coordinate_value=3.0,
)
print_geo_b = LiftTestEvidence(
    tested_entity="Print",
    observed_lift=180.0,
    matched_mta_estimate=170.0,
    test_id="print-geo-b",
    outcome_name="orders",
    population="all customers",
    unit="orders",
    coordinate_name="week",
    coordinate_value=7.0,
)

conflict_calibration = TemporalAttributionCalibration(
    evidence=[print_geo_a, print_geo_b]
)
conflict_estimates = {"E1": {"Print": 100.0, "Search": 50.0}}
conflict_windows = {"E1": (0.0, 10.0)}
```

### No conflict rule is inferred

When several observations map to the same tested-entity/epoch pair,
`evidence_conflict_policy is required`. A selection policy also reports that
`selected_evidence is required` until the caller identifies one retained
evidence record. Explicit combination weights must sum to one.

```python
with pytest.raises(ValueError, match="evidence_conflict_policy is required"):
    conflict_calibration.apply_estimates(
        conflict_estimates,
        ratio_policy="observed_only",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        epoch_windows=conflict_windows,
    )

with pytest.raises(ValueError, match="selected_evidence is required"):
    conflict_calibration.apply_estimates(
        conflict_estimates,
        ratio_policy="observed_only",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        epoch_windows=conflict_windows,
        evidence_conflict_policy="select_evidence",
    )

with pytest.raises(ValueError, match="sum to one"):
    conflict_calibration.apply_estimates(
        conflict_estimates,
        ratio_policy="observed_only",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        epoch_windows=conflict_windows,
        evidence_conflict_policy="weighted_ratio",
        evidence_weights={
            ("Print", "E1"): {
                "print-geo-a": 0.25,
                "print-geo-b": 0.25,
            }
        },
    )
```

The three supported resolutions retain both evidence points and their conflict
group. They differ only in the explicitly applied reduction.

```python
selected = conflict_calibration.apply_estimates(
    conflict_estimates,
    ratio_policy="observed_only",
    epoch_ratio_policy="direct",
    residual_policy="allow_total_change",
    epoch_windows=conflict_windows,
    evidence_conflict_policy="select_evidence",
    selected_evidence={("Print", "E1"): "print-geo-a"},
)

weighted = conflict_calibration.apply_estimates(
    conflict_estimates,
    ratio_policy="observed_only",
    epoch_ratio_policy="direct",
    residual_policy="allow_total_change",
    epoch_windows=conflict_windows,
    evidence_conflict_policy="weighted_ratio",
    evidence_weights={
        ("Print", "E1"): {
            "print-geo-a": 0.25,
            "print-geo-b": 0.75,
        }
    },
)

pooled = conflict_calibration.apply_estimates(
    conflict_estimates,
    ratio_policy="observed_only",
    epoch_ratio_policy="direct",
    residual_policy="allow_total_change",
    epoch_windows=conflict_windows,
    evidence_conflict_policy="pooled_lift",
)

assert selected.epoch_ratio_schedule[0].effective_ratio == 0.9
assert abs(
    weighted.epoch_ratio_schedule[0].effective_ratio
    - (0.25 * 0.9 + 0.75 * (180.0 / 170.0))
) < 1e-12

pooled_resolution = pooled.ratio_provenance[
    "conflict_resolution_by_pair"
][("Print", "E1")]["resolution"]
assert pooled_resolution["pooled_observed_lift"] == 252.0
assert pooled_resolution["pooled_matched_mta_estimate"] == 250.0
assert pooled.epoch_ratio_schedule[0].effective_ratio == 252.0 / 250.0
```

Selecting one observation is not the same operation as weighting all
observations. Neither
reduction is ever inferred.

## Reducing a coordinate schedule to one epoch ratio

The next evidence records are observed anchors at weeks zero and ten. The
authoritative epoch covers weeks two through eight. A direct explicit ratio
and each of the three coordinate reductions below produce exactly one final
effective ratio for the `("Print", "E1")` pair.

```python
print_week_0 = LiftTestEvidence(
    tested_entity="Print",
    observed_lift=80.0,
    matched_mta_estimate=100.0,
    test_id="print-geo-week0",
    coordinate_name="week",
    coordinate_value=0.0,
)
print_week_10 = LiftTestEvidence(
    tested_entity="Print",
    observed_lift=120.0,
    matched_mta_estimate=100.0,
    test_id="print-geo-week10",
    coordinate_name="week",
    coordinate_value=10.0,
)

reduction_calibration = TemporalAttributionCalibration(
    evidence=[print_week_0, print_week_10]
)
reduction_estimates = {"E1": {"Print": 100.0, "Search": 50.0}}
reduction_windows = {"E1": (2.0, 8.0)}

direct = reduction_calibration.apply_estimates(
    reduction_estimates,
    ratio_policy="explicit_ratio",
    epoch_ratio_policy="direct",
    residual_policy="allow_total_change",
    epoch_windows=reduction_windows,
    ratio_schedule={("Print", "E1"): 0.90},
)

representative = reduction_calibration.apply_estimates(
    reduction_estimates,
    ratio_policy="coordinate_linear",
    epoch_ratio_policy="representative_coordinate",
    residual_policy="allow_total_change",
    epoch_windows=reduction_windows,
    epoch_coordinates={"E1": 4.0},
)

averaged = reduction_calibration.apply_estimates(
    reduction_estimates,
    ratio_policy="coordinate_linear",
    epoch_ratio_policy="coordinate_average",
    residual_policy="allow_total_change",
    epoch_windows=reduction_windows,
)

support_weighted = reduction_calibration.apply_estimates(
    reduction_estimates,
    ratio_policy="coordinate_linear",
    epoch_ratio_policy="support_weighted_average",
    residual_policy="allow_total_change",
    epoch_windows=reduction_windows,
    epoch_coordinate_weights={"E1": [(3.0, 1.0), (7.0, 3.0)]},
)

assert direct.epoch_ratio_schedule[0].effective_ratio == 0.90
assert representative.epoch_ratio_schedule[0].effective_ratio == 0.96
assert averaged.epoch_ratio_schedule[0].effective_ratio == 1.00
assert support_weighted.epoch_ratio_schedule[0].effective_ratio == 1.04
```

## An epoch average is not an observed test result

The representative, coordinate-average, and support-weighted rows are
constructed from observed anchors. They are constructed values at attribution
epoch resolution; it is not a lift-test result. In particular, it
must not be reported as "the measured lift
in E1". The result provenance retains the source anchors, interpolation
interval, representative coordinate or weights, and applied reduction policy.
