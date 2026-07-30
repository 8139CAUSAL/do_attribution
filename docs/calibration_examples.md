# Static calibration examples

Static calibration applies one evidence record to each tested entity, then
handles the resulting residual under one required policy. The package does not
choose policy for you: it validates the
choices you declare. It does not select a
residual policy, eligible recipients, weights, bucket, or explicit adjustment.

The shared example starts with 1,000 attributed units. The Print experiment
supplies a ratio of 0.90, so Print's direct value moves from 200 to 180. That
direct adjustment creates a residual of 20.

```python
import pytest

from do_attribution.calibration import AttributionCalibration, LiftTestEvidence

print_geo = LiftTestEvidence(
    tested_entity="Print",
    observed_lift=90.0,
    matched_mta_estimate=100.0,
    test_id="print-geo-2024",
)
calibration = AttributionCalibration(evidence=[print_geo])
estimates = {"Print": 200.0, "Search": 500.0, "Social": 300.0}

def rows_by_entity(result):
    return {row.entity: row for row in result.rows}
```

## Allow the total to change

No residual is allocated. The calibrated total is the raw post-test total,
980.

```python
result_total_change = calibration.apply_estimates(
    estimates,
    residual_policy="allow_total_change",
)
assert result_total_change.original_total == 1000.0
assert result_total_change.raw_post_test_total == 980.0
assert result_total_change.allocated_residual == 0.0
assert result_total_change.calibrated_total == 980.0
```

## Allocate proportionally over named untested entities

Search and Social originally comprise 5/8 and 3/8 of their combined value.
They therefore receive 12.5 and 7.5 of the residual.

```python
result_proportional = calibration.apply_estimates(
    estimates,
    residual_policy="proportional_untested",
    eligible_entities=["Search", "Social"],
)
proportional_rows = rows_by_entity(result_proportional)
assert proportional_rows["Print"].calibrated_estimate == 180.0
assert proportional_rows["Search"].residual_adjustment == 12.5
assert proportional_rows["Social"].residual_adjustment == 7.5
assert result_proportional.calibrated_total == 1000.0
```

## Allocate with custom weights

The declared 25/75 weights allocate 5 to Search and 15 to Social.

```python
result_custom = calibration.apply_estimates(
    estimates,
    residual_policy="custom_weights",
    residual_weights={"Search": 0.25, "Social": 0.75},
)
custom_rows = rows_by_entity(result_custom)
assert custom_rows["Search"].residual_adjustment == 5.0
assert custom_rows["Social"].residual_adjustment == 15.0
assert result_custom.allocated_residual == 20.0
```

## Put the residual in a named bucket

The new bucket is an explicit accounting recipient, not a tested entity.

```python
result_bucket = calibration.apply_estimates(
    estimates,
    residual_policy="residual_bucket",
    residual_entity="Calibration residual",
)
bucket_rows = rows_by_entity(result_bucket)
assert bucket_rows["Calibration residual"].original_estimate == 0.0
assert bucket_rows["Calibration residual"].residual_adjustment == 20.0
assert bucket_rows["Calibration residual"].role == "residual_bucket"
```

## Supply explicit residual adjustments

The caller may state the complete allocation directly, provided the
adjustments sum to the required residual.

```python
result_explicit = calibration.apply_estimates(
    estimates,
    residual_policy="explicit_adjustments",
    residual_adjustments={"Search": 8.0, "Social": 12.0},
)
explicit_rows = rows_by_entity(result_explicit)
assert explicit_rows["Search"].residual_adjustment == 8.0
assert explicit_rows["Social"].residual_adjustment == 12.0
assert result_explicit.allocated_residual == 20.0
```

## Calibrate normalized shares

`apply_shares` converts valid shares to estimates using the declared total,
then applies the same policy. Here it is exactly equivalent to the
proportional estimate call.

```python
result_shares = calibration.apply_shares(
    {"Print": 0.20, "Search": 0.50, "Social": 0.30},
    current_total=1000.0,
    residual_policy="proportional_untested",
    eligible_entities=["Search", "Social"],
)
assert result_shares == result_proportional
assert rows_by_entity(result_shares)["Print"].calibrated_share == 0.18
```

## Invalid policy inputs raise

Inputs belonging to a different policy are rejected rather than ignored.

```python
with pytest.raises(ValueError, match="does not accept residual-allocation inputs"):
    calibration.apply_estimates(
        estimates,
        residual_policy="allow_total_change",
        eligible_entities=["Search", "Social"],
    )

with pytest.raises(ValueError, match="requires residual_weights"):
    calibration.apply_estimates(
        estimates,
        residual_policy="custom_weights",
    )
```

An untested recipient's change is a residual-allocation effect. It is not an
experimentally identified effect for that recipient:

- The lift test
  did not measure those entities; it measured Print.
