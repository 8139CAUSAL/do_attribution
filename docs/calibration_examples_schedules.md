# Calibration schedule, share, bound, and mix examples

In these examples, the package validates the choices you
declare; it does not select a ratio policy, an epoch-ratio policy, a residual
policy, an evidence-strength schedule, or a ratio bound for you.

The only measured quantity anywhere in this document is the single
`Print` ratio of `0.90`. Every other ratio is constructed by an explicit
policy, and every untested-entity change is a residual allocation.

## Part 1 — An explicit evidence-strength schedule

The retained Print experiment has observed lift 90 and a matched attribution
estimate of 100, so its observed calibration ratio is 0.90.

```python
import pytest

from do_attribution.calibration import (
    LiftTestEvidence,
    TemporalAttributionCalibration,
)

print_geo = LiftTestEvidence(
    tested_entity="Print",
    observed_lift=90.0,
    matched_mta_estimate=100.0,
    test_id="print-geo-2024",
    outcome_name="orders",
    population="all customers",
    unit="orders",
)
calibration = TemporalAttributionCalibration(evidence=[print_geo])

EPOCHS = ("E1", "E2", "E3", "E4", "E5")
estimates_by_epoch = {
    epoch: {"Print": 200.0, "Search": 500.0, "Social": 300.0}
    for epoch in EPOCHS
}
declared_strengths = {
    ("Print", "E1"): 1.00,
    ("Print", "E2"): 0.75,
    ("Print", "E3"): 0.50,
    ("Print", "E4"): 0.25,
    ("Print", "E5"): 0.00,
}

STRENGTH_INPUTS = dict(
    ratio_policy="explicit_strength",
    epoch_ratio_policy="direct",
    residual_policy="proportional_untested",
    source_evidence={"Print": "print-geo-2024"},
    evidence_strength=declared_strengths,
    eligible_entities=["Search", "Social"],
)

strength = calibration.apply_estimates(
    estimates_by_epoch,
    **STRENGTH_INPUTS,
)
assert [
    round(row.effective_ratio, 3)
    for row in strength.epoch_ratio_schedule
] == [0.900, 0.925, 0.950, 0.975, 1.000]
```

### The library does not infer the strength schedule

The explicit-strength policy requires inputs naming both the source evidence
and every requested pair's strength. Each strength must be in \[0, 1\], and
the policy requires epoch_ratio_policy='direct'. Invalid or irrelevant inputs
raise.

```python
with pytest.raises(ValueError, match="requires inputs"):
    calibration.apply_estimates(
        estimates_by_epoch,
        ratio_policy="explicit_strength",
        epoch_ratio_policy="direct",
        residual_policy="proportional_untested",
        eligible_entities=["Search", "Social"],
    )

with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
    calibration.apply_estimates(
        estimates_by_epoch,
        ratio_policy="explicit_strength",
        epoch_ratio_policy="direct",
        residual_policy="proportional_untested",
        source_evidence={"Print": "print-geo-2024"},
        evidence_strength={("Print", "E1"): 1.25},
        eligible_entities=["Search", "Social"],
    )

with pytest.raises(ValueError, match="requires epoch_ratio_policy='direct'"):
    calibration.apply_estimates(
        estimates_by_epoch,
        ratio_policy="explicit_strength",
        epoch_ratio_policy="coordinate_average",
        residual_policy="proportional_untested",
        source_evidence={"Print": "print-geo-2024"},
        evidence_strength=declared_strengths,
        eligible_entities=["Search", "Social"],
    )
```

### Missing strengths are not filled in

Partial schedule coverage calibrates only the named pairs. Uncovered epochs
retain their original rows and have no residual policy.

```python
partial = calibration.apply_estimates(
    estimates_by_epoch,
    ratio_policy="explicit_strength",
    epoch_ratio_policy="direct",
    residual_policy="proportional_untested",
    source_evidence={"Print": "print-geo-2024"},
    evidence_strength={
        ("Print", "E1"): 1.00,
        ("Print", "E2"): 0.75,
    },
    eligible_entities=["Search", "Social"],
)

assert [row.epoch for row in partial.epoch_ratio_schedule] == ["E1", "E2"]
```

## Part 2 — Temporal shares plus totals

`apply_shares` accepts a unit-sum attribution vector and an absolute total for
each epoch. It is equivalent to passing their products through
`apply_estimates`.

```python
shares_by_epoch = {
    epoch: {"Print": 0.20, "Search": 0.50, "Social": 0.30}
    for epoch in EPOCHS
}
totals_by_epoch = {epoch: 1000.0 for epoch in EPOCHS}

from_shares = calibration.apply_shares(
    shares_by_epoch,
    totals_by_epoch=totals_by_epoch,
    **STRENGTH_INPUTS,
)
assert from_shares == strength
assert next(
    row for row in from_shares.rows["E1"] if row.entity == "Print"
).original_share == 0.20
assert next(
    row for row in from_shares.rows["E1"] if row.entity == "Print"
).calibrated_share == 0.18
```

## Part 3 — Raise-only ratio bounds

### Bounds raise; they never clip

There is no default ratio bound. When `ratio_bounds=(0.95, 1.05)` is declared,
the first strength-derived ratio is rejected as outside declared raise-only
bounds. The error identifies
`tested_entity='Print'`, `epoch='E1'`, `effective_ratio=0.9`, `lower=0.95`,
`upper=1.05`, `ratio_policy='explicit_strength'`,
`epoch_ratio_policy='direct'`, `ratio_source='strength_derived'`, and
`source_evidence_ids=('print-geo-2024',)`. It is never quietly replaced by
the nearest bound.

```python
with pytest.raises(ValueError, match="outside declared raise-only bounds"):
    calibration.apply_estimates(
        estimates_by_epoch,
        ratio_policy="explicit_strength",
        epoch_ratio_policy="direct",
        residual_policy="proportional_untested",
        source_evidence={"Print": "print-geo-2024"},
        evidence_strength=declared_strengths,
        eligible_entities=["Search", "Social"],
        ratio_bounds=(0.95, 1.05),
    )

bounded = calibration.apply_estimates(
    estimates_by_epoch,
    ratio_policy="explicit_strength",
    epoch_ratio_policy="direct",
    residual_policy="proportional_untested",
    source_evidence={"Print": "print-geo-2024"},
    evidence_strength=declared_strengths,
    eligible_entities=["Search", "Social"],
    ratio_bounds=(0.90, 1.00),
)
assert [
    round(row.effective_ratio, 3)
    for row in bounded.epoch_ratio_schedule
] == [0.900, 0.925, 0.950, 0.975, 1.000]
```

## Part 4 — Changing entity mix with a constant effective ratio

### The ratio schedule and the epochs do not move

Both calls transport the same observed Print ratio to the same two
authoritative epochs. Only the untested mix changes.

```python
MIX_INPUTS = dict(
    ratio_policy="constant",
    epoch_ratio_policy="direct",
    residual_policy="proportional_untested",
    source_evidence={"Print": "print-geo-2024"},
    tested_entity_epochs={"Print": ["E1", "E2"]},
    epoch_windows={"E1": (0.0, 10.0), "E2": (10.0, 20.0)},
    eligible_entities=["Search", "Social"],
)

steady_mix = calibration.apply_estimates(
    {
        "E1": {"Print": 200.0, "Search": 500.0, "Social": 300.0},
        "E2": {"Print": 200.0, "Search": 500.0, "Social": 300.0},
    },
    **MIX_INPUTS,
)
shifted_mix = calibration.apply_estimates(
    {
        "E1": {"Print": 200.0, "Search": 500.0, "Social": 300.0},
        "E2": {"Print": 200.0, "Search": 200.0, "Social": 600.0},
    },
    **MIX_INPUTS,
)

assert steady_mix.epoch_ratio_schedule == shifted_mix.epoch_ratio_schedule
```

### The residual allocation does move

The required residual remains 20 in each epoch. Under the steady mix it is
allocated 12.5 to Search and 7.5 to Social. Under the shifted second-epoch mix,
it is allocated 5 to Search and 15 to Social.
Neither number is an experimentally identified effect on `Social`. No test measured `Social`.
