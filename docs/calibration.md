# Calibration

Calibration is a deterministic post-attribution layer. It applies external
lift evidence to an existing attribution result; it does not turn observational
attribution into a causal estimate. A calibration ratio is:

```text
observed experimental lift / matched attribution estimate
```

That ratio directly adjusts only the experimentally tested entity. Any change
to an untested entity is a separate, practitioner-selected residual-allocation
policy, not an effect measured by the experiment.

`standard_error` records uncertainty reported by the external experiment. It is
validated and retained with the complete evidence provenance, but it does not
change the calibration ratio, direct adjustment, residual allocation, or
calibrated point estimate. Calibration does not propagate an uncertainty
interval from this field.

## Construction and explicit policy selection

Construct the static and temporal calibrators with evidence:

```python
static = AttributionCalibration(evidence=evidence)
temporal = TemporalAttributionCalibration(evidence=evidence)
```

`AttributionCalibration(evidence=evidence)` requires exactly one evidence
record per tested entity. `TemporalAttributionCalibration(evidence=evidence)`
permits multiple evidence records for an entity so long as their identifiers
and coordinate metadata satisfy the temporal evidence contract.

The policy arguments on the `apply_estimates` and `apply_shares` entry points
are keyword-only and have no defaults. Static calls require
`residual_policy`. Temporal calls require `ratio_policy`,
`epoch_ratio_policy`, and `residual_policy`. Historical or implicit switches
such as `normalize`, `preserve_total`, `interpolate`, and `transport` are not
accepted policy aliases.

Omitting a required keyword or passing an unknown keyword produces a native
binding-time `TypeError`. A recognized call with invalid values, missing
policy-specific inputs, incompatible policies, or mismatched mapping coverage
produces an actionable `ValueError`.

## Residual policy matrix

Direct evidence adjustment and residual allocation are deliberately separate.
Tested entities cannot receive residual allocation in this phase.

| Residual policy | Behavior | Required policy inputs | Temporal form |
| --- | --- | --- | --- |
| `allow_total_change` | Allocates no residual; the post-test total may differ from the original total. | None. Residual-allocation controls are not permitted. | May be selected globally or for specific calibrated epochs. |
| `proportional_untested` | Allocates the residual over explicitly eligible untested entities in proportion to their original values. | Exactly one of `eligible_entities` or `all_untested=True`. | `eligible_entities_by_epoch` may replace the global selector and must exactly cover the epochs using this policy. |
| `custom_weights` | Allocates the residual using explicit, finite, nonnegative unit-sum recipient weights. | `residual_weights`. | `residual_weights_by_epoch` may replace global weights and must exactly cover the epochs using this policy. |
| `residual_bucket` | Assigns the residual to one named untested bucket. | A nonempty, hashable `residual_entity`; `allow_bucket_collision` is optional and applies only here. | `residual_entity_by_epoch` may replace the global entity and must exactly cover the epochs using this policy. |
| `explicit_adjustments` | Applies practitioner-supplied residual adjustments whose total must match the required residual within tolerance. | `residual_adjustments`. | `residual_adjustments_by_epoch` is required and must exactly cover the epochs using this policy. |

A global temporal residual control is valid only when every calibrated epoch
uses its corresponding policy. A global control and its per-epoch counterpart
may not both be supplied. Missing and extra rows raise rather than being
silently filled or ignored.

## Temporal ratio policy matrix

Ratio policies state how retained evidence becomes a schedule. Constructed or
transported values are policy outputs, not new experimental observations.

| Ratio policy | Meaning | Required policy inputs | Coverage and compatible epoch-ratio policy |
| --- | --- | --- | --- |
| `observed_only` | Uses observed evidence at its declared coordinate and aligns it to an authoritative epoch. | Evidence coordinates and `epoch_windows`; explicit conflict resolution when more than one observation maps to a tested-entity/epoch pair. | Observed pairs only; compatible with the direct epoch policy. |
| `constant` | Transports one explicitly selected observed ratio to declared entity/epoch pairs. | `source_evidence` and `tested_entity_epochs`. | Exactly the declared pairs; compatible with the direct epoch policy. |
| `explicit_ratio` | Uses practitioner-supplied ratios for declared pairs. | `ratio_schedule`; `tested_entity_epochs` is permitted to constrain coverage. | Exactly the supplied schedule; compatible with the direct epoch policy. |
| `explicit_strength` | Moves explicitly selected evidence ratios toward one using practitioner-supplied strengths. | `source_evidence` and `evidence_strength`; `tested_entity_epochs` is permitted to constrain coverage. | Exactly the strength schedule; compatible with the direct epoch policy. |
| `coordinate_linear` | Linearly interpolates between observed coordinate anchors without extrapolation. | At least two retained coordinate-bearing evidence points per selected entity and `epoch_windows`; `tested_entity_epochs` is permitted. | Within observed support only; compatible with representative-coordinate, coordinate-average, or support-weighted-average reduction. |

Inputs not listed as required or permitted for the selected ratio policy raise
rather than being ignored. Ratio bounds, when supplied, are an explicit final
bounding operation; the library does not silently clip ratios.

## Epoch-ratio policy matrix

An epoch-ratio policy reduces a ratio schedule to one final effective ratio for
each calibrated tested-entity/epoch pair.

| Epoch-ratio policy | Reduction | Required inputs and support | Compatible ratio schedule |
| --- | --- | --- | --- |
| `direct` | Uses a ratio already defined at epoch resolution. | No coordinate-reduction input; coordinate metadata and coordinate weights are not permitted. | Observed-only, constant, explicit-ratio, or explicit-strength schedules. |
| `representative_coordinate` | Evaluates a coordinate-linear schedule at one declared representative coordinate per affected epoch. | `epoch_coordinates`, inside observed support. | Coordinate-linear schedules only. |
| `coordinate_average` | Averages the piecewise-linear schedule over each authoritative epoch window. | `epoch_windows`; no representative coordinates or coordinate weights. | Coordinate-linear schedules only. |
| `support_weighted_average` | Averages the coordinate-linear schedule using explicit within-epoch coordinate weights. | `epoch_windows` and `epoch_coordinate_weights`; scalar `epoch_support` is diagnostic metadata and is not a substitute. | Coordinate-linear schedules only. |

Interpolation is confined to observed support. Calibration refuses
extrapolation and does not infer an epoch coordinate.

## Evidence-conflict policy matrix

Conflict policies apply only when the observed-only/direct pipeline places
multiple evidence records in one tested-entity/epoch group. The selected
policy may be supplied globally for all conflict groups or by a mapping whose
keys exactly match those groups.

| Evidence-conflict policy | Resolution | Required inputs | Provenance retained |
| --- | --- | --- | --- |
| `select_evidence` | Uses one explicitly selected evidence record from each conflict group. | `selected_evidence`, keyed exactly by applicable conflict group. | Candidate evidence identifiers and ratios, plus the selected identifier. |
| `weighted_ratio` | Computes an explicit weighted combination of observed ratios. | `evidence_weights`, keyed exactly by applicable group and then by all retained evidence identifiers; weights must sum to one. | Candidate identifiers and ratios, combination weights, and weighted terms. |
| `pooled_lift` | Divides pooled observed lift by pooled matched attribution after validating compatible tested entity, outcome, population, and unit metadata. | No additional mapping. | Component lifts, matched estimates, observed ratios, pooled numerator and denominator, and compatibility metadata. |

Irrelevant or extra conflict mappings raise. `selected_evidence` and
`evidence_weights` are accepted only for groups using their corresponding
conflict policies.

## Coverage and unchanged epochs

The temporal result distinguishes calibrated pairs from uncovered pairs.
Uncovered epochs are reported as uncalibrated. They retain their complete
original allocation, receive no direct or residual adjustment, and remain
wholly unchanged in calibrated output. Calibration never invents evidence or
silently extends a schedule to achieve complete coverage.

## Result and audit contract

Results keep the calculation layers separate and inspectable:

- original attribution;
- evidence-backed direct adjustments;
- residual adjustments;
- calibrated attribution;
- final effective ratios;
- complete provenance for evidence, ratio construction, epoch reduction, and
  residual allocation;
- original/raw/required/allocated/calibrated totals;
- warnings.

Untested residual recipients remain identified as policy-derived recipients,
not experimentally measured entities. Stored evidence retains the observed
lift, matched attribution estimate, test identifier, experimental provenance,
and supplied uncertainty metadata.

## Error, inference, and mutation boundaries

The calibration layer never implicitly fills missing values, selects evidence,
chooses weights, clips ratios, extrapolates, renormalizes tested values, or
infers an epoch coordinate. Those choices require an explicit supported policy
and its declared inputs.

Caller inputs and fitted attribution objects are never mutated. Repeating the
same call with equal inputs and policies must produce identical rows, exports,
summaries, plots, warnings, and provenance.

Sparse schedules remain sparse and are not expanded into a dense grid. Plots
and exports use stored result state; they do not refit calibration, choose a
new policy, or recompute evidence matches.
