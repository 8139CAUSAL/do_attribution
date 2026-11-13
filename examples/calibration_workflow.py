"""Shared GA4 experiment-to-calibration workflow used by public examples.

The experimental layer is:

    A deterministic semi-synthetic randomized experiment constructed on
    public GA4 journey data.

The Google Analytics sample supplies journey structure, timestamps, and the
baseline conversion outcome.  Google did not conduct the experiment modeled
here: campaign assignment and incremental conversions are generated locally
with a fixed seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from do_attribution import MarkovAttribution, TemporalMarkovAttribution
from do_attribution.calibration import (
    AttributionCalibration,
    CalibrationResult,
    LiftTestEvidence,
    TemporalAttributionCalibration,
    TemporalCalibrationResult,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GA4_PATH = REPOSITORY_ROOT / "examples" / "ga4_attribution_journeys.csv"
EXPERIMENT_DESCRIPTION = (
    "A deterministic semi-synthetic randomized experiment constructed on "
    "public GA4 journey data."
)
MARKOV_REMOVAL_STRATEGY = "fail"


@dataclass(frozen=True)
class ExperimentWindow:
    """One pre-specified randomized-test window.

    ``end`` is exclusive, matching pandas filtering and the calibration
    package's half-open authoritative epoch-window convention.
    """

    name: str
    test_id: str
    start: datetime
    end: datetime
    incremental_probability: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ExperimentWindow.name must be nonempty.")
        if not self.test_id.strip():
            raise ValueError("ExperimentWindow.test_id must be nonempty.")
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("ExperimentWindow boundaries must be timezone-aware.")
        if self.start >= self.end:
            raise ValueError("ExperimentWindow.start must be before end.")
        if not 0.0 <= self.incremental_probability <= 1.0:
            raise ValueError("incremental_probability must lie in [0, 1].")

    @property
    def coordinate(self) -> date:
        """Return the declared evidence coordinate for temporal calibration."""

        return (self.start + (self.end - self.start) / 2).date()


@dataclass(frozen=True)
class ExperimentConfig:
    """Deterministic assignment and outcome-generation parameters."""

    tested_entity: str = "Experimental Media"
    random_seed: int = 20260726
    assignment_probability: float = 0.5
    windows: tuple[ExperimentWindow, ...] = (
        ExperimentWindow(
            name="November 2020",
            test_id="experimental-media-ga4-2020-11",
            start=datetime(2020, 11, 1, tzinfo=timezone.utc),
            end=datetime(2020, 12, 1, tzinfo=timezone.utc),
            incremental_probability=0.012,
        ),
        ExperimentWindow(
            name="January 2021",
            test_id="experimental-media-ga4-2021-01",
            start=datetime(2021, 1, 1, tzinfo=timezone.utc),
            end=datetime(2021, 2, 1, tzinfo=timezone.utc),
            incremental_probability=0.006,
        ),
    )

    def __post_init__(self) -> None:
        if not self.tested_entity.strip():
            raise ValueError("tested_entity must be nonempty.")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise TypeError("random_seed must be an integer.")
        if not 0.0 < self.assignment_probability < 1.0:
            raise ValueError("assignment_probability must lie strictly between 0 and 1.")
        if not self.windows:
            raise ValueError("At least one experiment window is required.")
        ordered = sorted(self.windows, key=lambda item: item.start)
        if tuple(ordered) != self.windows:
            raise ValueError("Experiment windows must be chronological.")
        for previous, current in zip(self.windows, self.windows[1:]):
            if previous.end > current.start:
                raise ValueError("Experiment windows must not overlap.")
        test_ids = [window.test_id for window in self.windows]
        if len(set(test_ids)) != len(test_ids):
            raise ValueError("Experiment window test_id values must be unique.")


DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()
STATIC_TEST_ID = "experimental-media-ga4-full-eligible-period"


@dataclass
class StaticCalibrationWorkflow:
    """Static attribution, matched evidence, and calibrated result."""

    attribution_model: MarkovAttribution
    attribution_fit: Mapping[str, Any]
    attribution_estimates: Mapping[str, float]
    experiment_estimate: pd.Series
    evidence: LiftTestEvidence
    calibration_result: CalibrationResult


@dataclass
class TemporalCalibrationWorkflow:
    """Temporal attribution, window matches, evidence, and calibration."""

    attribution_model: TemporalMarkovAttribution
    attribution_fit: Mapping[str, Any]
    attribution_estimates_by_epoch: Mapping[Any, Mapping[str, float]]
    matched_window_estimates: Mapping[str, float]
    matched_window_models: Mapping[str, MarkovAttribution]
    evidence: tuple[LiftTestEvidence, ...]
    epoch_windows: Mapping[Any, tuple[date, date]]
    epoch_coordinates: Mapping[Any, date]
    epoch_support: Mapping[Any, float]
    calibration_result: TemporalCalibrationResult


@dataclass
class CalibrationWorkflow:
    """Complete shared state consumed by the notebook and image generator."""

    journeys: pd.DataFrame
    experiment_data: pd.DataFrame
    experimental_estimates: pd.DataFrame
    static: StaticCalibrationWorkflow
    temporal: TemporalCalibrationWorkflow
    comparison: pd.DataFrame


def load_ga4_journeys(
    path: str | Path = DEFAULT_GA4_PATH,
) -> pd.DataFrame:
    """Load and validate the public GA4 journey extract.

    The returned frame retains the supplied path, timestamp, and ``conversion``
    columns.  Timestamps are parsed as UTC so window membership is explicit.
    """

    source = Path(path)
    journeys = pd.read_csv(source)
    required = {
        "journey_id",
        "path",
        "conversion",
        "journey_start",
        "anchor_timestamp",
    }
    missing = sorted(required - set(journeys.columns))
    if missing:
        raise ValueError(f"GA4 journey extract is missing columns: {missing!r}.")
    if journeys.empty:
        raise ValueError("GA4 journey extract must not be empty.")
    if journeys["journey_id"].duplicated().any():
        raise ValueError("journey_id must be unique at the journey-row grain.")
    if journeys[list(required)].isna().any().any():
        raise ValueError("Required GA4 journey fields must not contain nulls.")
    if not set(journeys["conversion"].unique()) <= {0, 1}:
        raise ValueError("conversion must be binary.")
    if (journeys["path"].astype(str).str.strip() == "").any():
        raise ValueError("path must be nonempty.")

    journeys = journeys.copy(deep=True)
    journeys["conversion"] = journeys["conversion"].astype("int8")
    journeys["journey_start"] = pd.to_datetime(
        journeys["journey_start"], utc=True, errors="raise"
    )
    journeys["anchor_timestamp"] = pd.to_datetime(
        journeys["anchor_timestamp"], utc=True, errors="raise"
    )
    journeys.attrs["source_path"] = str(source.resolve())
    journeys.attrs["source_description"] = (
        "Public obfuscated GA4 ecommerce journey extract."
    )
    return journeys


def construct_randomized_campaign(
    journeys: pd.DataFrame,
    *,
    config: ExperimentConfig = DEFAULT_EXPERIMENT_CONFIG,
) -> pd.DataFrame:
    """Add deterministic randomized exposure and semi-synthetic outcomes.

    Eligibility is restricted to the declared test windows.  Each eligible
    journey is independently assigned to campaign or holdout.  The tested
    entity is appended to the observed path only for campaign-assigned rows.
    Incremental conversions are Bernoulli draws only among treated baseline
    non-converters; all original conversions remain conversions.
    """

    required = {"journey_id", "path", "conversion", "anchor_timestamp"}
    missing = sorted(required - set(journeys.columns))
    if missing:
        raise ValueError(f"journeys is missing columns: {missing!r}.")
    if not set(journeys["conversion"].unique()) <= {0, 1}:
        raise ValueError("conversion must be binary.")

    frame = journeys.copy(deep=True)
    timestamps = pd.to_datetime(frame["anchor_timestamp"], utc=True, errors="raise")
    entity_pattern = rf"(?:^| > ){re.escape(config.tested_entity)}(?: > |$)"
    if frame["path"].astype(str).str.contains(entity_pattern, regex=True).any():
        raise ValueError(
            f"tested_entity {config.tested_entity!r} collides with an existing "
            "journey-path token."
        )

    eligible = np.zeros(len(frame), dtype=bool)
    incremental_probability = np.zeros(len(frame), dtype=float)
    window_labels = np.full(len(frame), None, dtype=object)
    for window in config.windows:
        mask = (
            (timestamps >= pd.Timestamp(window.start))
            & (timestamps < pd.Timestamp(window.end))
        ).to_numpy()
        if np.any(eligible & mask):
            raise ValueError("Experiment windows produce overlapping row eligibility.")
        eligible[mask] = True
        incremental_probability[mask] = window.incremental_probability
        window_labels[mask] = window.name

    seed_sequence = np.random.SeedSequence(config.random_seed)
    assignment_seed, outcome_seed = seed_sequence.spawn(2)
    assignment_rng = np.random.default_rng(assignment_seed)
    outcome_rng = np.random.default_rng(outcome_seed)
    assignment_draw = assignment_rng.random(len(frame))
    outcome_draw = outcome_rng.random(len(frame))

    campaign_assigned = eligible & (
        assignment_draw < config.assignment_probability
    )
    baseline_conversion = frame["conversion"].to_numpy(dtype=np.int8, copy=True)
    incremental_conversion = (
        campaign_assigned
        & (baseline_conversion == 0)
        & (outcome_draw < incremental_probability)
    )
    observed_conversion = np.maximum(
        baseline_conversion,
        incremental_conversion.astype(np.int8),
    )

    assignments = np.full(len(frame), "not eligible", dtype=object)
    assignments[eligible] = "holdout"
    assignments[campaign_assigned] = "campaign"
    augmented_paths = frame["path"].astype(str).to_numpy(dtype=object, copy=True)
    augmented_paths[campaign_assigned] = (
        augmented_paths[campaign_assigned]
        + " > "
        + config.tested_entity
    )

    frame["experiment_eligible"] = eligible
    frame["experiment_window"] = pd.Series(
        window_labels,
        index=frame.index,
        dtype="string",
    )
    frame["experiment_assignment"] = pd.Series(
        assignments,
        index=frame.index,
        dtype="string",
    )
    frame["campaign_assigned"] = campaign_assigned
    frame["declared_incremental_probability"] = incremental_probability
    frame["baseline_conversion"] = baseline_conversion
    frame["incremental_conversion"] = incremental_conversion.astype("int8")
    frame["observed_conversion"] = observed_conversion.astype("int8")
    frame["augmented_path"] = augmented_paths
    frame.attrs.update(journeys.attrs)
    frame.attrs["experimental_layer"] = EXPERIMENT_DESCRIPTION
    frame.attrs["tested_entity"] = config.tested_entity
    frame.attrs["random_seed"] = config.random_seed
    return frame


def _standard_error_of_lift(
    treatment_outcomes: np.ndarray,
    control_outcomes: np.ndarray,
    eligible_count: int,
) -> float:
    """Return the randomization standard error on the full-population scale."""

    if len(treatment_outcomes) < 2 or len(control_outcomes) < 2:
        raise ValueError("Both experiment arms need at least two journeys.")
    treatment_variance = float(np.var(treatment_outcomes, ddof=1))
    control_variance = float(np.var(control_outcomes, ddof=1))
    rate_difference_se = np.sqrt(
        treatment_variance / len(treatment_outcomes)
        + control_variance / len(control_outcomes)
    )
    return float(eligible_count * rate_difference_se)


def _estimate_one_experiment(
    frame: pd.DataFrame,
    mask: np.ndarray,
    *,
    scope: str,
    evidence_id: str,
    window_name: str,
    window_start: datetime,
    window_end: datetime,
) -> dict[str, Any]:
    eligible = frame["experiment_eligible"].to_numpy(dtype=bool) & mask
    treated = eligible & frame["campaign_assigned"].to_numpy(dtype=bool)
    control = eligible & ~frame["campaign_assigned"].to_numpy(dtype=bool)
    outcomes = frame["observed_conversion"].to_numpy(dtype=float)

    eligible_count = int(eligible.sum())
    treatment_count = int(treated.sum())
    control_count = int(control.sum())
    if eligible_count == 0 or treatment_count == 0 or control_count == 0:
        raise ValueError(f"Experiment {evidence_id!r} has an empty analysis arm.")

    treatment_rate = float(outcomes[treated].mean())
    control_rate = float(outcomes[control].mean())
    rate_difference = treatment_rate - control_rate
    observed_lift = float(eligible_count * rate_difference)
    standard_error = _standard_error_of_lift(
        outcomes[treated],
        outcomes[control],
        eligible_count,
    )

    baseline_nonconverter = (
        eligible
        & (frame["baseline_conversion"].to_numpy(dtype=np.int8) == 0)
    )
    probabilities = frame["declared_incremental_probability"].to_numpy(
        dtype=float
    )
    true_expected_lift = float(probabilities[baseline_nonconverter].sum())
    baseline_nonconverter_count = int(baseline_nonconverter.sum())
    average_incremental_probability = (
        true_expected_lift / baseline_nonconverter_count
        if baseline_nonconverter_count
        else 0.0
    )

    return {
        "scope": scope,
        "evidence_id": evidence_id,
        "experiment_window": window_name,
        "window_start": window_start.date().isoformat(),
        "window_end_exclusive": window_end.date().isoformat(),
        "eligible_journeys": eligible_count,
        "treatment_journeys": treatment_count,
        "control_journeys": control_count,
        "treatment_conversions": int(outcomes[treated].sum()),
        "control_conversions": int(outcomes[control].sum()),
        "treatment_conversion_rate": treatment_rate,
        "control_conversion_rate": control_rate,
        "conversion_rate_difference": rate_difference,
        "observed_lift": observed_lift,
        "standard_error": standard_error,
        "ci95_lower": observed_lift - 1.96 * standard_error,
        "ci95_upper": observed_lift + 1.96 * standard_error,
        "declared_incremental_probability": average_incremental_probability,
        "true_expected_lift": true_expected_lift,
        "realized_treated_incremental_conversions": int(
            frame.loc[treated, "incremental_conversion"].sum()
        ),
        "experimental_layer": EXPERIMENT_DESCRIPTION,
    }


def estimate_experimental_evidence(
    experiment_data: pd.DataFrame,
    *,
    config: ExperimentConfig = DEFAULT_EXPERIMENT_CONFIG,
) -> pd.DataFrame:
    """Estimate full-period and window-specific randomized lift.

    The estimator follows the calibration design exactly:

    ``N_eligible * (mean(observed_conversion | campaign) -
                    mean(observed_conversion | holdout))``.
    """

    required = {
        "experiment_eligible",
        "campaign_assigned",
        "baseline_conversion",
        "incremental_conversion",
        "observed_conversion",
        "declared_incremental_probability",
        "anchor_timestamp",
    }
    missing = sorted(required - set(experiment_data.columns))
    if missing:
        raise ValueError(f"experiment_data is missing columns: {missing!r}.")

    frame = experiment_data
    timestamps = pd.to_datetime(frame["anchor_timestamp"], utc=True, errors="raise")
    all_rows = np.ones(len(frame), dtype=bool)
    records = [
        _estimate_one_experiment(
            frame,
            all_rows,
            scope="static_full_period",
            evidence_id=STATIC_TEST_ID,
            window_name="All pre-specified test windows",
            window_start=config.windows[0].start,
            window_end=config.windows[-1].end,
        )
    ]
    for window in config.windows:
        mask = (
            (timestamps >= pd.Timestamp(window.start))
            & (timestamps < pd.Timestamp(window.end))
        ).to_numpy()
        records.append(
            _estimate_one_experiment(
                frame,
                mask,
                scope="temporal_window",
                evidence_id=window.test_id,
                window_name=window.name,
                window_start=window.start,
                window_end=window.end,
            )
        )
    return pd.DataFrame.from_records(records)


def _fit_markov_estimates(
    paths: pd.Series,
    conversions: pd.Series,
) -> tuple[MarkovAttribution, Mapping[str, Any], dict[str, float]]:
    model = MarkovAttribution(
        order=1,
        # Calibration matches the original experimental population, so the
        # channel-removal question must preserve every journey and its weight.
        removal_strategy=MARKOV_REMOVAL_STRATEGY,
    )
    fit = model.fit(paths, conversions)
    attribution = _require_normalized_attribution(
        fit,
        context="Static or matched-window Markov fit",
    )
    total_conversions = float(conversions.sum())
    estimates = {
        entity: float(share * total_conversions)
        for entity, share in attribution.items()
    }
    return model, fit, estimates


def _require_normalized_attribution(
    fit: Mapping[str, Any],
    *,
    context: str,
) -> Mapping[str, float]:
    """Return shares or stop before raw diagnostics reach calibration."""

    attribution = fit.get("attribution")
    if attribution is None:
        reason = fit.get("normalization_reason", "unknown")
        raise ValueError(
            f"{context} normalized attribution is unavailable ({reason}); "
            "raw signed diagnostics are not shares and cannot be calibrated "
            "as shares."
        )
    return attribution


def fit_static_attribution_and_calibration(
    experiment_data: pd.DataFrame,
    *,
    experimental_estimates: pd.DataFrame | None = None,
    config: ExperimentConfig = DEFAULT_EXPERIMENT_CONFIG,
) -> StaticCalibrationWorkflow:
    """Fit matched static Markov attribution and apply static calibration."""

    estimates_table = (
        estimate_experimental_evidence(experiment_data, config=config)
        if experimental_estimates is None
        else experimental_estimates
    )
    static_rows = estimates_table.loc[
        estimates_table["scope"] == "static_full_period"
    ]
    if len(static_rows) != 1:
        raise ValueError("experimental_estimates must contain one static row.")
    experiment_estimate = static_rows.iloc[0].copy()

    eligible = experiment_data["experiment_eligible"].to_numpy(dtype=bool)
    analysis = experiment_data.loc[eligible]
    model, fit, attribution_estimates = _fit_markov_estimates(
        analysis["augmented_path"],
        analysis["observed_conversion"],
    )
    matched_mta_estimate = float(
        attribution_estimates.get(config.tested_entity, 0.0)
    )
    if matched_mta_estimate <= 0.0:
        raise ValueError("The matched tested-entity MTA estimate must be positive.")

    evidence = LiftTestEvidence(
        tested_entity=config.tested_entity,
        observed_lift=float(experiment_estimate["observed_lift"]),
        matched_mta_estimate=matched_mta_estimate,
        test_id=STATIC_TEST_ID,
        outcome_name="journey conversion",
        window_start=experiment_estimate["window_start"],
        window_end=experiment_estimate["window_end_exclusive"],
        population=(
            "GA4 journeys eligible for the pre-specified November 2020 and "
            "January 2021 experiment windows"
        ),
        unit="incremental conversions",
        standard_error=float(experiment_estimate["standard_error"]),
        metadata={
            "experimental_layer": EXPERIMENT_DESCRIPTION,
            "random_seed": config.random_seed,
            "assignment_probability": config.assignment_probability,
            "eligible_journeys": int(experiment_estimate["eligible_journeys"]),
            "true_expected_lift": float(
                experiment_estimate["true_expected_lift"]
            ),
        },
    )
    residual_recipients = [
        entity
        for entity, value in attribution_estimates.items()
        if entity != config.tested_entity and value > 0.0
    ]
    calibration_result = AttributionCalibration([evidence]).apply_estimates(
        attribution_estimates,
        residual_policy="proportional_untested",
        eligible_entities=residual_recipients,
    )
    return StaticCalibrationWorkflow(
        attribution_model=model,
        attribution_fit=fit,
        attribution_estimates=attribution_estimates,
        experiment_estimate=experiment_estimate,
        evidence=evidence,
        calibration_result=calibration_result,
    )


def _epoch_design(
    temporal_model: TemporalMarkovAttribution,
    timestamps: pd.Series,
) -> tuple[
    dict[Any, tuple[date, date]],
    dict[Any, date],
    dict[Any, float],
]:
    naive_timestamps = pd.to_datetime(timestamps, utc=True).dt.tz_localize(None)
    epoch_periods = naive_timestamps.dt.to_period("M")
    epoch_windows: dict[Any, tuple[date, date]] = {}
    epoch_coordinates: dict[Any, date] = {}
    epoch_support: dict[Any, float] = {}
    for epoch in temporal_model.epochs:
        start = epoch.start_time.date()
        end = (epoch + 1).start_time.date()
        epoch_windows[epoch] = (start, end)
        epoch_coordinates[epoch] = start + timedelta(days=(end - start).days // 2)
        epoch_support[epoch] = float((epoch_periods == epoch).sum())
    return epoch_windows, epoch_coordinates, epoch_support


def fit_temporal_attribution_and_calibration(
    experiment_data: pd.DataFrame,
    *,
    experimental_estimates: pd.DataFrame | None = None,
    config: ExperimentConfig = DEFAULT_EXPERIMENT_CONFIG,
) -> TemporalCalibrationWorkflow:
    """Fit monthly Markov attribution and coordinate-linear calibration."""

    estimates_table = (
        estimate_experimental_evidence(experiment_data, config=config)
        if experimental_estimates is None
        else experimental_estimates
    )
    timestamps = pd.to_datetime(
        experiment_data["anchor_timestamp"], utc=True, errors="raise"
    )
    naive_timestamps = timestamps.dt.tz_localize(None)
    temporal_model = TemporalMarkovAttribution(
        freq="M",
        min_samples_per_epoch=30,
        epoch_assignment="first_touch",
        order=1,
        # Each epoch remains aligned to its original experimental population.
        removal_strategy=MARKOV_REMOVAL_STRATEGY,
    )
    temporal_fit = temporal_model.fit(
        experiment_data["augmented_path"],
        naive_timestamps,
        experiment_data["observed_conversion"],
    )

    period_labels = naive_timestamps.dt.to_period("M")
    all_entities = sorted(set(temporal_model.channels) | {config.tested_entity})
    attribution_estimates_by_epoch: dict[Any, dict[str, float]] = {}
    for epoch in temporal_model.epochs:
        total_conversions = float(
            experiment_data.loc[
                period_labels == epoch,
                "observed_conversion",
            ].sum()
        )
        attribution = _require_normalized_attribution(
            temporal_model.epoch_results[epoch],
            context=f"Temporal Markov epoch {epoch!s}",
        )
        attribution_estimates_by_epoch[epoch] = {
            entity: float(attribution.get(entity, 0.0) * total_conversions)
            for entity in all_entities
        }

    matched_window_estimates: dict[str, float] = {}
    matched_window_models: dict[str, MarkovAttribution] = {}
    evidence_records: list[LiftTestEvidence] = []
    for window in config.windows:
        window_mask = (
            (timestamps >= pd.Timestamp(window.start))
            & (timestamps < pd.Timestamp(window.end))
            & experiment_data["experiment_eligible"]
        )
        window_data = experiment_data.loc[window_mask]
        window_model, _, window_estimates = _fit_markov_estimates(
            window_data["augmented_path"],
            window_data["observed_conversion"],
        )
        matched_mta_estimate = float(
            window_estimates.get(config.tested_entity, 0.0)
        )
        if matched_mta_estimate <= 0.0:
            raise ValueError(
                f"Matched MTA estimate is not positive for {window.name}."
            )
        matched_window_models[window.test_id] = window_model
        matched_window_estimates[window.test_id] = matched_mta_estimate

        estimate_rows = estimates_table.loc[
            estimates_table["evidence_id"] == window.test_id
        ]
        if len(estimate_rows) != 1:
            raise ValueError(
                f"experimental_estimates must contain one row for {window.test_id}."
            )
        estimate = estimate_rows.iloc[0]
        evidence_records.append(
            LiftTestEvidence(
                tested_entity=config.tested_entity,
                observed_lift=float(estimate["observed_lift"]),
                matched_mta_estimate=matched_mta_estimate,
                test_id=window.test_id,
                outcome_name="journey conversion",
                window_start=estimate["window_start"],
                window_end=estimate["window_end_exclusive"],
                population=f"GA4 journeys eligible in {window.name}",
                unit="incremental conversions",
                coordinate_name="date",
                coordinate_value=window.coordinate,
                standard_error=float(estimate["standard_error"]),
                metadata={
                    "experimental_layer": EXPERIMENT_DESCRIPTION,
                    "random_seed": config.random_seed,
                    "assignment_probability": config.assignment_probability,
                    "declared_incremental_probability": (
                        window.incremental_probability
                    ),
                    "eligible_journeys": int(estimate["eligible_journeys"]),
                    "true_expected_lift": float(estimate["true_expected_lift"]),
                },
            )
        )

    epoch_windows, epoch_coordinates, epoch_support = _epoch_design(
        temporal_model,
        timestamps,
    )
    eligible_entities_by_epoch = {
        epoch: [
            entity
            for entity, value in attribution_estimates_by_epoch[epoch].items()
            if entity != config.tested_entity and value > 0.0
        ]
        for epoch in temporal_model.epochs
    }
    calibration_result = TemporalAttributionCalibration(
        evidence_records
    ).apply_estimates(
        attribution_estimates_by_epoch,
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="representative_coordinate",
        residual_policy="proportional_untested",
        epoch_windows=epoch_windows,
        epoch_coordinates=epoch_coordinates,
        epoch_support=epoch_support,
        eligible_entities_by_epoch=eligible_entities_by_epoch,
    )
    return TemporalCalibrationWorkflow(
        attribution_model=temporal_model,
        attribution_fit=temporal_fit,
        attribution_estimates_by_epoch=attribution_estimates_by_epoch,
        matched_window_estimates=matched_window_estimates,
        matched_window_models=matched_window_models,
        evidence=tuple(evidence_records),
        epoch_windows=epoch_windows,
        epoch_coordinates=epoch_coordinates,
        epoch_support=epoch_support,
        calibration_result=calibration_result,
    )


def calibration_comparison_table(
    static: StaticCalibrationWorkflow,
    temporal: TemporalCalibrationWorkflow,
    experimental_estimates: pd.DataFrame,
    *,
    config: ExperimentConfig = DEFAULT_EXPERIMENT_CONFIG,
) -> pd.DataFrame:
    """Compare randomized lift, matched MTA, and calibrated attribution."""

    rows: list[dict[str, Any]] = []
    static_row = next(
        row
        for row in static.calibration_result.rows
        if row.entity == config.tested_entity
    )
    rows.append(
        {
            "scope": "Static: all eligible windows",
            "experimental_lift": static.evidence.observed_lift,
            "experimental_standard_error": static.evidence.standard_error,
            "true_expected_lift": float(
                static.experiment_estimate["true_expected_lift"]
            ),
            "matched_mta_estimate": static.evidence.matched_mta_estimate,
            "calibration_ratio": static.evidence.calibration_ratio,
            "calibrated_attribution": static_row.calibrated_estimate,
        }
    )

    estimate_by_id = {
        row["evidence_id"]: row
        for row in experimental_estimates.to_dict(orient="records")
    }
    evidence_by_id = {record.test_id: record for record in temporal.evidence}
    for window in config.windows:
        evidence = evidence_by_id[window.test_id]
        epoch = next(
            epoch
            for epoch, coordinate in temporal.epoch_coordinates.items()
            if coordinate == window.coordinate
        )
        calibrated_row = next(
            row
            for row in temporal.calibration_result.rows[epoch]
            if row.entity == config.tested_entity
        )
        rows.append(
            {
                "scope": f"Temporal: {window.name}",
                "experimental_lift": evidence.observed_lift,
                "experimental_standard_error": evidence.standard_error,
                "true_expected_lift": float(
                    estimate_by_id[window.test_id]["true_expected_lift"]
                ),
                "matched_mta_estimate": evidence.matched_mta_estimate,
                "calibration_ratio": evidence.calibration_ratio,
                "calibrated_attribution": calibrated_row.calibrated_estimate,
            }
        )
    return pd.DataFrame.from_records(rows)


def plot_experimental_evidence(
    temporal: TemporalCalibrationWorkflow,
    experimental_estimates: pd.DataFrame,
    *,
    config: ExperimentConfig = DEFAULT_EXPERIMENT_CONFIG,
) -> Any:
    """Plot the randomized evidence and its translation into calibration.

    The three panels follow the chain this workflow computes:

    A. the randomized outcomes of each test window: conversion rate,
       conversions, and journeys of the campaign-assigned and holdout arms;
    B. the estimated incremental lift with its 95% confidence interval,
       beside the matched attribution estimate it is compared with;
    C. the calibration ratio of each authoritative epoch, separating ratios
       observed in an experiment from ratios the applied policy constructed.

    The interval in panel B is the experiment's own: lift plus or minus 1.96
    standard errors.  Calibration does not propagate the standard error, so
    panel C draws no interval on any ratio and the figure shows no interval
    for a calibrated estimate.
    """

    import matplotlib.pyplot as plt  # plotting is optional for the workflow
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    experiment_color = "#1f77b4"
    holdout_color = "#8ab8e0"
    constructed_color = "#959ca8"
    interval_color = "#0d3d66"
    edge = {"edgecolor": "black", "linewidth": 0.4}

    estimate_by_id = {
        row["evidence_id"]: row
        for row in experimental_estimates.to_dict(orient="records")
    }
    evidence_by_id = {record.test_id: record for record in temporal.evidence}
    windows = config.windows
    missing = [
        window.test_id
        for window in windows
        if window.test_id not in estimate_by_id
        or window.test_id not in evidence_by_id
    ]
    if missing:
        raise ValueError(
            "Every experiment window needs one experimental estimate and one "
            f"evidence record; missing={missing!r}."
        )
    estimates = [estimate_by_id[window.test_id] for window in windows]
    evidence = [evidence_by_id[window.test_id] for window in windows]

    result = temporal.calibration_result
    ratio_rows = {
        row.epoch: row
        for row in result.epoch_ratio_schedule
        if row.tested_entity == config.tested_entity
    }
    ratio_epochs = [epoch for epoch in result.epochs if epoch in ratio_rows]
    evidence_by_coordinate = {
        record.coordinate_value: record for record in temporal.evidence
    }

    figure = plt.figure(figsize=(12, 9.5))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.15))
    outcomes_ax = figure.add_subplot(grid[0, 0])
    lift_ax = figure.add_subplot(grid[0, 1])
    ratio_ax = figure.add_subplot(grid[1, :])
    figure.suptitle(
        "Experimental Lift Evidence and Its Translation into "
        "Attribution Calibration",
        fontsize=14,
    )

    bar_height = 0.34
    offset = 0.2
    centers = [float(index) for index in range(len(windows))]
    upper = [center - offset for center in centers]
    lower = [center + offset for center in centers]
    # Panels A and B keep an empty band under the last window for their
    # legends, so a legend never sits on a bar or on a value label.
    legend_band = 0.95
    bar_axis_limits = (len(windows) - 0.5 + legend_band, -0.5)

    # A. Randomized outcomes -------------------------------------------------
    treatment_rates = [row["treatment_conversion_rate"] for row in estimates]
    control_rates = [row["control_conversion_rate"] for row in estimates]
    outcomes_ax.barh(
        upper,
        treatment_rates,
        height=bar_height,
        color=experiment_color,
        label="Treatment: campaign-assigned journeys",
        **edge,
    )
    outcomes_ax.barh(
        lower,
        control_rates,
        height=bar_height,
        color=holdout_color,
        label="Control: holdout journeys",
        **edge,
    )
    for positions, rates, prefix in (
        (upper, treatment_rates, "treatment"),
        (lower, control_rates, "control"),
    ):
        for position, rate, row in zip(positions, rates, estimates):
            outcomes_ax.annotate(
                f"{rate:.2%}  ({row[f'{prefix}_conversions']:,} of "
                f"{row[f'{prefix}_journeys']:,})",
                (rate, position),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
            )
    outcomes_ax.set_yticks(centers)
    outcomes_ax.set_yticklabels(
        [
            f"{window.name}\n{row['eligible_journeys']:,} eligible journeys"
            for window, row in zip(windows, estimates)
        ]
    )
    outcomes_ax.set_ylim(*bar_axis_limits)
    outcomes_ax.set_xlim(0.0, 1.9 * max(treatment_rates + control_rates))
    outcomes_ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    outcomes_ax.set_xlabel("Conversion rate")
    outcomes_ax.set_title("A. Experimental outcomes by test window")
    outcomes_ax.grid(axis="x", alpha=0.2)
    outcomes_ax.set_axisbelow(True)
    outcomes_ax.legend(loc="lower center", fontsize=9)

    # B. Estimated lift beside the matched attribution estimate ---------------
    lifts = [record.observed_lift for record in evidence]
    half_widths = [1.96 * record.standard_error for record in evidence]
    matched = [record.matched_mta_estimate for record in evidence]
    lift_ax.barh(
        upper,
        lifts,
        height=bar_height,
        color=experiment_color,
        label="Experimental lift (incremental conversions)",
        **edge,
    )
    lift_ax.errorbar(
        lifts,
        upper,
        xerr=half_widths,
        fmt="none",
        ecolor=interval_color,
        elinewidth=2.0,
        capsize=6,
        capthick=2.0,
        label="95% confidence interval of the lift (± 1.96 SE)",
    )
    lift_ax.barh(
        lower,
        matched,
        height=bar_height,
        color=constructed_color,
        label="Matched MTA estimate (attributed conversions)",
        **edge,
    )
    for position, lift, half_width in zip(upper, lifts, half_widths):
        lift_ax.annotate(
            f"{lift:,.1f}  (95% CI {lift - half_width:,.1f}"
            f"\N{EN DASH}{lift + half_width:,.1f})",
            (lift + half_width, position),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    for position, value in zip(lower, matched):
        lift_ax.annotate(
            f"{value:,.1f}",
            (value, position),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    lift_ax.set_yticks(centers)
    lift_ax.set_yticklabels([window.name for window in windows])
    lift_ax.set_ylim(*bar_axis_limits)
    lift_ax.set_xlim(
        0.0,
        2.0 * max(lift + half_width for lift, half_width in zip(lifts, half_widths)),
    )
    lift_ax.set_xlabel("Incremental or attributed conversions")
    lift_ax.set_title("B. Experimental lift versus matched MTA estimate")
    lift_ax.grid(axis="x", alpha=0.2)
    lift_ax.set_axisbelow(True)
    lift_ax.legend(loc="lower center", fontsize=9)

    # C. Calibration ratios across time ---------------------------------------
    positions = [float(index) for index in range(len(ratio_epochs))]
    ratios = [ratio_rows[epoch].effective_ratio for epoch in ratio_epochs]
    ratio_ax.axhline(
        1.0,
        color="0.45",
        linewidth=1.0,
        linestyle="--",
        zorder=1,
    )
    ratio_ax.plot(
        positions,
        ratios,
        color=experiment_color,
        linewidth=2.0,
        zorder=2,
    )
    constructed_sources: list[str] = []
    for position, epoch, ratio in zip(positions, ratio_epochs, ratios):
        row = ratio_rows[epoch]
        observed = row.ratio_source == "observed"
        ratio_ax.scatter(
            [position],
            [ratio],
            marker="o" if observed else "D",
            s=150 if observed else 120,
            color=experiment_color if observed else constructed_color,
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
        record = evidence_by_coordinate.get(row.epoch_coordinate)
        if observed and record is not None:
            note = (
                f"Observed\n{ratio:.3f} = {record.observed_lift:,.1f} / "
                f"{record.matched_mta_estimate:,.1f}"
            )
        elif observed:
            note = f"Observed\n{ratio:.3f}"
        else:
            source = row.ratio_source.replace("_", " ").capitalize()
            if source not in constructed_sources:
                constructed_sources.append(source)
            note = f"{source}\n{ratio:.3f}, no experiment ran in this epoch"
        ratio_ax.annotate(
            note,
            (position, ratio),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ratio_ax.set_xticks(positions)
    ratio_ax.set_xticklabels(
        [
            epoch.strftime("%b %Y") if hasattr(epoch, "strftime") else str(epoch)
            for epoch in ratio_epochs
        ]
    )
    ratio_ax.set_xlim(-0.5, len(ratio_epochs) - 0.5)
    # The band above the highest annotation is kept free for the legend.
    span = max(ratios + [1.0]) - min(ratios + [1.0])
    ratio_ax.set_ylim(
        min(ratios + [1.0]) - 0.12 * span,
        max(ratios + [1.0]) + 1.0 * span,
    )
    ratio_ax.set_xlabel("Authoritative attribution epoch")
    ratio_ax.set_ylabel("Calibration ratio\n(experimental lift / matched MTA)")
    ratio_ax.set_title(
        f"C. Calibration ratios across time: {config.tested_entity}, "
        f"ratio policy {result.ratio_policy}"
    )
    ratio_ax.grid(axis="y", alpha=0.2)
    ratio_ax.set_axisbelow(True)
    ratio_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=10,
            markerfacecolor=experiment_color,
            markeredgecolor="black",
            markeredgewidth=0.6,
            label="Observed: experimental lift / matched MTA of a test window",
        ),
        Line2D(
            [],
            [],
            marker="D",
            linestyle="none",
            markersize=9,
            markerfacecolor=constructed_color,
            markeredgecolor="black",
            markeredgewidth=0.6,
            label=(
                f"{' / '.join(constructed_sources) or 'Constructed'}: built by "
                "the ratio policy, not measured"
            ),
        ),
        Line2D(
            [],
            [],
            color=experiment_color,
            linewidth=2.0,
            label="Effective ratio applied to each epoch",
        ),
        Line2D(
            [],
            [],
            color="0.45",
            linewidth=1.0,
            linestyle="--",
            label="1.0: experimental lift equals the matched MTA estimate",
        ),
    ]
    ratio_ax.legend(
        handles=ratio_handles,
        loc="upper center",
        ncol=2,
        fontsize=9,
        title=(
            "Ratios carry no interval: calibration does not propagate the "
            "experiment's standard error into ratios or calibrated estimates."
        ),
        title_fontsize=9,
    )
    figure.tight_layout()
    return figure


def run_calibration_workflow(
    *,
    path: str | Path = DEFAULT_GA4_PATH,
    config: ExperimentConfig = DEFAULT_EXPERIMENT_CONFIG,
) -> CalibrationWorkflow:
    """Execute the canonical workflow once for notebooks and image exports."""

    journeys = load_ga4_journeys(path)
    experiment_data = construct_randomized_campaign(journeys, config=config)
    experimental_estimates = estimate_experimental_evidence(
        experiment_data,
        config=config,
    )
    static = fit_static_attribution_and_calibration(
        experiment_data,
        experimental_estimates=experimental_estimates,
        config=config,
    )
    temporal = fit_temporal_attribution_and_calibration(
        experiment_data,
        experimental_estimates=experimental_estimates,
        config=config,
    )
    comparison = calibration_comparison_table(
        static,
        temporal,
        experimental_estimates,
        config=config,
    )
    return CalibrationWorkflow(
        journeys=journeys,
        experiment_data=experiment_data,
        experimental_estimates=experimental_estimates,
        static=static,
        temporal=temporal,
        comparison=comparison,
    )


__all__ = [
    "CalibrationWorkflow",
    "DEFAULT_EXPERIMENT_CONFIG",
    "DEFAULT_GA4_PATH",
    "EXPERIMENT_DESCRIPTION",
    "ExperimentConfig",
    "ExperimentWindow",
    "STATIC_TEST_ID",
    "StaticCalibrationWorkflow",
    "TemporalCalibrationWorkflow",
    "calibration_comparison_table",
    "construct_randomized_campaign",
    "estimate_experimental_evidence",
    "fit_static_attribution_and_calibration",
    "fit_temporal_attribution_and_calibration",
    "load_ga4_journeys",
    "plot_experimental_evidence",
    "run_calibration_workflow",
]
