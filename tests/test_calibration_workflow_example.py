"""Contracts for the public GA4 experiment-calibration workflow."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from do_attribution.calibration import AttributionCalibration
from examples.calibration_workflow import (
    DEFAULT_EXPERIMENT_CONFIG,
    EXPERIMENT_DESCRIPTION,
    STATIC_TEST_ID,
    _fit_markov_estimates,
    construct_randomized_campaign,
    estimate_experimental_evidence,
    load_ga4_journeys,
    run_calibration_workflow,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def workflow():
    return run_calibration_workflow()


def test_ga4_source_is_loaded_at_the_documented_journey_grain() -> None:
    journeys = load_ga4_journeys()
    assert journeys.shape == (272_792, 8)
    assert journeys["journey_id"].nunique() == len(journeys)
    assert int(journeys["conversion"].sum()) == 4_848
    assert journeys["anchor_timestamp"].min().date().isoformat() == "2020-11-01"
    assert journeys["anchor_timestamp"].max().date().isoformat() == "2021-01-31"


def test_campaign_construction_is_deterministic_and_preserves_baseline() -> None:
    journeys = load_ga4_journeys()
    first = construct_randomized_campaign(journeys)
    second = construct_randomized_campaign(journeys)
    generated_columns = [
        "experiment_eligible",
        "experiment_window",
        "experiment_assignment",
        "campaign_assigned",
        "declared_incremental_probability",
        "baseline_conversion",
        "incremental_conversion",
        "observed_conversion",
        "augmented_path",
    ]
    pd.testing.assert_frame_equal(first[generated_columns], second[generated_columns])
    pd.testing.assert_series_equal(
        first["baseline_conversion"].astype(journeys["conversion"].dtype),
        journeys["conversion"],
        check_names=False,
    )
    assert (
        first["observed_conversion"].to_numpy()
        == np.maximum(
            first["baseline_conversion"].to_numpy(),
            first["incremental_conversion"].to_numpy(),
        )
    ).all()
    incremental = first["incremental_conversion"].eq(1)
    assert first.loc[incremental, "campaign_assigned"].all()
    assert first.loc[incremental, "baseline_conversion"].eq(0).all()
    treated = first["campaign_assigned"]
    assert first.loc[treated, "augmented_path"].str.endswith(
        f" > {DEFAULT_EXPERIMENT_CONFIG.tested_entity}"
    ).all()
    assert first.loc[~treated, "augmented_path"].equals(
        first.loc[~treated, "path"]
    )
    assert not first.loc[~first["experiment_eligible"], "campaign_assigned"].any()
    assert first.attrs["experimental_layer"] == EXPERIMENT_DESCRIPTION


def test_randomized_estimates_are_calculated_from_observed_arms() -> None:
    experiment = construct_randomized_campaign(load_ga4_journeys())
    estimates = estimate_experimental_evidence(experiment)
    assert estimates["evidence_id"].tolist() == [
        STATIC_TEST_ID,
        "experimental-media-ga4-2020-11",
        "experimental-media-ga4-2021-01",
    ]
    for estimate in estimates.itertuples(index=False):
        expected = estimate.eligible_journeys * (
            estimate.treatment_conversion_rate
            - estimate.control_conversion_rate
        )
        assert estimate.observed_lift == pytest.approx(expected)
        assert estimate.standard_error > 0.0
        assert estimate.ci95_lower < estimate.observed_lift < estimate.ci95_upper
        assert estimate.true_expected_lift > 0.0
        assert estimate.experimental_layer == EXPERIMENT_DESCRIPTION


def test_workflow_stops_when_normalized_attribution_is_unavailable() -> None:
    with pytest.warns(UserWarning, match="fail_refit"):
        with pytest.raises(
            ValueError,
            match=(
                r"normalized attribution is unavailable "
                r"\(zero_raw_magnitude\).*raw signed diagnostics are not shares"
            ),
        ):
            _fit_markov_estimates(
                pd.Series(["A", "B"]),
                pd.Series([0, 0]),
            )


def test_static_and_temporal_evidence_are_matched_to_attribution(
    workflow,
) -> None:
    tested_entity = DEFAULT_EXPERIMENT_CONFIG.tested_entity
    static = workflow.static
    assert static.attribution_model.removal_strategy == "fail"
    assert static.attribution_fit["population_policy"] == "fixed"
    for diagnostic in static.attribution_fit["population_diagnostics"].values():
        assert (
            diagnostic["transformed_journey_count"]
            == diagnostic["original_journey_count"]
        )
        assert (
            diagnostic["transformed_total_weight"]
            == diagnostic["original_total_weight"]
        )
        assert diagnostic["dropped_path_count"] == 0
    assert static.evidence.observed_lift == pytest.approx(
        static.experiment_estimate["observed_lift"]
    )
    assert static.evidence.matched_mta_estimate == pytest.approx(
        static.attribution_estimates[tested_entity]
    )
    static_tested_row = next(
        row
        for row in static.calibration_result.rows
        if row.entity == tested_entity
    )
    assert static_tested_row.calibrated_estimate == pytest.approx(
        static.evidence.observed_lift
    )
    assert static.calibration_result.calibrated_total == pytest.approx(
        static.calibration_result.original_total
    )

    temporal = workflow.temporal
    assert temporal.attribution_model.removal_strategy == "fail"
    assert all(
        result["population_policy"] == "fixed"
        for result in temporal.attribution_model.epoch_results.values()
    )
    assert all(
        model.removal_strategy == "fail"
        for model in temporal.matched_window_models.values()
    )
    assert tuple(str(epoch) for epoch in temporal.calibration_result.epochs) == (
        "2020-11",
        "2020-12",
        "2021-01",
    )
    assert [record.test_id for record in temporal.evidence] == [
        "experimental-media-ga4-2020-11",
        "experimental-media-ga4-2021-01",
    ]
    assert [
        row.ratio_source for row in temporal.calibration_result.epoch_ratio_schedule
    ] == ["observed", "interpolated", "observed"]
    for record in temporal.evidence:
        assert record.matched_mta_estimate == pytest.approx(
            temporal.matched_window_estimates[record.test_id]
        )
        assert record.metadata["experimental_layer"] == EXPERIMENT_DESCRIPTION

    december = temporal.calibration_result.epochs[1]
    december_tested_row = next(
        row
        for row in temporal.calibration_result.rows[december]
        if row.entity == tested_entity
    )
    assert december_tested_row.original_estimate == 0.0
    assert december_tested_row.direct_test_adjustment == 0.0
    for epoch in temporal.calibration_result.epochs:
        assert temporal.calibration_result.calibrated_totals[epoch] == pytest.approx(
            temporal.calibration_result.original_totals[epoch]
        )


def test_comparison_reconciles_experimental_and_calibrated_values(
    workflow,
) -> None:
    comparison = workflow.comparison
    assert comparison["scope"].tolist() == [
        "Static: all eligible windows",
        "Temporal: November 2020",
        "Temporal: January 2021",
    ]
    assert np.allclose(
        comparison["experimental_lift"],
        comparison["calibrated_attribution"],
    )
    assert np.allclose(
        comparison["experimental_lift"] / comparison["matched_mta_estimate"],
        comparison["calibration_ratio"],
    )


def test_workflow_point_calibration_is_independent_of_standard_error(
    workflow,
) -> None:
    static = workflow.static
    eligible_entities = [
        entity
        for entity, value in static.attribution_estimates.items()
        if entity != static.evidence.tested_entity and value > 0.0
    ]

    def calibrate(standard_error: float):
        evidence = replace(static.evidence, standard_error=standard_error)
        return AttributionCalibration([evidence]).apply_estimates(
            static.attribution_estimates,
            residual_policy="proportional_untested",
            eligible_entities=eligible_entities,
        )

    low_error = calibrate(1.0)
    high_error = calibrate(1000.0)

    assert low_error.rows == high_error.rows
    assert low_error.original_total == high_error.original_total
    assert low_error.required_residual == high_error.required_residual
    assert low_error.allocated_residual == high_error.allocated_residual
    assert low_error.calibrated_total == high_error.calibrated_total
    assert low_error.evidence[0].standard_error == 1.0
    assert high_error.evidence[0].standard_error == 1000.0


def test_notebook_and_generator_share_the_workflow_module() -> None:
    generator = (ROOT / "examples" / "generate_calibration_visuals.py").read_text(
        encoding="utf-8"
    )
    assert "from examples.calibration_workflow import" in generator
    assert "run_calibration_workflow()" in generator

    notebook = ROOT / "examples" / "attribution_experiment_calibration.ipynb"
    assert notebook.is_file()
    text = notebook.read_text(encoding="utf-8")
    assert "from calibration_workflow import" in text
    assert "sys.path" not in text
    assert "Path.cwd()" not in text
    assert EXPERIMENT_DESCRIPTION in text
