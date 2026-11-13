"""Contracts for the public GA4 experiment-calibration workflow."""

from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from do_attribution.calibration import AttributionCalibration
from examples.calibration_workflow import (
    DEFAULT_EXPERIMENT_CONFIG,
    EXPERIMENT_DESCRIPTION,
    MARKOV_REMOVAL_STRATEGY,
    STATIC_TEST_ID,
    _fit_markov_estimates,
    construct_randomized_campaign,
    estimate_experimental_evidence,
    load_ga4_journeys,
    plot_experimental_evidence,
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


# --------------------------------------------------------------------------- #
# Published numbers are the numbers this workflow computes                     #
# --------------------------------------------------------------------------- #
# The notebook's saved outputs and the README example are copies of workflow
# results.  They once kept the values of an earlier Markov removal strategy
# after the workflow moved to ``removal_strategy="fail"``; these contracts
# tie every published copy to the code.
def _saved_notebook_outputs(source_marker: str) -> str:
    """Return the saved text outputs of the code cells containing a marker."""
    notebook = json.loads(
        (ROOT / "examples" / "attribution_experiment_calibration.ipynb").read_text(
            encoding="utf-8"
        )
    )
    texts: list[str] = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        if source_marker not in "".join(cell["source"]):
            continue
        for output in cell.get("outputs", []):
            texts.append("".join(output.get("text", "")))
            texts.append("".join(output.get("data", {}).get("text/plain", "")))
    assert texts, f"no executed cell contains {source_marker!r}"
    return "\n".join(texts)


def _notebook_number(value: float) -> str:
    """Format a float the way the notebook's display option does."""
    return f"{value:,.4f}"


def test_notebook_saved_outputs_match_the_workflow(workflow) -> None:
    static = workflow.static
    temporal = workflow.temporal
    assert static.attribution_model.removal_strategy == MARKOV_REMOVAL_STRATEGY

    uncalibrated = _saved_notebook_outputs('name="uncalibrated_attribution"')
    for value in static.attribution_estimates.values():
        assert _notebook_number(value) in uncalibrated

    matched = _saved_notebook_outputs('"matched_mta_estimate": [')
    assert MARKOV_REMOVAL_STRATEGY in matched.split()
    assert _notebook_number(static.evidence.matched_mta_estimate) in matched

    evidence = _saved_notebook_outputs("static_evidence = static_workflow.evidence")
    assert _notebook_number(static.evidence.observed_lift) in evidence
    assert _notebook_number(static.evidence.matched_mta_estimate) in evidence
    assert _notebook_number(static.evidence.calibration_ratio) in evidence

    by_epoch = _saved_notebook_outputs("attribution_estimates_by_epoch")
    tested_entity = DEFAULT_EXPERIMENT_CONFIG.tested_entity
    for estimates in temporal.attribution_estimates_by_epoch.values():
        assert _notebook_number(estimates[tested_entity]) in by_epoch

    window_evidence = _saved_notebook_outputs("temporal_evidence_table = ")
    for record in temporal.evidence:
        assert _notebook_number(record.observed_lift) in window_evidence
        assert _notebook_number(record.matched_mta_estimate) in window_evidence
        assert _notebook_number(record.calibration_ratio) in window_evidence

    comparison = _saved_notebook_outputs("comparison = calibration_comparison_table(")
    for column in (
        "experimental_lift",
        "experimental_standard_error",
        "true_expected_lift",
        "matched_mta_estimate",
        "calibration_ratio",
        "calibrated_attribution",
    ):
        for value in workflow.comparison[column]:
            assert _notebook_number(value) in comparison, column


def test_readme_static_example_uses_the_workflow_values(workflow) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index("attributed_conversions = {")
    end = readme.index("}", start) + 1
    quoted = ast.literal_eval(readme[start:end].split("=", 1)[1].strip())

    static = workflow.static
    assert quoted == {
        entity: round(value, 4)
        for entity, value in static.attribution_estimates.items()
    }
    assert f"observed_lift={static.evidence.observed_lift:.4f}," in readme
    assert f'test_id="{STATIC_TEST_ID}"' in readme


# --------------------------------------------------------------------------- #
# The experimental-evidence figure plots the workflow's own numbers            #
# --------------------------------------------------------------------------- #
@pytest.fixture
def evidence_figure(workflow):
    matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")
    matplotlib.use("Agg")
    plt = pytest.importorskip("matplotlib.pyplot")
    figure = plot_experimental_evidence(
        workflow.temporal,
        workflow.experimental_estimates,
    )
    yield figure
    plt.close(figure)


def _window_estimates(workflow) -> list[dict]:
    by_id = {
        row["evidence_id"]: row
        for row in workflow.experimental_estimates.to_dict(orient="records")
    }
    return [by_id[window.test_id] for window in DEFAULT_EXPERIMENT_CONFIG.windows]


def _bar_values(ax) -> dict[str, list[float]]:
    return {
        container.get_label(): [float(patch.get_width()) for patch in container]
        for container in ax.containers
        if hasattr(container, "patches")
    }


def test_evidence_figure_shows_each_randomized_arm(workflow, evidence_figure) -> None:
    outcomes_ax = evidence_figure.axes[0]
    estimates = _window_estimates(workflow)
    assert [text.get_text() for text in evidence_figure.texts] == [
        "Experimental Lift Evidence and Its Translation into "
        "Attribution Calibration"
    ]
    assert outcomes_ax.get_title() == "A. Experimental outcomes by test window"
    assert _bar_values(outcomes_ax) == {
        "Treatment: campaign-assigned journeys": [
            row["treatment_conversion_rate"] for row in estimates
        ],
        "Control: holdout journeys": [
            row["control_conversion_rate"] for row in estimates
        ],
    }
    # Every arm states its rate, its conversions and its journeys; every
    # window states its eligible sample.
    labels = [text.get_text() for text in outcomes_ax.texts]
    for row in estimates:
        for arm in ("treatment", "control"):
            assert (
                f"{row[f'{arm}_conversion_rate']:.2%}  "
                f"({row[f'{arm}_conversions']:,} of {row[f'{arm}_journeys']:,})"
            ) in labels
    assert len(labels) == 2 * len(estimates)
    assert [tick.get_text() for tick in outcomes_ax.get_yticklabels()] == [
        f"{window.name}\n{row['eligible_journeys']:,} eligible journeys"
        for window, row in zip(DEFAULT_EXPERIMENT_CONFIG.windows, estimates)
    ]
    assert labels[0] == "3.25%  (1,211 of 37,278)"


def test_evidence_figure_sets_the_lift_and_its_interval_beside_matched_mta(
    workflow, evidence_figure
) -> None:
    lift_ax = evidence_figure.axes[1]
    evidence = workflow.temporal.evidence
    assert lift_ax.get_title() == "B. Experimental lift versus matched MTA estimate"
    assert lift_ax.get_xlabel() == "Incremental or attributed conversions"
    assert _bar_values(lift_ax) == {
        "Experimental lift (incremental conversions)": [
            record.observed_lift for record in evidence
        ],
        "Matched MTA estimate (attributed conversions)": [
            record.matched_mta_estimate for record in evidence
        ],
    }

    # The whiskers are the experiment's 95% interval: lift ± 1.96 SE, the
    # bounds the estimates table reports.
    (interval,) = [
        container
        for container in lift_ax.containers
        if not hasattr(container, "patches")
    ]
    assert interval.get_label() == (
        "95% confidence interval of the lift (± 1.96 SE)"
    )
    (bars,) = interval.lines[2]
    estimates = _window_estimates(workflow)
    for segment, row, record in zip(bars.get_segments(), estimates, evidence):
        low, high = sorted(point[0] for point in segment)
        assert low == pytest.approx(row["ci95_lower"])
        assert high == pytest.approx(row["ci95_upper"])
        assert low == pytest.approx(record.observed_lift - 1.96 * record.standard_error)
        assert high == pytest.approx(record.observed_lift + 1.96 * record.standard_error)

    labels = [text.get_text() for text in lift_ax.texts]
    assert labels == [
        "776.8  (95% CI 601.4\N{EN DASH}952.1)",
        "603.5  (95% CI 459.1\N{EN DASH}747.8)",
        "630.4",
        "408.0",
    ]


def test_evidence_figure_separates_observed_ratios_from_constructed_ones(
    workflow, evidence_figure
) -> None:
    ratio_ax = evidence_figure.axes[2]
    result = workflow.temporal.calibration_result
    rows = [
        row
        for row in result.epoch_ratio_schedule
        if row.tested_entity == DEFAULT_EXPERIMENT_CONFIG.tested_entity
    ]
    assert [row.ratio_source for row in rows] == [
        "observed",
        "interpolated",
        "observed",
    ]
    assert ratio_ax.get_title() == (
        "C. Calibration ratios across time: Experimental Media, "
        "ratio policy coordinate_linear"
    )
    assert [tick.get_text() for tick in ratio_ax.get_xticklabels()] == [
        "Nov 2020",
        "Dec 2020",
        "Jan 2021",
    ]

    # One marker per epoch at the applied effective ratio.
    markers = [
        tuple(collection.get_offsets()[0]) for collection in ratio_ax.collections
    ]
    assert markers == [
        (float(index), pytest.approx(row.effective_ratio))
        for index, row in enumerate(rows)
    ]
    # Observed anchors and the constructed value do not share a marker style.
    colors = [
        tuple(collection.get_facecolor()[0]) for collection in ratio_ax.collections
    ]
    assert colors[0] == colors[2] != colors[1]
    shapes = [
        len(collection.get_paths()[0].vertices) for collection in ratio_ax.collections
    ]
    assert shapes[0] == shapes[2] != shapes[1]

    notes = [text.get_text() for text in ratio_ax.texts]
    assert notes == [
        "Observed\n1.232 = 776.8 / 630.4",
        "Interpolated\n1.354, no experiment ran in this epoch",
        "Observed\n1.479 = 603.5 / 408.0",
    ]
    for note, row in zip(notes, rows):
        assert f"{row.effective_ratio:.3f}" in note


def test_evidence_figure_draws_no_interval_on_ratios_or_calibrated_values(
    evidence_figure,
) -> None:
    """The standard error is not propagated, so only the lift has an interval."""
    outcomes_ax, lift_ax, ratio_ax = evidence_figure.axes
    assert not ratio_ax.containers
    assert not [
        container
        for container in outcomes_ax.containers
        if not hasattr(container, "patches")
    ]
    # Lines on the ratio panel are the reference at 1 and the schedule only.
    assert len(ratio_ax.lines) == 2
    legend = ratio_ax.get_legend()
    assert " ".join(legend.get_title().get_text().split()) == (
        "Ratios carry no interval: calibration does not propagate the "
        "experiment's standard error into ratios or calibrated estimates."
    )
    assert [text.get_text() for text in legend.get_texts()] == [
        "Observed: experimental lift / matched MTA of a test window",
        "Interpolated: built by the ratio policy, not measured",
        "Effective ratio applied to each epoch",
        "1.0: experimental lift equals the matched MTA estimate",
    ]
    # Nothing in the figure plots a calibrated estimate.
    every_text = " ".join(
        text.get_text()
        for ax in evidence_figure.axes
        for text in [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts]
    ).lower()
    assert "calibrated" not in every_text


def test_evidence_figure_legends_cover_no_bar_marker_or_value_label(
    evidence_figure,
) -> None:
    from matplotlib.transforms import Bbox

    def drawn_box(ax, collection):
        """Display box of an interval's segments or of scatter markers."""
        if hasattr(collection, "get_segments"):
            points = [point for segment in collection.get_segments() for point in segment]
        else:
            points = list(collection.get_offsets())
        pixels = ax.transData.transform(np.asarray(points, dtype=float))
        # Ten pixels cover a marker's radius and an interval's caps.
        return Bbox([pixels.min(axis=0), pixels.max(axis=0)]).padded(10.0)

    evidence_figure.canvas.draw()
    renderer = evidence_figure.canvas.get_renderer()
    for ax in evidence_figure.axes:
        legend_box = ax.get_legend().get_window_extent(renderer)
        occupied = [patch.get_window_extent(renderer) for patch in ax.patches]
        occupied += [text.get_window_extent(renderer) for text in ax.texts]
        occupied += [drawn_box(ax, collection) for collection in ax.collections]
        assert occupied
        assert not [box for box in occupied if legend_box.overlaps(box)]
        # The legend stays inside its own panel.
        panel = ax.get_window_extent(renderer)
        assert panel.x0 <= legend_box.x0 and legend_box.x1 <= panel.x1
        assert panel.y0 <= legend_box.y0 and legend_box.y1 <= panel.y1


def test_published_policy_comparison_values_are_the_ones_the_package_computes(
    workflow,
) -> None:
    """The docs and notebook quote the comparison they chose not to plot."""
    matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")
    matplotlib.use("Agg")
    plt = pytest.importorskip("matplotlib.pyplot")

    result = workflow.temporal.calibration_result
    entity = DEFAULT_EXPERIMENT_CONFIG.tested_entity
    december = result.epochs[1]
    assert str(december) == "2020-12"
    figure = result.plot_evidence_strength(
        diagnostic_epoch_ratio_policies={(entity, december): ("coordinate_average",)}
    )
    try:
        # One panel: the comparison, with no empty evidence-strength panel.
        (comparison_ax,) = figure.axes
        values = {
            collection.get_label(): float(collection.get_offsets()[0][1])
            for collection in comparison_ax.collections
        }
    finally:
        plt.close(figure)
    applied = values["Stored applied effective ratio"]
    window_average = values["Diagnostic only — not applied: coordinate_average"]
    assert f"{applied:.4f}" == "1.3536"
    assert f"{window_average:.4f}" == "1.3557"

    diagnostics = (ROOT / "docs" / "calibration_diagnostics.md").read_text(
        encoding="utf-8"
    )
    assert "(1.3536 applied, 1.3557 as the window average)" in diagnostics
    notebook = json.loads(
        (ROOT / "examples" / "attribution_experiment_calibration.ipynb").read_text(
            encoding="utf-8"
        )
    )
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    assert "is 1.3557 against the applied 1.3536" in markdown
    # The notebook plots the evidence figure and not the policy comparison.
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "plot_experimental_evidence(" in code
    assert "plot_evidence_strength" not in code

