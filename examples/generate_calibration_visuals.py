"""Generate the public GA4 experiment-calibration visualization bank.

Run from the repository root:

    python examples/generate_calibration_visuals.py

The default output directory is ``examples/images``.  Pass ``--output-dir``
to render the same deterministic image set elsewhere for verification.

Every figure is computed from a deterministic semi-synthetic randomized
experiment constructed on the public GA4 journey rows.  Google did not
conduct the generated experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from do_attribution.calibration import CalibrationResult, TemporalCalibrationResult
from examples.calibration_workflow import (
    EXPERIMENT_DESCRIPTION,
    construct_randomized_campaign,
    estimate_experimental_evidence,
    fit_static_attribution_and_calibration,
    fit_temporal_attribution_and_calibration,
    load_ga4_journeys,
    run_calibration_workflow,
)


DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "examples" / "images"
ARTIFACT_FILENAMES = (
    "calibration_static_effects.png",
    "calibration_temporal_epoch_alignment.png",
    "calibration_temporal_ratio_schedule.png",
    "calibration_temporal_evidence_uncertainty.png",
    "calibration_temporal_effects.png",
    "calibration_temporal_residual_allocation.png",
    "calibration_temporal_entity_mix.png",
    "calibration_temporal_evidence_strength.png",
)


def build_static_result() -> CalibrationResult:
    """Build the GA4 experiment's matched static calibration result."""

    journeys = load_ga4_journeys()
    experiment_data = construct_randomized_campaign(journeys)
    experimental_estimates = estimate_experimental_evidence(experiment_data)
    return fit_static_attribution_and_calibration(
        experiment_data,
        experimental_estimates=experimental_estimates,
    ).calibration_result


def build_temporal_result() -> TemporalCalibrationResult:
    """Build the GA4 experiment's coordinate-linear temporal result."""

    journeys = load_ga4_journeys()
    experiment_data = construct_randomized_campaign(journeys)
    experimental_estimates = estimate_experimental_evidence(experiment_data)
    return fit_temporal_attribution_and_calibration(
        experiment_data,
        experimental_estimates=experimental_estimates,
    ).calibration_result


def build_strength_result() -> TemporalCalibrationResult:
    """Backward-compatible alias for the shared temporal result.

    The applied policy is coordinate-linear, so its evidence-strength
    diagnostic correctly states that an explicit strength schedule is not
    applicable.  A separate toy result is intentionally not constructed.
    """

    return build_temporal_result()


def _label_experimental_layer(figure: Any) -> Any:
    """Apply the required provenance label to a generated figure."""

    figure.text(
        0.5,
        0.01,
        EXPERIMENT_DESCRIPTION,
        ha="center",
        va="bottom",
        fontsize=8,
        color="0.25",
        style="italic",
    )
    # Rotated date ticks in the alignment diagnostic need enough separation
    # from this provenance line in tight-bbox exports.
    figure.subplots_adjust(bottom=0.20)
    return figure


def _save_figure(figure: Any, destination: Path) -> None:
    figure.savefig(
        destination,
        dpi=120,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "do-attribution"},
    )
    plt.close(figure)


def generate_calibration_visuals(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, ...]:
    """Generate all eight public calibration PNG artifacts."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    workflow = run_calibration_workflow()
    static_result = workflow.static.calibration_result
    temporal_result = workflow.temporal.calibration_result
    tested_entity = workflow.temporal.evidence[0].tested_entity
    middle_epoch = temporal_result.epochs[len(temporal_result.epochs) // 2]

    figures = (
        static_result.plot_calibration_effects(),
        temporal_result.plot_epoch_alignment(),
        temporal_result.plot_ratio_schedule(),
        temporal_result.plot_evidence_uncertainty(),
        temporal_result.plot_calibration_effects(),
        temporal_result.plot_residual_allocation(),
        temporal_result.plot_entity_mix(),
        temporal_result.plot_evidence_strength(
            diagnostic_epoch_ratio_policies={
                (tested_entity, middle_epoch): ("coordinate_average",)
            },
        ),
    )
    figures = tuple(_label_experimental_layer(figure) for figure in figures)
    paths = tuple(destination / name for name in ARTIFACT_FILENAMES)
    for figure, path in zip(figures, paths):
        _save_figure(figure, path)
    return paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that receives the eight deterministic PNG files.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    for artifact in generate_calibration_visuals(args.output_dir):
        print(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
