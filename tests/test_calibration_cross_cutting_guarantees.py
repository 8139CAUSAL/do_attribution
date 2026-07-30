"""Cross-cutting determinism, deep immutability, provenance, and
no-hidden-inference guarantees for CAL-087.

Four scenarios cover the static engine and the temporal engine's
interpolated, conflict-resolved, and strength-attenuated paths. Every
scenario is built twice from independently constructed inputs so repeated
execution, caller mutation, nested result mutation, exported-DataFrame
mutation, and diagnostic rendering can all be checked against an equal
pristine result.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")

from do_attribution.calibration import (  # noqa: E402
    AttributionCalibration,
    LiftTestEvidence,
    TemporalAttributionCalibration,
)


def _static_inputs() -> dict:
    # The caller-owned nested metadata object is kept as a separate handle so
    # tests can mutate it after result creation; LiftTestEvidence deep-copies
    # and freezes its own view at construction.
    caller_metadata = {"design": {"cells": ["geo-1", "geo-2"]}}
    return {
        "caller_metadata": caller_metadata,
        "evidence": [
            LiftTestEvidence(
                tested_entity="A",
                observed_lift=90.0,
                matched_mta_estimate=100.0,
                test_id="a-static",
                metadata=caller_metadata,
            ),
            LiftTestEvidence(
                tested_entity="B",
                observed_lift=115.0,
                matched_mta_estimate=100.0,
                test_id="b-static",
            ),
        ],
        "estimates": {"A": 400.0, "B": 200.0, "C": 250.0, "D": 150.0},
        "kwargs": {
            "residual_policy": "custom_weights",
            "residual_weights": {"C": 0.7, "D": 0.3},
        },
    }


def _static_result(inputs: dict):
    return AttributionCalibration(evidence=inputs["evidence"]).apply_estimates(
        inputs["estimates"],
        **inputs["kwargs"],
    )


def _interpolated_inputs() -> dict:
    return {
        "evidence": [
            LiftTestEvidence(
                tested_entity="A",
                observed_lift=0.9,
                matched_mta_estimate=1.0,
                test_id="a-early",
                coordinate_name="x",
                coordinate_value=2.0,
                metadata={"window": {"cells": ["north"]}},
            ),
            LiftTestEvidence(
                tested_entity="A",
                observed_lift=1.2,
                matched_mta_estimate=1.0,
                test_id="a-late",
                coordinate_name="x",
                coordinate_value=8.0,
            ),
        ],
        "estimates": {
            "E1": {"A": 100.0, "B": 50.0},
            "E2": {"A": 80.0, "B": 50.0},
            "E3": {"A": 60.0, "B": 40.0},
        },
        "windows": {"E1": (0.0, 5.0), "E2": (5.0, 10.0), "E3": (10.0, 15.0)},
        "coordinates": {"E1": 2.5, "E2": 7.0, "E3": 12.0},
        "eligible": ["B"],
    }


def _interpolated_result(inputs: dict):
    return TemporalAttributionCalibration(inputs["evidence"]).apply_estimates(
        inputs["estimates"],
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="representative_coordinate",
        residual_policy="proportional_untested",
        epoch_windows=inputs["windows"],
        epoch_coordinates=inputs["coordinates"],
        eligible_entities=inputs["eligible"],
    )


def _conflict_inputs() -> dict:
    return {
        "evidence": [
            LiftTestEvidence(
                tested_entity="A",
                observed_lift=0.8,
                matched_mta_estimate=1.0,
                test_id="a-first",
                coordinate_name="x",
                coordinate_value=1.0,
            ),
            LiftTestEvidence(
                tested_entity="A",
                observed_lift=1.1,
                matched_mta_estimate=1.0,
                test_id="a-second",
                coordinate_name="x",
                coordinate_value=3.0,
            ),
        ],
        "estimates": {
            "E1": {"A": 100.0, "B": 60.0},
            "E2": {"A": 90.0, "B": 60.0},
        },
        "windows": {"E1": (0.0, 5.0), "E2": (5.0, 10.0)},
        "conflict": {("A", "E1"): "weighted_ratio"},
        "weights": {("A", "E1"): {"a-first": 0.25, "a-second": 0.75}},
    }


def _conflict_result(inputs: dict):
    return TemporalAttributionCalibration(inputs["evidence"]).apply_estimates(
        inputs["estimates"],
        ratio_policy="observed_only",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        epoch_windows=inputs["windows"],
        evidence_conflict_policy=inputs["conflict"],
        evidence_weights=inputs["weights"],
    )


def _strength_inputs() -> dict:
    return {
        "evidence": [
            LiftTestEvidence(
                tested_entity="A",
                observed_lift=0.8,
                matched_mta_estimate=1.0,
                test_id="a-strength",
            )
        ],
        "estimates": {
            "E1": {"A": 100.0, "B": 40.0},
            "E2": {"A": 100.0, "B": 40.0},
        },
        "source": {"A": "a-strength"},
        "strength": {("A", "E1"): 1.0, ("A", "E2"): 0.25},
    }


def _strength_result(inputs: dict):
    return TemporalAttributionCalibration(inputs["evidence"]).apply_estimates(
        inputs["estimates"],
        ratio_policy="explicit_strength",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        source_evidence=inputs["source"],
        evidence_strength=inputs["strength"],
    )


TEMPORAL_SCENARIOS = (
    ("interpolated", _interpolated_inputs, _interpolated_result),
    ("conflict", _conflict_inputs, _conflict_result),
    ("strength", _strength_inputs, _strength_result),
)


def _line_data(figure_or_axes) -> list:
    axes = (
        figure_or_axes.axes
        if hasattr(figure_or_axes, "axes")
        and not hasattr(figure_or_axes, "get_lines")
        else [figure_or_axes]
    )
    data = []
    for axis in axes:
        for line in axis.get_lines():
            data.append([tuple(point) for point in line.get_xydata()])
    return data


def test_repeated_static_execution_is_identical() -> None:
    first = _static_result(_static_inputs())
    second = _static_result(_static_inputs())
    assert first == second
    assert first.summary() == second.summary()
    assert first.to_dataframe().equals(second.to_dataframe())


@pytest.mark.parametrize(
    ("name", "make_inputs", "make_result"),
    TEMPORAL_SCENARIOS,
    ids=[name for name, _, _ in TEMPORAL_SCENARIOS],
)
def test_repeated_temporal_execution_is_identical(
    name: str, make_inputs, make_result
) -> None:
    first = make_result(make_inputs())
    second = make_result(make_inputs())
    assert first == second
    assert first.summary() == second.summary()
    first_frames = first.to_dataframes()
    second_frames = second.to_dataframes()
    assert set(first_frames) == set(second_frames)
    for key in first_frames:
        assert first_frames[key].equals(second_frames[key])
    assert first.warnings == second.warnings
    if name == "interpolated":
        first_plot = _line_data(first.plot_ratio_schedule())
        second_plot = _line_data(second.plot_ratio_schedule())
        assert first_plot == second_plot
        plt.close("all")


def test_caller_mutation_after_static_result_has_no_effect() -> None:
    inputs = _static_inputs()
    result = _static_result(inputs)
    pristine = _static_result(_static_inputs())

    inputs["caller_metadata"]["design"]["cells"].append("mutated")
    inputs["caller_metadata"]["design"] = None
    inputs["evidence"].clear()
    inputs["estimates"]["A"] = -1.0
    inputs["estimates"]["Z"] = 999.0
    inputs["kwargs"]["residual_weights"]["C"] = 0.0

    assert result == pristine
    assert result.evidence[0].metadata["design"]["cells"] == ("geo-1", "geo-2")
    assert result.to_dataframe().equals(pristine.to_dataframe())


@pytest.mark.parametrize(
    ("name", "make_inputs", "make_result"),
    TEMPORAL_SCENARIOS,
    ids=[name for name, _, _ in TEMPORAL_SCENARIOS],
)
def test_caller_mutation_after_temporal_result_has_no_effect(
    name: str, make_inputs, make_result
) -> None:
    inputs = make_inputs()
    result = make_result(inputs)
    pristine = make_result(make_inputs())

    inputs["evidence"].clear()
    for epoch_estimates in inputs["estimates"].values():
        for entity in list(epoch_estimates):
            epoch_estimates[entity] = -123.0
    inputs["estimates"].clear()
    if "windows" in inputs:
        for epoch in list(inputs["windows"]):
            inputs["windows"][epoch] = (-99.0, 99.0)
    if "coordinates" in inputs:
        inputs["coordinates"].clear()
    if "eligible" in inputs:
        inputs["eligible"].clear()
    if "conflict" in inputs:
        inputs["conflict"].clear()
    if "weights" in inputs:
        for pair in list(inputs["weights"]):
            inputs["weights"][pair]["a-first"] = 1.0
    if "source" in inputs:
        inputs["source"].clear()
    if "strength" in inputs:
        for pair in list(inputs["strength"]):
            inputs["strength"][pair] = 0.0

    assert result == pristine
    frames = result.to_dataframes()
    pristine_frames = pristine.to_dataframes()
    for key in frames:
        assert frames[key].equals(pristine_frames[key])


def test_nested_result_mutation_is_rejected_everywhere() -> None:
    static_result = _static_result(_static_inputs())
    temporal_result = _interpolated_result(_interpolated_inputs())

    with pytest.raises(FrozenInstanceError):
        static_result.calibrated_total = 0.0
    with pytest.raises(FrozenInstanceError):
        static_result.rows[0].calibrated_estimate = 0.0
    with pytest.raises(TypeError):
        static_result.evidence[0].metadata["new"] = "value"
    assert isinstance(static_result.rows, tuple)

    with pytest.raises(FrozenInstanceError):
        temporal_result.ratio_policy = "constant"
    with pytest.raises(FrozenInstanceError):
        temporal_result.epoch_ratio_schedule[0].effective_ratio = 5.0
    with pytest.raises(TypeError):
        temporal_result.rows["E1"] = ()
    with pytest.raises(TypeError):
        temporal_result.residual_provenance["E1"]["status"] = "hacked"
    with pytest.raises(FrozenInstanceError):
        temporal_result.rows["E1"][0].calibrated_estimate = 0.0
    with pytest.raises((TypeError, AttributeError)):
        temporal_result.epochs.append("E9")  # tuple has no append


def test_exported_dataframe_mutation_does_not_leak_back() -> None:
    static_result = _static_result(_static_inputs())
    static_frame = static_result.to_dataframe()
    static_snapshot = static_frame.copy(deep=True)
    static_frame.iloc[0, static_frame.columns.get_loc("calibrated_estimate")] = -1.0
    static_frame["injected"] = "junk"
    fresh_static = static_result.to_dataframe()
    assert fresh_static.equals(static_snapshot)
    assert "injected" not in fresh_static.columns

    temporal_result = _interpolated_result(_interpolated_inputs())
    frames = temporal_result.to_dataframes()
    snapshots = {key: frame.copy(deep=True) for key, frame in frames.items()}
    for frame in frames.values():
        if len(frame):
            frame.iloc[0, 0] = "mutated"
        frame["injected"] = "junk"
    fresh_frames = temporal_result.to_dataframes()
    assert set(fresh_frames) == set(snapshots)
    for key, fresh in fresh_frames.items():
        assert fresh.equals(snapshots[key])
        assert "injected" not in fresh.columns


def test_every_effective_ratio_records_complete_provenance() -> None:
    interpolated = _interpolated_result(_interpolated_inputs())
    conflict = _conflict_result(_conflict_inputs())
    strength = _strength_result(_strength_inputs())

    for result in (interpolated, conflict, strength):
        for row in result.epoch_ratio_schedule:
            assert row.ratio_policy is not None
            assert row.epoch_ratio_policy is not None
            assert row.ratio_source
    interpolated_rows = {
        row.epoch: row for row in interpolated.epoch_ratio_schedule
    }
    assert set(interpolated_rows) == {"E1", "E2"}
    for row in interpolated_rows.values():
        assert row.ratio_source == "interpolated"
        assert row.source_evidence_ids == ("a-early", "a-late")
        assert row.left_source_evidence_id == "a-early"
        assert row.right_source_evidence_id == "a-late"
        assert row.inside_observed_support
    conflict_rows = {
        (row.tested_entity, row.epoch): row
        for row in conflict.epoch_ratio_schedule
    }
    weighted = conflict_rows[("A", "E1")]
    assert weighted.evidence_conflict_policy == "weighted_ratio"
    assert sorted(weighted.source_evidence_ids) == ["a-first", "a-second"]
    assert dict(weighted.evidence_weights) == {"a-first": 0.25, "a-second": 0.75}
    assert weighted.effective_ratio == 0.25 * 0.8 + 0.75 * 1.1
    strength_rows = {row.epoch: row for row in strength.epoch_ratio_schedule}
    assert strength_rows["E1"].source_evidence_ids == ("a-strength",)
    assert strength_rows["E1"].effective_ratio == 0.8
    assert strength_rows["E2"].effective_ratio == 1.0 + 0.25 * (0.8 - 1.0)


def test_every_residual_adjustment_records_policy_and_recipients() -> None:
    static_result = _static_result(_static_inputs())
    for row in static_result.rows:
        if row.residual_adjustment != 0.0:
            assert row.residual_policy == "custom_weights"
            assert row.role == "residual_recipient"
    assert static_result.residual_policy == "custom_weights"

    temporal_result = _interpolated_result(_interpolated_inputs())
    for epoch in ("E1", "E2"):
        provenance = temporal_result.residual_provenance[epoch]
        assert provenance["status"] == "calibrated"
        assert provenance["residual_policy"] == "proportional_untested"
        assert provenance["recipients"] == ("B",)
        adjustments = dict(provenance["adjustments"])
        assert set(adjustments) == {"B"}
        assert adjustments["B"] == pytest.approx(
            provenance["allocated_residual"], abs=1e-9
        )
        rows = {row.entity: row for row in temporal_result.rows[epoch]}
        assert rows["B"].residual_adjustment == pytest.approx(
            adjustments["B"], abs=1e-9
        )
        assert rows["B"].residual_policy == "proportional_untested"
    outside = temporal_result.residual_provenance["E3"]
    assert outside["status"] == "not_calibrated"
    for row in temporal_result.rows["E3"]:
        assert row.residual_policy is None
        assert row.residual_adjustment == 0.0
        assert row.role == "untouched"


def test_diagnostics_do_not_mutate_results_or_become_applied_policy() -> None:
    inputs = _interpolated_inputs()
    result = _interpolated_result(inputs)
    pristine = _interpolated_result(_interpolated_inputs())

    result.summary()
    result.to_dataframes()
    result.plot_epoch_alignment()
    result.plot_ratio_schedule()
    result.plot_calibration_effects()
    result.plot_residual_allocation()
    result.plot_entity_mix()
    plt.close("all")

    assert result == pristine
    assert result.ratio_policy == "coordinate_linear"
    assert result.epoch_ratio_policy == "representative_coordinate"
    assert [row.ratio_source for row in result.epoch_ratio_schedule] == [
        "interpolated",
        "interpolated",
    ]

    strength_result = _strength_result(_strength_inputs())
    pristine_strength = _strength_result(_strength_inputs())
    strength_result.plot_evidence_strength()
    plt.close("all")
    assert strength_result == pristine_strength
    assert strength_result.ratio_policy == "explicit_strength"
