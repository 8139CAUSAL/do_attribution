"""Evidence-strength and nonapplied diagnostic plotting tests for CAL-078.

``plot_evidence_strength()`` is a calibration-policy comparison.  A strength
schedule exists only under ``explicit_strength`` and nonapplied alternatives
can be requested only under ``observed_only`` or ``coordinate_linear``, so the
figure is one axes holding the view that applies to the result.
"""

import pytest

matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")

from matplotlib.figure import Figure  # noqa: E402

from do_attribution.calibration import (  # noqa: E402
    LiftTestEvidence,
    TemporalAttributionCalibration,
)


def _evidence(
    evidence_id: str,
    ratio: float,
    *,
    coordinate: float | None = None,
):
    return LiftTestEvidence(
        tested_entity="A",
        observed_lift=100.0 * ratio,
        matched_mta_estimate=100.0,
        test_id=evidence_id,
        coordinate_name=None if coordinate is None else "time",
        coordinate_value=coordinate,
    )


def _container_values(ax):
    return {
        container.get_label(): list(container.datavalues)
        for container in ax.containers
    }


def _collection_values(ax):
    return {
        collection.get_label(): collection.get_offsets()[:, 1].tolist()
        for collection in ax.collections
        if not collection.get_label().startswith("_")
    }


def test_explicit_strength_pairs_strength_source_and_effective_ratios() -> None:
    result = TemporalAttributionCalibration([_evidence("a-source", 0.8)]).apply_estimates(
        {
            "E1": {"A": 100.0, "B": 100.0},
            "E2": {"A": 100.0, "B": 100.0},
            "E3": {"A": 100.0, "B": 100.0},
        },
        ratio_policy="explicit_strength",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        source_evidence={"A": "a-source"},
        evidence_strength={
            ("A", "E1"): 0.0,
            ("A", "E2"): 0.5,
            ("A", "E3"): 1.0,
        },
    )
    snapshot = (result.rows, result.epoch_ratio_schedule, result.ratio_provenance)

    figure = result.plot_evidence_strength()
    try:
        assert isinstance(figure, Figure)
        # The strength schedule is the whole figure: nonapplied alternatives
        # cannot be requested for this policy, so no panel is kept for them.
        (strength_ax,) = figure.axes
        assert strength_ax.get_title() == "Explicit Evidence Strength Schedule"
        values = _container_values(strength_ax)
        assert values["Explicit evidence strength"] == [0.0, 0.5, 1.0]
        assert values["Source observed calibration ratio"] == [0.8, 0.8, 0.8]
        assert values["Applied effective ratio"] == pytest.approx([1.0, 0.9, 0.8])
        assert [tick.get_text() for tick in strength_ax.get_xticklabels()] == [
            "A / E1",
            "A / E2",
            "A / E3",
        ]
        assert strength_ax.get_ylabel() == "Strength or ratio"
        assert strength_ax.get_xlabel() == "Tested entity / authoritative epoch"
        assert not strength_ax.texts
        assert not strength_ax.collections
        assert (
            result.rows,
            result.epoch_ratio_schedule,
            result.ratio_provenance,
        ) == snapshot
    finally:
        plt.close(figure)


def test_result_with_no_applicable_view_states_so_without_a_value_scale() -> None:
    """Neither view applies, so the figure says that and plots nothing."""
    result = TemporalAttributionCalibration([_evidence("a-test", 0.8)]).apply_estimates(
        {"E1": {"A": 100.0, "B": 100.0}},
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        ratio_schedule={("A", "E1"): 0.8},
    )

    figure = result.plot_evidence_strength()
    try:
        (ax,) = figure.axes
        assert ax.get_title() == "Evidence Strength Not Applicable"
        assert [text.get_text() for text in ax.texts] == [
            "Evidence strength is not applicable to the applied ratio policy: "
            "explicit_ratio.",
            "No diagnostic alternatives were requested; stored applied "
            "policies and ratios are unchanged.",
        ]
        assert not ax.containers
        assert not ax.collections
        # A statement is not an empty data plot: no scale, label or line.
        assert list(ax.get_xticks()) == []
        assert list(ax.get_yticks()) == []
        assert ax.get_xlabel() == ""
        assert ax.get_ylabel() == ""
        assert not ax.lines
    finally:
        plt.close(figure)


def test_selected_weighted_and_pooled_diagnostics_are_labeled_nonapplied() -> None:
    result = TemporalAttributionCalibration(
        [_evidence("a-first", 0.8, coordinate=2.0), _evidence("a-second", 0.6, coordinate=8.0)]
    ).apply_estimates(
        {"E1": {"A": 100.0, "B": 100.0}},
        ratio_policy="observed_only",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        epoch_windows={"E1": (0.0, 10.0)},
        evidence_conflict_policy="select_evidence",
        selected_evidence={("A", "E1"): "a-second"},
    )
    snapshot = (
        result.rows,
        result.epoch_ratio_schedule,
        result.ratio_provenance,
        result.epoch_reduction_provenance,
    )

    figure = result.plot_evidence_strength(
        diagnostic_conflict_policies={
            ("A", "E1"): (
                "select_evidence",
                "weighted_ratio",
                "pooled_lift",
            )
        },
        diagnostic_selected_evidence={("A", "E1"): "a-first"},
        diagnostic_evidence_weights={
            ("A", "E1"): {"a-first": 0.25, "a-second": 0.75}
        },
    )
    try:
        (diagnostic_ax,) = figure.axes
        values = _collection_values(diagnostic_ax)
        assert values["Stored applied effective ratio"] == [0.6]
        assert values[
            "Diagnostic only — not applied: select_evidence"
        ] == [0.8]
        assert values[
            "Diagnostic only — not applied: weighted_ratio"
        ] == pytest.approx([0.65])
        assert values["Diagnostic only — not applied: pooled_lift"] == [0.7]
        assert (
            result.rows,
            result.epoch_ratio_schedule,
            result.ratio_provenance,
            result.epoch_reduction_provenance,
        ) == snapshot
    finally:
        plt.close(figure)


def test_representative_and_coordinate_average_diagnostics_are_nonapplied() -> None:
    result = TemporalAttributionCalibration(
        [_evidence("a-left", 0.8, coordinate=0.0), _evidence("a-right", 1.2, coordinate=10.0)]
    ).apply_estimates(
        {"E1": {"A": 100.0, "B": 100.0}},
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="support_weighted_average",
        residual_policy="allow_total_change",
        epoch_windows={"E1": (0.0, 10.0)},
        epoch_coordinate_weights={"E1": ((0.0, 1.0), (10.0, 3.0))},
    )
    snapshot = (
        result.rows,
        result.epoch_ratio_schedule,
        result.ratio_provenance,
        result.epoch_reduction_provenance,
    )

    figure = result.plot_evidence_strength(
        diagnostic_epoch_ratio_policies={
            ("A", "E1"): (
                "representative_coordinate",
                "coordinate_average",
            )
        },
        diagnostic_epoch_coordinates={"E1": 2.5},
    )
    try:
        (diagnostic_ax,) = figure.axes
        values = _collection_values(diagnostic_ax)
        assert values["Stored applied effective ratio"] == pytest.approx([1.1])
        assert values[
            "Diagnostic only — not applied: representative_coordinate"
        ] == pytest.approx([0.9])
        assert values[
            "Diagnostic only — not applied: coordinate_average"
        ] == pytest.approx([1.0])
        assert (
            result.rows,
            result.epoch_ratio_schedule,
            result.ratio_provenance,
            result.epoch_reduction_provenance,
        ) == snapshot
    finally:
        plt.close(figure)


def test_material_diagnostic_inputs_are_never_ignored() -> None:
    result = TemporalAttributionCalibration([_evidence("a-test", 0.8)]).apply_estimates(
        {"E1": {"A": 100.0, "B": 100.0}},
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        ratio_schedule={("A", "E1"): 0.8},
    )

    with pytest.raises(ValueError, match="require diagnostic_conflict_policies"):
        result.plot_evidence_strength(
            diagnostic_selected_evidence={("A", "E1"): "a-test"}
        )
    with pytest.raises(ValueError, match="applicable only"):
        result.plot_evidence_strength(
            diagnostic_conflict_policies={("A", "E1"): ("pooled_lift",)}
        )


def _two_epoch_coordinate_linear_result():
    return TemporalAttributionCalibration(
        [_evidence("a-left", 0.8, coordinate=0.0), _evidence("a-right", 1.2, coordinate=10.0)]
    ).apply_estimates(
        {
            "E1": {"A": 100.0, "B": 100.0},
            "E2": {"A": 100.0, "B": 100.0},
        },
        ratio_policy="coordinate_linear",
        epoch_ratio_policy="coordinate_average",
        residual_policy="allow_total_change",
        epoch_windows={"E1": (0.0, 5.0), "E2": (5.0, 10.0)},
    )


def _marker_positions(ax):
    return [
        x
        for collection in ax.collections
        if not collection.get_label().startswith("_")
        for x, _ in collection.get_offsets().tolist()
    ]


def test_one_requested_pair_keeps_its_markers_beside_its_tick() -> None:
    """Autoscaled limits put one pair's markers on opposite panel edges."""
    result = _two_epoch_coordinate_linear_result()

    figure = result.plot_evidence_strength(
        diagnostic_epoch_ratio_policies={
            ("A", "E1"): ("representative_coordinate", "coordinate_average")
        },
        diagnostic_epoch_coordinates={"E1": 1.0},
    )
    try:
        (diagnostic_ax,) = figure.axes
        assert diagnostic_ax.get_xlim() == (-0.5, 0.5)
        assert list(diagnostic_ax.get_xticks()) == [0]
        assert [tick.get_text() for tick in diagnostic_ax.get_xticklabels()] == [
            "A / E1"
        ]
        positions = _marker_positions(diagnostic_ax)
        assert len(positions) == 3
        # Every marker is within a quarter band of the tick it belongs to.
        assert all(abs(position) < 0.25 for position in positions)
    finally:
        plt.close(figure)


def test_each_requested_pair_has_its_own_band() -> None:
    result = _two_epoch_coordinate_linear_result()

    figure = result.plot_evidence_strength(
        diagnostic_epoch_ratio_policies={
            ("A", "E1"): ("representative_coordinate",),
            ("A", "E2"): ("representative_coordinate",),
        },
        diagnostic_epoch_coordinates={"E1": 1.0, "E2": 9.0},
    )
    try:
        (diagnostic_ax,) = figure.axes
        assert diagnostic_ax.get_xlim() == (-0.5, 1.5)
        assert [tick.get_text() for tick in diagnostic_ax.get_xticklabels()] == [
            "A / E1",
            "A / E2",
        ]
        values = _collection_values(diagnostic_ax)
        assert values["Stored applied effective ratio"] == pytest.approx([0.9, 1.1])
        assert values[
            "Diagnostic only — not applied: representative_coordinate"
        ] == pytest.approx([0.84, 1.16])
        positions = sorted(_marker_positions(diagnostic_ax))
        assert [round(position) for position in positions] == [0, 0, 1, 1]
        assert all(
            abs(position - round(position)) < 0.25 for position in positions
        )
    finally:
        plt.close(figure)


def test_requested_alternatives_are_the_whole_figure_for_a_non_strength_result() -> None:
    """No empty evidence-strength panel accompanies the ratio comparison."""
    result = _two_epoch_coordinate_linear_result()

    figure = result.plot_evidence_strength(
        diagnostic_epoch_ratio_policies={("A", "E1"): ("representative_coordinate",)},
        diagnostic_epoch_coordinates={"E1": 1.0},
    )
    try:
        (diagnostic_ax,) = figure.axes
        assert diagnostic_ax.get_title() == (
            "Applied Ratios and Explicitly Requested Diagnostic Alternatives"
        )
        assert diagnostic_ax.get_ylabel() == "Ratio"
        assert diagnostic_ax.get_xlabel() == "Tested entity / authoritative epoch"
        # Nothing in the figure presents the comparison as evidence strength.
        assert not diagnostic_ax.texts
        assert not diagnostic_ax.containers
        assert "strength" not in diagnostic_ax.get_title().lower()
        legend_labels = [
            text.get_text() for text in diagnostic_ax.get_legend().get_texts()
        ]
        assert legend_labels == [
            "Stored applied effective ratio",
            "Diagnostic only — not applied: representative_coordinate",
        ]
    finally:
        plt.close(figure)
