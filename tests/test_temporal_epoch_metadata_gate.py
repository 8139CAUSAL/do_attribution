"""Cross-cutting authoritative epoch-metadata gate for CAL-039."""

import ast
import inspect
from pathlib import Path

import pytest

import do_attribution.calibration as calibration_module
from do_attribution.calibration import (
    LiftTestEvidence,
    TemporalAttributionCalibration,
    _normalize_authoritative_epoch_windows,
    _normalize_epoch_coordinate_weights,
    _normalize_epoch_coordinates,
    _normalize_epoch_support,
    _normalize_estimates_by_epoch,
)


def _evidence(entity: str, test_id: str, coordinate: float, ratio: float):
    return LiftTestEvidence(
        tested_entity=entity,
        observed_lift=ratio * 100.0,
        matched_mta_estimate=100.0,
        test_id=test_id,
        coordinate_name="time",
        coordinate_value=coordinate,
    )


def _metadata_snapshot(epochs=("E1", "E2")):
    windows = _normalize_authoritative_epoch_windows(
        {"E1": (0.0, 1.0), "E2": (1.0, 2.0)},
        epochs=epochs,
    )
    coordinates = _normalize_epoch_coordinates(
        {"E1": 0.5, "E2": 1.5},
        epochs=epochs,
        epoch_windows=windows,
        required_epochs=epochs,
    )
    support = _normalize_epoch_support(
        {"E1": 10.0, "E2": 20.0},
        epochs=epochs,
    )
    weights = _normalize_epoch_coordinate_weights(
        {"E1": [(0.25, 1.0)], "E2": [(1.75, 2.0)]},
        epochs=epochs,
        epoch_windows=windows,
        required_epochs=epochs,
    )
    return windows, coordinates, support, weights


def test_print_or_social_evidence_cannot_change_epoch_metadata() -> None:
    baseline = _metadata_snapshot()
    print_calibrator = TemporalAttributionCalibration(
        evidence=[_evidence("Print", "print-a", 0.25, 0.8)]
    )
    after_print = _metadata_snapshot()
    social_calibrator = TemporalAttributionCalibration(
        evidence=[
            _evidence("Social", "social-a", 0.75, 1.2),
            _evidence("Social", "social-b", 1.75, 0.9),
        ]
    )
    after_social = _metadata_snapshot()

    assert tuple(item.test_id for item in print_calibrator._evidence) != tuple(
        item.test_id for item in social_calibrator._evidence
    )
    assert baseline == after_print == after_social
    assert all(point.matched_epoch is None for point in print_calibrator._evidence_points)
    assert all(point.matched_epoch is None for point in social_calibrator._evidence_points)


def test_changing_entity_mix_still_uses_one_top_level_epoch_grid() -> None:
    print_heavy = _normalize_estimates_by_epoch(
        {
            "E1": {"Print": 90.0, "Social": 10.0},
            "E2": {"Print": 20.0, "Social": 80.0},
        }
    )
    social_only_later = _normalize_estimates_by_epoch(
        {
            "E1": {"Print": 100.0},
            "E2": {"Social": 100.0},
        }
    )
    print_epochs = tuple(epoch for epoch, _ in print_heavy)
    mixed_epochs = tuple(epoch for epoch, _ in social_only_later)

    assert print_epochs == mixed_epochs == ("E1", "E2")
    for metadata in _metadata_snapshot(print_epochs):
        assert tuple(metadata) == print_epochs
    for helper in (
        _normalize_authoritative_epoch_windows,
        _normalize_epoch_coordinates,
        _normalize_epoch_support,
        _normalize_epoch_coordinate_weights,
    ):
        assert not any(
            "entity" in parameter
            for parameter in inspect.signature(helper).parameters
        )


def test_string_epoch_labels_are_keys_never_coordinates_or_distances() -> None:
    with pytest.raises((TypeError, ValueError), match="epoch_coordinates"):
        _normalize_epoch_coordinates(
            {"E1": "E1"},
            epochs=("E1",),
            epoch_windows={"E1": (0.0, 1.0)},
        )
    with pytest.raises((TypeError, ValueError), match="coordinate"):
        _normalize_epoch_coordinate_weights(
            {"E1": [("E1", 1.0)]},
            epochs=("E1",),
            epoch_windows={"E1": (0.0, 1.0)},
        )


def test_calibration_module_has_no_epoch_discovery_dependency_or_call() -> None:
    source = Path(calibration_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert not imported_modules.intersection(
        {
            "do_attribution.temporal",
            "do_attribution.temporal_diagnostics",
            "temporal",
            "temporal_diagnostics",
        }
    )
    assert called_names.isdisjoint(
        {"recommend_epochs", "suggest_freq", "assign_paths_to_epochs"}
    )
