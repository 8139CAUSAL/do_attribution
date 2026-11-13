"""Generated calibration image-bank contract."""

from __future__ import annotations

from hashlib import sha256
import importlib.util
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")
Image = pytest.importorskip("PIL.Image")

from do_attribution.calibration import (  # noqa: E402
    CalibrationResult,
    TemporalCalibrationResult,
)


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "examples" / "generate_calibration_visuals.py"
IMAGE_DIR = ROOT / "examples" / "images"
DIAGNOSTICS = ROOT / "docs" / "calibration_diagnostics.md"
EXPECTED_FILENAMES = (
    "calibration_experimental_evidence.png",
    "calibration_static_effects.png",
    "calibration_temporal_epoch_alignment.png",
    "calibration_temporal_ratio_schedule.png",
    "calibration_temporal_evidence_uncertainty.png",
    "calibration_temporal_effects.png",
    "calibration_temporal_residual_allocation.png",
    "calibration_temporal_entity_mix.png",
)


def _generator_module():
    spec = importlib.util.spec_from_file_location(
        "generate_calibration_visuals",
        GENERATOR_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_checked_in_artifact_bank_has_exact_readable_png_inventory() -> None:
    artifacts = tuple(IMAGE_DIR / name for name in EXPECTED_FILENAMES)
    assert all(path.is_file() and path.stat().st_size > 10_000 for path in artifacts)
    for path in artifacts:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.width >= 900
            assert image.height >= 450


def test_generator_renders_the_evidence_figure_and_seven_public_plot_methods(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bank leads with the experimental evidence, then the package plots.

    ``plot_evidence_strength()`` is deliberately absent: the coordinate-linear
    example has no strength schedule, and its only requestable alternative is
    one interpolated ratio beside a near-identical one.
    """
    generator = _generator_module()
    calls: list[str] = []

    evidence_figure = generator.plot_experimental_evidence

    def evidence_wrapper(*args, **kwargs):
        calls.append("plot_experimental_evidence")
        return evidence_figure(*args, **kwargs)

    monkeypatch.setattr(generator, "plot_experimental_evidence", evidence_wrapper)

    static_method = CalibrationResult.plot_calibration_effects
    temporal_methods = {
        name: getattr(TemporalCalibrationResult, name)
        for name in (
            "plot_epoch_alignment",
            "plot_ratio_schedule",
            "plot_evidence_uncertainty",
            "plot_calibration_effects",
            "plot_residual_allocation",
            "plot_entity_mix",
            "plot_evidence_strength",
        )
    }
    assert set(temporal_methods) == {
        name
        for name in dir(TemporalCalibrationResult)
        if name.startswith("plot_")
    }

    def static_wrapper(self):
        calls.append("CalibrationResult.plot_calibration_effects")
        return static_method(self)

    monkeypatch.setattr(
        CalibrationResult,
        "plot_calibration_effects",
        static_wrapper,
    )

    for name, method in temporal_methods.items():
        def wrapper(self, *args, _method=method, _name=name, **kwargs):
            calls.append(f"TemporalCalibrationResult.{_name}")
            return _method(self, *args, **kwargs)

        monkeypatch.setattr(TemporalCalibrationResult, name, wrapper)

    generated = generator.generate_calibration_visuals(tmp_path)
    assert tuple(path.name for path in generated) == EXPECTED_FILENAMES
    assert calls == [
        "plot_experimental_evidence",
        "CalibrationResult.plot_calibration_effects",
        "TemporalCalibrationResult.plot_epoch_alignment",
        "TemporalCalibrationResult.plot_ratio_schedule",
        "TemporalCalibrationResult.plot_evidence_uncertainty",
        "TemporalCalibrationResult.plot_calibration_effects",
        "TemporalCalibrationResult.plot_residual_allocation",
        "TemporalCalibrationResult.plot_entity_mix",
    ]
    assert plt.get_fignums() == []


def test_generation_is_byte_deterministic_for_identical_inputs(
    tmp_path: Path,
) -> None:
    generator = _generator_module()
    first = generator.generate_calibration_visuals(tmp_path / "first")
    second = generator.generate_calibration_visuals(tmp_path / "second")
    assert [_digest(path) for path in first] == [_digest(path) for path in second]


def test_public_diagnostics_surface_every_stable_artifact_path() -> None:
    text = DIAGNOSTICS.read_text(encoding="utf-8")
    for filename in EXPECTED_FILENAMES:
        reference = f"../examples/images/{filename}"
        assert text.count(reference) == 1
    assert "generated calibration visual artifact bank" in text.lower()
