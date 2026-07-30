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
    "calibration_static_effects.png",
    "calibration_temporal_epoch_alignment.png",
    "calibration_temporal_ratio_schedule.png",
    "calibration_temporal_effects.png",
    "calibration_temporal_residual_allocation.png",
    "calibration_temporal_entity_mix.png",
    "calibration_temporal_evidence_strength.png",
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


def test_generator_calls_all_seven_public_plot_methods(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _generator_module()
    calls: list[str] = []

    static_method = CalibrationResult.plot_calibration_effects
    temporal_methods = {
        name: getattr(TemporalCalibrationResult, name)
        for name in (
            "plot_epoch_alignment",
            "plot_ratio_schedule",
            "plot_calibration_effects",
            "plot_residual_allocation",
            "plot_entity_mix",
            "plot_evidence_strength",
        )
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
        "CalibrationResult.plot_calibration_effects",
        "TemporalCalibrationResult.plot_epoch_alignment",
        "TemporalCalibrationResult.plot_ratio_schedule",
        "TemporalCalibrationResult.plot_calibration_effects",
        "TemporalCalibrationResult.plot_residual_allocation",
        "TemporalCalibrationResult.plot_entity_mix",
        "TemporalCalibrationResult.plot_evidence_strength",
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
