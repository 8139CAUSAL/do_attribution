"""Package-root export tests for CAL-089.

Verifies that the seven calibration core records/results and both
calibrator classes are importable from the package root, that the
calibration type aliases remain module-level only (no package convention
supports type aliases at the root), that no obsolete global-ratio or
channel-specific name is exported, and that importing the package root
does not change attribution behavior.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import do_attribution
import do_attribution.calibration as calibration


REPO_ROOT = Path(__file__).resolve().parent.parent

# The nine calibration plan objects exported at the package root: the
# seven core records/results plus both calibrator classes.
CALIBRATION_ROOT_EXPORTS = [
    "LiftTestEvidence",
    "EvidencePointRecord",
    "RatioScheduleRow",
    "EpochRatioRow",
    "CalibrationRow",
    "CalibrationResult",
    "TemporalCalibrationResult",
    "AttributionCalibration",
    "TemporalAttributionCalibration",
]

# Calibration type aliases stay importable from the calibration module
# only; no other module exports type aliases at the package root.
CALIBRATION_MODULE_ONLY_ALIASES = [
    "Entity",
    "Epoch",
    "EvidenceId",
    "CoordinateValue",
    "ResidualPolicy",
    "RatioPolicy",
    "EpochRatioPolicy",
    "EvidenceConflictPolicy",
    "EstimateInput",
    "ShareInput",
    "EstimatesByEpochInput",
    "SharesByEpochInput",
    "TotalsByEpochInput",
    "EpochWindowsInput",
    "EpochCoordinatesInput",
    "EpochSupportInput",
    "EpochCoordinateWeightsInput",
    "RatioScheduleInput",
    "EvidenceStrengthInput",
    "SourceEvidenceInput",
    "SelectedEvidenceInput",
    "EvidenceWeightsInput",
]

# Obsolete global-ratio and channel-specific names that must not exist at
# the package root or in the calibration module.  Exact names only:
# legitimate attribution utilities such as ``extract_channels`` contain
# the substring "channel" and remain exported.
OBSOLETE_NAMES = [
    "GlobalRatioCalibration",
    "GlobalCalibration",
    "global_ratio",
    "global_calibration_ratio",
    "apply_global_ratio",
    "ChannelCalibration",
    "ChannelAttributionCalibration",
    "CalibratedChannel",
    "channel_calibration",
    "calibrate_channel",
    "target",
    "target_epochs",
    "eligible_channels",
    "residual_channel",
]


def test_all_nine_calibration_objects_import_from_package_root() -> None:
    for name in CALIBRATION_ROOT_EXPORTS:
        assert hasattr(do_attribution, name)
        assert name in do_attribution.__all__


def test_package_root_objects_are_the_module_objects() -> None:
    for name in CALIBRATION_ROOT_EXPORTS:
        assert getattr(do_attribution, name) is getattr(calibration, name)


def test_all_module_exports_import_from_calibration_module() -> None:
    for name in calibration.__all__:
        assert hasattr(calibration, name)


def test_type_aliases_are_not_exported_at_package_root() -> None:
    for name in CALIBRATION_MODULE_ONLY_ALIASES:
        assert name not in do_attribution.__all__


def test_no_obsolete_global_ratio_or_channel_specific_name_is_exported() -> None:
    for name in OBSOLETE_NAMES:
        assert name not in do_attribution.__all__
        assert name not in calibration.__all__
        assert not hasattr(do_attribution, name)
        assert not hasattr(calibration, name)


def test_package_root_all_is_complete_and_duplicate_free() -> None:
    assert len(do_attribution.__all__) == len(set(do_attribution.__all__))
    for name in do_attribution.__all__:
        assert hasattr(do_attribution, name)


def test_package_import_is_side_effect_free_in_fresh_interpreter() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "import do_attribution"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""


def test_package_import_does_not_change_attribution_behavior() -> None:
    # The package-root classes are the attribution module classes
    # themselves, so the calibration import cannot have wrapped or
    # shadowed them.
    import do_attribution.base
    import do_attribution.compare
    import do_attribution.markov
    import do_attribution.shapley

    assert do_attribution.BaseAttribution is do_attribution.base.BaseAttribution
    assert do_attribution.MarkovAttribution is do_attribution.markov.MarkovAttribution
    assert (
        do_attribution.ShapleyAttribution
        is do_attribution.shapley.ShapleyAttribution
    )
    assert (
        do_attribution.CompareAttribution
        is do_attribution.compare.CompareAttribution
    )

    # Calibration classes do not inherit from BaseAttribution.
    assert not issubclass(
        do_attribution.AttributionCalibration, do_attribution.BaseAttribution
    )
    assert not issubclass(
        do_attribution.TemporalAttributionCalibration,
        do_attribution.BaseAttribution,
    )

    # A deterministic fit through the package root returns a result and
    # leaves the caller's inputs unchanged.
    paths = ["A > B > C", "A > C", "B > C"]
    conversions = [1, 0, 1]
    result = do_attribution.MarkovAttribution(order=1).fit(paths, conversions)
    assert set(result["attribution"]) == {"A", "B", "C"}
    assert paths == ["A > B > C", "A > C", "B > C"]
    assert conversions == [1, 0, 1]
