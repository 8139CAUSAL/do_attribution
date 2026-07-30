"""Contract checks for the CAL-005 user-facing policy matrix."""

from pathlib import Path


CONTRACT = (
    Path(__file__).resolve().parents[1] / "docs" / "calibration.md"
).read_text(encoding="utf-8")
NORMALIZED_CONTRACT = " ".join(CONTRACT.split())


def test_all_supported_policies_appear_once_in_their_matrix_rows() -> None:
    policy_groups = {
        "residual": (
            "allow_total_change",
            "proportional_untested",
            "custom_weights",
            "residual_bucket",
            "explicit_adjustments",
        ),
        "ratio": (
            "observed_only",
            "constant",
            "explicit_ratio",
            "explicit_strength",
            "coordinate_linear",
        ),
        "epoch": (
            "direct",
            "representative_coordinate",
            "coordinate_average",
            "support_weighted_average",
        ),
        "conflict": ("select_evidence", "weighted_ratio", "pooled_lift"),
    }
    for policies in policy_groups.values():
        for policy in policies:
            assert CONTRACT.count(f"| `{policy}` |") == 1


def test_every_policy_matrix_declares_requirements_and_compatibility() -> None:
    assert "Required policy inputs" in CONTRACT
    assert "Coverage and compatible epoch-ratio policy" in CONTRACT
    assert "Required inputs and support" in CONTRACT
    assert "Provenance retained" in CONTRACT
    assert "Inputs not listed as required or permitted" in CONTRACT


def test_constructor_and_mandatory_policy_contracts_are_explicit() -> None:
    assert "AttributionCalibration(evidence=evidence)" in NORMALIZED_CONTRACT
    assert "TemporalAttributionCalibration(evidence=evidence)" in NORMALIZED_CONTRACT
    assert "exactly one evidence record per tested entity" in NORMALIZED_CONTRACT
    assert "keyword-only and have no defaults" in NORMALIZED_CONTRACT
    for alias in ("normalize", "preserve_total", "interpolate", "transport"):
        assert f"`{alias}`" in NORMALIZED_CONTRACT


def test_irrelevant_conflicting_and_global_per_epoch_inputs_raise() -> None:
    assert "raise rather than being ignored" in NORMALIZED_CONTRACT
    assert "may not both be supplied" in NORMALIZED_CONTRACT
    assert "Irrelevant or extra conflict mappings raise" in NORMALIZED_CONTRACT
    assert "Missing and extra rows raise" in NORMALIZED_CONTRACT


def test_uncovered_epochs_are_explicitly_unchanged() -> None:
    assert "reported as uncalibrated" in NORMALIZED_CONTRACT
    assert "retain their complete original allocation" in NORMALIZED_CONTRACT
    assert "remain wholly unchanged" in NORMALIZED_CONTRACT


def test_result_audit_surface_contains_all_required_layers() -> None:
    for phrase in (
        "original attribution",
        "evidence-backed direct adjustments",
        "residual adjustments",
        "calibrated attribution",
        "final effective ratios",
        "complete provenance",
        "original/raw/required/allocated/calibrated totals",
        "warnings",
    ):
        assert phrase in NORMALIZED_CONTRACT


def test_error_boundary_and_no_hidden_inference_are_explicit() -> None:
    assert "native binding-time `TypeError`" in NORMALIZED_CONTRACT
    assert "actionable `ValueError`" in NORMALIZED_CONTRACT
    for forbidden in (
        "fills missing values",
        "selects evidence",
        "chooses weights",
        "clips ratios",
        "extrapolates",
        "renormalizes tested values",
        "infers an epoch coordinate",
    ):
        assert forbidden in NORMALIZED_CONTRACT


def test_determinism_immutability_and_performance_are_explicit() -> None:
    assert "Caller inputs and fitted attribution objects are never mutated" in NORMALIZED_CONTRACT
    assert "produce identical rows, exports, summaries" in NORMALIZED_CONTRACT
    assert "not expanded into a dense grid" in NORMALIZED_CONTRACT
    assert "Plots and exports use stored result state" in NORMALIZED_CONTRACT
