"""Terminology and causal-boundary contract for calibration."""

import inspect

import do_attribution.calibration as calibration


def _raw_module_contract() -> str:
    contract = inspect.getdoc(calibration)
    assert contract is not None
    return contract


def _module_contract() -> str:
    return " ".join(_raw_module_contract().split())


def test_all_normative_terms_are_defined() -> None:
    contract = _raw_module_contract()
    required_terms = (
        "outcome",
        "entity",
        "tested entity",
        "authoritative attribution epoch",
        "evidence point",
        "evidence coordinate",
        "ratio schedule",
        "epoch-ratio policy",
        "effective epoch ratio",
        "evidence-conflict policy",
        "direct test adjustment",
        "residual",
        "residual recipient",
        "residual policy",
    )
    for term in required_terms:
        assert f"``{term}``\n" in contract


def test_evidence_coordinates_are_independent_of_epochs() -> None:
    contract = _module_contract()
    assert "independent of the attribution epoch definition" in contract
    assert "several evidence points may occur inside one epoch" in contract


def test_interpolation_is_missing_evidence_not_an_observation_or_partition() -> None:
    contract = _module_contract()
    assert "Interpolation is an explicit missing-evidence rule" in contract
    assert "neither discovers epochs nor creates experimental observations" in contract


def test_residual_effects_are_not_experiment_identified() -> None:
    contract = _module_contract()
    assert (
        "Changes to untested entities are residual-allocation adjustments, not effects "
        "identified by the experiment"
    ) in contract


def test_library_provides_mechanics_without_selecting_strategy() -> None:
    contract = _module_contract()
    assert "provides transparent calibration mechanics" in contract
    assert "does not select a calibration strategy" in contract


def test_no_entity_owns_or_changes_authoritative_epochs() -> None:
    contract = _module_contract()
    assert "An entity never owns an epoch" in contract
    for forbidden_change in ("creates", "moves", "splits", "merges", "relabels", "reorders"):
        assert forbidden_change in contract


def test_no_legacy_public_product_symbols_are_added() -> None:
    public = {name for name in vars(calibration) if not name.startswith("_")}
    assert public.isdisjoint(
        {
            "Target",
            "eligible_channels",
            "residual_channel",
            "target_epochs",
            "observed_epochs",
            "plot_channel_mix",
        }
    )
