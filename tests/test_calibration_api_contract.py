"""Exact alias and signature contract tests for CAL-006."""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
API_CONTRACT = ROOT / "tests" / "fixtures" / "calibration_api_contract.py"
MODULE = ROOT / "do_attribution" / "calibration.py"


def read_normalized(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")


def _api_source() -> str:
    return read_normalized(API_CONTRACT)


def _alias_nodes(tree: ast.Module) -> dict[str, ast.AnnAssign]:
    return {
        node.target.id: node
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and isinstance(node.annotation, ast.Name)
        and node.annotation.id == "TypeAlias"
    }


def _method_nodes(tree: ast.Module) -> dict[tuple[str, str], ast.FunctionDef]:
    methods: dict[tuple[str, str], ast.FunctionDef] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for member in node.body:
            if isinstance(member, ast.FunctionDef):
                methods[(node.name, member.name)] = member
    return methods


def _one_evidence():
    from do_attribution.calibration import LiftTestEvidence

    return [
        LiftTestEvidence(
            tested_entity="A",
            observed_lift=1.0,
            matched_mta_estimate=1.0,
            test_id="api-test",
        )
    ]


def test_type_alias_definitions_match_frozen_api_ast_exactly() -> None:
    expected = _alias_nodes(ast.parse(_api_source()))
    actual = _alias_nodes(ast.parse(read_normalized(MODULE)))
    assert actual.keys() == expected.keys()
    for name in expected:
        assert ast.dump(actual[name].annotation) == ast.dump(expected[name].annotation)
        assert ast.dump(actual[name].value) == ast.dump(expected[name].value)


def test_four_method_signatures_match_frozen_api_ast_exactly() -> None:
    expected = _method_nodes(ast.parse(_api_source()))
    actual = _method_nodes(ast.parse(read_normalized(MODULE)))
    assert expected.keys() <= actual.keys()
    calibrator_methods = {
        key
        for key in actual
        if key[0] in {"AttributionCalibration", "TemporalAttributionCalibration"}
    }
    assert calibrator_methods - expected.keys() == {
        ("AttributionCalibration", "__init__"),
        ("TemporalAttributionCalibration", "__init__"),
    }
    for method in expected:
        assert ast.dump(actual[method].args) == ast.dump(expected[method].args)
        assert ast.dump(actual[method].returns) == ast.dump(expected[method].returns)


@pytest.mark.skipif(sys.version_info < (3, 10), reason="package requires Python >=3.10")
def test_required_policies_are_inspectable_keyword_only_without_defaults() -> None:
    from do_attribution.calibration import (
        AttributionCalibration,
        TemporalAttributionCalibration,
    )

    cases = {
        AttributionCalibration.apply_estimates: ("residual_policy",),
        AttributionCalibration.apply_shares: ("residual_policy",),
        TemporalAttributionCalibration.apply_estimates: (
            "ratio_policy",
            "epoch_ratio_policy",
            "residual_policy",
        ),
        TemporalAttributionCalibration.apply_shares: (
            "ratio_policy",
            "epoch_ratio_policy",
            "residual_policy",
        ),
    }
    for method, required in cases.items():
        parameters = inspect.signature(method).parameters
        for name in required:
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
            assert parameters[name].default is inspect.Parameter.empty


@pytest.mark.skipif(sys.version_info < (3, 10), reason="package requires Python >=3.10")
def test_temporal_entry_point_rejects_empty_epoch_input_after_binding() -> None:
    from do_attribution.calibration import TemporalAttributionCalibration

    with pytest.raises(ValueError, match="estimates_by_epoch must not be empty"):
        TemporalAttributionCalibration(evidence=_one_evidence()).apply_estimates(
            {},
            ratio_policy="observed_only",
            epoch_ratio_policy="direct",
            residual_policy="allow_total_change",
        )


def test_no_legacy_or_substitute_parameter_names_entered_signatures() -> None:
    forbidden = {
        "Target",
        "eligible_channels",
        "residual_channel",
        "target_epochs",
        "observed_epochs",
        "normalize",
        "preserve_total",
        "interpolate",
        "transport",
    }
    methods = _method_nodes(ast.parse(read_normalized(MODULE)))
    observed = {
        argument.arg
        for method in methods.values()
        for argument in (
            method.args.posonlyargs
            + method.args.args
            + method.args.kwonlyargs
        )
    }
    assert observed.isdisjoint(forbidden)


@pytest.mark.skipif(sys.version_info < (3, 10), reason="package requires Python >=3.10")
@pytest.mark.parametrize(
    ("method_name", "args", "kwargs", "missing"),
    [
        ("static_estimates", ({"A": 1.0},), {}, ("residual_policy",)),
        (
            "static_shares",
            ({"A": 1.0},),
            {"current_total": 1.0},
            ("residual_policy",),
        ),
        (
            "temporal_estimates",
            ({"E1": {"A": 1.0}},),
            {},
            ("ratio_policy", "epoch_ratio_policy", "residual_policy"),
        ),
        (
            "temporal_estimates",
            ({"E1": {"A": 1.0}},),
            {"residual_policy": "allow_total_change"},
            ("ratio_policy", "epoch_ratio_policy"),
        ),
        (
            "temporal_estimates",
            ({"E1": {"A": 1.0}},),
            {"ratio_policy": "observed_only", "residual_policy": "allow_total_change"},
            ("epoch_ratio_policy",),
        ),
        (
            "temporal_estimates",
            ({"E1": {"A": 1.0}},),
            {"ratio_policy": "observed_only", "epoch_ratio_policy": "direct"},
            ("residual_policy",),
        ),
        (
            "temporal_shares",
            ({"E1": {"A": 1.0}},),
            {"totals_by_epoch": {"E1": 1.0}},
            ("ratio_policy", "epoch_ratio_policy", "residual_policy"),
        ),
    ],
)
def test_policy_omission_raises_native_binding_type_error(
    method_name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    missing: tuple[str, ...],
) -> None:
    from do_attribution.calibration import (
        AttributionCalibration,
        TemporalAttributionCalibration,
    )

    static = AttributionCalibration(evidence=_one_evidence())
    temporal = TemporalAttributionCalibration(evidence=_one_evidence())
    methods = {
        "static_estimates": static.apply_estimates,
        "static_shares": static.apply_shares,
        "temporal_estimates": temporal.apply_estimates,
        "temporal_shares": temporal.apply_shares,
    }
    with pytest.raises(TypeError) as error:
        methods[method_name](*args, **kwargs)
    for parameter in missing:
        assert parameter in str(error.value)
    assert not isinstance(error.value, NotImplementedError)


@pytest.mark.skipif(sys.version_info < (3, 10), reason="package requires Python >=3.10")
@pytest.mark.parametrize(
    "alias", ["normalize", "preserve_total", "interpolate", "transport"]
)
def test_implicit_policy_aliases_are_rejected_during_binding(alias: str) -> None:
    from do_attribution.calibration import AttributionCalibration

    calibrator = AttributionCalibration(evidence=_one_evidence())
    with pytest.raises(TypeError, match=alias):
        calibrator.apply_estimates(
            {"A": 1.0},
            residual_policy="allow_total_change",
            **{alias: True},
        )


@pytest.mark.skipif(sys.version_info < (3, 10), reason="package requires Python >=3.10")
def test_constructors_expose_no_policy_fallbacks() -> None:
    from do_attribution.calibration import (
        AttributionCalibration,
        TemporalAttributionCalibration,
    )

    for calibrator_type in (AttributionCalibration, TemporalAttributionCalibration):
        signature = inspect.signature(calibrator_type)
        assert list(signature.parameters) == ["evidence"]
        assert signature.parameters["evidence"].default is inspect.Parameter.empty
