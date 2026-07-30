"""Explicit static residual-policy dispatch tests for CAL-022."""

import inspect

import pytest

import do_attribution.calibration as calibration
from do_attribution.calibration import AttributionCalibration, LiftTestEvidence


def _calibrator() -> AttributionCalibration:
    return AttributionCalibration(
        evidence=[
            LiftTestEvidence(
                tested_entity="A",
                observed_lift=90.0,
                matched_mta_estimate=100.0,
                test_id="a-test",
            )
        ]
    )


def _request(policy: str, **overrides: object):
    values = {
        "residual_policy": policy,
        "tested_entities": ("A",),
        "eligible_entities": None,
        "all_untested": None,
        "residual_weights": None,
        "residual_entity": None,
        "residual_adjustments": None,
        "allow_bucket_collision": False,
    }
    values.update(overrides)
    return calibration._static_residual_request(**values)


def _state():
    calibrator = _calibrator()
    return calibration._static_direct_state(
        {"A": 100.0, "B": 50.0},
        evidence_by_entity=calibrator._evidence_by_entity,
        allow_signed=False,
        tolerance=1e-9,
    )


@pytest.mark.parametrize(
    ("policy", "handler_name", "policy_inputs"),
    [
        ("allow_total_change", "_allocate_allow_total_change", {}),
        (
            "proportional_untested",
            "_allocate_proportional_untested",
            {"eligible_entities": ["B"]},
        ),
        (
            "custom_weights",
            "_allocate_custom_weights",
            {"residual_weights": {"B": 1.0}},
        ),
        (
            "residual_bucket",
            "_allocate_residual_bucket",
            {"residual_entity": "baseline"},
        ),
        (
            "explicit_adjustments",
            "_allocate_explicit_adjustments",
            {"residual_adjustments": {"B": 10.0}},
        ),
    ],
)
def test_each_exact_policy_routes_to_only_its_declared_handler(
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
    handler_name: str,
    policy_inputs: dict[str, object],
) -> None:
    sentinel = object()
    called: list[tuple[object, object, bool, float]] = []

    def selected_handler(state, request, *, allow_signed, tolerance):
        called.append((state, request, allow_signed, tolerance))
        return sentinel

    def wrong_handler(*args, **kwargs):
        raise AssertionError("dispatcher called an unselected residual policy")

    for name in (
        "_allocate_allow_total_change",
        "_allocate_proportional_untested",
        "_allocate_custom_weights",
        "_allocate_residual_bucket",
        "_allocate_explicit_adjustments",
    ):
        monkeypatch.setattr(
            calibration,
            name,
            selected_handler if name == handler_name else wrong_handler,
        )

    state = _state()
    request = _request(policy, **policy_inputs)
    result = calibration._dispatch_static_residual(
        state, request, allow_signed=False, tolerance=1e-9
    )

    assert result is sentinel
    assert called == [(state, request, False, 1e-9)]


def test_unknown_policy_raises_and_static_public_policy_has_no_default() -> None:
    with pytest.raises(ValueError, match="residual_policy must be one of"):
        _request("renormalize")

    parameter = inspect.signature(
        AttributionCalibration.apply_estimates
    ).parameters["residual_policy"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


@pytest.mark.parametrize(
    "overrides",
    [
        {"eligible_entities": ["B"]},
        {"all_untested": True},
        {"residual_weights": {"B": 1.0}},
        {"residual_entity": "baseline"},
        {"residual_adjustments": {"B": 10.0}},
        {"allow_bucket_collision": True},
    ],
)
def test_allow_total_change_rejects_every_residual_allocation_input(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="does not accept residual-allocation inputs"):
        _request("allow_total_change", **overrides)


def test_policy_dependent_required_and_irrelevant_inputs_are_not_ignored() -> None:
    with pytest.raises(ValueError, match="requires exactly one"):
        _request("proportional_untested")
    with pytest.raises(ValueError, match="requires exactly one"):
        _request(
            "proportional_untested",
            eligible_entities=["B"],
            all_untested=True,
        )
    with pytest.raises(ValueError, match="requires residual_weights"):
        _request("custom_weights")
    with pytest.raises(ValueError, match="requires a nonempty residual_entity"):
        _request("residual_bucket")
    with pytest.raises(ValueError, match="requires residual_adjustments"):
        _request("explicit_adjustments")
    with pytest.raises(ValueError, match="accepts only residual_weights"):
        _request(
            "custom_weights",
            residual_weights={"B": 1.0},
            eligible_entities=["B"],
        )


@pytest.mark.parametrize(
    ("policy", "policy_inputs"),
    [
        ("proportional_untested", {"eligible_entities": ["A"]}),
        ("custom_weights", {"residual_weights": {"A": 1.0}}),
        ("residual_bucket", {"residual_entity": "A"}),
        ("explicit_adjustments", {"residual_adjustments": {"A": 10.0}}),
    ],
)
def test_tested_entities_are_rejected_as_residual_recipients(
    policy: str, policy_inputs: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match="Tested entities cannot receive residual"):
        _request(policy, **policy_inputs)
