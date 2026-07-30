"""Temporal residual-policy normalization and dispatch tests for CAL-059."""

from dataclasses import FrozenInstanceError
import inspect

import pytest

from do_attribution import calibration


EPOCHS = ("E1", "E2", "E3")
CALIBRATED = ("E3", "E1")


def _request(**overrides):
    values = {
        "epochs": EPOCHS,
        "calibrated_epochs": CALIBRATED,
        "residual_policy": "allow_total_change",
    }
    values.update(overrides)
    return calibration._normalize_temporal_residual_request(**values)


def test_scalar_policy_applies_only_to_calibrated_epochs_in_authoritative_order() -> None:
    request = _request()

    assert request.epochs == EPOCHS
    assert request.calibrated_epochs == ("E1", "E3")
    assert request.uncalibrated_epochs == ("E2",)
    assert tuple(request.policies_by_epoch) == ("E1", "E3")
    assert tuple(request.requests_by_epoch) == ("E1", "E3")
    assert all(
        item.policy == "allow_total_change"
        for item in request.requests_by_epoch.values()
    )
    assert "E2" not in request.requests_by_epoch


def test_mixed_policy_mapping_resolves_each_control_only_in_its_epoch() -> None:
    request = _request(
        calibrated_epochs=EPOCHS,
        residual_policy={
            "E1": "proportional_untested",
            "E2": "custom_weights",
            "E3": "residual_bucket",
        },
        eligible_entities_by_epoch={"E1": ["B", "C"]},
        residual_weights_by_epoch={"E2": {"B": 0.25, "C": 0.75}},
        residual_entity_by_epoch={"E3": "baseline"},
        allow_bucket_collision=True,
    )

    proportional = request.requests_by_epoch["E1"]
    custom = request.requests_by_epoch["E2"]
    bucket = request.requests_by_epoch["E3"]
    assert proportional.eligible_entities == ("B", "C")
    assert proportional.residual_weights is None
    assert custom.residual_weights == {"B": 0.25, "C": 0.75}
    assert custom.eligible_entities is None
    assert bucket.residual_entity == "baseline"
    assert bucket.allow_bucket_collision is True
    assert not proportional.allow_bucket_collision
    assert not custom.allow_bucket_collision


@pytest.mark.parametrize(
    "policy, expected",
    [
        ({"E1": "allow_total_change"}, "missing"),
        (
            {
                "E1": "allow_total_change",
                "E2": "allow_total_change",
                "E3": "allow_total_change",
            },
            "extra",
        ),
        (
            {"E1": "allow_total_change", "E3": "preserve_total"},
            r"residual_policy\['E3'\]",
        ),
    ],
)
def test_policy_mapping_requires_exact_calibrated_coverage_and_known_values(
    policy, expected
) -> None:
    with pytest.raises(ValueError, match=expected):
        _request(residual_policy=policy)


@pytest.mark.parametrize(
    "global_name, global_value, by_name, by_value",
    [
        (
            "eligible_entities",
            ["B"],
            "eligible_entities_by_epoch",
            {"E1": ["B"], "E3": ["B"]},
        ),
        (
            "residual_weights",
            {"B": 1.0},
            "residual_weights_by_epoch",
            {"E1": {"B": 1.0}, "E3": {"B": 1.0}},
        ),
        (
            "residual_entity",
            "baseline",
            "residual_entity_by_epoch",
            {"E1": "base-1", "E3": "base-3"},
        ),
    ],
)
def test_global_and_per_epoch_counterparts_are_always_ambiguous(
    global_name, global_value, by_name, by_value
) -> None:
    with pytest.raises(ValueError, match="cannot both be supplied"):
        _request(**{global_name: global_value, by_name: by_value})


@pytest.mark.parametrize(
    "overrides, control",
    [
        (
            {
                "residual_policy": {
                    "E1": "proportional_untested",
                    "E3": "allow_total_change",
                },
                "eligible_entities": ["B"],
            },
            "eligible_entities",
        ),
        (
            {
                "residual_policy": {
                    "E1": "custom_weights",
                    "E3": "allow_total_change",
                },
                "residual_weights": {"B": 1.0},
            },
            "residual_weights",
        ),
        (
            {
                "residual_policy": {
                    "E1": "residual_bucket",
                    "E3": "allow_total_change",
                },
                "residual_entity": "baseline",
            },
            "residual_entity",
        ),
    ],
)
def test_global_policy_dependent_values_raise_for_mixed_policy_epochs(
    overrides, control
) -> None:
    with pytest.raises(ValueError, match=f"Global {control}"):
        _request(**overrides)


@pytest.mark.parametrize(
    "overrides, control",
    [
        (
            {
                "residual_policy": "proportional_untested",
                "eligible_entities_by_epoch": {"E1": ["B"]},
            },
            "eligible_entities_by_epoch",
        ),
        (
            {
                "residual_policy": "custom_weights",
                "residual_weights_by_epoch": {
                    "E1": {"B": 1.0},
                    "E3": {"B": 1.0},
                    "E2": {"B": 1.0},
                },
            },
            "residual_weights_by_epoch",
        ),
        (
            {
                "residual_policy": "residual_bucket",
                "residual_entity_by_epoch": {
                    "E1": "base-1",
                    "E3": "base-3",
                    "unknown": "base-x",
                },
            },
            "residual_entity_by_epoch",
        ),
        (
            {
                "residual_policy": "explicit_adjustments",
                "residual_adjustments_by_epoch": {
                    "E1": {"B": 1.0},
                },
            },
            "residual_adjustments_by_epoch",
        ),
    ],
)
def test_per_epoch_controls_exactly_cover_only_their_policy_epochs(
    overrides, control
) -> None:
    with pytest.raises(ValueError, match=control):
        _request(**overrides)


def test_all_untested_is_explicit_and_applies_only_to_proportional_epochs() -> None:
    request = _request(
        residual_policy={
            "E1": "proportional_untested",
            "E3": "allow_total_change",
        },
        all_untested=True,
    )
    assert request.requests_by_epoch["E1"].all_untested is True
    assert request.requests_by_epoch["E3"].all_untested is None

    with pytest.raises(ValueError, match="must be True"):
        _request(
            residual_policy="proportional_untested",
            all_untested=False,
        )
    with pytest.raises(ValueError, match="exactly one"):
        _request(
            residual_policy="proportional_untested",
            all_untested=True,
            eligible_entities_by_epoch={"E1": ["B"], "E3": ["B"]},
        )


def test_global_controls_are_valid_for_one_uniform_relevant_policy() -> None:
    proportional = _request(
        residual_policy="proportional_untested",
        eligible_entities=["B", "C"],
    )
    custom = _request(
        residual_policy="custom_weights",
        residual_weights={"B": 0.4, "C": 0.6},
    )
    bucket = _request(
        residual_policy="residual_bucket",
        residual_entity="baseline",
    )

    assert all(
        item.eligible_entities == ("B", "C")
        for item in proportional.requests_by_epoch.values()
    )
    assert all(
        item.residual_weights == {"B": 0.4, "C": 0.6}
        for item in custom.requests_by_epoch.values()
    )
    assert all(
        item.residual_entity == "baseline"
        for item in bucket.requests_by_epoch.values()
    )


def test_nested_inputs_are_copied_and_every_normalized_layer_is_read_only() -> None:
    policies = {"E1": "custom_weights", "E3": "custom_weights"}
    weights = {"E1": {"B": 1.0}, "E3": {"C": 1.0}}
    request = _request(
        residual_policy=policies,
        residual_weights_by_epoch=weights,
    )
    policies["E1"] = "allow_total_change"
    weights["E1"]["B"] = 0.0

    assert request.policies_by_epoch["E1"] == "custom_weights"
    assert request.requests_by_epoch["E1"].residual_weights["B"] == 1.0
    with pytest.raises(TypeError):
        request.requests_by_epoch["E1"] = request.requests_by_epoch["E3"]
    with pytest.raises(TypeError):
        request.requests_by_epoch["E1"].residual_weights["B"] = 0.0
    with pytest.raises(FrozenInstanceError):
        request.calibrated_epochs = ()


def test_empty_calibrated_epoch_set_needs_no_residual_arguments() -> None:
    request = _request(
        calibrated_epochs=(),
        residual_policy={},
    )
    assert request.calibrated_epochs == ()
    assert request.uncalibrated_epochs == EPOCHS
    assert request.requests_by_epoch == {}


def test_residual_normalization_has_no_ratio_policy_inputs_or_state() -> None:
    parameters = inspect.signature(
        calibration._normalize_temporal_residual_request
    ).parameters
    assert "ratio_policy" not in parameters
    assert "epoch_ratio_policy" not in parameters
    assert "ratio_schedule" not in parameters
    assert not any("ratio" in field for field in vars(_request()))


@pytest.mark.parametrize(
    "calibrated_epochs, expected",
    [
        (("E1", "E1"), "duplicate"),
        (("E4",), "unknown"),
        ("E1", "sequence"),
    ],
)
def test_calibrated_epoch_subset_is_unique_and_authoritative(
    calibrated_epochs, expected
) -> None:
    with pytest.raises((TypeError, ValueError), match=expected):
        _request(calibrated_epochs=calibrated_epochs)
