"""Ratio-policy dispatch and compatibility validation for CAL-040."""

from dataclasses import fields

import pytest

from do_attribution.calibration import (
    LiftTestEvidence,
    _validate_ratio_policy_request,
)


def _evidence(*, coordinate: bool = True):
    return [
        LiftTestEvidence(
            tested_entity="A",
            observed_lift=90.0,
            matched_mta_estimate=100.0,
            test_id="a-test",
            coordinate_name="time" if coordinate else None,
            coordinate_value=0.5 if coordinate else None,
        )
    ]


def _request(ratio_policy, epoch_ratio_policy, **overrides):
    values = {
        "evidence": _evidence(),
        "ratio_policy": ratio_policy,
        "epoch_ratio_policy": epoch_ratio_policy,
        "epoch_windows": None,
        "epoch_coordinates": None,
        "epoch_coordinate_weights": None,
        "epoch_support": None,
        "source_evidence": None,
        "tested_entity_epochs": None,
        "ratio_schedule": None,
        "evidence_strength": None,
    }
    values.update(overrides)
    return _validate_ratio_policy_request(**values)


@pytest.mark.parametrize(
    ("ratio_policy", "epoch_ratio_policy", "inputs"),
    [
        ("observed_only", "direct", {"epoch_windows": {"E1": (0.0, 1.0)}}),
        (
            "constant",
            "direct",
            {
                "source_evidence": {"A": "a-test"},
                "tested_entity_epochs": {"A": ["E1"]},
            },
        ),
        (
            "explicit_ratio",
            "direct",
            {"ratio_schedule": {("A", "E1"): 0.9}},
        ),
        (
            "explicit_strength",
            "direct",
            {
                "source_evidence": {"A": "a-test"},
                "evidence_strength": {("A", "E1"): 0.5},
            },
        ),
        (
            "coordinate_linear",
            "representative_coordinate",
            {
                "epoch_windows": {"E1": (0.0, 1.0)},
                "epoch_coordinates": {"E1": 0.5},
            },
        ),
        (
            "coordinate_linear",
            "coordinate_average",
            {"epoch_windows": {"E1": (0.0, 1.0)}},
        ),
        (
            "coordinate_linear",
            "support_weighted_average",
            {
                "epoch_windows": {"E1": (0.0, 1.0)},
                "epoch_coordinate_weights": {"E1": [(0.5, 1.0)]},
            },
        ),
    ],
)
def test_every_supported_policy_combination_is_explicitly_accepted(
    ratio_policy, epoch_ratio_policy, inputs
) -> None:
    request = _request(
        ratio_policy,
        epoch_ratio_policy,
        epoch_support={"E1": 10.0},
        **inputs,
    )
    assert request.ratio_policy == ratio_policy
    assert request.epoch_ratio_policy == epoch_ratio_policy
    assert [field.name for field in fields(request)] == [
        "ratio_policy",
        "epoch_ratio_policy",
    ]


@pytest.mark.parametrize(
    ("ratio_policy", "epoch_ratio_policy"),
    [
        ("observed_only", "representative_coordinate"),
        ("constant", "coordinate_average"),
        ("explicit_ratio", "support_weighted_average"),
        ("explicit_strength", "representative_coordinate"),
        ("coordinate_linear", "direct"),
    ],
)
def test_invalid_direct_vs_coordinate_policy_pairs_raise(
    ratio_policy, epoch_ratio_policy
) -> None:
    with pytest.raises(ValueError, match="requires|incompatible"):
        _request(ratio_policy, epoch_ratio_policy)


def test_unknown_policies_raise_before_schedule_work() -> None:
    with pytest.raises(ValueError, match="ratio_policy must be one of"):
        _request("nearest", "direct")
    with pytest.raises(ValueError, match="epoch_ratio_policy must be one of"):
        _request("explicit_ratio", "midpoint", ratio_schedule={})


@pytest.mark.parametrize(
    ("ratio_policy", "inputs", "missing_name"),
    [
        ("observed_only", {}, "epoch_windows"),
        ("constant", {"tested_entity_epochs": {"A": ["E1"]}}, "source_evidence"),
        ("constant", {"source_evidence": {"A": "a-test"}}, "tested_entity_epochs"),
        ("explicit_ratio", {}, "ratio_schedule"),
        (
            "explicit_strength",
            {"evidence_strength": {("A", "E1"): 0.5}},
            "source_evidence",
        ),
        (
            "explicit_strength",
            {"source_evidence": {"A": "a-test"}},
            "evidence_strength",
        ),
    ],
)
def test_missing_policy_inputs_raise_without_fill(
    ratio_policy, inputs, missing_name
) -> None:
    with pytest.raises(ValueError, match=missing_name):
        _request(ratio_policy, "direct", **inputs)


@pytest.mark.parametrize(
    ("ratio_policy", "valid", "irrelevant"),
    [
        (
            "observed_only",
            {"epoch_windows": {"E1": (0.0, 1.0)}},
            {"tested_entity_epochs": {"A": ["E1"]}},
        ),
        (
            "constant",
            {
                "source_evidence": {"A": "a-test"},
                "tested_entity_epochs": {"A": ["E1"]},
            },
            {"ratio_schedule": {("A", "E1"): 0.9}},
        ),
        (
            "explicit_ratio",
            {"ratio_schedule": {("A", "E1"): 0.9}},
            {"source_evidence": {"A": "a-test"}},
        ),
        (
            "explicit_strength",
            {
                "source_evidence": {"A": "a-test"},
                "evidence_strength": {("A", "E1"): 0.5},
            },
            {"ratio_schedule": {("A", "E1"): 0.9}},
        ),
        (
            "coordinate_linear",
            {"epoch_windows": {"E1": (0.0, 1.0)}},
            {"source_evidence": {"A": "a-test"}},
        ),
    ],
)
def test_policy_specific_inputs_are_rejected_when_irrelevant(
    ratio_policy, valid, irrelevant
) -> None:
    with pytest.raises(ValueError, match="does not accept"):
        _request(ratio_policy, "direct" if ratio_policy != "coordinate_linear" else "coordinate_average", **valid, **irrelevant)


def test_coordinate_evidence_and_reduction_metadata_are_required_explicitly() -> None:
    with pytest.raises(ValueError, match="explicit evidence coordinates"):
        _request(
            "observed_only",
            "direct",
            evidence=_evidence(coordinate=False),
            epoch_windows={"E1": (0.0, 1.0)},
        )
    with pytest.raises(ValueError, match="epoch_coordinates"):
        _request(
            "coordinate_linear",
            "representative_coordinate",
            epoch_windows={"E1": (0.0, 1.0)},
        )
    with pytest.raises(ValueError, match="epoch_coordinate_weights"):
        _request(
            "coordinate_linear",
            "support_weighted_average",
            epoch_windows={"E1": (0.0, 1.0)},
            epoch_support={"E1": 100.0},
        )


@pytest.mark.parametrize(
    ("epoch_ratio_policy", "inputs", "name"),
    [
        ("direct", {"epoch_coordinates": {"E1": 0.5}}, "epoch_coordinates"),
        (
            "coordinate_average",
            {"epoch_coordinate_weights": {"E1": [(0.5, 1.0)]}},
            "epoch_coordinate_weights",
        ),
        (
            "representative_coordinate",
            {
                "epoch_coordinates": {"E1": 0.5},
                "epoch_coordinate_weights": {"E1": [(0.5, 1.0)]},
            },
            "epoch_coordinate_weights",
        ),
        (
            "support_weighted_average",
            {
                "epoch_coordinate_weights": {"E1": [(0.5, 1.0)]},
                "epoch_coordinates": {"E1": 0.5},
            },
            "epoch_coordinates",
        ),
    ],
)
def test_epoch_reduction_metadata_is_not_silently_ignored(
    epoch_ratio_policy, inputs, name
) -> None:
    ratio_policy = "explicit_ratio" if epoch_ratio_policy == "direct" else "coordinate_linear"
    base = (
        {"ratio_schedule": {("A", "E1"): 0.9}}
        if ratio_policy == "explicit_ratio"
        else {"epoch_windows": {"E1": (0.0, 1.0)}}
    )
    with pytest.raises(ValueError, match=name):
        _request(ratio_policy, epoch_ratio_policy, **base, **inputs)
