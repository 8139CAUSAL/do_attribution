"""Explicit evidence-combination weighted ratios for CAL-050."""

from math import nan

import pytest

from do_attribution.calibration import (
    LiftTestEvidence,
    _build_observed_only_schedule,
    _resolve_weighted_ratio,
    _validate_evidence_conflict_request,
)


def _evidence(entity, test_id, coordinate, ratio):
    return LiftTestEvidence(
        tested_entity=entity,
        observed_lift=ratio * 100.0,
        matched_mta_estimate=100.0,
        test_id=test_id,
        coordinate_name="index",
        coordinate_value=coordinate,
    )


def _request(ratios=(0.5, 1.5), policies="weighted_ratio"):
    schedule = _build_observed_only_schedule(
        [
            _evidence("A", "a-first", 2.0, ratios[0]),
            _evidence("A", "a-second", 4.0, ratios[1]),
            _evidence("B", "b-first", 12.0, 0.8),
            _evidence("B", "b-second", 14.0, 1.2),
        ],
        epochs=("E1", "E2"),
        epoch_windows={"E1": (0.0, 10.0), "E2": (10.0, 20.0)},
    )
    return _validate_evidence_conflict_request(
        schedule,
        evidence_conflict_policy=policies,
    )


def test_weighted_ratio_identity_and_complete_provenance() -> None:
    resolutions = _resolve_weighted_ratio(
        _request(),
        evidence_weights={
            ("A", "E1"): {"a-second": 0.75, "a-first": 0.25},
            ("B", "E2"): {"b-first": 0.4, "b-second": 0.6},
        },
    )
    a = resolutions[("A", "E1")]
    assert a.effective_ratio == 0.25 * 0.5 + 0.75 * 1.5
    assert a.evidence_conflict_policy == "weighted_ratio"
    assert a.ratio_source == "weighted_from_multiple"
    assert a.source_evidence_ids == ("a-first", "a-second")
    assert a.provenance == {
        "candidate_evidence_ids": ("a-first", "a-second"),
        "candidate_observed_ratios": {"a-first": 0.5, "a-second": 1.5},
        "evidence_combination_weights": {
            "a-first": 0.25,
            "a-second": 0.75,
        },
        "weighted_ratio_terms": {"a-first": 0.125, "a-second": 1.125},
    }
    assert "epoch_coordinate_weights" not in a.provenance
    assert "evidence_strength" not in a.provenance
    assert "residual_weights" not in a.provenance


@pytest.mark.parametrize(
    "ratios, weights",
    [
        ((0.0, 2.0), (0.5, 0.5)),
        ((-1.0, 3.0), (0.25, 0.75)),
        ((1.25, 0.75), (1.0, 0.0)),
        ((100.0, 0.01), (0.001, 0.999)),
    ],
)
def test_weighted_ratio_property_cases_follow_exact_identity(ratios, weights) -> None:
    resolution = _resolve_weighted_ratio(
        _request(ratios=ratios),
        evidence_weights={
            ("A", "E1"): {"a-first": weights[0], "a-second": weights[1]},
            ("B", "E2"): {"b-first": 0.5, "b-second": 0.5},
        },
    )[("A", "E1")]
    assert resolution.effective_ratio == pytest.approx(
        ratios[0] * weights[0] + ratios[1] * weights[1]
    )


@pytest.mark.parametrize(
    "group_weights, expected",
    [
        ({"a-first": 1.0}, "missing"),
        ({"a-first": 0.5, "a-second": 0.4, "unknown": 0.1}, "extra"),
        ({"a-first": -0.1, "a-second": 1.1}, "nonnegative"),
        ({"a-first": 0.4, "a-second": 0.4}, "sum to one"),
        ({"a-first": nan, "a-second": 0.0}, "finite"),
        ({"a-first": True, "a-second": 0.0}, "real number"),
    ],
)
def test_invalid_group_weights_raise(group_weights, expected) -> None:
    with pytest.raises((TypeError, ValueError), match=expected):
        _resolve_weighted_ratio(
            _request(),
            evidence_weights={
                ("A", "E1"): group_weights,
                ("B", "E2"): {"b-first": 0.5, "b-second": 0.5},
            },
        )


@pytest.mark.parametrize(
    "weights, expected",
    [
        (None, "required"),
        ({("A", "E1"): {"a-first": 0.5, "a-second": 0.5}}, "missing"),
        (
            {
                ("A", "E1"): {"a-first": 0.5, "a-second": 0.5},
                ("B", "E2"): {"b-first": 0.5, "b-second": 0.5},
                ("A", "E2"): {"a-first": 0.5, "a-second": 0.5},
            },
            "extra",
        ),
    ],
)
def test_outer_mapping_exactly_covers_weighted_groups(weights, expected) -> None:
    with pytest.raises(ValueError, match=expected):
        _resolve_weighted_ratio(_request(), evidence_weights=weights)


def test_mixed_request_consumes_weights_only_for_weighted_groups() -> None:
    request = _request(
        policies={
            ("A", "E1"): "select_evidence",
            ("B", "E2"): "weighted_ratio",
        }
    )
    resolutions = _resolve_weighted_ratio(
        request,
        evidence_weights={
            ("B", "E2"): {"b-first": 0.5, "b-second": 0.5}
        },
    )
    assert tuple(resolutions) == (("B", "E2"),)
    with pytest.raises(ValueError, match="extra"):
        _resolve_weighted_ratio(
            request,
            evidence_weights={
                ("A", "E1"): {"a-first": 0.5, "a-second": 0.5},
                ("B", "E2"): {"b-first": 0.5, "b-second": 0.5},
            },
        )


def test_caller_tolerance_is_honored_without_renormalizing_weights() -> None:
    resolution = _resolve_weighted_ratio(
        _request(),
        evidence_weights={
            ("A", "E1"): {"a-first": 0.5, "a-second": 0.4999999995},
            ("B", "E2"): {"b-first": 0.5, "b-second": 0.5},
        },
        tolerance=1e-9,
    )[("A", "E1")]
    assert resolution.provenance["evidence_combination_weights"] == {
        "a-first": 0.5,
        "a-second": 0.4999999995,
    }
    with pytest.raises(ValueError, match="sum to one"):
        _resolve_weighted_ratio(
            _request(),
            evidence_weights={
                ("A", "E1"): {"a-first": 0.5, "a-second": 0.4999999995},
                ("B", "E2"): {"b-first": 0.5, "b-second": 0.5},
            },
            tolerance=1e-12,
        )


def test_weighted_resolution_is_deterministic_immutable_and_input_isolated() -> None:
    weights = {
        ("A", "E1"): {"a-first": 0.25, "a-second": 0.75},
        ("B", "E2"): {"b-first": 0.5, "b-second": 0.5},
    }
    first = _resolve_weighted_ratio(_request(), evidence_weights=weights)
    weights[("A", "E1")]["a-first"] = 1.0
    second = _resolve_weighted_ratio(
        _request(),
        evidence_weights={
            ("A", "E1"): {"a-first": 0.25, "a-second": 0.75},
            ("B", "E2"): {"b-first": 0.5, "b-second": 0.5},
        },
    )
    assert first == second
    with pytest.raises(TypeError):
        first[("A", "E1")] = second[("A", "E1")]
    with pytest.raises(TypeError):
        first[("A", "E1")].provenance["evidence_combination_weights"][
            "a-first"
        ] = 1.0
