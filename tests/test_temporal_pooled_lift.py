"""Compatible numerator/denominator pooling for CAL-051."""

import pytest

from do_attribution.calibration import (
    LiftTestEvidence,
    _build_observed_only_schedule,
    _resolve_pooled_lift,
    _validate_evidence_conflict_request,
)


def _evidence(
    entity,
    test_id,
    coordinate,
    observed_lift,
    matched_estimate,
    *,
    outcome_name="sales",
    population="all-customers",
    unit="orders",
):
    return LiftTestEvidence(
        tested_entity=entity,
        observed_lift=observed_lift,
        matched_mta_estimate=matched_estimate,
        test_id=test_id,
        outcome_name=outcome_name,
        population=population,
        unit=unit,
        coordinate_name="index",
        coordinate_value=coordinate,
    )


def _request(
    first=None,
    second=None,
    *,
    policies="pooled_lift",
):
    first = first or _evidence("A", "a-first", 2.0, 90.0, 100.0)
    second = second or _evidence("A", "a-second", 4.0, 100.0, 200.0)
    schedule = _build_observed_only_schedule(
        [
            first,
            second,
            _evidence("B", "b-first", 12.0, 40.0, 100.0),
            _evidence("B", "b-second", 14.0, 180.0, 200.0),
        ],
        epochs=("E1", "E2"),
        epoch_windows={"E1": (0.0, 10.0), "E2": (10.0, 20.0)},
    )
    return _validate_evidence_conflict_request(
        schedule,
        evidence_conflict_policy=policies,
    )


def test_pooled_identity_differs_from_unweighted_ratio_mean() -> None:
    resolution = _resolve_pooled_lift(_request())[("A", "E1")]
    expected = (90.0 + 100.0) / (100.0 + 200.0)
    unweighted_mean = (0.9 + 0.5) / 2.0
    assert resolution.effective_ratio == pytest.approx(expected)
    assert resolution.effective_ratio != pytest.approx(unweighted_mean)
    assert resolution.evidence_conflict_policy == "pooled_lift"
    assert resolution.ratio_source == "pooled_from_multiple"
    assert resolution.source_evidence_ids == ("a-first", "a-second")


def test_pooled_provenance_records_every_component_and_compatibility_field() -> None:
    resolution = _resolve_pooled_lift(_request())[("A", "E1")]
    assert resolution.provenance == {
        "candidate_evidence_ids": ("a-first", "a-second"),
        "candidate_observed_ratios": {"a-first": 0.9, "a-second": 0.5},
        "observed_lifts": {"a-first": 90.0, "a-second": 100.0},
        "matched_mta_estimates": {"a-first": 100.0, "a-second": 200.0},
        "pooled_observed_lift": 190.0,
        "pooled_matched_mta_estimate": 300.0,
        "compatibility": {
            "tested_entity": "A",
            "outcome_name": "sales",
            "population": "all-customers",
            "unit": "orders",
        },
    }


@pytest.mark.parametrize(
    "ratios",
    [
        ((0.0, 10.0), (20.0, 10.0)),
        ((-5.0, 10.0), (15.0, 30.0)),
        ((1e100, 2e100), (3e100, 4e100)),
    ],
)
def test_pooled_property_cases_use_sums_of_components(ratios) -> None:
    (lift_a, matched_a), (lift_b, matched_b) = ratios
    request = _request(
        _evidence("A", "a-first", 2.0, lift_a, matched_a),
        _evidence("A", "a-second", 4.0, lift_b, matched_b),
    )
    resolution = _resolve_pooled_lift(request)[("A", "E1")]
    assert resolution.effective_ratio == pytest.approx(
        (lift_a + lift_b) / (matched_a + matched_b)
    )


@pytest.mark.parametrize(
    "field, replacement",
    [
        ("outcome_name", "revenue"),
        ("population", "new-customers"),
        ("unit", "dollars"),
    ],
)
def test_incompatible_outcome_population_or_unit_raises(field, replacement) -> None:
    values = {
        "outcome_name": "sales",
        "population": "all-customers",
        "unit": "orders",
    }
    values[field] = replacement
    second = _evidence(
        "A",
        "a-second",
        4.0,
        100.0,
        200.0,
        **values,
    )
    with pytest.raises(ValueError, match=field):
        _resolve_pooled_lift(_request(second=second))


def test_explicitly_omitted_compatibility_metadata_is_compatible() -> None:
    request = _request(
        _evidence(
            "A", "a-first", 2.0, 90.0, 100.0,
            outcome_name=None, population=None, unit=None,
        ),
        _evidence(
            "A", "a-second", 4.0, 100.0, 200.0,
            outcome_name=None, population=None, unit=None,
        ),
    )
    assert _resolve_pooled_lift(request)[("A", "E1")].effective_ratio == (
        190.0 / 300.0
    )


def test_zero_pooled_denominator_raises_even_when_components_are_nonzero() -> None:
    request = _request(
        _evidence("A", "a-first", 2.0, 90.0, 100.0),
        _evidence("A", "a-second", 4.0, -50.0, -100.0),
    )
    with pytest.raises(ValueError, match="zero pooled"):
        _resolve_pooled_lift(request)


def test_mixed_policy_request_pools_only_declared_pooled_groups() -> None:
    request = _request(
        policies={
            ("A", "E1"): "select_evidence",
            ("B", "E2"): "pooled_lift",
        }
    )
    resolutions = _resolve_pooled_lift(request)
    assert tuple(resolutions) == (("B", "E2"),)
    assert resolutions[("B", "E2")].effective_ratio == pytest.approx(
        (40.0 + 180.0) / (100.0 + 200.0)
    )


def test_no_pooled_groups_returns_empty_without_altering_request() -> None:
    request = _request(policies="weighted_ratio")
    before = request.evidence_points
    assert _resolve_pooled_lift(request) == {}
    assert request.evidence_points == before


def test_pooled_resolution_is_deterministic_deeply_read_only_and_retains_all() -> None:
    request = _request()
    first = _resolve_pooled_lift(request)
    second = _resolve_pooled_lift(_request())
    assert first == second
    assert [point.evidence_id for point in request.evidence_points] == [
        "a-first",
        "a-second",
        "b-first",
        "b-second",
    ]
    with pytest.raises(TypeError):
        first[("A", "E1")] = second[("A", "E1")]
    with pytest.raises(TypeError):
        first[("A", "E1")].provenance["observed_lifts"]["a-first"] = 0.0
