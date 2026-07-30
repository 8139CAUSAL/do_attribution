"""Static residual-bucket allocation tests for CAL-026."""

import pytest

from do_attribution.calibration import AttributionCalibration, LiftTestEvidence


def _calibrator(observed_lift: float = 360.0) -> AttributionCalibration:
    return AttributionCalibration(
        evidence=[
            LiftTestEvidence(
                tested_entity="A",
                observed_lift=observed_lift,
                matched_mta_estimate=400.0,
                test_id="a-geo",
            )
        ]
    )


def _rows(result):
    return {row.entity: row for row in result.rows}


def test_canonical_new_residual_bucket_receives_the_complete_residual() -> None:
    result = _calibrator().apply_estimates(
        {"A": 400.0, "B": 350.0, "C": 250.0},
        residual_policy="residual_bucket",
        residual_entity="Unattributed / baseline",
    )
    rows = _rows(result)
    bucket = rows["Unattributed / baseline"]

    assert [row.entity for row in result.rows] == [
        "A",
        "B",
        "C",
        "Unattributed / baseline",
    ]
    assert rows["A"].calibrated_estimate == 360.0
    assert rows["A"].residual_adjustment == 0.0
    assert rows["B"].calibrated_estimate == 350.0
    assert rows["C"].calibrated_estimate == 250.0
    assert rows["B"].role == rows["C"].role == "untouched"
    assert bucket.original_estimate == 0.0
    assert bucket.direct_test_adjustment == 0.0
    assert bucket.residual_adjustment == 40.0
    assert bucket.calibrated_estimate == 40.0
    assert bucket.role == "residual_bucket"
    assert bucket.evidence_ids == ()
    assert bucket.residual_policy == "residual_bucket"
    assert result.required_residual == result.allocated_residual == 40.0
    assert result.calibrated_total == 1000.0


def test_existing_untested_bucket_requires_explicit_collision_permission() -> None:
    estimates = {"A": 400.0, "B": 350.0, "baseline": 250.0}
    with pytest.raises(ValueError, match="collides with an existing entity"):
        _calibrator().apply_estimates(
            estimates,
            residual_policy="residual_bucket",
            residual_entity="baseline",
        )

    result = _calibrator().apply_estimates(
        estimates,
        residual_policy="residual_bucket",
        residual_entity="baseline",
        allow_bucket_collision=True,
    )
    rows = _rows(result)

    assert [row.entity for row in result.rows] == ["A", "B", "baseline"]
    assert rows["baseline"].original_estimate == 250.0
    assert rows["baseline"].residual_adjustment == 40.0
    assert rows["baseline"].calibrated_estimate == 290.0
    assert rows["baseline"].role == "residual_bucket"
    assert rows["B"].calibrated_estimate == 350.0


def test_tested_entity_collision_is_always_disallowed() -> None:
    with pytest.raises(ValueError, match="Tested entities cannot receive residual"):
        _calibrator().apply_estimates(
            {"A": 400.0, "B": 600.0},
            residual_policy="residual_bucket",
            residual_entity="A",
            allow_bucket_collision=True,
        )


@pytest.mark.parametrize("bucket", [None, "", "   "])
def test_residual_bucket_requires_a_nonempty_name(bucket) -> None:
    with pytest.raises(ValueError, match="requires a nonempty residual_entity"):
        _calibrator().apply_estimates(
            {"A": 400.0, "B": 600.0},
            residual_policy="residual_bucket",
            residual_entity=bucket,
        )


def test_new_bucket_negative_residual_requires_signed_outputs() -> None:
    with pytest.raises(ValueError, match="cannot be satisfied simultaneously"):
        _calibrator(500.0).apply_estimates(
            {"A": 400.0, "B": 600.0},
            residual_policy="residual_bucket",
            residual_entity="baseline",
        )

    result = _calibrator(500.0).apply_estimates(
        {"A": 400.0, "B": 600.0},
        residual_policy="residual_bucket",
        residual_entity="baseline",
        allow_signed=True,
    )
    assert _rows(result)["baseline"].calibrated_estimate == -100.0
    assert result.calibrated_total == 1000.0
