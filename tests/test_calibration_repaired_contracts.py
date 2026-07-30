"""Focused integration tests for calibration against repaired attribution APIs."""

import pytest

from do_attribution import ShapleyAttribution
from do_attribution.calibration import (
    AttributionCalibration,
    LiftTestEvidence,
    TemporalAttributionCalibration,
)


def _evidence() -> LiftTestEvidence:
    return LiftTestEvidence(
        tested_entity="A",
        observed_lift=1.0,
        matched_mta_estimate=1.0,
        test_id="a-lift",
    )


def test_static_calibration_refuses_unavailable_normalized_attribution() -> None:
    paths = ["A"] * 90 + ["B"] * 10 + ["A > B"]
    conversions = [1] * len(paths)
    with pytest.warns(UserWarning, match="normalization is unavailable"):
        attribution_result = ShapleyAttribution().fit(paths, conversions)

    raw_values = dict(attribution_result["shapley_values"])
    assert attribution_result["attribution"] is None

    with pytest.raises(
        ValueError,
        match=r"attribution is unavailable .* normalized attribution shares",
    ):
        AttributionCalibration([_evidence()]).apply_shares(
            attribution_result["attribution"],
            current_total=101.0,
            residual_policy="allow_total_change",
        )

    with pytest.raises(ValueError, match="must be nonnegative"):
        AttributionCalibration([_evidence()]).apply_shares(
            attribution_result["shapley_values"],
            current_total=101.0,
            residual_policy="allow_total_change",
        )

    assert attribution_result["shapley_values"] == raw_values


def test_temporal_calibration_identifies_epochs_without_attribution() -> None:
    attribution_by_epoch = {
        "E1": {"A": 1.0},
        "E2": None,
    }

    with pytest.raises(
        ValueError,
        match=r"attribution_by_epoch\['E2'\] is unavailable",
    ):
        TemporalAttributionCalibration([_evidence()]).apply_shares(
            attribution_by_epoch,
            totals_by_epoch={"E1": 10.0, "E2": 20.0},
            ratio_policy="constant",
            epoch_ratio_policy="direct",
            residual_policy="allow_total_change",
        )
