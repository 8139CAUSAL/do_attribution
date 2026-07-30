"""Static allocation DataFrame export tests for CAL-066."""

from dataclasses import fields

import pandas as pd
import pytest

from do_attribution.calibration import (
    AttributionCalibration,
    CalibrationRow,
    LiftTestEvidence,
)


def _calibrator() -> AttributionCalibration:
    return AttributionCalibration(
        evidence=[
            LiftTestEvidence(
                tested_entity="A",
                observed_lift=360.0,
                matched_mta_estimate=400.0,
                test_id="a-geo",
            )
        ]
    )


POLICY_CASES = [
    ("allow_total_change", {}),
    ("proportional_untested", {"eligible_entities": ["B", "C"]}),
    ("custom_weights", {"residual_weights": {"B": 0.7, "C": 0.3}}),
    ("residual_bucket", {"residual_entity": "baseline"}),
    (
        "explicit_adjustments",
        {"residual_adjustments": {"B": 25.0, "C": 15.0}},
    ),
]


@pytest.mark.parametrize(("policy", "policy_inputs"), POLICY_CASES)
def test_export_is_an_exact_projection_of_every_stored_row_field(
    policy: str,
    policy_inputs: dict[str, object],
) -> None:
    result = _calibrator().apply_estimates(
        {"A": 400.0, "B": 350.0, "C": 250.0},
        residual_policy=policy,
        **policy_inputs,
    )

    frame = result.to_dataframe()

    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == [field.name for field in fields(CalibrationRow)]
    assert frame.index.equals(pd.RangeIndex(len(result.rows)))
    assert frame["entity"].tolist() == [row.entity for row in result.rows]
    for position, row in enumerate(result.rows):
        exported = frame.iloc[position]
        for column in frame.columns:
            assert exported[column] == getattr(row, column)


def test_canonical_export_keeps_direct_and_residual_audit_layers_separate() -> None:
    result = _calibrator().apply_estimates(
        {"A": 400.0, "B": 350.0, "C": 250.0},
        residual_policy="explicit_adjustments",
        residual_adjustments={"B": 25.0, "C": 15.0},
    )

    rows = result.to_dataframe().set_index("entity")

    assert rows.loc["A"].to_dict() == {
        "original_estimate": 400.0,
        "direct_test_adjustment": -40.0,
        "residual_adjustment": 0.0,
        "net_adjustment": -40.0,
        "calibrated_estimate": 360.0,
        "original_share": 0.4,
        "calibrated_share": 0.36,
        "role": "tested",
        "evidence_ids": ("a-geo",),
        "residual_policy": "explicit_adjustments",
    }
    assert rows.loc["B", "direct_test_adjustment"] == 0.0
    assert rows.loc["B", "residual_adjustment"] == 25.0
    assert rows.loc["B", "net_adjustment"] == 25.0
    assert rows.loc["B", "calibrated_estimate"] == 375.0
    assert rows.loc["B", "calibrated_share"] == 0.375
    assert rows.loc["B", "role"] == "residual_recipient"
    assert rows.loc["B", "evidence_ids"] == ()
    assert rows.loc["C", "residual_adjustment"] == 15.0
    assert rows.loc["C", "calibrated_share"] == 0.265


def test_new_residual_bucket_stays_after_the_original_entity_order() -> None:
    result = _calibrator().apply_estimates(
        {"C": 250.0, "A": 400.0, "B": 350.0},
        residual_policy="residual_bucket",
        residual_entity="baseline",
    )

    frame = result.to_dataframe()

    assert frame["entity"].tolist() == ["C", "A", "B", "baseline"]
    assert frame.iloc[-1]["role"] == "residual_bucket"
    assert frame.iloc[-1]["original_estimate"] == 0.0
    assert frame.iloc[-1]["direct_test_adjustment"] == 0.0
    assert frame.iloc[-1]["residual_adjustment"] == 40.0
    assert frame.iloc[-1]["calibrated_estimate"] == 40.0
    assert frame.iloc[-1]["residual_policy"] == "residual_bucket"


class _MutableHashableEntity:
    def __init__(self, label: str) -> None:
        self.label = label

    def __hash__(self) -> int:
        return hash(self.label)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _MutableHashableEntity)
            and self.label == other.label
        )


def test_each_export_is_a_fresh_snapshot_isolated_from_dataframe_mutation() -> None:
    tested = _MutableHashableEntity("tested")
    other = _MutableHashableEntity("other")
    result = AttributionCalibration(
        evidence=[
            LiftTestEvidence(
                tested_entity=tested,
                observed_lift=50.0,
                matched_mta_estimate=100.0,
                test_id="mutable-entity-test",
            )
        ]
    ).apply_estimates(
        [(tested, 100.0), (other, 100.0)],
        residual_policy="allow_total_change",
    )
    expected_rows = result.rows

    first = result.to_dataframe()
    first.at[0, "entity"].label = "mutated"
    first.at[0, "direct_test_adjustment"] = 999.0
    first.at[0, "evidence_ids"] = ("changed",)
    first.drop(index=1, inplace=True)

    second = result.to_dataframe()

    assert result.rows == expected_rows
    assert result.rows[0].entity.label == "tested"
    assert second["entity"].iloc[0].label == "tested"
    assert second["direct_test_adjustment"].tolist() == [-50.0, 0.0]
    assert second["evidence_ids"].tolist() == [
        ("mutable-entity-test",),
        (),
    ]
    assert second.index.equals(pd.RangeIndex(2))
