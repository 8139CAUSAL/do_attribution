"""Final temporal epoch-ratio DataFrame export tests for CAL-070."""

import pandas as pd
import pytest

from do_attribution.calibration import (
    LiftTestEvidence,
    TemporalAttributionCalibration,
)


EPOCH_RATIO_COLUMNS = [
    "tested_entity",
    "epoch",
    "epoch_start",
    "epoch_end",
    "epoch_coordinate",
    "epoch_support",
    "observed_ratio_count",
    "effective_ratio",
    "ratio_policy",
    "epoch_ratio_policy",
    "ratio_source",
    "evidence_conflict_policy",
    "source_evidence_ids",
    "evidence_weights",
    "left_source_evidence_id",
    "right_source_evidence_id",
    "ratio_bound_lower",
    "ratio_bound_upper",
]


def _evidence(
    evidence_id: str,
    ratio: float,
    coordinate: float = 1.0,
    entity: object = "A",
) -> LiftTestEvidence:
    return LiftTestEvidence(
        tested_entity=entity,
        observed_lift=100.0 * ratio,
        matched_mta_estimate=100.0,
        test_id=evidence_id,
        coordinate_name="time",
        coordinate_value=coordinate,
    )


def test_exact_schema_omits_uncalibrated_epochs_and_retains_raise_only_bounds() -> None:
    result = TemporalAttributionCalibration([_evidence("a-source", 0.8)]).apply_estimates(
        {
            "E1": {"A": 100.0, "B": 100.0},
            "E2": {"A": 100.0, "B": 100.0},
        },
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        epoch_windows={"E1": (0.0, 5.0), "E2": (5.0, 10.0)},
        epoch_support={"E1": 10.0, "E2": 20.0},
        ratio_schedule={("A", "E2"): 0.9},
        tested_entity_epochs={"A": ["E2"]},
        ratio_bounds=(0.8, 1.0),
    )

    tables = result.to_dataframes()
    ratios = tables["epoch_ratios"]

    assert list(tables) == [
        "evidence",
        "coordinate_schedule",
        "epoch_ratios",
        "allocations",
        "totals",
        "ratio_provenance",
        "epoch_reduction_provenance",
        "residual_provenance",
        "warnings",
    ]
    assert list(ratios.columns) == EPOCH_RATIO_COLUMNS
    assert ratios.index.equals(pd.RangeIndex(1))
    assert ratios.loc[0].to_dict() == {
        "tested_entity": "A",
        "epoch": "E2",
        "epoch_start": 5.0,
        "epoch_end": 10.0,
        "epoch_coordinate": None,
        "epoch_support": 20.0,
        "observed_ratio_count": 0,
        "effective_ratio": 0.9,
        "ratio_policy": "explicit_ratio",
        "epoch_ratio_policy": "direct",
        "ratio_source": "explicit",
        "evidence_conflict_policy": None,
        "source_evidence_ids": (),
        "evidence_weights": None,
        "left_source_evidence_id": None,
        "right_source_evidence_id": None,
        "ratio_bound_lower": 0.8,
        "ratio_bound_upper": 1.0,
    }
    assert "E1" not in ratios["epoch"].tolist()


def _conflict_result(policy: str):
    evidence = [
        _evidence("a-low", 0.8, 2.0),
        _evidence("a-high", 1.2, 8.0),
    ]
    extra = {}
    if policy == "select_evidence":
        extra["selected_evidence"] = {("A", "E1"): "a-high"}
    elif policy == "weighted_ratio":
        extra["evidence_weights"] = {
            ("A", "E1"): {"a-low": 0.25, "a-high": 0.75}
        }
    return TemporalAttributionCalibration(evidence).apply_estimates(
        {"E1": {"A": 100.0, "B": 100.0}},
        ratio_policy="observed_only",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        epoch_windows={"E1": (0.0, 10.0)},
        evidence_conflict_policy=policy,
        **extra,
    )


@pytest.mark.parametrize(
    ("policy", "expected_ratio", "expected_ids"),
    [
        ("select_evidence", 1.2, ("a-high",)),
        ("weighted_ratio", 1.1, ("a-low", "a-high")),
        ("pooled_lift", 1.0, ("a-low", "a-high")),
    ],
)
def test_conflict_exports_count_every_observation_and_retain_applied_provenance(
    policy: str,
    expected_ratio: float,
    expected_ids: tuple[str, ...],
) -> None:
    row = _conflict_result(policy).to_dataframes()["epoch_ratios"].iloc[0]

    assert row["observed_ratio_count"] == 2
    assert row["effective_ratio"] == pytest.approx(expected_ratio)
    assert row["evidence_conflict_policy"] == policy
    assert row["source_evidence_ids"] == expected_ids
    if policy == "weighted_ratio":
        assert row["evidence_weights"] == {"a-low": 0.25, "a-high": 0.75}
    else:
        assert pd.isna(row["evidence_weights"])


def test_all_direct_ratio_policies_have_policy_specific_observation_counts() -> None:
    evidence = [_evidence("a-source", 0.8)]
    estimates = {"E1": {"A": 100.0, "B": 100.0}}
    observed = TemporalAttributionCalibration(evidence).apply_estimates(
        estimates,
        ratio_policy="observed_only",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        epoch_windows={"E1": (0.0, 2.0)},
    )
    constant = TemporalAttributionCalibration(evidence).apply_estimates(
        estimates,
        ratio_policy="constant",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        source_evidence={"A": "a-source"},
        tested_entity_epochs={"A": ["E1"]},
    )
    explicit = TemporalAttributionCalibration(evidence).apply_estimates(
        estimates,
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        ratio_schedule={("A", "E1"): 0.9},
    )
    strength = TemporalAttributionCalibration(evidence).apply_estimates(
        estimates,
        ratio_policy="explicit_strength",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        source_evidence={"A": "a-source"},
        evidence_strength={("A", "E1"): 0.5},
        tested_entity_epochs={"A": ["E1"]},
    )

    assert [
        result.to_dataframes()["epoch_ratios"].loc[0, "observed_ratio_count"]
        for result in (observed, constant, explicit, strength)
    ] == [1, 1, 0, 1]


def _coordinate_result(epoch_policy: str):
    evidence = [
        _evidence("a-left", 0.8, 0.0),
        _evidence("a-right", 1.2, 10.0),
    ]
    common = {
        "ratio_policy": "coordinate_linear",
        "epoch_ratio_policy": epoch_policy,
        "residual_policy": "allow_total_change",
        "epoch_windows": {"E1": (2.0, 8.0)},
        "epoch_support": {"E1": 40.0},
        "tested_entity_epochs": {"A": ["E1"]},
        "ratio_bounds": (0.5, 1.5),
    }
    if epoch_policy == "representative_coordinate":
        common["epoch_coordinates"] = {"E1": 5.0}
    elif epoch_policy == "support_weighted_average":
        common["epoch_coordinate_weights"] = {
            "E1": [(3.0, 1.0), (7.0, 3.0)]
        }
    return TemporalAttributionCalibration(evidence).apply_estimates(
        {"E1": {"A": 100.0, "B": 100.0}},
        **common,
    )


@pytest.mark.parametrize(
    ("epoch_policy", "ratio_source", "coordinate"),
    [
        ("representative_coordinate", "interpolated", 5.0),
        ("coordinate_average", "coordinate_average", None),
        ("support_weighted_average", "support_weighted_average", None),
    ],
)
def test_coordinate_reduction_rows_export_metadata_sources_anchors_and_bounds(
    epoch_policy: str,
    ratio_source: str,
    coordinate: float | None,
) -> None:
    row = _coordinate_result(epoch_policy).to_dataframes()["epoch_ratios"].iloc[0]

    assert row["epoch_start"] == 2.0
    assert row["epoch_end"] == 8.0
    assert row["epoch_coordinate"] == coordinate or (
        coordinate is None and pd.isna(row["epoch_coordinate"])
    )
    assert row["epoch_support"] == 40.0
    assert row["observed_ratio_count"] == 2
    assert row["effective_ratio"] == pytest.approx(1.0 if epoch_policy != "support_weighted_average" else 1.04)
    assert row["ratio_source"] == ratio_source
    assert row["source_evidence_ids"] == ("a-left", "a-right")
    assert row["left_source_evidence_id"] == "a-left"
    assert row["right_source_evidence_id"] == "a-right"
    assert row["ratio_bound_lower"] == 0.5
    assert row["ratio_bound_upper"] == 1.5


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


def test_epoch_ratio_exports_deep_copy_entities_and_weight_mappings() -> None:
    weighted = _conflict_result("weighted_ratio")
    first = weighted.to_dataframes()["epoch_ratios"]
    first.at[0, "evidence_weights"]["a-low"] = 1.0
    first.at[0, "source_evidence_ids"] = ("changed",)

    second = weighted.to_dataframes()["epoch_ratios"]
    assert second.loc[0, "evidence_weights"] == {"a-low": 0.25, "a-high": 0.75}
    assert second.loc[0, "source_evidence_ids"] == ("a-low", "a-high")

    tested = _MutableHashableEntity("A")
    mutable_result = TemporalAttributionCalibration(
        [_evidence("mutable-a", 0.8, entity=tested)]
    ).apply_estimates(
        {"E1": [(tested, 100.0)]},
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy="allow_total_change",
        ratio_schedule={(tested, "E1"): 0.8},
    )
    exported = mutable_result.to_dataframes()["epoch_ratios"]
    exported.at[0, "tested_entity"].label = "mutated"

    assert mutable_result.epoch_ratio_schedule[0].tested_entity.label == "A"
    assert (
        mutable_result.to_dataframes()["epoch_ratios"].loc[0, "tested_entity"].label
        == "A"
    )
