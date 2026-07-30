"""Public five-policy temporal residual accounting gate for CAL-065."""

import pytest

from do_attribution.calibration import (
    LiftTestEvidence,
    TemporalAttributionCalibration,
)


EPOCHS = ("E1", "E2")
ESTIMATES = {
    "E1": {"A": 100.0, "B": 30.0, "C": 70.0},
    "E2": {"A": 100.0, "B": 80.0, "C": 20.0},
}
RATIOS = {("A", "E1"): 0.8, ("A", "E2"): 0.8}


def _calibrator():
    return TemporalAttributionCalibration(
        [
            LiftTestEvidence(
                tested_entity="A",
                observed_lift=80.0,
                matched_mta_estimate=100.0,
                test_id="a-test",
            )
        ]
    )


def _apply(policy, **controls):
    return _calibrator().apply_estimates(
        ESTIMATES,
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy=policy,
        ratio_schedule=RATIOS,
        **controls,
    )


@pytest.mark.parametrize(
    "policy, controls, expected_e1, expected_e2, totals",
    [
        (
            "allow_total_change",
            {},
            {"A": 80.0, "B": 30.0, "C": 70.0},
            {"A": 80.0, "B": 80.0, "C": 20.0},
            {"E1": 180.0, "E2": 180.0},
        ),
        (
            "proportional_untested",
            {"eligible_entities": ["B", "C"]},
            {"A": 80.0, "B": 36.0, "C": 84.0},
            {"A": 80.0, "B": 96.0, "C": 24.0},
            {"E1": 200.0, "E2": 200.0},
        ),
        (
            "custom_weights",
            {"residual_weights": {"B": 0.25, "C": 0.75}},
            {"A": 80.0, "B": 35.0, "C": 85.0},
            {"A": 80.0, "B": 85.0, "C": 35.0},
            {"E1": 200.0, "E2": 200.0},
        ),
        (
            "residual_bucket",
            {"residual_entity": "baseline"},
            {"A": 80.0, "B": 30.0, "C": 70.0, "baseline": 20.0},
            {"A": 80.0, "B": 80.0, "C": 20.0, "baseline": 20.0},
            {"E1": 200.0, "E2": 200.0},
        ),
        (
            "explicit_adjustments",
            {
                "residual_adjustments_by_epoch": {
                    "E1": {"B": 5.0, "C": 15.0},
                    "E2": {"B": 12.0, "C": 8.0},
                }
            },
            {"A": 80.0, "B": 35.0, "C": 85.0},
            {"A": 80.0, "B": 92.0, "C": 28.0},
            {"E1": 200.0, "E2": 200.0},
        ),
    ],
)
def test_all_five_policies_return_exact_public_rows_and_totals(
    policy, controls, expected_e1, expected_e2, totals
) -> None:
    result = _apply(policy, **controls)

    assert {row.entity: row.calibrated_estimate for row in result.rows["E1"]} == expected_e1
    assert {row.entity: row.calibrated_estimate for row in result.rows["E2"]} == expected_e2
    assert result.calibrated_totals == totals
    assert result.required_residuals == {"E1": 20.0, "E2": 20.0}


def test_tested_values_are_fixed_and_direct_residual_fields_never_blur() -> None:
    result = _apply(
        "proportional_untested",
        eligible_entities=["B", "C"],
    )

    for epoch in EPOCHS:
        tested, b, c = result.rows[epoch]
        assert tested.entity == "A"
        assert tested.role == "tested"
        assert tested.direct_test_adjustment == -20.0
        assert tested.residual_adjustment == 0.0
        assert tested.evidence_ids == ("a-test",)
        for recipient in (b, c):
            assert recipient.role == "residual_recipient"
            assert recipient.direct_test_adjustment == 0.0
            assert recipient.evidence_ids == ()
            assert recipient.residual_adjustment != 0.0


def test_row_arithmetic_totals_and_shares_hold_independently_per_epoch() -> None:
    result = _apply(
        "custom_weights",
        residual_weights={"B": 0.4, "C": 0.6},
    )

    for epoch in EPOCHS:
        rows = result.rows[epoch]
        assert sum(row.original_estimate for row in rows) == result.original_totals[epoch]
        assert sum(row.calibrated_estimate for row in rows) == result.calibrated_totals[epoch]
        assert sum(row.original_share for row in rows) == pytest.approx(1.0)
        assert sum(row.calibrated_share for row in rows) == pytest.approx(1.0)
        for row in rows:
            assert row.net_adjustment == pytest.approx(
                row.direct_test_adjustment + row.residual_adjustment
            )
            assert row.calibrated_estimate == pytest.approx(
                row.original_estimate + row.net_adjustment
            )


@pytest.mark.parametrize("ratio", [0.0, 0.5, 1.0, 1.5])
def test_closed_policy_property_cases_balance_both_residual_directions(ratio) -> None:
    estimates = {"E1": {"A": 100.0, "B": 40.0, "C": 60.0}}
    residual = 100.0 * (1.0 - ratio)
    for policy, controls in (
        ("proportional_untested", {"eligible_entities": ["B", "C"]}),
        ("custom_weights", {"residual_weights": {"B": 0.4, "C": 0.6}}),
        (
            "explicit_adjustments",
            {
                "residual_adjustments_by_epoch": {
                    "E1": {"B": 0.4 * residual, "C": 0.6 * residual}
                }
            },
        ),
    ):
        result = _calibrator().apply_estimates(
            estimates,
            ratio_policy="explicit_ratio",
            epoch_ratio_policy="direct",
            residual_policy=policy,
            ratio_schedule={("A", "E1"): ratio},
            **controls,
        )
        assert result.required_residuals["E1"] == pytest.approx(residual)
        assert result.calibrated_totals["E1"] == pytest.approx(200.0)
        assert result.rows["E1"][0].calibrated_estimate == pytest.approx(
            100.0 * ratio
        )


def test_changing_mix_changes_only_residual_layers_not_ratio_or_epoch_state() -> None:
    calibrator = _calibrator()
    common = {
        "ratio_policy": "explicit_ratio",
        "epoch_ratio_policy": "direct",
        "residual_policy": "proportional_untested",
        "ratio_schedule": RATIOS,
        "eligible_entities": ["B", "C"],
        "epoch_windows": {"E1": (0.0, 1.0), "E2": (1.0, 2.0)},
    }
    first = calibrator.apply_estimates(ESTIMATES, **common)
    second = calibrator.apply_estimates(
        {
            "E1": {"A": 100.0, "B": 50.0, "C": 50.0},
            "E2": {"A": 100.0, "B": 10.0, "C": 90.0},
        },
        **common,
    )

    assert first.epochs == second.epochs
    assert first.epoch_windows == second.epoch_windows
    assert first.epoch_ratio_schedule == second.epoch_ratio_schedule
    assert first.ratio_provenance == second.ratio_provenance
    assert first.epoch_reduction_provenance == second.epoch_reduction_provenance
    assert first.residual_provenance != second.residual_provenance
    assert [row.residual_adjustment for row in first.rows["E1"][1:]] == [6.0, 14.0]
    assert [row.residual_adjustment for row in second.rows["E1"][1:]] == [10.0, 10.0]


def test_public_mixed_policy_mapping_routes_every_epoch_exactly_once() -> None:
    epochs = ("E1", "E2", "E3", "E4", "E5")
    estimates = {
        epoch: {"A": 100.0, "B": 60.0, "C": 40.0}
        for epoch in epochs
    }
    result = _calibrator().apply_estimates(
        estimates,
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy={
            "E1": "allow_total_change",
            "E2": "proportional_untested",
            "E3": "custom_weights",
            "E4": "residual_bucket",
            "E5": "explicit_adjustments",
        },
        ratio_schedule={("A", epoch): 0.8 for epoch in epochs},
        eligible_entities_by_epoch={"E2": ["B", "C"]},
        residual_weights_by_epoch={"E3": {"B": 0.25, "C": 0.75}},
        residual_entity_by_epoch={"E4": "baseline"},
        residual_adjustments_by_epoch={"E5": {"B": 7.0, "C": 13.0}},
    )

    assert tuple(result.residual_policy) == epochs
    assert [result.residual_provenance[epoch]["residual_policy"] for epoch in epochs] == [
        "allow_total_change",
        "proportional_untested",
        "custom_weights",
        "residual_bucket",
        "explicit_adjustments",
    ]
    assert result.calibrated_totals == {
        "E1": 180.0,
        "E2": 200.0,
        "E3": 200.0,
        "E4": 200.0,
        "E5": 200.0,
    }


def test_all_untested_selection_tracks_epoch_specific_entity_entry_and_exit() -> None:
    result = _calibrator().apply_estimates(
        {
            "E1": {"A": 100.0, "B": 25.0, "C": 75.0},
            "E2": {"A": 100.0, "B": 60.0, "D": 40.0},
        },
        ratio_policy="explicit_ratio",
        epoch_ratio_policy="direct",
        residual_policy="proportional_untested",
        ratio_schedule=RATIOS,
        all_untested=True,
    )

    assert [row.entity for row in result.rows["E1"]] == ["A", "B", "C"]
    assert [row.entity for row in result.rows["E2"]] == ["A", "B", "D"]
    assert [row.residual_adjustment for row in result.rows["E1"][1:]] == [5.0, 15.0]
    assert [row.residual_adjustment for row in result.rows["E2"][1:]] == [12.0, 8.0]
