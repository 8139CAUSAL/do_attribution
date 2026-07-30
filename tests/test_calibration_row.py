"""Immutable direct-versus-residual entity-row tests for CAL-016."""

from dataclasses import FrozenInstanceError, fields

import pytest

from do_attribution.calibration import CalibrationRow


EXPECTED_FIELDS = [
    "entity",
    "original_estimate",
    "direct_test_adjustment",
    "residual_adjustment",
    "net_adjustment",
    "calibrated_estimate",
    "original_share",
    "calibrated_share",
    "role",
    "evidence_ids",
    "residual_policy",
]


def _row(**overrides) -> CalibrationRow:
    values = {
        "entity": "A",
        "original_estimate": 100.0,
        "direct_test_adjustment": 0.0,
        "residual_adjustment": 0.0,
        "net_adjustment": 0.0,
        "calibrated_estimate": 100.0,
        "original_share": 0.1,
        "calibrated_share": 0.1,
        "role": "untouched",
        "evidence_ids": (),
        "residual_policy": "allow_total_change",
    }
    values.update(overrides)
    return CalibrationRow(**values)


@pytest.mark.parametrize(
    ("role", "overrides"),
    [
        (
            "tested",
            {
                "direct_test_adjustment": -10.0,
                "net_adjustment": -10.0,
                "calibrated_estimate": 90.0,
                "evidence_ids": ("a-test",),
            },
        ),
        (
            "residual_recipient",
            {
                "residual_adjustment": 10.0,
                "net_adjustment": 10.0,
                "calibrated_estimate": 110.0,
                "residual_policy": "proportional_untested",
            },
        ),
        ("untouched", {}),
        (
            "residual_bucket",
            {
                "original_estimate": 0.0,
                "residual_adjustment": 10.0,
                "net_adjustment": 10.0,
                "calibrated_estimate": 10.0,
                "residual_policy": "residual_bucket",
            },
        ),
    ],
)
def test_allowed_roles_match_direct_and_residual_provenance(
    role: str, overrides: dict
) -> None:
    row = _row(role=role, **overrides)
    assert [item.name for item in fields(row)] == EXPECTED_FIELDS
    assert row.role == role
    assert row.net_adjustment == (
        row.direct_test_adjustment + row.residual_adjustment
    )
    assert row.calibrated_estimate == row.original_estimate + row.net_adjustment


def test_reserved_tested_recipient_role_is_disallowed() -> None:
    with pytest.raises(ValueError, match="reserved and disallowed"):
        _row(role="tested_and_residual_recipient")


def test_tested_entity_cannot_receive_residual_allocation() -> None:
    with pytest.raises(ValueError, match="cannot receive residual"):
        _row(
            role="tested",
            direct_test_adjustment=-10.0,
            residual_adjustment=5.0,
            net_adjustment=-5.0,
            calibrated_estimate=95.0,
            evidence_ids=("a-test",),
            residual_policy="custom_weights",
        )


def test_arithmetic_identities_are_enforced_without_recalculation() -> None:
    with pytest.raises(ValueError, match="net_adjustment must equal"):
        _row(net_adjustment=1.0, calibrated_estimate=101.0)
    with pytest.raises(ValueError, match="calibrated_estimate must equal"):
        _row(calibrated_estimate=99.0)


def test_roles_require_consistent_evidence_and_policy_state() -> None:
    with pytest.raises(ValueError, match="tested rows require"):
        _row(role="tested")
    with pytest.raises(ValueError, match="only tested rows may contain evidence"):
        _row(evidence_ids=("unexpected",))
    with pytest.raises(ValueError, match="recipient allocation policy"):
        _row(role="residual_recipient")
    with pytest.raises(ValueError, match="requires residual_bucket policy"):
        _row(role="residual_bucket")
    with pytest.raises(ValueError, match="allow_total_change"):
        _row(
            role="untouched",
            residual_adjustment=1.0,
            net_adjustment=1.0,
            calibrated_estimate=101.0,
        )


def test_nested_evidence_and_policy_state_are_immutable() -> None:
    evidence_ids = ["a-test"]
    row = _row(
        role="tested",
        evidence_ids=evidence_ids,
        direct_test_adjustment=-10.0,
        net_adjustment=-10.0,
        calibrated_estimate=90.0,
    )
    evidence_ids.append("mutated")
    assert row.evidence_ids == ("a-test",)
    assert row.residual_policy == "allow_total_change"
    with pytest.raises(FrozenInstanceError):
        row.residual_policy = "custom_weights"
    with pytest.raises(ValueError, match="residual_policy must be one of"):
        _row(residual_policy="automatic")
