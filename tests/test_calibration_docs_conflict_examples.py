"""Executable-documentation tests for the conflict and reduction examples.

Extracts every fenced Python block from
``docs/calibration_examples_conflicts.md`` and executes them in order in
one shared namespace.  The documented examples carry their own exact
assertions, so a passing run proves every documented example executes and
its expected values match the tested behavior.  Additional tests pin the
statements CAL-092 requires: all three evidence-conflict policies and all
four epoch-ratio policies are documented with explicit inputs, no conflict
rule is inferred, every observation and weight and pooled component and
interval and coordinate and anchor appears in the documented outputs, and
an epoch average is not presented as an observed test result.

This suite executes all blocks present in the document, so sections added
by later documentation items extend it without changing its contract.
"""

from __future__ import annotations

import re
from pathlib import Path

DOC_PATH = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "calibration_examples_conflicts.md"
)

_PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)

CONFLICT_POLICIES = ("select_evidence", "weighted_ratio", "pooled_lift")

EPOCH_RATIO_POLICIES = (
    "direct",
    "representative_coordinate",
    "coordinate_average",
    "support_weighted_average",
)


def _python_blocks() -> list[str]:
    return _PYTHON_BLOCK.findall(DOC_PATH.read_text(encoding="utf-8"))


def _executed_namespace() -> dict[str, object]:
    namespace: dict[str, object] = {}
    for index, block in enumerate(_python_blocks()):
        try:
            exec(compile(block, f"{DOC_PATH.name}[block {index}]", "exec"), namespace)
        except Exception as error:  # pragma: no cover - failure reporting only
            raise AssertionError(
                f"Documented conflict example block {index} failed: {error!r}"
            ) from error
    return namespace


def test_every_documented_example_executes_with_expected_values() -> None:
    namespace = _executed_namespace()
    for name in (
        "print_geo_a",
        "print_geo_b",
        "conflict_calibration",
        "selected",
        "weighted",
        "pooled",
        "reduction_calibration",
        "direct",
        "representative",
        "averaged",
        "support_weighted",
    ):
        assert name in namespace

    # Two evidence points really do land in the same epoch.
    conflict_points = namespace["selected"].evidence_points  # type: ignore[attr-defined]
    assert len(conflict_points) == 2
    assert {point.matched_epoch for point in conflict_points} == {"E1"}


def test_every_conflict_policy_is_documented_with_explicit_inputs() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    for policy in CONFLICT_POLICIES:
        assert f'evidence_conflict_policy="{policy}"' in text

    # Each policy's own declared inputs appear, not just the policy name.
    assert 'selected_evidence={("Print", "E1"): "print-geo-a"}' in text
    assert '"print-geo-a": 0.25' in text
    assert '"print-geo-b": 0.75' in text
    assert "pooled_observed_lift" in text
    assert "pooled_matched_mta_estimate" in text


def test_every_epoch_ratio_policy_is_documented_with_explicit_inputs() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    for policy in EPOCH_RATIO_POLICIES:
        assert f'epoch_ratio_policy="{policy}"' in text

    assert 'ratio_schedule={("Print", "E1"): 0.90}' in text
    assert 'epoch_coordinates={"E1": 4.0}' in text
    assert 'epoch_coordinate_weights={"E1": [(3.0, 1.0), (7.0, 3.0)]}' in text


def test_every_documented_call_states_its_mandatory_policies() -> None:
    calls = [block for block in _python_blocks() if "apply_estimates(" in block]
    assert calls
    for block in calls:
        assert "ratio_policy=" in block
        assert "epoch_ratio_policy=" in block
        assert "residual_policy=" in block


def test_no_conflict_rule_is_inferred() -> None:
    text = DOC_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert "### No conflict rule is inferred" in text
    assert "evidence_conflict_policy is required" in text
    assert "selected_evidence is required" in text
    assert "must sum to one" in text
    assert "Neither\nreduction is ever inferred." in text

    # The rejections are executable, not merely described.
    namespace = _executed_namespace()
    assert "pytest" in namespace


def test_outputs_show_evidence_weights_pooled_components_intervals_and_anchors() -> None:
    namespace = _executed_namespace()

    # Every observation is retained under every conflict policy.
    for name in ("selected", "weighted", "pooled"):
        result = namespace[name]
        assert tuple(
            point.evidence_id for point in result.evidence_points
        ) == ("print-geo-a", "print-geo-b")
        for point in result.evidence_points:
            assert point.conflict_group == ("print-geo-a", "print-geo-b")

    # Declared combination weights.
    weighted_row = namespace["weighted"].epoch_ratio_schedule[0]
    assert weighted_row.evidence_weights == {
        "print-geo-a": 0.25,
        "print-geo-b": 0.75,
    }

    # Pooled components, exposed separately.
    pooled_resolution = namespace["pooled"].ratio_provenance[
        "conflict_resolution_by_pair"
    ][("Print", "E1")]["resolution"]
    assert pooled_resolution["pooled_observed_lift"] == 252.0
    assert pooled_resolution["pooled_matched_mta_estimate"] == 250.0

    # Intervals, coordinates, and anchors on the reduction rows.
    for name in (
        "direct",
        "representative",
        "averaged",
        "support_weighted",
    ):
        row = namespace[name].epoch_ratio_schedule[0]
        assert (row.epoch_start, row.epoch_end) == (2.0, 8.0)

    assert namespace["representative"].epoch_ratio_schedule[0].epoch_coordinate == 4.0
    for name in ("representative", "averaged", "support_weighted"):
        row = namespace[name].epoch_ratio_schedule[0]
        assert row.left_source_evidence_id == "print-geo-week0"
        assert row.right_source_evidence_id == "print-geo-week10"


def test_documentation_states_an_epoch_average_is_not_an_observed_result() -> None:
    text = DOC_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert "## An epoch average is not an observed test result" in text
    assert "They are constructed values" in text
    assert "it is not a lift-test result" in text
    assert 'must not be reported as "the measured lift\nin E1"' in text

    # And the record backs the claim: no reduced ratio claims observation.
    namespace = _executed_namespace()
    for name in ("direct", "representative", "averaged", "support_weighted"):
        assert namespace[name].epoch_ratio_schedule[0].ratio_source != "observed"


def test_documented_reductions_produce_one_ratio_per_pair() -> None:
    namespace = _executed_namespace()
    expected = {
        "direct": 0.90,
        "representative": 0.96,
        "averaged": 1.00,
        "support_weighted": 1.04,
    }
    for name, ratio in expected.items():
        schedule = namespace[name].epoch_ratio_schedule
        assert len(schedule) == 1
        assert (schedule[0].tested_entity, schedule[0].epoch) == ("Print", "E1")
        assert abs(schedule[0].effective_ratio - ratio) < 1e-9


def test_documentation_states_validation_without_selection() -> None:
    text = DOC_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert (
        "The package validates the choices you declare;\nit does not select a"
        " conflict policy, an epoch-ratio policy, or a residual\npolicy for you."
        in text
    )
