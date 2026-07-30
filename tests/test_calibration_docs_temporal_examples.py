"""Executable-documentation tests for the temporal calibration examples.

Extracts every fenced Python block from
``docs/calibration_examples_temporal.md`` and executes them in order in
one shared namespace.  The documented examples carry their own exact
assertions, so a passing run proves every documented temporal example
executes and its expected values match the tested behavior.  Additional
tests pin the statements the plan requires: all three temporal policy
layers are explicit, Print does not own the epochs, interpolation is a
missing-evidence rule, and no midpoint or string-coordinate inference
appears.

This suite executes all blocks present in the document, so sections
added by later documentation items extend it without changing its
contract.
"""

from __future__ import annotations

import re
from pathlib import Path

DOC_PATH = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "calibration_examples_temporal.md"
)

_PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _python_blocks() -> list[str]:
    return _PYTHON_BLOCK.findall(DOC_PATH.read_text(encoding="utf-8"))


def _executed_namespace() -> dict[str, object]:
    namespace: dict[str, object] = {}
    for index, block in enumerate(_python_blocks()):
        try:
            exec(compile(block, f"{DOC_PATH.name}[block {index}]", "exec"), namespace)
        except Exception as error:  # pragma: no cover - failure reporting only
            raise AssertionError(
                f"Documented temporal example block {index} failed: {error!r}"
            ) from error
    return namespace


def test_every_documented_temporal_example_executes_with_expected_values() -> None:
    namespace = _executed_namespace()
    for name in (
        "may_print",
        "dec_print",
        "temporal_calibration",
        "result",
        "result_outside",
    ):
        assert name in namespace


def test_all_three_temporal_policy_layers_are_explicit_in_every_call() -> None:
    calls = [
        block for block in _python_blocks() if "apply_estimates(" in block
    ]
    assert calls
    for block in calls:
        assert 'ratio_policy="coordinate_linear"' in block
        assert 'epoch_ratio_policy="representative_coordinate"' in block
        assert 'residual_policy="proportional_untested"' in block


def test_evidence_points_declare_explicit_dated_coordinates() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    assert text.count('coordinate_name="date"') >= 2
    assert "coordinate_value=date(2023, 5, 16)" in text
    assert "coordinate_value=date(2023, 12, 16)" in text


def test_documentation_states_no_epoch_ownership_and_interpolation_role() -> None:
    text = DOC_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert "Print\ndoes not own the epochs" in text
    assert "Interpolation is a\nmissing-evidence rule" in text
    assert "not entity-specific time\npartitioning" in text
    assert (
        "calibration never creates,\nmoves, splits, merges, or reassigns epoch"
        " boundaries" in text
    )


def test_no_midpoint_or_string_coordinate_inference_appears() -> None:
    text = DOC_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
    # The document states both boundaries explicitly.
    assert "never silently converts" in text
    assert "midpoint" in text
    assert "cannot serve as interpolation coordinates" in text
    # And no documented code block infers a coordinate: every coordinate
    # in code is an explicit date literal.
    for block in _python_blocks():
        assert "window_midpoint" not in block
        assert ".mid" not in block
