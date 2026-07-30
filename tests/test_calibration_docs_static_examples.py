"""Executable-documentation tests for CAL-090.

Extracts every fenced Python block from
``docs/calibration_examples.md`` and executes them in order in one
shared namespace.  The documented examples carry their own exact
assertions (the canonical static arithmetic), so a passing run proves
every static example executes and its expected values match the tested
values.  Additional tests pin the documentation statements required by
the plan: the package validates choices but does not select them, and
residual-recipient changes are described as residual-allocation
effects, never as experiment-identified effects.
"""

from __future__ import annotations

import re
from pathlib import Path

DOC_PATH = Path(__file__).resolve().parent.parent / "docs" / "calibration_examples.md"

_PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _python_blocks() -> list[str]:
    return _PYTHON_BLOCK.findall(DOC_PATH.read_text(encoding="utf-8"))


def test_document_exists_and_contains_all_static_policy_examples() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    for policy in (
        "allow_total_change",
        "proportional_untested",
        "custom_weights",
        "residual_bucket",
        "explicit_adjustments",
    ):
        assert f'residual_policy="{policy}"' in text
    assert "apply_shares" in text
    assert "current_total=1000.0" in text


def test_every_documented_static_example_executes_with_expected_values() -> None:
    blocks = _python_blocks()
    assert len(blocks) == 8
    namespace: dict[str, object] = {}
    for index, block in enumerate(blocks):
        try:
            exec(compile(block, f"{DOC_PATH.name}[block {index}]", "exec"), namespace)
        except Exception as error:  # pragma: no cover - failure reporting only
            raise AssertionError(
                f"Documented example block {index} failed: {error!r}"
            ) from error
    # The shared session produced every documented result object.
    for name in (
        "result_total_change",
        "result_proportional",
        "result_custom",
        "result_bucket",
        "result_explicit",
        "result_shares",
    ):
        assert name in namespace


def test_documentation_states_validation_without_selection() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    assert "it validates the\nchoices you declare" in text.replace("\r\n", "\n")
    assert "It does not select a\nresidual policy" in text.replace("\r\n", "\n")


def test_residual_changes_are_labeled_residual_allocation_not_experimental() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    assert "residual-allocation effect" in text
    assert "The lift test\n  did not measure those entities" in text.replace(
        "\r\n", "\n"
    )
    # The only occurrences of "experimentally identified" phrasing must be
    # negations; the document never affirms an experimental effect for a
    # residual recipient.
    for match in re.finditer(r"experiment[a-z]* identified", text):
        preceding = text[max(0, match.start() - 60) : match.start()]
        assert "not" in preceding or "never" in preceding
