"""Executable-documentation tests for the strength, share, bounds, and mix examples.

Extracts every fenced Python block from
``docs/calibration_examples_schedules.md`` and executes them in order in one
shared namespace.  The documented examples carry their own exact assertions,
so a passing run proves every documented example executes and its expected
values match the tested behavior.  Additional tests pin the statements
CAL-093 requires: an explicit-strength schedule is documented with its source
evidence and strengths, shares plus totals by epoch are documented and shown
equivalent to absolute estimates, optional ratio bounds are documented as
raise-only and never clipping, and a changing entity mix is documented as
changing residual allocation without changing ratio schedules or epochs.

This suite executes all blocks present in the document, so sections added by
later documentation items extend it without changing its contract.
"""

from __future__ import annotations

import re
from pathlib import Path

DOC_PATH = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "calibration_examples_schedules.md"
)

_PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)

EPOCHS = ("E1", "E2", "E3", "E4", "E5")

DOCUMENTED_STRENGTHS = (1.00, 0.75, 0.50, 0.25, 0.00)

DOCUMENTED_EFFECTIVE_RATIOS = (0.900, 0.925, 0.950, 0.975, 1.000)


def _python_blocks() -> list[str]:
    return _PYTHON_BLOCK.findall(DOC_PATH.read_text(encoding="utf-8"))


def _executed_namespace() -> dict[str, object]:
    namespace: dict[str, object] = {}
    for index, block in enumerate(_python_blocks()):
        try:
            exec(compile(block, f"{DOC_PATH.name}[block {index}]", "exec"), namespace)
        except Exception as error:  # pragma: no cover - failure reporting only
            raise AssertionError(
                f"Documented schedule example block {index} failed: {error!r}"
            ) from error
    return namespace


def test_every_documented_example_executes() -> None:
    namespace = _executed_namespace()
    for name in (
        "print_geo",
        "calibration",
        "strength",
        "partial",
        "from_shares",
        "bounded",
        "steady_mix",
        "shifted_mix",
    ):
        assert name in namespace

    # The scenario really is a single observation about a single tested entity.
    evidence_points = namespace["strength"].evidence_points  # type: ignore[attr-defined]
    assert len(evidence_points) == 1
    assert evidence_points[0].tested_entity == "Print"
    assert evidence_points[0].observed_ratio == 0.90


def test_explicit_strength_is_documented_with_source_evidence_and_strengths() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    assert 'ratio_policy="explicit_strength"' in text
    assert 'source_evidence={"Print": "print-geo-2024"}' in text
    assert "evidence_strength=" in text
    for strength in ("1.00", "0.75", "0.50", "0.25", "0.00"):
        assert strength in text

    namespace = _executed_namespace()
    schedule = namespace["strength"].epoch_ratio_schedule  # type: ignore[attr-defined]
    assert len(schedule) == len(EPOCHS)

    by_pair = namespace["strength"].epoch_reduction_provenance["by_pair"]  # type: ignore[attr-defined]
    for epoch, declared, effective in zip(
        EPOCHS, DOCUMENTED_STRENGTHS, DOCUMENTED_EFFECTIVE_RATIOS
    ):
        row = next(row for row in schedule if row.epoch == epoch)
        assert row.ratio_policy == "explicit_strength"
        assert row.epoch_ratio_policy == "direct"
        assert row.ratio_source == "strength_derived"
        # The source evidence point is named on every derived row.
        assert row.source_evidence_ids == ("print-geo-2024",)
        # Both the observation and the declared strength survive in provenance.
        provenance = by_pair[("Print", epoch)]
        assert provenance["observed_ratio"] == 0.90
        assert provenance["evidence_strength"] == declared
        assert abs(provenance["effective_ratio"] - effective) < 1e-9


def test_strength_schedule_is_never_inferred() -> None:
    text = DOC_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert "### The library does not infer the strength schedule" in text
    assert "requires inputs" in text
    assert "must be in \\[0, 1\\]" in text
    assert "requires epoch_ratio_policy='direct'" in text
    assert "### Missing strengths are not filled in" in text

    # The refusals are executable, not merely described.
    namespace = _executed_namespace()
    assert "pytest" in namespace

    # Partial coverage calibrates only the covered epochs; nothing is filled in.
    partial = namespace["partial"]
    assert [row.epoch for row in partial.epoch_ratio_schedule] == ["E1", "E2"]  # type: ignore[attr-defined]
    assert partial.ratio_provenance["uncalibrated_pairs"] == (  # type: ignore[attr-defined]
        ("Print", "E3"),
        ("Print", "E4"),
        ("Print", "E5"),
    )
    for epoch in ("E3", "E4", "E5"):
        for row in partial.rows[epoch]:  # type: ignore[attr-defined]
            assert row.role == "untouched"
            assert row.residual_policy is None


def test_temporal_shares_plus_totals_are_documented_and_equivalent() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    assert "apply_shares(" in text
    assert "totals_by_epoch=" in text

    namespace = _executed_namespace()
    from_shares = namespace["from_shares"]
    from_estimates = namespace["strength"]

    # Equivalence is asserted in the document; assert it here independently.
    assert from_shares == from_estimates
    for epoch in EPOCHS:
        assert from_shares.rows[epoch] == from_estimates.rows[epoch]  # type: ignore[attr-defined]
        assert (
            from_shares.calibrated_totals[epoch]  # type: ignore[attr-defined]
            == from_estimates.calibrated_totals[epoch]  # type: ignore[attr-defined]
        )
    assert from_shares.epoch_ratio_schedule == from_estimates.epoch_ratio_schedule  # type: ignore[attr-defined]

    # Shares are recorded on both sides of the calculation.
    row = next(r for r in from_shares.rows["E1"] if r.entity == "Print")  # type: ignore[attr-defined]
    assert row.original_share == 0.20
    assert row.calibrated_share == 0.18


def test_ratio_bounds_are_documented_as_raise_only_and_never_clipping() -> None:
    text = DOC_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert "### Bounds raise; they never clip" in text
    assert "ratio_bounds=(0.95, 1.05)" in text
    assert "ratio_bounds=(0.90, 1.00)" in text
    assert "outside declared raise-only bounds" in text
    assert "There is no default ratio bound." in text
    assert "It is never quietly replaced by\nthe nearest bound" in text

    namespace = _executed_namespace()
    bounded = namespace["bounded"]
    unbounded = namespace["strength"]

    # In-bound ratios are retained exactly, not modified in any way.
    assert [row.effective_ratio for row in bounded.epoch_ratio_schedule] == [  # type: ignore[attr-defined]
        row.effective_ratio for row in unbounded.epoch_ratio_schedule  # type: ignore[attr-defined]
    ]
    for epoch in EPOCHS:
        assert [row.calibrated_estimate for row in bounded.rows[epoch]] == [  # type: ignore[attr-defined]
            row.calibrated_estimate for row in unbounded.rows[epoch]  # type: ignore[attr-defined]
        ]

    # Inclusive endpoints both pass, and the declared bounds are recorded.
    ratios = [row.effective_ratio for row in bounded.epoch_ratio_schedule]  # type: ignore[attr-defined]
    assert min(ratios) == 0.90
    assert max(ratios) == 1.00
    for row in bounded.epoch_ratio_schedule:  # type: ignore[attr-defined]
        assert row.ratio_bound_lower == 0.90
        assert row.ratio_bound_upper == 1.00
    assert (
        bounded.epoch_reduction_provenance["by_pair"][("Print", "E1")][  # type: ignore[attr-defined]
            "bounds_action"
        ]
        == "validated_without_clipping"
    )

    # Bounds reach the exports without clipping the exported ratios.
    epoch_ratios = bounded.to_dataframes()["epoch_ratios"]  # type: ignore[attr-defined]
    assert set(epoch_ratios["ratio_bound_lower"]) == {0.90}
    assert set(epoch_ratios["ratio_bound_upper"]) == {1.00}
    assert [round(value, 3) for value in epoch_ratios["effective_ratio"]] == list(
        DOCUMENTED_EFFECTIVE_RATIOS
    )


def test_bounds_error_identifies_every_required_field() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    for fragment in (
        "tested_entity='Print'",
        "epoch='E1'",
        "effective_ratio=0.9",
        "lower=0.95",
        "upper=1.05",
        "ratio_policy='explicit_strength'",
        "epoch_ratio_policy='direct'",
        "ratio_source='strength_derived'",
        "source_evidence_ids=('print-geo-2024',)",
    ):
        assert fragment in text


def test_changing_mix_moves_allocation_without_moving_ratios_or_epochs() -> None:
    text = DOC_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert "## Part 4 — Changing entity mix with a constant effective ratio" in text
    assert "### The ratio schedule and the epochs do not move" in text
    assert "### The residual allocation does move" in text

    namespace = _executed_namespace()
    steady = namespace["steady_mix"]
    shifted = namespace["shifted_mix"]

    # Nothing about the evidence, the ratios, or the epochs changed.
    assert steady.epoch_ratio_schedule == shifted.epoch_ratio_schedule  # type: ignore[attr-defined]
    assert steady.evidence_points == shifted.evidence_points  # type: ignore[attr-defined]
    assert tuple(steady.rows) == tuple(shifted.rows) == ("E1", "E2")  # type: ignore[attr-defined]
    for row in shifted.epoch_ratio_schedule:  # type: ignore[attr-defined]
        assert row.effective_ratio == 0.90
        assert row.ratio_source == "constant"
    assert [
        (row.epoch, row.epoch_start, row.epoch_end)
        for row in shifted.epoch_ratio_schedule  # type: ignore[attr-defined]
    ] == [("E1", 0.0, 10.0), ("E2", 10.0, 20.0)]

    # The tested entity's direct effect is identical under both mixes.
    for result in (steady, shifted):
        for epoch in ("E1", "E2"):
            row = next(r for r in result.rows[epoch] if r.entity == "Print")  # type: ignore[attr-defined]
            assert row.calibrated_estimate == 180.0
            assert row.direct_test_adjustment == -20.0
            assert row.residual_adjustment == 0.0

    # Only the residual allocation differs, and only where the mix differs.
    def residuals(result, epoch):
        return {
            row.entity: round(row.residual_adjustment, 6)
            for row in result.rows[epoch]
        }

    assert residuals(steady, "E1") == residuals(shifted, "E1")
    assert residuals(steady, "E2") == {"Print": 0.0, "Search": 12.5, "Social": 7.5}
    assert residuals(shifted, "E2") == {"Print": 0.0, "Search": 5.0, "Social": 15.0}

    # The residual requirement is unchanged; only its distribution moved.
    for result in (steady, shifted):
        for epoch in ("E1", "E2"):
            provenance = result.residual_provenance[epoch]  # type: ignore[attr-defined]
            assert provenance["required_residual"] == 20.0
            assert provenance["allocated_residual"] == 20.0


def test_residual_changes_are_not_described_as_experimentally_identified() -> None:
    text = DOC_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert "Neither number is an experimentally identified effect on `Social`." in text
    assert "No test measured `Social`." in text
    assert (
        "The only measured quantity anywhere in this document is the single\n"
        "`Print` ratio of `0.90`." in text
    )

    # Any "experiment* identified" phrasing must be a negation.
    for match in re.finditer(r"[^.]*experiment\w*\s+identified[^.]*\.", text):
        sentence = match.group(0)
        assert any(
            negation in sentence
            for negation in ("Neither", "not ", "no ", "No ", "never")
        ), sentence


def test_every_documented_temporal_call_states_its_mandatory_policies() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    # Each policy bundle passed to a temporal call names all three layers.
    for bundle in ("STRENGTH_INPUTS = dict(", "MIX_INPUTS = dict("):
        start = text.index(bundle)
        block = text[start : text.index(")\n", start)]
        assert "ratio_policy=" in block
        assert "epoch_ratio_policy=" in block
        assert "residual_policy=" in block

    # The partial-coverage call states all three inline.
    partial_start = text.index("partial = calibration.apply_estimates(")
    partial_block = text[partial_start : text.index(")\n\nassert", partial_start)]
    assert 'ratio_policy="explicit_strength"' in partial_block
    assert 'epoch_ratio_policy="direct"' in partial_block
    assert 'residual_policy="proportional_untested"' in partial_block


def test_documentation_states_validation_without_selection() -> None:
    text = DOC_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert (
        "the package validates the choices you\ndeclare; it does not select a ratio"
        " policy, an epoch-ratio policy, a residual\npolicy, an evidence-strength"
        " schedule, or a ratio bound for you." in text
    )
