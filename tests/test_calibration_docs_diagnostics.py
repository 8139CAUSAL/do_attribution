"""Executable-documentation tests for the diagnostics, plotting, and positioning docs.

Extracts every fenced Python block from ``docs/calibration_diagnostics.md``
and executes them in order in one shared namespace.  The documented examples
carry their own exact assertions, so a passing run proves every documented
example executes, every documented plot method and artist exists, and the
documented values match the tested behavior.

Additional tests pin the statements CAL-094 requires: all seven plotting methods
and the resolved diagnostic-alternative surface are documented; diagnostics are
stated to expose alternatives without applying them; epochs are stated to be
outcome-driven or user-supplied and never channel-owned; interpolation is
stated to be a missing-evidence rule; residual effects are stated not to be
experiment-identified; the excluded-scope list is present; and no excluded
behavior is advertised.

This suite executes all blocks present in the document, so sections added by
later documentation items extend it without changing its contract.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")
matplotlib.use("Agg")
pytest.importorskip("matplotlib.pyplot")

DOC_PATH = (
    Path(__file__).resolve().parent.parent / "docs" / "calibration_diagnostics.md"
)

_PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)

PLOT_METHODS = (
    "plot_epoch_alignment",
    "plot_ratio_schedule",
    "plot_evidence_uncertainty",
    "plot_calibration_effects",
    "plot_residual_allocation",
    "plot_entity_mix",
    "plot_evidence_strength",
)

DIAGNOSTIC_CONTROLS = (
    "diagnostic_conflict_policies",
    "diagnostic_selected_evidence",
    "diagnostic_evidence_weights",
    "diagnostic_epoch_ratio_policies",
    "diagnostic_epoch_coordinates",
)

EXCLUDED_TOPICS = (
    "default calibration strategy",
    "automatic pooling",
    "automatic interpolation",
    "bayesian updating",
    "mmm",
    "epoch boundary movement",
    "extrapolation",
    "hidden normalization",
)

NEGATION = re.compile(
    r"\b(?:not|never|no|none|nothing|neither|nor|without|excluded|"
    r"deliberately|refuses?|rejects?|raises?|instead of|rather than|would)\b",
    re.IGNORECASE,
)


def _document() -> str:
    return DOC_PATH.read_text(encoding="utf-8")


def _python_blocks() -> list[str]:
    return _PYTHON_BLOCK.findall(_document())


def _executed_namespace() -> dict[str, object]:
    namespace: dict[str, object] = {}
    for index, block in enumerate(_python_blocks()):
        try:
            exec(compile(block, f"{DOC_PATH.name}[block {index}]", "exec"), namespace)
        except Exception as error:  # pragma: no cover - failure reporting only
            raise AssertionError(
                f"Documented diagnostics example block {index} failed: {error!r}"
            ) from error
    return namespace


def _sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
        if sentence.strip()
    ]


def test_every_documented_example_executes() -> None:
    namespace = _executed_namespace()
    for name in (
        "result",
        "strength_result",
        "conflict_result",
        "reduction_result",
        "outside_support",
    ):
        assert name in namespace, f"documented object {name!r} was not produced"

    # The shared scenario is the one described in the document.
    result = namespace["result"]
    assert result.epochs == ("E1", "E2", "E3")
    assert [
        (row.epoch, row.effective_ratio, row.ratio_source)
        for row in result.epoch_ratio_schedule
    ] == [
        ("E1", 0.90, "observed"),
        ("E2", 1.00, "interpolated"),
        ("E3", 1.10, "observed"),
    ]


def test_all_seven_plotting_methods_are_documented_and_exist() -> None:
    document = _document()
    namespace = _executed_namespace()
    result = namespace["result"]

    for method_name in PLOT_METHODS:
        assert f"{method_name}()" in document, f"{method_name} is not documented"
        assert callable(
            getattr(result, method_name)
        ), f"{method_name} does not exist on the result"

    # Exactly seven plotting methods exist; the document claims no others.
    actual = {
        name
        for name in dir(result)
        if name.startswith("plot_") and callable(getattr(result, name))
    }
    assert actual == set(PLOT_METHODS)
    assert re.search(r"exactly seven plotting methods", document, re.IGNORECASE)


def test_diagnostic_alternative_surface_is_documented_with_every_control() -> None:
    document = _document()
    for control in DIAGNOSTIC_CONTROLS:
        assert control in document, f"diagnostic control {control!r} is not documented"

    # All three conflict policies and both coordinate reductions are shown as
    # alternatives against a stored applied ratio.
    for policy in (
        "select_evidence",
        "weighted_ratio",
        "pooled_lift",
        "representative_coordinate",
        "coordinate_average",
    ):
        assert (
            f"Diagnostic only — not applied: {policy}" in document
        ), f"{policy} is not documented as a nonapplied alternative"
    assert "Stored applied effective ratio" in document


def test_documentation_states_diagnostics_do_not_apply_alternatives() -> None:
    document = _document()
    assert (
        "Diagnostics expose alternatives without applying them" in document
    ), "the diagnostics-do-not-apply statement is missing"
    assert "never becomes an applied policy" in document
    assert "never replaces a stored ratio" in document


def test_diagnostic_calls_do_not_change_result_state() -> None:
    """Behavioral backing for the documented nonapplied guarantee."""

    namespace = _executed_namespace()
    conflict_result = namespace["conflict_result"]
    plt = namespace["plt"]

    snapshot = (
        conflict_result.rows,
        conflict_result.epoch_ratio_schedule,
        conflict_result.evidence_points,
        conflict_result.ratio_provenance,
        conflict_result.epoch_reduction_provenance,
        conflict_result.residual_provenance,
        conflict_result.calibrated_totals,
    )
    figure = conflict_result.plot_evidence_strength(
        diagnostic_conflict_policies={
            ("Print", "E1"): ("weighted_ratio", "pooled_lift")
        },
        diagnostic_evidence_weights={
            ("Print", "E1"): {"print-geo-a": 0.25, "print-geo-b": 0.75}
        },
    )
    plt.close(figure)

    assert (
        conflict_result.rows,
        conflict_result.epoch_ratio_schedule,
        conflict_result.evidence_points,
        conflict_result.ratio_provenance,
        conflict_result.epoch_reduction_provenance,
        conflict_result.residual_provenance,
        conflict_result.calibrated_totals,
    ) == snapshot
    assert conflict_result.epoch_ratio_schedule[0].effective_ratio == 0.80
    assert conflict_result.epoch_ratio_schedule[0].ratio_source == (
        "selected_from_multiple"
    )


def test_documentation_states_epochs_are_never_entity_owned() -> None:
    document = _document()
    assert "Epochs are outcome-driven or user-supplied, never channel-owned" in document
    assert "does not own the epochs" in document

    # Any ownership sentence must be a denial, never a claim.
    for sentence in _sentences(document):
        if re.search(r"\bowns? the epochs?\b", sentence, re.IGNORECASE):
            assert re.search(
                r"\b(not|never|neither|nor|does not|do not)\b",
                sentence,
                re.IGNORECASE,
            ), f"epoch-ownership sentence is not a denial: {sentence!r}"


def test_documentation_states_interpolation_is_a_missing_evidence_rule() -> None:
    document = _document()
    assert "Interpolation is a missing-evidence rule" in document
    assert "Interpolation addresses missing evidence" in document
    assert "not entity-specific time" in document

    # The constructed value is never described as an observation.
    namespace = _executed_namespace()
    result = namespace["result"]
    interpolated = next(
        row for row in result.epoch_ratio_schedule if row.epoch == "E2"
    )
    assert interpolated.ratio_source == "interpolated"
    assert interpolated.ratio_source != "observed"


def test_documentation_states_residual_effects_are_not_experiment_identified() -> None:
    document = _document()
    assert "Residual effects are not experiment-identified" in document
    assert "Neither of those numbers is an experimentally identified effect" in document

    # Any "experiment*-identified" sentence must be a negation.
    for sentence in _sentences(document):
        if re.search(r"experiment\w*[ -]identified", sentence, re.IGNORECASE):
            assert re.search(
                r"\b(not|never|no|neither|nor|would be)\b", sentence, re.IGNORECASE
            ), f"residual causal-language sentence is not a negation: {sentence!r}"


def test_residual_recipients_receive_no_direct_test_adjustment() -> None:
    """Behavioral backing for the residual-is-not-measured statement."""

    namespace = _executed_namespace()
    result = namespace["result"]
    for epoch in result.epochs:
        for row in result.rows[epoch]:
            if row.role == "residual_recipient":
                assert row.direct_test_adjustment == 0.0
                assert row.evidence_ids == ()
            if row.role == "tested":
                assert row.residual_adjustment == 0.0


def _excluded_scope_section() -> str:
    document = _document()
    heading = "## Part 4 — Excluded scope"
    assert heading in document, "the excluded-scope section is missing"
    section = document[document.index(heading) + len(heading) :]
    end = section.find("\n## ")
    return section if end == -1 else section[:end]


def _excluded_scope_headings() -> str:
    """The bolded lead of every bullet in the excluded-scope section.

    Checking the whole section would pass on an incidental prose mention
    elsewhere in a bullet's body, so only the named exclusions are collected.
    """

    section = _excluded_scope_section()
    bullets = [line for line in section.splitlines() if line.startswith("- ")]
    assert bullets, "the excluded-scope section contains no exclusions"
    assert all(bullet.startswith("- **") for bullet in bullets)
    headings = re.findall(r"^- \*\*(.+?)\*\*", section, re.MULTILINE | re.DOTALL)
    assert len(headings) == len(bullets)
    return " | ".join(headings).lower()


def test_excluded_scope_is_listed() -> None:
    headings = _excluded_scope_headings()
    for topic in EXCLUDED_TOPICS:
        assert topic in headings, (
            f"excluded topic {topic!r} is not named by any exclusion in the "
            f"excluded-scope section"
        )


def test_no_excluded_behavior_is_advertised() -> None:
    """The document must never affirmatively offer an excluded behavior.

    Each pattern names a behavior the feature excludes.  A sentence matching a
    pattern is acceptable only when it denies or refuses that behavior; an
    affirmative sentence would advertise it.
    """

    forbidden = (
        r"\bby default\b",
        r"\bdefaults? to\b",
        r"\bautomatically (?:pools?|interpolates?|selects?|weights?|chooses?)\b",
        r"\bfalls? back to\b",
        r"\bextrapolat\w+\b",
        r"\brenormaliz\w+\b",
        r"\bcarry[- ]forward\b",
        r"\btriangulat\w+\b",
    )
    for pattern in forbidden:
        advertised = [
            sentence
            for sentence in _sentences(_document())
            if re.search(pattern, sentence, re.IGNORECASE)
            and not NEGATION.search(sentence)
        ]
        assert not advertised, (
            f"excluded behavior advertised via {pattern!r}: {advertised}"
        )


def test_documentation_states_validation_without_selection() -> None:
    document = _document()
    assert (
        "the package validates the choices you\ndeclare and applies exactly those "
        "choices" in document
    )
    assert "It does not select a ratio policy" in document
    assert "There is no default calibration strategy" in document


def test_every_documented_temporal_call_states_its_mandatory_policies() -> None:
    for index, block in enumerate(_python_blocks()):
        for call in ("apply_estimates(", "apply_shares("):
            for offset in [m.start() for m in re.finditer(re.escape(call), block)]:
                window = block[offset : offset + 1600]
                assert "residual_policy=" in window, (
                    f"block {index}: {call} does not state residual_policy"
                )
                assert "ratio_policy=" in window, (
                    f"block {index}: {call} does not state ratio_policy"
                )
                assert "epoch_ratio_policy=" in window, (
                    f"block {index}: {call} does not state epoch_ratio_policy"
                )


REPOSITORY_IMAGE_DIRECTORIES = ("examples/", "docs/", "images/", "assets/")

RAW_PREFIX = "https://raw.githubusercontent.com/8139CAUSAL/do_attribution/main/"


def _readme_image_sources() -> list[str]:
    text = (Path(__file__).resolve().parent.parent / "README.md").read_text(
        encoding="utf-8"
    )
    markdown_images = re.findall(r"!\[[^\]]*\]\(\s*<?([^)\s>]+)", text)
    html_images = re.findall(r"<img[^>]*\ssrc=[\"']([^\"']+)[\"']", text)
    return markdown_images + html_images


def test_readme_image_links_are_absolute_for_pypi() -> None:
    """CAL-094 also requires PyPI-renderable README images.

    PyPI renders the README outside the repository, so a relative or
    root-relative image path resolves to nothing.  Every image source must be
    absolute.
    """

    sources = _readme_image_sources()
    assert sources, "README.md contains no image links to validate"
    for source in sources:
        assert source.startswith(("https://", "http://")), (
            f"README image link is not absolute and will not render on PyPI: "
            f"{source!r}"
        )


def test_readme_repository_images_use_absolute_raw_github_urls() -> None:
    """Images served from this repository must use the raw GitHub host."""

    repository_images = [
        source
        for source in _readme_image_sources()
        if not source.startswith(("https://", "http://"))
        or "githubusercontent.com" in source
        or "github.com" in source
        or source.lstrip("./").startswith(REPOSITORY_IMAGE_DIRECTORIES)
    ]
    assert repository_images, "README.md references no repository images"
    for source in repository_images:
        assert source.startswith(RAW_PREFIX), (
            f"README repository image does not use the absolute raw GitHub "
            f"URL prefix {RAW_PREFIX!r}: {source!r}"
        )


def test_readme_repository_images_exist_in_the_repository() -> None:
    """An absolute raw URL must still point at a file that exists here."""

    root = Path(__file__).resolve().parent.parent
    for source in _readme_image_sources():
        if not source.startswith(RAW_PREFIX):
            continue
        relative = source[len(RAW_PREFIX) :]
        assert (root / relative).is_file(), (
            f"README image URL points at a missing repository file: {relative!r}"
        )
