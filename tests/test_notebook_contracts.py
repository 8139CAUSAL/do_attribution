"""Static contracts for installed-package notebook validation."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
NOTEBOOKS = (
    "attribution_experiment_calibration.ipynb",
    "basic_markov.ipynb",
    "basic_shapley.ipynb",
    "methodological_stress_tests.ipynb",
    "side_by_side.ipynb",
    "temporal_markov.ipynb",
    "temporal_shapley.ipynb",
)


def _notebook(name: str) -> dict:
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def _all_source(name: str) -> str:
    return "\n".join(_source(cell) for cell in _notebook(name)["cells"])


def _all_text_outputs(name: str) -> str:
    text_outputs: list[str] = []
    for cell in _notebook(name)["cells"]:
        for output in cell.get("outputs", []):
            text = output.get("text", "")
            text_outputs.append("".join(text) if isinstance(text, list) else text)
            plain = output.get("data", {}).get("text/plain", "")
            text_outputs.append("".join(plain) if isinstance(plain, list) else plain)
    return "\n".join(text_outputs)


def test_notebooks_do_not_inject_or_discover_the_repository_source_path() -> None:
    forbidden = ("sys.path", "Path.cwd()", "REPOSITORY_ROOT", "os.chdir", "chdir(")
    for notebook_name in NOTEBOOKS:
        code = "\n".join(
            _source(cell)
            for cell in _notebook(notebook_name)["cells"]
            if cell["cell_type"] == "code"
        )
        for token in forbidden:
            assert token not in code, f"{notebook_name} contains {token!r}"


def test_notebook_outputs_are_current_and_contain_no_execution_errors() -> None:
    for notebook_name in NOTEBOOKS:
        code_cells = [
            cell
            for cell in _notebook(notebook_name)["cells"]
            if cell["cell_type"] == "code"
        ]
        assert code_cells
        assert all(cell.get("execution_count") is not None for cell in code_cells)
        assert not [
            output
            for cell in code_cells
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]


def test_repaired_normalization_contracts_are_encoded_in_notebooks() -> None:
    stress_test = _all_source("methodological_stress_tests.ipynb")
    comparison = _all_source("side_by_side.ipynb")
    temporal_shapley = _all_source("temporal_shapley.ipynb")

    assert '"near_cancellation", "negative_share"' in stress_test
    assert "population_refit" in stress_test
    assert "comparison.unavailable_models()" in comparison
    assert "raw_shapley_value" in comparison
    assert "get_shapley_values_matrix()" in temporal_shapley
    assert "Raw Shapley game value" in temporal_shapley


def test_executed_outputs_report_unavailable_normalization_as_diagnostic() -> None:
    comparison_output = _all_text_outputs("side_by_side.ipynb")
    temporal_output = _all_text_outputs("temporal_shapley.ipynb")

    assert "Unavailable normalized models: {'shapley': 'near_cancellation'}" in (
        comparison_output
    )
    assert "Normalized differences are unavailable" in comparison_output
    assert "No epoch produced a valid normalized attribution vector" in temporal_output
    assert "raw signed Shapley game values, not shares or percentages" in temporal_output


def test_installed_notebook_runner_covers_every_public_notebook() -> None:
    runner = (ROOT / "scripts" / "execute_notebooks.py").read_text(encoding="utf-8")
    for notebook_name in NOTEBOOKS:
        assert f'"{notebook_name}"' in runner
    assert "package_path.is_relative_to(SOURCE_PACKAGE_DIRECTORY)" in runner
