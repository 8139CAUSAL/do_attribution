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
    assert 'for characteristic_function in ("cumulative", "exact")' in stress_test
    assert "population_refit" in stress_test
    assert "comparison.unavailable_models()" in comparison
    assert "raw_shapley_value" in comparison
    assert "get_shapley_values_matrix()" in temporal_shapley
    assert "normalization_status" in temporal_shapley


def test_executed_outputs_report_available_shapley_attribution() -> None:
    static_output = _all_text_outputs("basic_shapley.ipynb")
    comparison_output = _all_text_outputs("side_by_side.ipynb")
    temporal_output = _all_text_outputs("temporal_shapley.ipynb")

    assert "characteristic function: cumulative" in static_output
    assert "normalization status   : available" in static_output
    assert "shares sum to 1.000000" in static_output
    assert "raw values sum to 4,848.00; observed conversions: 4,848" in static_output
    # The former presentation defect: an empty share table "summing to zero".
    assert "Empty DataFrame" not in static_output
    assert "shares sum to 0.000000" not in static_output

    assert "Unavailable normalized models: {}" in comparison_output
    assert "Largest disagreement: shapley_minus_markov on" in comparison_output
    assert "Normalized differences are unavailable" not in comparison_output

    assert "No epoch produced a valid normalized attribution vector" not in (
        temporal_output
    )
    assert "unavailable" not in temporal_output
    assert "Empty DataFrame" not in temporal_output


def test_executed_stress_test_keeps_the_legacy_game_refusal_visible() -> None:
    stress_output = _all_text_outputs("methodological_stress_tests.ipynb")
    assert "cumulative" in stress_output
    assert "near_cancellation" in stress_output
    assert "negative_share" in stress_output


def test_temporal_notebooks_assign_every_journey_to_one_weekly_epoch() -> None:
    """A week spanning 2020/2021 is one epoch holding all of its journeys."""
    for notebook_name in ("temporal_markov.ipynb", "temporal_shapley.ipynb"):
        output = _all_text_outputs(notebook_name)
        assert "epochs fitted: 14 " in output, notebook_name
        assert "journeys assigned to an epoch: 272,792 of 272,792" in output
        assert "Overall: 272792 paths, 4848 conversions" in output
        assert "Overall: 269123 paths" not in output


def test_every_figure_is_displayed_exactly_once() -> None:
    """No figure is rendered twice, and every plotting cell shows its figure.

    A trailing figure expression under ``%matplotlib inline`` produced an
    image-bearing ``execute_result`` plus the backend's own ``display_data``
    for the same figure.
    """
    expected_image_counts = {
        "attribution_experiment_calibration.ipynb": 8,
        "basic_markov.ipynb": 3,
        "basic_shapley.ipynb": 2,
        "methodological_stress_tests.ipynb": 0,
        "side_by_side.ipynb": 1,
        "temporal_markov.ipynb": 2,
        "temporal_shapley.ipynb": 2,
    }
    assert set(expected_image_counts) == set(NOTEBOOKS)
    for notebook_name, expected in expected_image_counts.items():
        images: list[str] = []
        for cell in _notebook(notebook_name)["cells"]:
            cell_images = []
            for output in cell.get("outputs", []):
                image = output.get("data", {}).get("image/png")
                if image is None:
                    continue
                assert output["output_type"] == "display_data", (
                    f"{notebook_name}: a figure is stored as "
                    f"{output['output_type']!r}"
                )
                cell_images.append(image)
            assert len(set(cell_images)) == len(cell_images), (
                f"{notebook_name}: a cell stores the same figure twice"
            )
            images.extend(cell_images)
        assert len(images) == expected, notebook_name


def test_plotting_cells_show_figures_instead_of_returning_them() -> None:
    for notebook_name in NOTEBOOKS:
        notebook = _notebook(notebook_name)
        code_cells = [
            _source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]
        code = "\n".join(code_cells)
        if "plt." in code or "savefig" in code:
            assert "%matplotlib inline" in code, notebook_name
        for cell_source in code_cells:
            last_line = cell_source.rstrip().splitlines()[-1].strip()
            assert last_line not in {"fig", "figure", "plt.gcf()"}, (
                f"{notebook_name}: a cell ends with the bare figure "
                f"expression {last_line!r}"
            )


def test_saved_outputs_contain_no_warnings_or_machine_specific_paths() -> None:
    for notebook_name in NOTEBOOKS:
        notebook = _notebook(notebook_name)
        stderr = [
            output
            for cell in notebook["cells"]
            for output in cell.get("outputs", [])
            if output.get("output_type") == "stream" and output.get("name") == "stderr"
        ]
        assert not stderr, f"{notebook_name} stores stderr output"
        outputs = _all_text_outputs(notebook_name)
        for token in ("FigureCanvasAgg", "UserWarning", "site-packages", "ipykernel_"):
            assert token not in outputs, f"{notebook_name} output contains {token!r}"


def test_installed_notebook_runner_covers_every_public_notebook() -> None:
    runner = (ROOT / "scripts" / "execute_notebooks.py").read_text(encoding="utf-8")
    for notebook_name in NOTEBOOKS:
        assert f'"{notebook_name}"' in runner
    assert "package_path.is_relative_to(SOURCE_PACKAGE_DIRECTORY)" in runner


def test_installed_notebook_runner_uses_a_backend_that_can_show_figures() -> None:
    """``plt.show()`` on a bare Agg canvas only warns and embeds no image."""
    runner = (ROOT / "scripts" / "execute_notebooks.py").read_text(encoding="utf-8")
    assert '"MPLBACKEND": MATPLOTLIB_BACKEND' in runner
    assert 'MATPLOTLIB_BACKEND = "module://matplotlib_inline.backend_inline"' in runner
    assert '"MPLBACKEND": "Agg"' not in runner
    assert '"--in-place"' in runner
