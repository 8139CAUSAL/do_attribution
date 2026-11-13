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
    "weighted_attribution.ipynb",
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


def test_weighted_notebook_calls_fit_with_the_optional_weights() -> None:
    source = _all_source("weighted_attribution.ipynb")

    # The unweighted call is shown unchanged, beside the weighted ones.
    assert "unweighted_model.fit(paths, conversions)" in source
    assert 'weights=frequency_table["journeys"]' in source
    assert "weights=relative_frequency" in source
    assert "weights=zero_weights" in source
    assert "comparison.fit(paths, conversions, weights=weights)" in source
    assert "CompareAttribution(include_baselines=True).fit(" in source
    assert "rtol=1e-10" in source and "atol=1e-12" in source


def test_executed_weighted_notebook_shows_the_weighting_contracts() -> None:
    output = _all_text_outputs("weighted_attribution.ipynb")

    # 1. A fit without weights reports its counts as the weight totals.
    assert "False      272792.0                4848.0" in output

    # 2. Frequency weights reproduce the one-row-per-journey fit.
    assert "272,792 journey rows -> 1,546 distinct (path, conversion) rows" in output
    for quantity in ("shares         ", "raw values     ", "coalition table"):
        assert f"weighted table reproduces the row-level {quantity}: True" in output
    assert "True      272792.0                4848.0" in output
    # Observed rows and weighted totals are reported separately.
    assert "Overall: 1546 paths, 276 conversions, 17.85% conversion rate" in output
    assert "| Total observation weight      | 272792 |" in output
    assert "| Weighted conversions          | 4848   |" in output
    assert "Top coalitions by weighted conversions" in output

    # 3. Fractional weights are kept; zero weights remove journeys.
    assert "shares unchanged by rescaling the weights: True" in output
    assert "weighted conversions 0.017772; observed conversion rate 0.017772" in output
    assert "journeys touching Affiliate (weight 0): 1,037" in output
    assert (
        "channels with zero weight: Direct, Email, Organic Search, "
        "Organic Social, Other, Paid Search, Referral"
    ) in output
    assert "same shares as deleting those rows: True" in output
    assert "Overall: 272792 paths, 4848 conversions, 1.78% conversion rate" in output
    assert "| Zero-weight journeys excluded | 1037   |" in output

    # 4. The weighted comparison recovers the fit on all journeys.
    assert "rows kept           : 129,749 of 272,792" in output
    assert "sample, weighted    0.0002      0.0" in output
    assert "sample, no weights  0.0369      0.0" in output
    assert "all journeys      : Markov P(conversion | start) = 0.0178" in output
    assert "sample, weighted  : Markov P(conversion | start) = 0.0177" in output
    assert "sample, no weights: Markov P(conversion | start) = 0.0374" in output
    assert (
        "ValueError: A weighted comparison supports only the built-in Markov "
        "and Shapley models."
    ) in output


def test_weighted_notebook_text_quotes_the_numbers_its_outputs_show() -> None:
    """The prose restates results; they must be the executed ones."""
    notebook = _notebook("weighted_attribution.ipynb")
    prose = "\n".join(
        _source(cell) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )
    normalized = " ".join(prose.split())

    assert "the 1,546 rows of the frequency table, 276 of them with" in normalized
    assert "the 272,792 journeys and 4,848 conversions" in normalized
    assert "Markov shares within 0.0002 of the fit on all journeys" in normalized
    assert "a conversion probability of 0.0177 against 0.0178" in normalized
    assert "roughly doubles to 0.0374" in normalized
    assert "up to 3.7 percentage points" in normalized
    assert "still counts all 272,792 rows" in normalized


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
        "weighted_attribution.ipynb": 0,
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
