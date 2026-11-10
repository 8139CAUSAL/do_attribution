"""Execute every public notebook against a non-editable installed package.

Run this script with the Python interpreter from an environment where a wheel
or source distribution of ``do-attribution`` and the notebook dependencies are
installed.  The script deliberately refuses to run when the package import
resolves to this repository's source tree.

By default the executed copies are written to a temporary directory and
discarded, so the run only validates that every notebook executes.  Pass
``--in-place`` to write the executed notebooks back over ``examples/*.ipynb``;
that is how the committed outputs are regenerated, and it keeps them tied to
the package and example code they were produced by.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import do_attribution


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIRECTORY = REPOSITORY_ROOT / "examples"
SOURCE_PACKAGE_DIRECTORY = REPOSITORY_ROOT / "do_attribution"
NOTEBOOKS = (
    "attribution_experiment_calibration.ipynb",
    "basic_markov.ipynb",
    "basic_shapley.ipynb",
    "methodological_stress_tests.ipynb",
    "side_by_side.ipynb",
    "temporal_markov.ipynb",
    "temporal_shapley.ipynb",
)
# Figures are rendered through the same backend a Jupyter kernel selects on
# its own, so ``plt.show()`` embeds the image in the notebook.  It needs no
# display, and it pins the backend against whatever ``MPLBACKEND`` the calling
# shell exports.  A bare ``Agg`` canvas cannot show figures: ``plt.show()``
# then only warns and the executed notebook contains no image.
MATPLOTLIB_BACKEND = "module://matplotlib_inline.backend_inline"


def _installed_package_path() -> Path:
    package_path = Path(do_attribution.__file__).resolve()
    if package_path.is_relative_to(SOURCE_PACKAGE_DIRECTORY):
        raise RuntimeError(
            "Notebook validation requires a non-editable wheel or source-"
            f"distribution install; import resolved to {package_path}."
        )
    return package_path


def _execute_notebook(
    notebook: Path,
    *,
    output_directory: Path,
    runtime_directory: Path,
    timeout: int,
) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "IPYTHONDIR": str(runtime_directory / "ipython"),
            "JUPYTER_CONFIG_DIR": str(runtime_directory / "jupyter-config"),
            "JUPYTER_DATA_DIR": str(runtime_directory / "jupyter-data"),
            "MPLBACKEND": MATPLOTLIB_BACKEND,
        }
    )
    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--execute",
        "--to",
        "notebook",
        "--output-dir",
        str(output_directory),
        f"--ExecutePreprocessor.timeout={timeout}",
        str(notebook),
    ]
    subprocess.run(
        command,
        cwd=runtime_directory,
        env=environment,
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Maximum seconds allowed for each notebook cell (default: 300).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help=(
            "Write the executed notebooks back to examples/, replacing the "
            "committed outputs. Without this flag the executed copies are "
            "discarded after validation."
        ),
    )
    args = parser.parse_args()

    package_path = _installed_package_path()
    print(f"Installed package: {package_path}")

    with TemporaryDirectory(prefix="do-attribution-notebooks-") as temporary:
        temporary_path = Path(temporary)
        output_directory = temporary_path / "executed"
        output_directory.mkdir()
        for notebook_name in NOTEBOOKS:
            notebook = EXAMPLES_DIRECTORY / notebook_name
            print(f"Executing {notebook.name}")
            _execute_notebook(
                notebook,
                output_directory=output_directory,
                runtime_directory=temporary_path,
                timeout=args.timeout,
            )
        if args.in_place:
            # Replace the sources only after every notebook has executed, so
            # a failure part-way through leaves examples/ untouched.
            for notebook_name in NOTEBOOKS:
                shutil.copyfile(
                    output_directory / notebook_name,
                    EXAMPLES_DIRECTORY / notebook_name,
                )
                print(f"Updated {EXAMPLES_DIRECTORY / notebook_name}")

    print(f"Executed {len(NOTEBOOKS)} notebooks against {package_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
