"""Execute every public notebook against a non-editable installed package.

Run this script with the Python interpreter from an environment where a wheel
or source distribution of ``do-attribution`` and the notebook dependencies are
installed.  The script deliberately refuses to run when the package import
resolves to this repository's source tree.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
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
            "MPLBACKEND": "Agg",
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

    print(f"Executed {len(NOTEBOOKS)} notebooks against {package_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
