"""Distribution metadata and package-root version contract."""

from __future__ import annotations

import ast
from importlib.metadata import metadata, version
from pathlib import Path

import do_attribution


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_runtime_version_comes_from_distribution_metadata() -> None:
    assert do_attribution.__version__ == version("do-attribution")


def test_installed_metadata_matches_the_public_package() -> None:
    package_metadata = metadata("do-attribution")

    assert package_metadata["Name"] == "do-attribution"
    assert package_metadata["Version"] == do_attribution.__version__
    assert package_metadata["Requires-Python"] == ">=3.10"


def test_setup_py_contains_no_duplicate_metadata() -> None:
    setup_tree = ast.parse((REPO_ROOT / "setup.py").read_text(encoding="utf-8"))
    setup_calls = [
        node
        for node in ast.walk(setup_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    ]

    assert len(setup_calls) == 1
    assert setup_calls[0].args == []
    assert setup_calls[0].keywords == []


def test_requirements_delegates_to_pyproject_metadata() -> None:
    requirement_lines = [
        line.strip()
        for line in (REPO_ROOT / "requirements.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert requirement_lines == ["-e .[dev]"]
