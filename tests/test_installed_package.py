"""Smoke contract shared by source-tree and installed-artifact validation."""

from __future__ import annotations

from importlib.metadata import files, metadata, version
from pathlib import Path
import sys

import do_attribution


REQUIRED_PACKAGE_MODULES = {
    "__init__.py",
    "_calibration_validation.py",
    "_temporal_helpers.py",
    "base.py",
    "calibration.py",
    "compare.py",
    "markov.py",
    "normalization.py",
    "reporting.py",
    "shapley.py",
    "temporal.py",
    "temporal_diagnostics.py",
    "temporal_markov.py",
    "temporal_shapley.py",
    "utils.py",
    "visualization.py",
}


def test_installed_package_contract() -> None:
    package_path = Path(do_attribution.__file__).resolve()
    distribution_files = files("do-attribution")
    assert distribution_files is not None

    packaged_modules = {
        Path(str(item)).name
        for item in distribution_files
        if Path(str(item)).parent == Path("do_attribution")
    }
    package_metadata = metadata("do-attribution")

    assert REQUIRED_PACKAGE_MODULES <= packaged_modules
    assert package_metadata["Name"] == "do-attribution"
    assert package_metadata["License-Expression"] == "MIT"
    assert do_attribution.__version__ == version("do-attribution")
    assert all(hasattr(do_attribution, name) for name in do_attribution.__all__)

    result = do_attribution.ShapleyAttribution().fit(
        ["Email", "Email > Search", "Email > Search"],
        [1, 1, 1],
    )
    assert result["attribution"] == {"Email": 0.75, "Search": 0.25}

    for forbidden_prefix in sys.argv[1:]:
        assert not package_path.is_relative_to(Path(forbidden_prefix).resolve())


if __name__ == "__main__":
    test_installed_package_contract()
    print(f"installed package smoke passed: {Path(do_attribution.__file__).resolve()}")
