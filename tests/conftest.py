"""Shared pytest fixtures for the do_attribution test suite."""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Shared small datasets                                                        #
# --------------------------------------------------------------------------- #
@pytest.fixture
def simple_paths():
    return ["A > B > C", "A > C", "B > C", "A > B", "C"]


@pytest.fixture
def simple_conversions():
    return [1, 0, 1, 1, 0]


@pytest.fixture
def example_paths():
    # This fixture intentionally retains the ``"Purchase"`` endpoint token
    # in the path strings so the conversion_labels regression tests can
    # exercise the parameter.  Callers who do NOT want the endpoint in the
    # data should use ``example_paths_no_endpoint``.
    return [
        "Email > Social > Purchase",
        "Search > Email > Purchase",
        "Social > Purchase",
        "Email > Search",
        "Search > Social > Email > Purchase",
        "Direct > Purchase",
        "Email > Social",
        "Search > Email > Social > Purchase",
        "Social > Email > Purchase",
        "Search",
    ]


@pytest.fixture
def example_conversions():
    return [1, 1, 1, 0, 1, 1, 0, 1, 1, 0]


# Companion fixture with the endpoint token already stripped.
# Numerically, ``MarkovAttribution(conversion_labels=["Purchase"]).fit(example_paths, ...)``
# must produce attribution values identical to
# ``MarkovAttribution().fit(example_paths_no_endpoint, ...)``.
@pytest.fixture
def example_paths_no_endpoint():
    return [
        "Email > Social",
        "Search > Email",
        "Social",
        "Email > Search",
        "Search > Social > Email",
        "Direct",
        "Email > Social",
        "Search > Email > Social",
        "Social > Email",
        "Search",
    ]


@pytest.fixture
def example_conversion_label():
    return "Purchase"


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT
