"""Compatibility shim for tools that still invoke ``setup.py``.

All package metadata and build configuration live in ``pyproject.toml``.
"""

from setuptools import setup


setup()
