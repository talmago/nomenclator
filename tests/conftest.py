"""Shared pytest configuration for the nomenclator test suite."""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: tests that call the live WCO nomenclature source",
    )
