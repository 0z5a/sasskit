"""Shared test fixtures for sasskit."""

import os
import pytest


# Path to test cubin — set via environment or auto-detect
TEST_CUBIN_PATH = os.environ.get('TEST_CUBIN', '')


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests that need a cubin file when it's not available."""
    if os.path.exists(TEST_CUBIN_PATH):
        return
    skip_no_cubin = pytest.mark.skip(reason=f"Test cubin not found: {TEST_CUBIN_PATH}")
    for item in items:
        if "cubin" in item.fixturenames:
            item.add_marker(skip_no_cubin)
