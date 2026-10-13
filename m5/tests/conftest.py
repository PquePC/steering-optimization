"""Shared fixtures for the M5 tests.

Run:  pytest m5/tests -q

Tests that need torch and transformers skip themselves when those are not installed (the
pipeline's logic is tested without them; the hooks are tested with them, on tiny random models).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from m5 import config  # noqa: E402


@pytest.fixture
def cfg():
    """The default config with RUN_ROOT pointed at nothing real (tests pass tmp_path explicitly)."""
    return config.load()


@pytest.fixture
def run_root(tmp_path):
    """A run root outside the repository, as m5.safety requires."""
    p = tmp_path / "m5_runs"
    p.mkdir()
    return p
