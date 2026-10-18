"""Shared training-test fixtures."""

import pytest

from kohakumoss_kernels.backends.compile import choice
from kohakumoss_kernels.tuning import tuner


@pytest.fixture
def fresh_tuning(tmp_path, monkeypatch):
    """TUNE="table" over an empty kernel cache, no settled or measured choices."""
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    tuner.set_mode("table")
    tuner._TABLES.clear()
    tuner._CHOSEN.clear()
    choice.reset()
    yield
    tuner._TABLES.clear()
    tuner._CHOSEN.clear()
    choice.reset()
