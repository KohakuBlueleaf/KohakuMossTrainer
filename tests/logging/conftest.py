"""Logging-test fixtures."""

import pytest

from tests._util.logs import collecting


@pytest.fixture
def collect():
    """Every project record at INFO and above, from all three package roots."""
    with collecting() as handler:
        yield handler
