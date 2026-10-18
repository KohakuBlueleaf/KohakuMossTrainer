"""`must_fail` counts a check that fails by assert or by a pytest check (`pytest.raises`
that sees no exception), and rejects a check that passes."""

import pytest

from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _raises_nothing() -> None:
    with pytest.raises(RuntimeError):
        pass


def test_a_failed_pytest_check_counts():
    must_fail(_raises_nothing, "pytest.raises saw no exception")
    must_fail(lambda: pytest.fail("explicit"), "pytest.fail")


def test_a_passing_check_is_rejected():
    with pytest.raises(AssertionError, match="passed on the broken variant"):
        must_fail(lambda: None, "nothing fails")
