"""`retrying`: transient failures retried within the attempt cap and the deadline, other
failures raised at once (pure function)."""

import pytest

from kohakumoss_common.retry import retrying
from tests._util.negative import must_fail

pytestmark = [pytest.mark.cpu, pytest.mark.budget(5)]


def _flaky(failures: int, exc=OSError):
    state = {"calls": 0}

    def call():
        state["calls"] += 1
        if state["calls"] <= failures:
            raise exc("busy")
        return "ok"

    return call, state


def check_recovers(attempts: int) -> None:
    call, state = _flaky(3)
    try:
        got = retrying(call, (OSError,), "x", 60.0, attempts, backoff=0.0)
    except OSError as err:
        raise AssertionError(f"not recovered: {err}") from err
    assert got == "ok" and state["calls"] == 4


def test_recovers_after_transient_failures():
    check_recovers(4)
    must_fail(lambda: check_recovers(3), "attempt cap below the failures")


def test_cap_and_deadline_raise_oserror():
    call, _ = _flaky(9)
    with pytest.raises(OSError, match="2 attempts"):
        retrying(call, (OSError,), "x", 60.0, 2, backoff=0.0)
    call, state = _flaky(9)
    with pytest.raises(OSError, match="1 attempts"):
        retrying(call, (OSError,), "x", 0.5, backoff=1.0)
    assert state["calls"] == 1


def test_other_failures_propagate():
    call, state = _flaky(1, ValueError)
    with pytest.raises(ValueError):
        retrying(call, (OSError,), "x", 60.0, backoff=0.0)
    assert state["calls"] == 1
    call, _ = _flaky(1)
    with pytest.raises(OSError, match="busy"):
        retrying(call, (OSError,), "x", 60.0, is_transient=lambda e: False)
