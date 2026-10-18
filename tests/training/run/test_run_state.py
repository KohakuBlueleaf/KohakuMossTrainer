"""A run starts from a fresh process's kernel state: a backend-priority override, an
in-memory backend failure and the decision log's live mode of an earlier run in the
same process are gone (model-free)."""

import pytest

from kohakumoss_kernels import registry, status
from kohakumoss_kernels.run_state import reset_run_state
from kohakumoss_kernels.tuning import decisions
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
KEY = "op|backend|arch|test"


def _earlier_run() -> None:
    registry.set_priority("test-arch", ("eager",))
    status._FAILED[KEY] = "an earlier run's failure"
    decisions.go_live("an earlier run")


def check_fresh() -> None:
    assert "test-arch" not in registry._STATE.priority
    assert KEY not in status._FAILED
    assert not decisions._STATE["live"]
    assert registry._STATE.priority == dict(registry.DEFAULT_PRIORITY)


def test_reset_clears_the_earlier_run():
    _earlier_run()
    reset_run_state()
    check_fresh()
    _earlier_run()
    try:
        must_fail(check_fresh, "state inherited from an earlier run")
    finally:
        reset_run_state()
