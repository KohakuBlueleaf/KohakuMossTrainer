"""Results of spawned model-free workers leave no directory behind (gloo on CPU): after
`selftest.spawn_results` the system temp directory holds nothing new, and the results
are every rank's. Broken variant: a results directory that is never removed."""

import contextlib
import os
import tempfile

import pytest

from kohakumoss.parallel.telemetry import selftest
from kohakumoss.types.plans import ParallelPlan
from tests._util.negative import must_fail

pytestmark = [pytest.mark.cpu, pytest.mark.budget(60)]


@contextlib.contextmanager
def _kept_dir(prefix: str = "tmp"):
    yield tempfile.mkdtemp(prefix=prefix)


def check_nothing_left(root) -> None:
    plan = ParallelPlan(kind="test", backend="gloo", devices=(0, 1))
    res = selftest.spawn_results(selftest.ddp_worker, plan, device_type="cpu")
    assert len(res) == 2 and all(r["max_err"] < 1e-6 for r in res), res
    assert not os.listdir(root), os.listdir(root)


def test_spawned_results_leave_no_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    check_nothing_left(tmp_path)
    monkeypatch.setattr(
        selftest.tempfile, "TemporaryDirectory", lambda prefix="": _kept_dir(prefix)
    )
    must_fail(lambda: check_nothing_left(tmp_path), "a results directory left behind")
