"""The spawn launcher leaves no rendezvous directory behind (gloo on CPU, no model): the
tcp init creates none, the file init's directory is gone after a normal job, after a job
whose rank raises, and after SIGTERM to the launcher, which also ends its rank
processes. Broken variants: the directory is never removed; SIGTERM kills the launcher
before its clean-up."""

import os
import signal
import subprocess
import sys
import tempfile
import time

import pytest

from kohakumoss.parallel import launch
from kohakumoss.parallel.telemetry import selftest
from kohakumoss.types.plans import ParallelPlan
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
PREFIX = "kohakumoss_dist_"
WAIT_S = 90.0
SCRIPT = (
    "import contextlib, os, sys\n"
    "from kohakumoss.parallel import launch\n"
    "from kohakumoss.parallel.telemetry import selftest\n"
    "from kohakumoss.types.plans import ParallelPlan\n"
    "if os.environ.get('KM_BROKEN') == '1':\n"
    "    launch._sigterm_exits = contextlib.nullcontext\n"
    "plan = ParallelPlan(kind='test', backend='gloo', init='file', devices=(0, 1))\n"
    "launch.spawn(selftest.comm_worker, plan, sys.argv[1], (256,) * 64, device_type='cpu')\n"
)


def leftovers(root) -> list[str]:
    return [n for n in os.listdir(root) if n.startswith(PREFIX)]


def check_clean(root) -> None:
    """No rendezvous directory is left under `root`."""
    assert not leftovers(root), leftovers(root)


def _job(init: str, worker, *args) -> None:
    plan = ParallelPlan(kind="test", backend="gloo", init=init, devices=(0, 1))
    launch.spawn(worker, plan, *args, device_type="cpu")


@pytest.mark.budget(200)  # spawns its gloo rank processes four times
def test_launch_removes_its_rendezvous_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    res = str(tmp_path / "res")
    for init in ("tcp", "file"):
        _job(init, selftest.ddp_worker, res)
        selftest.read_results(res, 2)
        check_clean(tmp_path)
    with pytest.raises(Exception):
        _job("file", selftest.raise_worker, False)
    check_clean(tmp_path)
    monkeypatch.setattr(launch.shutil, "rmtree", lambda *a, **k: None)
    _job("file", selftest.ddp_worker, res)
    must_fail(lambda: check_clean(tmp_path), "the directory is never removed")


def _group_gone(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    return False


def _terminated(tmp, broken: bool) -> tuple[str, int]:
    """Start the launcher in its own session, SIGTERM it once its rendezvous directory
    exists and its ranks run; (its temp root, its process group)."""
    root = tmp / ("broken" if broken else "fixed")
    root.mkdir()
    env = dict(os.environ, TMPDIR=str(root), KM_BROKEN="1" if broken else "0")
    proc = subprocess.Popen(
        [sys.executable, "-c", SCRIPT, str(root / "res")],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    end = time.monotonic() + WAIT_S
    while not leftovers(root) and time.monotonic() < end:
        time.sleep(0.1)
    time.sleep(5.0)
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=WAIT_S)
    return str(root), proc.pid


def check_ended(root: str, pgid: int) -> None:
    """The directory is gone and every rank process ended within a few seconds."""
    end = time.monotonic() + 10.0
    while not _group_gone(pgid) and time.monotonic() < end:
        time.sleep(0.1)
    assert _group_gone(pgid), "rank processes outlived the launcher"
    check_clean(root)


@pytest.mark.budget(240)  # two launcher processes with their gloo ranks
def test_sigterm_ends_the_job_and_removes_the_dir(tmp_path):
    if sys.platform == "win32":
        pytest.skip("POSIX signals and process groups")
    groups = []
    try:
        root, pgid = _terminated(tmp_path, broken=False)
        groups.append(pgid)
        check_ended(root, pgid)
        root, pgid = _terminated(tmp_path, broken=True)
        groups.append(pgid)
        must_fail(lambda: check_ended(root, pgid), "SIGTERM kills the launcher first")
    finally:
        for pgid in groups:
            if not _group_gone(pgid):
                os.killpg(pgid, signal.SIGKILL)
