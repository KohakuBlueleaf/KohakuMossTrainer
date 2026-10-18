"""A bench run's whole process group ends with it: `end` reaches a grandchild the run
started (a loader worker's place), `reap` one that outlived the run."""

import subprocess
import sys
import time

import psutil
import pytest

pytest.importorskip(
    "kohakumoss_bench", reason="not installed: uv pip install -e scripts/bench/matrix"
)

from kohakumoss_bench.run import proc
from tests._util.negative import must_fail

pytestmark = [pytest.mark.cpu, pytest.mark.budget(60)]
RUN = """
import subprocess, sys, time
g = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
with open(sys.argv[1], "w") as f:
    f.write(str(g.pid))
time.sleep(120 if sys.argv[2] == "stay" else 0)
"""


def _alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _gone(pid: int, secs: float = 10.0) -> bool:
    deadline = time.monotonic() + secs
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.1)
    return False


def _start(tmp_path, mode: str) -> tuple[subprocess.Popen, int]:
    path = tmp_path / f"grandchild-{mode}.pid"
    run = subprocess.Popen(
        [sys.executable, "-c", RUN, str(path), mode], **proc.session()
    )
    deadline = time.monotonic() + 30
    while not (path.is_file() and path.read_text()):
        assert time.monotonic() < deadline, "the run never started its grandchild"
        time.sleep(0.05)
    return run, int(path.read_text())


def _cleanup(pid: int) -> None:
    if _alive(pid):
        psutil.Process(pid).kill()


def check_end(tmp_path) -> None:
    run, grandchild = _start(tmp_path, "stay")
    try:
        proc.end(run, [])
        assert run.returncode is not None
        assert _gone(grandchild), "end left the run's grandchild alive"
    finally:
        _cleanup(grandchild)


def check_reap(tmp_path) -> None:
    run, grandchild = _start(tmp_path, "exit")
    try:
        run.wait(timeout=30)
        assert _alive(grandchild), "the grandchild should outlive the run"
        proc.reap(run, [])
        assert _gone(grandchild), "reap left the run's grandchild alive"
    finally:
        _cleanup(grandchild)


def _leader_only(run, sig, sudo, leader=True):
    if sig == "TERM":
        run.terminate()
    elif leader:
        run.kill()


def test_end_reaches_the_whole_group(tmp_path):
    check_end(tmp_path)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows loader workers exit with their parent; reap is a no-op there",
)
def test_reap_kills_what_outlived_the_run(tmp_path):
    check_reap(tmp_path)


def test_negative_signals_to_the_leader_only(tmp_path, monkeypatch):
    monkeypatch.setattr(proc, "signal_group", _leader_only)
    must_fail(lambda: check_end(tmp_path), "end signals only the run's own pid")
    if sys.platform != "win32":
        must_fail(lambda: check_reap(tmp_path), "reap signals only the run's own pid")
