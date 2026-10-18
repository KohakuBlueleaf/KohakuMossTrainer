"""A rank that raises ends the whole spawned job within seconds (non-zero exit, its
traceback printed) while its peer waits in a receive, and takes its worker processes
with it; gloo on CPU and nccl on two GPUs, no model. Broken variants: the failing rank
tears the process group down before raising (blocks on nccl while a receive is
posted), and a child hidden from the rank's process table (survives the rank)."""

import os
import signal
import subprocess
import sys
import tempfile
import time

import pytest
import torch

from kohakumoss.parallel.telemetry.selftest import FAILURE, RAISED
from tests._util.negative import must_fail

CAP_S = 180.0  # a job's whole wall time, start-up under load included
HANG_CAP_S = 40.0  # the hanging variant's cap
FAST_S = 15.0  # from the failing rank's raise to the job's exit
GONE_S = 15.0  # from the job's exit to its child's end
SCRIPT = (
    "from kohakumoss.parallel.telemetry import selftest\n"
    "from kohakumoss.parallel.launch import spawn\n"
    "from kohakumoss.types.plans import ParallelPlan\n"
    "plan = ParallelPlan(kind='test', backend={backend!r}, devices=(0, 1))\n"
    "spawn(selftest.{worker}, plan, *{args!r}, device_type={device_type!r})\n"
)


def _kill_tree(proc: subprocess.Popen) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], check=False)
    else:
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()


def _raised_at(stderr: str) -> float | None:
    for line in stderr.splitlines():
        if line.startswith(RAISED):
            return float(line.split()[-1])
    return None


def _job(
    backend: str,
    device_type: str,
    worker: str,
    *args,
    cap: float = CAP_S,
    group_kill: bool = True,
) -> dict:
    """Run the 2-process job: its exit code, stderr and the seconds from the failing
    rank's raise to the job's exit (code None: still running at `cap`, then killed with
    its workers, or alone without `group_kill`); the exit is the job process's own,
    whatever still holds its stderr."""
    script = SCRIPT.format(
        backend=backend, worker=worker, args=args, device_type=device_type
    )
    with tempfile.TemporaryFile("w+", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.DEVNULL,
            stderr=log,
            start_new_session=sys.platform != "win32",
        )
        try:
            code = proc.wait(timeout=cap)
        except subprocess.TimeoutExpired:
            if group_kill:
                _kill_tree(proc)
            else:
                proc.kill()
                proc.wait()
            code = None
        ended = time.time()
        log.seek(0)
        err = log.read()
    raised = _raised_at(err)
    after = None if raised is None or code is None else ended - raised
    return {"code": code, "after_raise_s": after, "stderr": err}


def check_failfast(res: dict) -> None:
    """The job ended, non-zero, within FAST_S of the failing rank's raise, with its
    traceback (process start-up, which load stretches, is not counted)."""
    assert res["code"] is not None and res["code"] != 0, res["code"]
    assert res["after_raise_s"] is not None and res["after_raise_s"] < FAST_S, res
    assert FAILURE in res["stderr"], res["stderr"][-2000:]


def _alive(pid: int) -> bool:
    """`pid` exists and is not a zombie (Linux /proc)."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def check_child_gone(pid: int) -> None:
    """The failing rank's child is gone within GONE_S of the job's end."""
    end = time.perf_counter() + GONE_S
    while _alive(pid) and time.perf_counter() < end:
        time.sleep(0.1)
    assert not _alive(pid), f"child {pid} outlived its rank"


def _orphan_job(detach: bool) -> tuple[dict, int]:
    """The orphan job; with `detach` (the surviving child may hold the parent's exit)
    the parent alone is killed at HANG_CAP_S, so the child stays for the check."""
    pid_path = os.path.join(tempfile.mkdtemp(), "child.pid")
    if detach:
        res = _job(
            "gloo",
            "cpu",
            "orphan_worker",
            pid_path,
            True,
            cap=HANG_CAP_S,
            group_kill=False,
        )
    else:
        res = _job("gloo", "cpu", "orphan_worker", pid_path, False)
    with open(pid_path, encoding="utf-8") as f:
        return res, int(f.read())


@pytest.mark.cpu
@pytest.mark.budget(
    200
)  # spawns its gloo rank processes (start-up stretches under load)
def test_raising_rank_ends_the_job_gloo():
    check_failfast(_job("gloo", "cpu", "raise_worker", False))


@pytest.mark.cpu
@pytest.mark.budget(400)  # spawns its gloo rank processes twice
def test_raising_rank_ends_its_children():
    if sys.platform == "win32":
        pytest.skip("forkserver loader workers are Linux-only")
    res, pid = _orphan_job(detach=False)
    check_failfast(res)
    check_child_gone(pid)
    res, pid = _orphan_job(detach=True)
    try:
        assert FAILURE in res["stderr"], res["stderr"][-2000:]
        must_fail(lambda: check_child_gone(pid), "a child left alive")
    finally:
        if _alive(pid):
            os.kill(pid, signal.SIGKILL)


@pytest.mark.cuda
@pytest.mark.multi_gpu
@pytest.mark.budget(260)  # the broken variant runs to HANG_CAP_S
def test_raising_rank_ends_the_job_nccl():
    if sys.platform == "win32" or torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices on Linux")
    check_failfast(_job("nccl", "cuda", "raise_worker", False))
    hung = _job("nccl", "cuda", "raise_worker", True, cap=HANG_CAP_S)
    must_fail(lambda: check_failfast(hung), "a group teardown before the exit")
