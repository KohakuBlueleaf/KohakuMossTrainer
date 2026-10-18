"""A run that gets SIGTERM ends its loader: with `kohakumoss.training.shutdown.install`
the forkserver DataLoader workers and the forkserver exit with the trainer."""

import json
import os
import signal
import subprocess
import sys
import time

import psutil
import pytest

from tests._util.negative import must_fail

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.budget(90),
    pytest.mark.skipif(
        sys.platform == "win32", reason="Windows loaders use spawn, not a forkserver"
    ),
]
TRAINER = """
import json, sys, time
import multiprocessing as mp
import torch
from torch.utils.data import DataLoader, TensorDataset
if sys.argv[2] == "install":
    from kohakumoss.training import shutdown
    shutdown.install()
loader = DataLoader(
    TensorDataset(torch.zeros(100000, 1)),
    num_workers=2,
    persistent_workers=True,
    multiprocessing_context=mp.get_context("forkserver"),
)
it = iter(loader)
next(it)
pids = [w.pid for w in it._workers] + [mp.forkserver._forkserver._forkserver_pid]
with open(sys.argv[1], "w") as f:
    f.write(json.dumps(pids))
try:
    time.sleep(120)
finally:
    with open(sys.argv[1] + ".unwound", "w") as f:
        f.write("1")
"""


def _alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _terminated(tmp_path, mode: str) -> tuple[list[int], bool]:
    """(worker and forkserver pids still alive 10 s after SIGTERM, whether the
    trainer's finally block ran)."""
    path = tmp_path / f"pids-{mode}.json"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    run = subprocess.Popen([sys.executable, "-c", TRAINER, str(path), mode], env=env)
    deadline = time.monotonic() + 60
    while not (path.is_file() and path.read_text()):
        assert time.monotonic() < deadline, "the trainer never started its loader"
        assert run.poll() is None, "the trainer died before its loader started"
        time.sleep(0.1)
    pids = json.loads(path.read_text())
    assert all(_alive(p) for p in pids), pids
    run.send_signal(signal.SIGTERM)
    run.wait(timeout=60)
    deadline = time.monotonic() + 10
    while any(_alive(p) for p in pids) and time.monotonic() < deadline:
        time.sleep(0.1)
    left = [p for p in pids if _alive(p)]
    for p in left:
        psutil.Process(p).kill()
    return left, (tmp_path / f"pids-{mode}.json.unwound").is_file()


def check_no_leak(left: list[int]) -> None:
    assert not left, f"loader processes outlived the trainer: {left}"


def check_unwound(unwound: bool) -> None:
    assert unwound, "SIGTERM did not unwind the trainer"


def test_sigterm_ends_the_loader_workers_and_forkserver(tmp_path):
    left, unwound = _terminated(tmp_path, "install")
    check_no_leak(left)
    check_unwound(unwound)


def test_negative_without_the_handler(tmp_path):
    left, unwound = _terminated(tmp_path, "plain")
    must_fail(lambda: check_no_leak(left), "no handler: the loader outlives")
    must_fail(lambda: check_unwound(unwound), "no handler: no unwinding")
