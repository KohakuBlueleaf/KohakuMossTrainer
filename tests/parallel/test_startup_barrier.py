"""The store gates (rank-0-first waits and the all-ranks barrier) wait for
DIST_GATE_TIMEOUT, not DIST_TIMEOUT, the barrier waits until every rank arrived, and
the rank-0-only phases (checkpoint write, sample grid) end in a store gate, not a
collective (model-free: a recording store; three threads over one FileStore)."""

import threading
import time
from datetime import timedelta

import pytest
import torch.distributed as dist

from kohakumoss.parallel import launch
from kohakumoss.parallel.launch import startup_barrier
from kohakumoss.training.callbacks.kernels import RetuneKernels
from kohakumoss.training.callbacks.runner import MeetRanks
from kohakumoss.training.checkpoint import manager
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
WORLD = 3


def check_save_gates(monkeypatch, tmp_path, barrier) -> None:
    met: list[str] = []
    monkeypatch.setattr(manager, "ranks_meet", met.append)
    monkeypatch.setattr(manager, "_barrier", barrier)
    monkeypatch.setattr(manager, "capture_rng", lambda *a: {})
    monkeypatch.setattr(manager, "save_rng", lambda *a: None)
    ckpt = manager.CheckpointManager(tmp_path, keep_last=1)
    monkeypatch.setattr(ckpt, "_save", lambda mgr, step, parts: None)
    ckpt.save(5, {"step_rng": None})
    assert met == ["checkpoint"] * 3, met


def _collective_barrier() -> None:
    """The NCCL barrier variant (DIST_TIMEOUT): no store gate."""


def test_rank0_phases_end_in_a_store_gate(monkeypatch, tmp_path):
    check_save_gates(monkeypatch, tmp_path, manager._barrier)
    must_fail(
        lambda: check_save_gates(monkeypatch, tmp_path, _collective_barrier),
        "checkpoint save gated by a collective",
    )
    met: list[str] = []
    MeetRanks(500, "sampled", met.append).on_step_end(None, None)
    assert met == ["sampled"]


def _retune_order(cls, pending: int) -> list[str]:
    order: list[str] = []
    cb = cls(lambda what: order.append(f"meet:{what}"), anywhere=lambda n: n > 0)
    cb.pending = lambda: pending
    cb.retune = lambda: order.append("retune")
    cb.agree = lambda: order.append("agree")
    cb.on_step_end(None, None)
    return order


def check_retune_gate(cls) -> None:
    assert _retune_order(cls, 0) == ["agree"], "met with nothing to retune"
    assert _retune_order(cls, 2) == ["retune", "meet:retuned", "agree"]


class _Ungated(RetuneKernels):
    """The retune callback without its store gate before `agree`."""

    def on_step_end(self, state, out) -> None:
        self.retune()
        self.agree()


class _EveryStep(RetuneKernels):
    """The retune callback meeting on the store every step."""

    def on_step_end(self, state, out) -> None:
        self.retune()
        self.meet("retuned")
        self.agree()


def test_retune_meets_on_the_store_only_when_pending():
    check_retune_gate(RetuneKernels)
    must_fail(lambda: check_retune_gate(_Ungated), "a retune waited for in agree")
    must_fail(lambda: check_retune_gate(_EveryStep), "a store meet every step")


def check_waits_for_all(barrier, path: str) -> None:
    left: list[float] = []
    late = time.monotonic() + 0.5

    def rank(delay: float) -> None:
        store = dist.FileStore(path, WORLD)
        time.sleep(delay)
        barrier("build", WORLD, store, 10.0)
        left.append(time.monotonic())

    threads = [threading.Thread(target=rank, args=(d,)) for d in (0.0, 0.0, 0.5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(15.0)
    assert len(left) == WORLD, "a rank never left the barrier"
    assert min(left) >= late, "a rank left before the last one arrived"


def _first_arrival(name, world, store, timeout_s):
    store.set(f"kohakumoss/{name}", "done")
    store.wait([f"kohakumoss/{name}"], timedelta(seconds=timeout_s))


class _Recording:
    """A store that records each wait's timeout (every key is there at once)."""

    def __init__(self):
        self.waits: list[timedelta] = []

    def add(self, key, n):
        return 1

    def set(self, key, value):
        return None

    def wait(self, keys, timeout):
        self.waits.append(timeout)

    def get(self, key):
        return b"done"


def check_startup_timeout(rank0_gate, barrier) -> None:
    launch.set_gate_timeout(1234)
    store = _Recording()
    rank0_gate("catalog-0", store)
    barrier("data-0", 1, store)
    assert store.waits == [timedelta(seconds=1234)] * 2, store.waits


def _fixed_gate(name, store):
    store.wait([f"kohakumoss/{name}"], timedelta(seconds=launch.TIMEOUT_S))


def test_startup_gates_use_the_startup_timeout(monkeypatch):
    monkeypatch.setitem(launch._STARTUP, "timeout_s", launch.RANK0_WAIT_S)
    check_startup_timeout(launch.wait_for_rank0, startup_barrier)
    must_fail(
        lambda: check_startup_timeout(_fixed_gate, startup_barrier),
        "a startup gate on DIST_TIMEOUT",
    )
    with pytest.raises(ValueError):
        launch.set_gate_timeout(0)


def test_barrier_waits_for_every_rank(tmp_path):
    check_waits_for_all(startup_barrier, str(tmp_path / "store"))
    must_fail(
        lambda: check_waits_for_all(_first_arrival, str(tmp_path / "broken")),
        "released on the first arrival",
    )
