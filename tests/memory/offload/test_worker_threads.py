"""Host worker threads (gradient accumulation, CPUAdamW updates) stop when their owner
is released. Model-free; a whole training run leaving none behind is checked on the real
model (tests/memory/real/test_profile_memory.py)."""

import gc
import threading

import pytest
import torch

from kohakumoss.offload import worker
from kohakumoss.offload.optim.cpu_optim import CPUAdamW
from kohakumoss.offload.worker import HostWorker
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _workers() -> set[threading.Thread]:
    return {t for t in threading.enumerate() if t.name.startswith("kohakumoss-")}


def _no_sentinel(jobs, thread, timeout) -> None:
    return None


def _check_none(left: int) -> None:
    assert left == 0, f"{left} worker threads left"


def _released() -> int:
    """Worker threads left after a HostWorker and a device CPUAdamW are dropped."""
    before = _workers()
    host = HostWorker()
    host.put(lambda: None)
    host.join()
    owners = [host]
    if torch.cuda.is_available():
        p = torch.nn.Parameter(torch.ones(8, device="cuda"))
        owners.append(CPUAdamW([p]))
    del host, owners
    gc.collect()
    return len(_workers() - before)


def test_released_owner_stops_its_worker(monkeypatch):
    _check_none(_released())
    monkeypatch.setattr(worker, "_stop", _no_sentinel)
    must_fail(lambda: _check_none(_released()), "a worker without sentinel")
