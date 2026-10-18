"""MEMORY_PROFILE='auto' under a process group: rank 0 plans, every rank resolves rank
0's profile, so ranks with different free memory build the same plan (model-free: two
threads as ranks over one FileStore)."""

import threading

import pytest
import torch
import torch.distributed as dist

from kohakumoss.memory.profiles import resolve
from kohakumoss.parallel import launch
from kohakumoss.training.build import model
from tests._util.negative import must_fail
from tests.training.knobs.test_knobs import runnable

pytestmark = pytest.mark.cpu
CHOICE = {0: "balanced", 1: "low"}  # what each rank's own free memory would pick


def _ranks_plan(monkeypatch, path: str, agree) -> dict:
    local = threading.local()
    monkeypatch.setattr(model, "dist_rank", lambda: (local.rank, 2))
    monkeypatch.setattr(model, "gate_name", lambda what: what)
    monkeypatch.setattr(model, "rank0_value", agree(local))
    monkeypatch.setattr(
        model,
        "auto_profile",
        lambda *a: (resolve(CHOICE[local.rank], device="cpu"), []),
    )
    out = {}

    def rank(r: int) -> None:
        local.rank = r
        local.store = dist.FileStore(path, 2)
        plan, _ = model.resolve_memory(
            runnable(MEMORY_PROFILE="auto"), torch.device("cpu")
        )
        out[r] = plan.profile

    threads = [threading.Thread(target=rank, args=(r,)) for r in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30.0)
    return out


def _through_store(local):
    return lambda n, c, r, w: launch.rank0_value(n, c, r, w, local.store, 10.0)


def _per_rank(local):
    return lambda n, c, r, w: c()


def check_agree(monkeypatch, path: str, agree) -> None:
    out = _ranks_plan(monkeypatch, path, agree)
    assert out == {0: "balanced", 1: "balanced"}, out


def test_ranks_resolve_rank0s_profile(monkeypatch, tmp_path):
    check_agree(monkeypatch, str(tmp_path / "store"), _through_store)
    must_fail(
        lambda: check_agree(monkeypatch, str(tmp_path / "per"), _per_rank),
        "each rank planned from its own free memory",
    )
