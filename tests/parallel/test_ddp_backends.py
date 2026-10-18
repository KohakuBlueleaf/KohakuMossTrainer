"""DDP gradient sums and backends, model-free (explicit parameters, one-weight
holders): 16-bit gradients are summed in fp32 by default (both reducers) and come back
in their parameter's dtype; DDP_BACKEND='torch' sums over ranks (not torch's mean) and
skips the sync on all micro-batches but the last; the rank-0 gates wait on the store;
the timeout / bind / backend knobs and their checks."""

import json

import pytest
import torch
import torch.distributed as dist
from torch import nn

from kohakumoss.parallel import ddp
from kohakumoss.parallel.ddp import AsyncReducer, GradReducer, wrap_torch_ddp
from kohakumoss.parallel.launch import release_ranks, wait_for_rank0
from kohakumoss.parallel.telemetry.selftest import spawn_results
from kohakumoss.training.build.parallel import _NoSync, reduce_dtype
from kohakumoss.training.knobs import plan_problems, validate
from kohakumoss.types.plans import MemoryPlan, ParallelPlan
from tests._util.negative import must_fail
from tests.training.knobs.test_knobs import runnable

TOL = 1e-6


@pytest.fixture
def group(tmp_path):
    store = dist.FileStore(str(tmp_path / "store"), 1)
    dist.init_process_group("gloo", store=store, rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def _bf16_params() -> list[nn.Parameter]:
    return [nn.Parameter(torch.ones(5, dtype=torch.bfloat16)) for _ in range(2)]


def check_wide_bucket(reducer: GradReducer, params) -> None:
    bucket = reducer.buckets[0]
    assert bucket.reduced_buffer().dtype == torch.float32, "16-bit sum"
    reducer.start(True)
    sum((p.float() * 3.0).sum() for p in params).backward()
    reducer.finish()
    for p in params:
        assert p.grad.dtype == torch.bfloat16
        assert torch.equal(p.grad.float(), torch.full((5,), 3.0))


@pytest.mark.cpu
def test_grad_reducer_sums_in_fp32(group):
    params = _bf16_params()
    check_wide_bucket(GradReducer(params), params)
    narrow = _bf16_params()
    reducer = GradReducer(narrow, reduce_dtype=reduce_dtype("param"))
    must_fail(lambda: check_wide_bucket(reducer, narrow), "parameter-dtype sum")


def check_async(reducer: AsyncReducer) -> None:
    p = nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    got = {}

    def then(q):
        got["grad"] = q.grad

    hook = reducer.reduced([p], then)
    assert reducer.slabs[0].dtype == torch.float32, "16-bit slab"
    reducer.start(True)
    p.grad = torch.full((4,), 2.0, dtype=torch.bfloat16)
    hook(p)
    reducer.finish()
    assert got["grad"].dtype == torch.bfloat16 and torch.equal(
        got["grad"].float(), torch.full((4,), 2.0)
    )


@pytest.mark.cpu
def test_async_reducer_sums_in_fp32(group):
    check_async(AsyncReducer("cpu"))
    must_fail(lambda: check_async(AsyncReducer("cpu", reduce_dtype=None)), "16-bit")
    with pytest.raises(ValueError):
        reduce_dtype("bf16")


class _Holder(nn.Module):
    """One explicit (1, 3) weight; forward is `x @ weight.T`."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.t()


def torch_ddp_worker(env, path: str, hooked: bool) -> None:
    """Two micro-batches (the first under no_sync); rank r's input is r + 1, so the
    summed gradient of every weight is world x (1 + 2) x mean input."""
    holder = _Holder()
    model = (
        wrap_torch_ddp(holder, env.device)
        if hooked
        else nn.parallel.DistributedDataParallel(holder)
    )
    accumulate = _NoSync(model)
    for last in (False, True):
        with accumulate(last):
            x = torch.full((2, 3), float(env.rank + 1))
            model(x).sum().backward()
    with open(f"{path}.{env.rank}.json", "w", encoding="utf-8") as f:
        json.dump({"grad": holder.weight.grad.tolist()}, f)


def _run_torch_ddp(hooked: bool) -> list[dict]:
    plan = ParallelPlan(kind="test", backend="gloo", devices=(0, 1))
    return spawn_results(torch_ddp_worker, plan, hooked, device_type="cpu")


def check_summed(results: list[dict]) -> None:
    want = 2 * (1.0 + 2.0) * 2  # two micro-batches x rows of 2, ranks 1 and 2 summed
    for r in results:
        for v in r["grad"][0]:
            assert abs(v - want) < TOL, r


@pytest.mark.cpu
@pytest.mark.budget(60)  # spawns two gloo rank processes twice
def test_torch_backend_sums_over_ranks():
    check_summed(_run_torch_ddp(True))
    must_fail(lambda: check_summed(_run_torch_ddp(False)), "torch's mean")


@pytest.mark.cpu
def test_rank0_gate_waits_on_the_store(tmp_path):
    store = dist.FileStore(str(tmp_path / "gate"), 1)
    release_ranks("catalog-0", store)
    wait_for_rank0("catalog-0", store, timeout_s=5.0)
    with pytest.raises(Exception):
        wait_for_rank0("catalog-1", store, timeout_s=0.5)


def _torch_problem(plan: MemoryPlan) -> bool:
    k = {
        "PARALLEL": "ddp",
        "OPTIMIZER": "adamw",
        "ADAPTER_MODE": "bypass",
        "ALGO": "lora",
        "PIPELINE_SCHEDULE": "1f1b",
        "DDP_BACKEND": "torch",
    }
    return any("DDP_BACKEND='torch'" in p for p in plan_problems(k, plan))


@pytest.mark.cpu
def test_knobs():
    assert ParallelPlan().timeout_s == 600.0 and ParallelPlan().bind == "lazy"
    assert _torch_problem(MemoryPlan(fused_step=True))
    assert _torch_problem(MemoryPlan(stream="ring", stream_resident=2))
    assert not _torch_problem(MemoryPlan())
    must_fail(lambda: _yes(_torch_problem(MemoryPlan())), "a device-gradient plan")
    bad = validate(runnable(DIST_TIMEOUT=0, DIST_BIND="eager", DDP_REDUCE_DTYPE="x"))
    for knob in ("DIST_TIMEOUT", "DIST_BIND", "DDP_REDUCE_DTYPE"):
        assert any(knob in p for p in bad), (knob, bad)
    module = validate(runnable(DDP_BACKEND="torch", DROPOUT={"module": 0.1}))
    assert any("DROPOUT['module']" in p for p in module), module
    assert ddp.MIB == 2**20


def _yes(flag: bool) -> None:
    assert flag
