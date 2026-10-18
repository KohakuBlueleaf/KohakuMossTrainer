"""The DDP reduction of gradients that leave backward early: static buckets (reverse
order, one dtype, the byte cap), host-tensor broadcast runs, and the side-stream
AsyncReducer summing every rank's last-micro-batch gradient into the delivered views
(gloo on CPU, nccl on two GPUs)."""

import sys

import pytest
import torch

from kohakumoss.parallel.ddp import _chunks, plan_buckets
from kohakumoss.parallel.telemetry import selftest
from kohakumoss.types.plans import ParallelPlan
from tests._util.negative import must_fail

TOL = 1e-6


@pytest.mark.cpu
def test_plan_buckets_reverse_dtype_cap():
    items = [(400, "a"), (400, "a"), (100, "b"), (1000, "b"), (50, "b")]
    want = [[4, 3], [2], [1, 0]]
    assert plan_buckets(items, 512) == want
    forward = [[0, 1], [2, 3], [4]]
    must_fail(lambda: _same(forward, want), "buckets in registration order")
    assert plan_buckets([], 512) == []


def _same(got, want) -> None:
    assert got == want, (got, want)


@pytest.mark.cpu
def test_broadcast_chunks_keep_order_and_cap():
    ts = [torch.zeros(n) for n in (10, 30, 70, 5, 100)]
    runs = _chunks(ts, 80 * 4)
    assert [[t.numel() for t in run] for run in runs] == [[10, 30], [70, 5], [100]]
    assert [t for run in runs for t in run] == ts
    assert _chunks(ts, None) == [ts]


def _run(worker, devices, device_type, *args, backend="auto"):
    plan = ParallelPlan(kind="test", backend=backend, devices=tuple(devices))
    return selftest.spawn_results(worker, plan, *args, device_type=device_type)


def check_sum(results: list[dict]) -> None:
    """Every rank received the summed gradients, shaped like their parameters, from
    several buckets."""
    assert all(r["max_err"] < TOL and r["views"] for r in results), results
    assert all(r["buckets"] > 1 for r in results), results


@pytest.mark.cpu
@pytest.mark.budget(30)  # spawns its gloo rank processes
def test_async_reducer_sums_cpu_gloo():
    check_sum(_run(selftest.async_worker, [0, 1], "cpu", backend="gloo"))
    local = _run(selftest.async_worker, [0, 1], "cpu", 2, 256, True, backend="gloo")
    must_fail(lambda: check_sum(local), "a reducer that does not reduce")


@pytest.mark.cuda
@pytest.mark.multi_gpu
def test_async_reducer_sums_cuda_nccl():
    if sys.platform == "win32" or torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices on Linux")
    check_sum(_run(selftest.async_worker, [0, 1], "cuda"))
