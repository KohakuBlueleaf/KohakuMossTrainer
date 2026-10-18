"""Model-free multi-process checks through the spawn launcher: the gradient reducer sums
every rank's and micro-batch's gradient into in-place bucket views (gloo on CPU, nccl
on two GPUs), and the link sweeps run on two GPUs."""

import sys

import pytest
import torch

from kohakumoss.parallel.telemetry import selftest
from kohakumoss.types.plans import ParallelPlan
from tests._util.negative import must_fail

TOL = 1e-6


def _run(worker, devices, device_type, *args, backend="auto"):
    plan = ParallelPlan(kind="test", backend=backend, devices=tuple(devices))
    return selftest.spawn_results(worker, plan, *args, device_type=device_type)


def check_sum(results: list[dict]) -> None:
    """Every rank holds the summed gradients, as views of its bucket buffers."""
    assert all(r["max_err"] < TOL and r["views"] for r in results), results


@pytest.mark.cpu
@pytest.mark.budget(30)  # spawns its gloo rank processes
def test_reducer_sums_cpu_gloo():
    check_sum(_run(selftest.ddp_worker, [0, 1], "cpu", backend="gloo"))
    averaged = _run(selftest.ddp_worker, [0, 1], "cpu", 2, 256, True, backend="gloo")
    must_fail(lambda: check_sum(averaged), "a reducer that averages")


def _two_gpus():
    if sys.platform == "win32" or torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices on Linux")


@pytest.mark.cuda
@pytest.mark.multi_gpu
def test_reducer_sums_cuda_nccl():
    _two_gpus()
    check_sum(_run(selftest.ddp_worker, [0, 1], "cuda"))


@pytest.mark.cuda
@pytest.mark.multi_gpu
@pytest.mark.parametrize(
    "worker",
    [selftest.comm_worker, selftest.collective_worker, selftest.hop_worker],
    ids=["all_reduce", "gather_scatter", "p2p_hop"],
)
def test_link_sweeps_cuda(worker):
    _two_gpus()
    res = _run(worker, [0, 1], "cuda")
    for rank in res:
        for row in rank.values():
            assert all(v > 0 for v in row.values()), row
