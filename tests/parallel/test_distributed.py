"""Multi-process checks via the spawn launcher: DDP reducer, pipeline schedules, FSDP2."""

import os
import tempfile

import pytest
import torch

from kohakumoss.parallel import selftest
from kohakumoss.parallel.launch import spawn
from kohakumoss.types.plans import ParallelPlan


def _run(worker, devices, device_type, *args, backend="auto"):
    path = os.path.join(tempfile.mkdtemp(), "res")
    plan = ParallelPlan(kind="test", backend=backend, devices=tuple(devices))
    spawn(worker, plan, path, *args, device_type=device_type)
    return selftest.read_results(path, len(devices))


@pytest.mark.cpu
def test_ddp_reducer_cpu_gloo():
    res = _run(selftest.ddp_worker, [0, 1], "cpu", backend="gloo")
    assert all(r["max_err"] < 1e-12 for r in res), res


@pytest.mark.cpu
@pytest.mark.parametrize(
    "schedule", ["gpipe", "1f1b", "interleaved-1f1b", "looped-bfs", "zbv"]
)
def test_pipeline_cpu_gloo(schedule):
    res = _run(selftest.pipeline_worker, [0, 1], "cpu", schedule, backend="gloo")
    assert all(r["max_err"] < 1e-10 and r["n"] > 0 for r in res), res
    losses = [r["loss_err"] for r in res if "loss_err" in r]
    assert len(losses) == 1 and losses[0] < 1e-12, res


@pytest.mark.multi_gpu
def test_ddp_reducer_cuda():
    res = _run(selftest.ddp_worker, [0, 1], "cuda")
    assert all(r["max_err"] < 1e-10 for r in res), res


@pytest.mark.multi_gpu
@pytest.mark.parametrize("schedule", ["gpipe", "1f1b"])
def test_pipeline_cuda(schedule):
    res = _run(selftest.pipeline_worker, [0, 1], "cuda", schedule)
    assert all(r["max_err"] < 1e-6 for r in res), res


@pytest.mark.multi_gpu
def test_fsdp2_cuda():
    if not torch.cuda.is_available():
        pytest.skip("cuda")
    res = _run(selftest.fsdp_worker, [0, 1], "cuda")
    assert all(r["max_err"] < 1e-5 for r in res), res
