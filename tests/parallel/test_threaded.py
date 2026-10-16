"""The single-process multi-device pipeline reproduces single-device training."""

import pytest
import torch

from kohakumoss.parallel.threaded import ThreadedPipeline
from kohakumoss.training.memory.synthetic import synthetic_batch
from tests._util.negative import must_fail
from tests.parallel._common import make_model

DENOM = 64.0


def _micro(device, seed):
    img, txt, sigma, seq, target = synthetic_batch(
        [(3, 4), (2, 2)], [4, 2], 8, "cpu", torch.float64, seed
    )
    return (img.to(device), txt.to(device), sigma.to(device), seq.to(device)), target


def _loss(pred, target):
    return (pred - target.to(pred.device)).square().sum() / DENOM


def _reference(n_mb):
    model = make_model()
    for s in range(n_mb):
        inputs, target = _micro("cpu", s)
        _loss(model(*inputs), target).backward()
    return {n: p.grad.clone() for n, p in model.named_parameters()}


def _threaded(devices, schedule, n_mb):
    model = make_model()
    pipe = ThreadedPipeline(
        model, devices, epilogue_modules=model.epilogue_modules(), schedule=schedule
    )
    pipe.step([_micro(devices[0], s) for s in range(n_mb)], _loss)
    return {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()}


def _max(a, b):
    return max((a[k] - b[k]).abs().max().item() for k in a)


@pytest.mark.cpu
@pytest.mark.parametrize("schedule", ["gpipe", "async"])
def test_threaded_cpu_matches(schedule):
    ref = _reference(3)
    assert _max(_threaded(["cpu", "cpu"], schedule, 3), ref) < 1e-10

    def check():
        assert _max(_threaded(["cpu", "cpu"], schedule, 2), ref) < 1e-10

    must_fail(check, "a dropped micro-batch")


@pytest.mark.multi_gpu
@pytest.mark.parametrize("schedule", ["gpipe", "async"])
def test_threaded_two_gpus(schedule):
    ref = _reference(3)
    assert _max(_threaded(["cuda:0", "cuda:1"], schedule, 3), ref) < 1e-6
