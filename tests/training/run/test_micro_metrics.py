"""A step's device metrics are the mean over its micro-batches (kept on the device),
not the last micro-batch's."""

from types import SimpleNamespace

import pytest
import torch

from kohakumoss.training.drivers.common import micro_steps
from kohakumoss.types.step import StepOut
from tests._util.negative import must_fail

pytestmark = [pytest.mark.cpu, pytest.mark.budget(5)]


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _run(steps, values) -> dict:
    w = torch.nn.Parameter(torch.ones(()))

    def step(batch, denom, index):
        loss = w * batch
        return StepOut(loss, torch.ones(()), {"m": torch.tensor(float(batch))})

    bundle = SimpleNamespace(backward=lambda loss: loss.backward())
    lasts = [False] * (len(values) - 1) + [True]
    out = steps(step, lambda hb: hb, lambda last: _Null(), lasts, bundle, values, 1, 0)
    return out.metrics


def _check(steps) -> None:
    got = _run(steps, [1.0, 2.0, 6.0])
    assert isinstance(got["m"], torch.Tensor)
    assert float(got["m"]) == pytest.approx(3.0)
    assert float(_run(steps, [4.0])["m"]) == pytest.approx(4.0)


def test_metrics_are_the_micro_batch_mean():
    _check(micro_steps)


def test_metrics_are_the_micro_batch_mean_negative():
    def last_only(step, batcher, accumulate, lasts, bundle, group, denom, index):
        out = None
        for hb, last in zip(group, lasts):
            with accumulate(last):
                out = step(batcher(hb), denom, index)
                bundle.backward(out.loss)
                out = out.detached()
        return out

    must_fail(lambda: _check(last_only), "the last micro-batch's metrics")
