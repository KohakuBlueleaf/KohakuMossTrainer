"""No driver keeps a step's autograd graph past its backward (model-free fakes): the
StepOut that micro_steps and the pipeline drivers return carries no graph, so a tensor
held only by that graph is freed when the step's backward is done."""

import weakref
from types import SimpleNamespace

import pytest
import torch

from kohakumoss.training.drivers import common
from kohakumoss.training.drivers.common import micro_steps
from kohakumoss.training.drivers.pipeline import PipelineDriver, ThreadedDriver
from kohakumoss.types.step import StepOut
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


class _Holds(torch.autograd.Function):
    """x * w with an activation-sized tensor kept as a ctx attribute (what the graph
    alone keeps alive)."""

    @staticmethod
    def forward(ctx, x, w):
        ctx.kept = torch.ones(4096)
        ctx.save_for_backward(x)
        return x * w

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return None, (g * x).sum().reshape(1)


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _run(steps) -> tuple[StepOut, list]:
    """One micro-step over `steps` (micro_steps or a variant); (out, weakrefs to every
    tensor its graph kept)."""
    w = torch.nn.Parameter(torch.ones(1))
    refs = []

    def step(batch, denom, index):
        y = _Holds.apply(torch.randn(8), w)
        refs.append(weakref.ref(y.grad_fn.kept))
        return StepOut(y.sum(), torch.ones(()), {"m": y.detach().mean()})

    bundle = SimpleNamespace(backward=lambda loss: loss.backward())
    out = steps(step, lambda hb: hb, lambda last: _Null(), [True], bundle, [0], 1, 0)
    return out, refs


def check_graph_free(steps) -> None:
    out, refs = _run(steps)
    assert out.loss.grad_fn is None, "the returned loss still carries its graph"
    assert all(r() is None for r in refs), "the step's graph outlived its backward"


def _old_micro_steps(step, batcher, accumulate, lasts, bundle, group, denom, index):
    out = None
    for hb, last in zip(group, lasts):
        with accumulate(last):
            out = step(batcher(hb), denom, index)
            bundle.backward(out.loss)
    return out


def test_micro_steps_drop_the_graph():
    check_graph_free(micro_steps)
    must_fail(lambda: check_graph_free(_old_micro_steps), "graph kept by the StepOut")
    assert common.micro_steps is micro_steps


def _prep():
    seq = SimpleNamespace(attn_text=SimpleNamespace(lengths=[2]))
    z = torch.zeros(2, 4)
    return SimpleNamespace(
        z=z, text=z, sigma=torch.ones(1), seq=seq, target=z, weight=torch.ones(1)
    )


def _driver(cls, total):
    d = object.__new__(cls)
    d.batcher = lambda hb: hb
    d.device = torch.device("cpu")
    d.normalizer = SimpleNamespace(denominator=lambda count, device: torch.ones(()))
    d.parts = SimpleNamespace(
        trainable=SimpleNamespace(step_state=dict, set_step_state=lambda s: None)
    )
    d.micro = lambda *a, **kw: a
    engine = SimpleNamespace(step=lambda *a: total)
    d.engine, d.pipe = engine, engine
    return d


def check_pipeline_graph_free(cls) -> None:
    """The pipeline drivers' StepOut carries no graph either."""
    w = torch.nn.Parameter(torch.ones(()))
    total = w * 2.0
    step = SimpleNamespace(
        noise=lambda batch, i: _prep(),
        pre_forward=lambda sigma, seq, i: None,
        autocast=_Null,
        loss=lambda pred, p, denom: pred.sum(),
    )
    bundle = SimpleNamespace(
        scaler=SimpleNamespace(scale=1.0), observe=lambda loss: None
    )
    hb = SimpleNamespace(hw=torch.tensor([[1, 2]]), img_keep=None)
    out = _driver(cls, total).run_group(step, [hb], 1, bundle, SimpleNamespace(step=0))
    assert out.loss.grad_fn is None, cls.__name__


def _as_is(self):
    return self


def test_pipeline_drivers_drop_the_graph(monkeypatch):
    for cls in (ThreadedDriver, PipelineDriver):
        check_pipeline_graph_free(cls)
    monkeypatch.setattr(StepOut, "detached", _as_is)
    for cls in (ThreadedDriver, PipelineDriver):
        must_fail(lambda: check_pipeline_graph_free(cls), "graph kept")
