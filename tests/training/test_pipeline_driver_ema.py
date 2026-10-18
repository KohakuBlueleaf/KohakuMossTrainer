"""The pipeline drivers take the EMA NativeDriver.run hands them and join a pending
background EMA update before the engine / pipe step (model-free fakes); host
gradients of a non-finite report carry their parameter names."""

import inspect
import itertools
from types import SimpleNamespace

import pytest
import torch

from kohakumoss.training.drivers.pipeline import PipelineDriver, ThreadedDriver
from kohakumoss.training.memory.optimizers import host_grad_names
from kohakumoss.training.optim.bundle.base import Bundle
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


class _OldThreaded(ThreadedDriver):
    def run_group(self, step, group, count, bundle, state):
        return None


def check_binds(cls) -> None:
    """`run_group` binds the arguments NativeDriver.run passes (ema included)."""
    try:
        inspect.signature(cls.run_group).bind(None, 1, [], 1, None, None, None)
    except TypeError as err:
        raise AssertionError(f"{cls.__name__}.run_group: {err}") from err


def test_run_group_takes_the_ema():
    for cls in (ThreadedDriver, PipelineDriver):
        check_binds(cls)
    must_fail(lambda: check_binds(_OldThreaded), "the signature without ema")


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _prep():
    seq = SimpleNamespace(attn_text=SimpleNamespace(lengths=[2]))
    z = torch.zeros(2, 4)
    return SimpleNamespace(
        z=z, text=z, sigma=torch.ones(1), seq=seq, target=z, weight=torch.ones(1)
    )


def _driver(cls, order: list):
    d = object.__new__(cls)
    d.batcher = lambda hb: hb
    d.device = torch.device("cpu")
    d.normalizer = SimpleNamespace(denominator=lambda count, device: torch.ones(()))
    d.parts = SimpleNamespace(
        trainable=SimpleNamespace(step_state=dict, set_step_state=lambda s: None)
    )
    d.micro = lambda *a, **kw: a
    engine = SimpleNamespace(step=lambda *a: order.append("step") or torch.zeros(()))
    d.engine, d.pipe = engine, engine
    return d


def check_join_first(cls, ema, order: list) -> None:
    """The pending EMA update is joined before the engine / pipe step."""
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
    _driver(cls, order).run_group(step, [hb], 1, bundle, SimpleNamespace(step=0), ema)
    assert order == ["join", "step"], order


def test_pending_ema_joined_before_the_step():
    for cls in (ThreadedDriver, PipelineDriver):
        order: list = []
        ema = SimpleNamespace(join=lambda: order.append("join"))
        check_join_first(cls, ema, order)
        late: list = []
        must_fail(lambda: check_join_first(cls, None, late), "no join before the step")


def _report(names) -> list[str]:
    bundle = object.__new__(Bundle)
    bundle._grad_names = names
    grads = [
        torch.tensor([float("nan")]),
        torch.tensor([1.0]),
        torch.tensor([float("inf")]),
    ]
    return [r["param"] for r in bundle._host_report(grads)]


def check_named(names) -> None:
    """Non-finite host gradients are reported under their parameter names."""
    assert _report(names) == ["block.0.w", "block.2.w"], _report(names)


def test_host_gradients_carry_parameter_names():
    a, b = torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))
    m = torch.nn.Parameter(torch.zeros(1))
    m.grad = torch.zeros(1)
    grads = SimpleNamespace(filled_params=lambda: [a, b])
    by_id = {id(a): "block.0.w", id(b): "block.1.w", id(m): "block.2.w"}
    names = host_grad_names(grads, [m], by_id)
    assert names == ["block.0.w", "block.1.w", "block.2.w"]
    check_named(lambda: list(names))
    must_fail(lambda: check_named(None), "host gradients named by index")
    assert list(itertools.islice(_report(None), 1)) == ["host_grad[0]"]
