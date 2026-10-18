"""Host EMA cost (model-free, explicit parameters): under DDP only rank 0 keeps an EMA;
a host EMA updates on its worker thread, a state read waits for that update, and the
driver joins it before each backward."""

import threading
from types import SimpleNamespace

import pytest
import torch

from kohakumoss.training.build.loop import build_ema_for
from kohakumoss.training.drivers.common import micro_steps
from kohakumoss.training.ema.averagers import BackgroundEMA, ConstantEMA, NoEMA
from kohakumoss.types.step import StepOut
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _built(n: int = 2):
    params = [torch.nn.Parameter(torch.full((3,), float(i))) for i in range(n)]
    named = [(f"p{i}", p) for i, p in enumerate(params)]
    return SimpleNamespace(trainable=SimpleNamespace(named_trainable=lambda: named))


def check_rank_zero_only(build) -> None:
    """Rank 1 of a DDP run allocates no EMA buffer; rank 0 and a single run do."""
    k = {"PARALLEL": "ddp", "EMA": 0.99}
    bundle = SimpleNamespace(blocks=None)
    other = build({**k}, _built(), bundle, None, 1)
    assert not getattr(
        other, "buffers", getattr(getattr(other, "inner", None), "buffers", {})
    )
    for kind, rank in (("ddp", 0), ("none", 0)):
        ema = build({**k, "PARALLEL": kind}, _built(), bundle, None, rank)
        inner = getattr(ema, "inner", ema)
        assert inner.buffers, (kind, rank)


def _every_rank(k, built, bundle, names, rank):
    return build_ema_for({**k, "PARALLEL": "none"}, built, bundle, names, 0)


def test_ddp_ema_on_rank_zero_only():
    check_rank_zero_only(build_ema_for)
    assert isinstance(
        build_ema_for(
            {"PARALLEL": "ddp", "EMA": 0.99},
            _built(),
            SimpleNamespace(blocks=None),
            None,
            1,
        ),
        NoEMA,
    )
    must_fail(lambda: check_rank_zero_only(_every_rank), "an EMA on every rank")


class _NoWaitRead(BackgroundEMA):
    def state_dict(self):
        return self.inner.state_dict()


def check_background(make) -> None:
    """The update runs on the worker (the caller returns while it is held), and a
    state read waits for it and sees the averaged values."""
    p = torch.nn.Parameter(torch.zeros(4))
    ema = make(ConstantEMA([("p", p)], (0.5,), warmup=False))
    gate, seen = threading.Event(), []

    def held():
        seen.append(threading.current_thread().name)
        gate.wait(5)

    ema.before_read = held
    with torch.no_grad():
        p.fill_(2.0)
    ema.update(0)
    assert not gate.is_set()
    got = {}
    reader = threading.Thread(target=lambda: got.update(ema.state_dict()))
    reader.start()
    reader.join(0.3)
    early = dict(got)
    gate.set()
    reader.join(5)
    assert seen and seen[0] != threading.current_thread().name, seen
    assert not early, "the state read did not wait for the update"
    assert torch.allclose(got["ema0.5/p"], torch.ones(4)), got


def test_host_ema_updates_in_the_background():
    check_background(BackgroundEMA)
    must_fail(lambda: check_background(_NoWaitRead), "a save that does not wait")


class _Recorder:
    def __init__(self):
        self.order = []

    def join(self):
        self.order.append("join")

    def backward(self, loss):
        self.order.append("backward")


def check_join_first(ema, rec: _Recorder) -> None:
    """Each backward follows a join of the background EMA."""
    out = StepOut(torch.zeros(()), torch.ones(()))
    micro_steps(
        lambda b, d, i: out,
        lambda hb: hb,
        lambda last: _Null(),
        [False, True],
        rec,
        [0, 1],
        torch.ones(()),
        0,
        ema,
    )
    assert rec.order == ["join", "backward", "join", "backward"], rec.order


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_driver_joins_before_backward():
    rec = _Recorder()
    check_join_first(rec, rec)
    other = _Recorder()
    must_fail(lambda: check_join_first(None, other), "no join before backward")
