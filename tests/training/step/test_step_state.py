"""Per-micro-batch adapter step state under the pipelines, model-free: a state drawn
by `begin` is captured and installed again exactly (rank masks, DyLoRA ranks,
module-dropout skips); the threaded gpipe runs each micro-batch's backward with its own
state installed; the async threaded schedule with a recompute rejects such adapters."""

from types import SimpleNamespace

import pytest
import torch

from kohakumoss.adapters.bind.context import Draws
from kohakumoss.adapters.train import tlora
from kohakumoss.adapters.train.steps import StepPlan
from kohakumoss.parallel.pipeline.threaded import ThreadedPipeline
from kohakumoss.training.knobs import plan_problems
from kohakumoss.types.plans import MemoryPlan
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _draw(t, gen: torch.Generator) -> None:
    t.draws.rank = int(torch.randint(1, 1000, (), generator=gen))


def _plan() -> StepPlan:
    masked = SimpleNamespace(
        record=SimpleNamespace(per_token_mask=True, draw_step=None),
        axis="sample",
        draws=Draws(8),
    )
    drawn = SimpleNamespace(
        record=SimpleNamespace(per_token_mask=False, draw_step=_draw),
        axis="joint",
        draws=Draws(8),
    )
    config = SimpleNamespace(seed=3, mask_kwargs=lambda: {}, dim=8, module_dropout=0.5)
    return StepPlan([masked, drawn], config)


def _live(plan: StepPlan) -> tuple:
    masks = {g: m.clone() for g, m in tlora.saved_masks().items()}
    return (
        masks,
        [t.draws.rank for t in plan.drawn],
        [t.draws.skip for t in plan.dropped],
    )


def _same(a: tuple, b: tuple) -> None:
    assert a[1] == b[1] and a[2] == b[2], (a[1:], b[1:])
    assert a[0].keys() == b[0].keys()
    for g in a[0]:
        assert torch.equal(a[0][g], b[0][g]), g


def check_install(plan: StepPlan) -> None:
    plan.begin(torch.tensor([0.1, 0.9]), None, 5)
    first, live0 = plan.capture(), _live(plan)
    plan.begin(torch.tensor([0.8, 0.2]), None, 5)
    live1 = _live(plan)
    must_fail(lambda: _same(live0, live1), "two micro-batches drew one state")
    plan.install(first)
    _same(_live(plan), live0)
    plan.end()


def test_captured_state_installs_back(monkeypatch):
    check_install(_plan())
    monkeypatch.setattr(StepPlan, "install", lambda self, state: None)
    must_fail(lambda: check_install(_plan()), "install ignored")


class _Probe(torch.autograd.Function):
    """Identity whose backward logs the state installed when it runs."""

    @staticmethod
    def forward(ctx, x, stub):
        ctx.stub = stub
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        ctx.stub.log.append(ctx.stub.current)
        return grad, None


class _Stub:
    """What `_gpipe` reads of the pipeline: `install` and `forward`."""

    def __init__(self):
        self.current = None
        self.log: list = []

    def install(self, state) -> None:
        self.current = state

    def forward(self, x):
        return _Probe.apply(x, self)


def _one_backward(self, microbatches, loss_fn, states):
    losses = []
    for (inp, ex), state in zip(microbatches, states):
        self.install(state)
        losses.append(loss_fn(self.forward(*inp), ex))
    torch.stack(losses).sum().backward()


def check_backward_states(gpipe) -> None:
    stub = _Stub()
    x = [torch.randn(3, requires_grad=True) for _ in range(3)]
    gpipe(stub, [((v,), None) for v in x], lambda pred, ex: pred.sum(), ["a", "b", "c"])
    assert stub.log == ["c", "b", "a"], stub.log


def test_gpipe_backward_sees_its_own_state():
    check_backward_states(ThreadedPipeline._gpipe)
    must_fail(lambda: check_backward_states(_one_backward), "one backward")


def _knobs(algo: str, schedule: str, module: float = 0.0) -> dict:
    return {
        "PARALLEL": "threaded",
        "OPTIMIZER": "adamw",
        "ADAPTER_MODE": "bypass",
        "ALGO": algo,
        "PIPELINE_SCHEDULE": schedule,
        "DROPOUT": {"module": module},
    }


def _rejected(k: dict, plan: MemoryPlan) -> bool:
    return any("async threaded" in p for p in plan_problems(k, plan))


def test_async_recompute_rejects_step_state():
    ckpt = MemoryPlan(checkpoint="block")
    assert _rejected(_knobs("tlora", "async"), ckpt)
    assert _rejected(_knobs("dylora", "async"), ckpt)
    assert _rejected(_knobs("lora", "async", module=0.1), ckpt)
    assert not _rejected(_knobs("lora", "async"), ckpt)
    assert not _rejected(_knobs("tlora", "gpipe"), ckpt)
    assert not _rejected(_knobs("tlora", "async"), MemoryPlan(checkpoint="none"))
    must_fail(lambda: _yes(_rejected(_knobs("lora", "async"), ckpt)), "plain LoRA")


def _yes(flag: bool) -> None:
    assert flag
