"""The candidate an autotuned adapter op binds in a checkpointed forward is the one its
recompute binds, when another layer with the same problem settles that problem to the
other candidate between the forward and the recompute."""

import pytest
import torch
from lycoris.modules import LoConModule
from torch import nn
from torch.utils.checkpoint import checkpoint

from kohakumoss.adapters.bind import autotuned, bodies, ops
from kohakumoss_kernels.tuning import tuner
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
GAMMA = 0.5


def _layer(seed: int) -> autotuned.AutotunedOp:
    torch.manual_seed(seed)
    lyco = LoConModule("l", nn.Linear(64, 48), 1.0, 4, 4.0)
    nn.init.normal_(lyco.lora_up.weight, std=0.1)
    return autotuned.AutotunedOp(
        ops.lora_bypass_with_base(lyco, GAMMA), "triton", "tokens"
    )


def _recording(op: autotuned.AutotunedOp) -> list[str]:
    """Replace `op`'s candidates by the eager body, logging which candidate ran."""
    picks: list[str] = []

    def candidate(name):
        def run(operands):
            picks.append(name)
            return bodies.lora_with_base(operands, GAMMA)

        return run

    op.candidates = {name: candidate(name) for name in op.candidates}
    return picks


def _inputs() -> tuple:
    x = torch.randn(300, 64, requires_grad=True)
    return x, torch.randn(300, 48, requires_grad=True)


def check_recompute_binds_the_forward_pick() -> None:
    tuner.reset()
    checkpointed, outside = _layer(0), _layer(1)
    picks = _recording(checkpointed)
    timed = _recording(outside)

    def measure(fn, device):
        fn()
        return 2.0 if timed.pop() == "kernel" else 1.0

    tuner.TIMINGS["burst"] = measure
    x, base = _inputs()
    out = checkpoint(lambda a, b: checkpointed((a, b)), x, base, use_reentrant=False)
    forward = list(picks)
    outside(_inputs())
    assert tuner.decided(outside.name, torch.device("cpu"), _key(outside))
    out.square().sum().backward()
    recompute = picks[len(forward) :]
    assert forward == ["kernel"], forward
    assert recompute == forward, f"forward ran {forward}, its recompute {recompute}"


def _key(op: autotuned.AutotunedOp) -> str:
    x, base = _inputs()
    return autotuned.problem_key(op._operands((x, base)), op.op.tokens)


@pytest.fixture
def tuning(monkeypatch, tmp_path):
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    monkeypatch.setattr(tuner, "_TABLES", {})
    monkeypatch.setattr(autotuned, "fusable", lambda operands: True)
    monkeypatch.setitem(tuner.TIMINGS, "burst", tuner.TIMINGS["burst"])
    tuner.set_mode("table")
    yield
    tuner.reset()


def test_recompute_binds_the_forward_pick(tuning):
    check_recompute_binds_the_forward_pick()


def test_negative_unbound_provisional_pick(tuning, monkeypatch):
    monkeypatch.setattr(autotuned, "_recomputed", lambda: False)
    must_fail(check_recompute_binds_the_forward_pick, "provisional pick left unbound")
