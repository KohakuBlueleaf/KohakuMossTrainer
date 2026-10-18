"""The adapter compile candidate bound for a bucket (`CompiledBody.at`) resolves its
`CompiledOp` runner once, not the call pattern on every call; a change of autocast state
resolves a second runner."""

import pytest
import torch
from lycoris.modules import LoConModule
from torch import nn

from kohakumoss.adapters.bind import autotuned, ops
from kohakumoss_kernels.backends.compile import ops as compile_ops
from kohakumoss_kernels.backends.compile.ops import CompiledOp
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _eager_graph(body, dynamic, shapes, backend="inductor"):
    return CompiledOp(f"adapter_{body.__name__}", dynamic, "eager", eager=body)


def _counting(monkeypatch) -> list:
    calls = []
    real = compile_ops._pattern

    def pattern(args):
        calls.append(1)
        return real(args)

    monkeypatch.setattr(compile_ops, "_pattern", pattern)
    return calls


def check_resolved_once(at, monkeypatch) -> None:
    lyco = LoConModule("l", nn.Linear(16, 12), 1.0, 4, 4.0)
    op = ops.lora_bypass_with_base(lyco, 0.5)
    body = autotuned.compiled(op, "tokens")
    t = (torch.randn(8, 16), torch.randn(8, 12))
    operands = op.operands(t)
    run = at(body, operands)
    calls = _counting(monkeypatch)
    for _ in range(4):
        run(operands)
    assert len(calls) == 1, f"{len(calls)} call patterns built for one bucket"
    with torch.autocast("cpu", dtype=torch.bfloat16):
        run(operands)
    assert len(calls) == 2, "a new autocast state reused the old runner"


def _per_call(body, operands):
    """The broken variant: the generic `CompiledOp` call, its pattern every call."""
    consts = body.consts
    graph = autotuned._graph(body.body, body.mode, ())
    return lambda o: graph(o, *consts)


def test_compile_candidate_resolves_once(monkeypatch):
    monkeypatch.setattr(autotuned, "_graph", _eager_graph)
    check_resolved_once(lambda body, o: body.at(o), monkeypatch)


def test_negative_pattern_per_call(monkeypatch):
    monkeypatch.setattr(autotuned, "_graph", _eager_graph)
    must_fail(lambda: check_resolved_once(_per_call, monkeypatch), "pattern per call")
