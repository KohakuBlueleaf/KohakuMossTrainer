"""The adapter compile candidate under activation checkpointing: a checkpointed forward
and its recompute run graphs that save the same tensors, whatever layer shapes, token
counts and grad patterns run in between (the threaded pipeline runs every micro-batch's
forward before any recompute); each compile cache holds one graph."""

import functools
import itertools

import pytest
import torch
import torch._dynamo
from torch.utils.checkpoint import CheckpointError, checkpoint

from kohakumoss.adapters.bind import autotuned, bodies
from kohakumoss_kernels.backends.compile.ops import CompiledOp
from tests._util.negative import must_fail

DEV = "cuda" if torch.cuda.is_available() else "cpu"
pytestmark = [pytest.mark.slow, pytest.mark.cuda if DEV == "cuda" else pytest.mark.cpu]
SHAPES = ((64, 64), (64, 96), (96, 64))
RANK = 4
OP = autotuned.Op(
    "lora_with_base", None, None, None, bodies.lora_with_base, (1.0,), tokens=(0, 3)
)


def _ops(n: int, din: int, dout: int, grads) -> tuple:
    g = torch.Generator().manual_seed(n + din + dout)
    x = torch.randn(n, din, generator=g)
    d = torch.randn(RANK, din, generator=g) * 0.1
    u = torch.randn(dout, RANK, generator=g) * 0.1
    base = torch.randn(n, dout, generator=g)
    ts = [t.to(DEV, torch.bfloat16) for t in (x, d, u, base)]
    for t, rg in zip(ts, grads):
        t.requires_grad_(rg)
    return tuple(ts)


def _between(run) -> None:
    """Other layers' shapes, token counts and grad patterns, as other micro-batches."""
    for n, (din, dout), grads in itertools.product(
        (40, 72), SHAPES, ((False, True, True, False), (True, True, True, True))
    ):
        ops = _ops(n, din, dout, grads)
        run(ops)
        with torch.no_grad():
            run(ops)


def _step(run) -> None:
    x, d, u, base = _ops(48, 64, 64, (True, True, True, True))
    out = checkpoint(
        lambda a, b, c, e: run((a, b, c, e)), x, d, u, base, use_reentrant=False
    )
    _between(run)
    try:
        out.float().square().sum().backward()
    except CheckpointError as err:
        raise AssertionError(f"recompute differs from the forward: {err}") from err
    assert d.grad is not None and torch.isfinite(d.grad.float()).all()


class _Graphs:
    """A Dynamo backend running the captured graph as is, counting the graphs."""

    def __init__(self):
        self.count = 0

    def __call__(self, gm, example_inputs):
        self.count += 1
        return gm.forward


def _counting(monkeypatch) -> list:
    made: list = []

    @functools.cache
    def graph(body, dynamic, shapes):
        op = CompiledOp(f"adapter_{body.__name__}", dynamic, _Graphs(), eager=body)
        made.append(op)
        return op

    monkeypatch.setattr(autotuned, "_graph", graph)
    return made


def check_one_graph_per_cache(made: list) -> None:
    for op in made:
        assert op.backend.count == len(
            op.compiled
        ), f"{op.backend.count} graphs over {len(op.compiled)} call patterns"


def test_checkpointed_body_recomputes_the_same_graph():
    torch._dynamo.reset()
    autotuned._graph.cache_clear()
    _step(autotuned.compiled(OP, "tokens"))


def test_each_cache_holds_one_graph(monkeypatch):
    torch._dynamo.reset()
    made = _counting(monkeypatch)
    _between(autotuned.compiled(OP, "tokens"))
    _between(autotuned.compiled(OP, "tokens"))
    check_one_graph_per_cache(made)


def test_negative_one_cache_across_shapes(monkeypatch):
    torch._dynamo.reset()
    shared = CompiledOp("adapter_shared", None, _Graphs(), eager=bodies.lora_with_base)
    monkeypatch.setattr(autotuned, "_graph", lambda body, dynamic, shapes: shared)
    _between(autotuned.compiled(OP, "tokens"))
    must_fail(lambda: check_one_graph_per_cache([shared]), "one cache, several graphs")
