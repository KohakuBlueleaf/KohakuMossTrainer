"""Compiled ops under activation checkpointing: a checkpointed forward and its recompute
run the same compiled graph, even when the op's other call patterns (dtype x
requires_grad x grad mode) exceed Dynamo's recompile limit in between (Dynamo then
stops running the op's cached graphs and runs it eagerly). A call pattern that needs
more than GRAPH_LIMIT graphs runs the eager reference from then on and never raises."""

import itertools

import pytest
import torch
import torch._dynamo
from torch.utils.checkpoint import CheckpointError, checkpoint

from kohakumoss_kernels.backends.compile import ops
from kohakumoss_kernels.backends.eager import norm
from tests._util.error_stats import assert_no_regression
from tests._util.negative import must_fail

DEV = "cuda" if torch.cuda.is_available() else "cpu"
pytestmark = [pytest.mark.slow, pytest.mark.cuda if DEV == "cuda" else pytest.mark.cpu]
DYNAMO_DEFAULT_LIMIT = 8


def _inputs(n: int, dtype, grads):
    g = torch.Generator().manual_seed(n)
    ts = [torch.randn(s, 64, generator=g).to(DEV, dtype) for s in (n, n, 2)]
    for t, rg in zip(ts, grads):
        t.requires_grad_(rg)
    seg = (torch.arange(n) % 2).to(DEV)
    return (*ts, seg)


def _exhaust(op) -> None:
    """More call patterns than Dynamo's default recompile limit."""
    for dtype in (torch.float32, torch.bfloat16):
        for grads in itertools.product((False, True), repeat=3):
            x, o, gate, seg = _inputs(40, dtype, grads)
            op(x, o, gate, seg, 0.5)
            with torch.no_grad():
                op(x, o, gate, seg, 0.5)


def _step(op) -> tuple:
    """Grads of a checkpointed bf16 call whose recompute comes after `_exhaust`."""
    x, o, gate, seg = _inputs(48, torch.bfloat16, (True, True, False))
    op(x, o, gate, seg, 1.0).float().sum().backward()
    x.grad = o.grad = None
    out = checkpoint(lambda a, b: op(a, b, gate, seg, 1.0), x, o, use_reentrant=False)
    _exhaust(op)
    try:
        out.float().square().sum().backward()
    except CheckpointError as err:
        raise AssertionError(f"recompute differs from the forward: {err}") from err
    return x.grad, o.grad


class _GraphRuns:
    """A Dynamo backend that runs the captured FX graph as is and counts its runs."""

    def __init__(self):
        self.count = 0

    def __call__(self, gm, example_inputs):
        def run(*args):
            self.count += 1
            return gm.forward(*args)

        return run


def _recompute_state_runs_a_graph(op, runs: _GraphRuns) -> None:
    """The forward's call, then the op's other patterns, then the same call under the
    global state of the autograd thread a checkpoint recompute runs on (another
    intra-op thread count): that call must run a compiled graph, not eager."""
    x, o, gate, seg = _inputs(48, torch.bfloat16, (True, False, False))
    op(x, o, gate, seg, 1.0)
    _exhaust(op)
    threads = torch.get_num_threads()
    torch.set_num_threads(max(1, threads // 2) if threads > 1 else 2)
    try:
        before = runs.count
        op(x, o, gate, seg, 1.0)
    finally:
        torch.set_num_threads(threads)
    assert runs.count == before + 1, "the recompute-state call ran eagerly"


def _reference(dtype) -> tuple:
    """Grads of the eager reference on the `_step` inputs cast to `dtype`."""
    x, o, gate, seg = _inputs(48, torch.bfloat16, (False, False, False))
    x, o = (t.to(dtype).requires_grad_() for t in (x, o))
    norm.gated_residual(
        x, o, gate.to(dtype), seg, 1.0
    ).float().square().sum().backward()
    return x.grad, o.grad


class _OneCache:
    """The broken variant: one torch.compile (one Dynamo cache) for every pattern."""

    def __init__(self, backend):
        self.run = torch.compile(
            ops._fresh(norm.gated_residual, "one_cache"), dynamic=True, backend=backend
        )

    def __call__(self, *args):
        return self.run(*args)


def test_checkpointed_call_matches_eager():
    torch._dynamo.reset()
    got = _step(ops.CompiledOp("gated_residual", True, "inductor"))
    eager, ref = _reference(torch.bfloat16), _reference(torch.float64)
    for name, g, e, r in zip(("dx", "do"), got, eager, ref):
        assert_no_regression(g, e, r, f"checkpointed compiled {name}", "compiled")


def test_recompute_state_never_falls_back_to_eager():
    torch._dynamo.reset()
    runs = _GraphRuns()
    _recompute_state_runs_a_graph(ops.CompiledOp("gated_residual", True, runs), runs)

    def broken():
        torch._dynamo.reset()
        broken_runs = _GraphRuns()
        with torch._dynamo.config.patch(recompile_limit=DYNAMO_DEFAULT_LIMIT):
            _recompute_state_runs_a_graph(_OneCache(broken_runs), broken_runs)

    must_fail(broken, "one Dynamo cache past its recompile limit")


def _two_shapes(op) -> None:
    for n in (40, 56):
        x, o, gate, seg = _inputs(n, torch.float32, (False, False, False))
        op(x, o, gate, seg, 1.0)


class _Compiles:
    """A Dynamo backend that runs the captured FX graph as is and counts its graphs."""

    def __init__(self):
        self.graphs = 0

    def __call__(self, gm, example_inputs):
        self.graphs += 1
        return gm.forward


def check_limit_runs_eager(monkeypatch, limit: int = 2) -> None:
    """Static shapes, more token counts than GRAPH_LIMIT allows graphs for one call
    pattern: nothing raises, the pattern runs the eager reference from the limit on
    (right results), and no graph is built past it."""
    torch._dynamo.reset()
    monkeypatch.setattr(ops, "GRAPH_LIMIT", limit)
    backend = _Compiles()
    op = ops.CompiledOp("gated_residual", False, backend)
    for n in range(40, 40 + 8 * (limit + 3), 8):
        x, o, gate, seg = _inputs(n, torch.float32, (False, False, False))
        try:
            out = op(x, o, gate, seg, 1.0)
        except ops.GraphLimitError as err:
            raise AssertionError(f"a production call raised: {err}") from err
        torch.testing.assert_close(out, norm.gated_residual(x, o, gate, seg, 1.0))
    assert backend.graphs <= limit + 1, backend.graphs
    assert op.limited, "the pattern was never bound to the eager reference"


def _raising_limit(self, key, err, args):
    """The broken variant: the limit raises out of the call."""
    raise err


def test_graph_limit_runs_eager(monkeypatch):
    check_limit_runs_eager(monkeypatch)
    monkeypatch.setattr(ops.CompiledOp, "_limit", _raising_limit)
    must_fail(lambda: check_limit_runs_eager(monkeypatch), "the limit raises")
