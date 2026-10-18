"""A checkpointed block's recompute runs under its forward's intra-op thread count, so a
compiled op inside it reruns the forward's call pattern and graph (none traced in the
recompute) when the backward runs at another count, as on the autograd device threads.
"""

import contextlib

import pytest
import torch
import torch._dynamo

from kohakumoss.adapters.bind import bodies
from kohakumoss.memory import runners
from kohakumoss_kernels.backends.compile.ops import CompiledOp
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
FORWARD, BACKWARD = 2, 1


class _Graphs:
    """A Dynamo backend running the captured graph as is."""

    def __call__(self, gm, example_inputs):
        return gm.forward


def _step(mode: str) -> tuple[list[int], int]:
    """(intra-op thread counts the block ran at, call patterns compiled) of one
    checkpointed forward at FORWARD threads and its backward at BACKWARD threads."""
    torch._dynamo.reset()
    op = CompiledOp("adapter_threads", None, _Graphs(), eager=bodies.lora_with_base)
    seen: list[int] = []

    def block(x, ctx):
        seen.append(torch.get_num_threads())
        return op((x, *ctx), 1.0)

    runner = runners.CheckpointRunner(mode).bind([block])
    g = torch.Generator().manual_seed(0)
    x = torch.randn(48, 32, generator=g, requires_grad=True)
    d = torch.randn(4, 32, generator=g, requires_grad=True)
    u = torch.randn(16, 4, generator=g, requires_grad=True)
    base = torch.randn(48, 16, generator=g)
    before = torch.get_num_threads()
    try:
        torch.set_num_threads(FORWARD)
        out = runner.run(0, block, x, (d, u, base))
        torch.set_num_threads(BACKWARD)
        out.square().sum().backward()
        after = torch.get_num_threads()
    finally:
        torch.set_num_threads(before)
    assert after == BACKWARD, "the recompute did not restore the caller's count"
    return seen, len(op.compiled)


def check_forward_threads(mode: str) -> None:
    seen, patterns = _step(mode)
    assert seen == [FORWARD, FORWARD], f"forward / recompute thread counts {seen}"
    assert patterns == 1, f"{patterns} call patterns compiled: the recompute traced one"


def test_recompute_runs_at_the_forward_thread_count():
    for mode in ("block", "selective"):
        check_forward_threads(mode)


def test_negative_recompute_at_the_backward_thread_count(monkeypatch):
    def plain(inner=None):
        return lambda: (contextlib.nullcontext(), contextlib.nullcontext())

    monkeypatch.setattr(runners, "forward_state", plain)
    must_fail(lambda: check_forward_threads("block"), "recompute at the caller's count")
