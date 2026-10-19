"""An adapter kernel whose first call runs a cold one-time tuning far above its own
working memory (LyCORIS's tile tuning) is still timed against the body: the tiles
settle in an untimed call outside the working-memory bound."""

import pytest
import torch
from lycoris.modules import LoConModule
from torch import nn

from kohakumoss.adapters.bind import autotuned, bodies, ops
from kohakumoss_kernels.tuning import tuner
from tests._util.negative import must_fail

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="working-memory bound"),
]
GAMMA = 0.5
COLD_BYTES = 768 << 20


def _layer() -> autotuned.AutotunedOp:
    torch.manual_seed(0)
    lyco = LoConModule("l", nn.Linear(256, 192), 1.0, 8, 8.0).cuda()
    nn.init.normal_(lyco.lora_up.weight, std=0.1)
    op = autotuned.AutotunedOp(
        ops.lora_bypass_with_base(lyco, GAMMA), "triton", "tokens"
    )
    cold = [True]

    def kernel(operands):
        if cold[0]:
            cold[0] = False
            scratch = torch.empty(COLD_BYTES, dtype=torch.uint8, device="cuda")
            scratch.zero_()
        return bodies.lora_with_base(operands, GAMMA)

    op.candidates = {**op.candidates, "kernel": kernel}
    return op


def kernel_failure() -> str | None:
    """The kernel candidate's recorded failure after one resolve, None when timed."""
    op = _layer()
    x = torch.randn(1024, 256, device="cuda", requires_grad=True)
    base = torch.randn(1024, 192, device="cuda", requires_grad=True)
    op((x, base))
    key = autotuned.problem_key(op._operands((x, base)), op.op.tokens)
    entry = tuner.row(op.name, x.device, key)
    assert entry is not None, "the problem was not timed"
    return entry.get("failures", {}).get("impl=kernel")


def check_kernel_timed() -> None:
    failure = kernel_failure()
    assert failure is None, failure


@pytest.fixture
def tuning(monkeypatch, tmp_path):
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    monkeypatch.setattr(tuner, "_TABLES", {})
    monkeypatch.setattr(autotuned, "fusable", lambda operands: True)
    tuner.set_mode("table")
    tuner.reset()
    yield
    tuner.reset()


def test_cold_kernel_is_timed(tuning):
    check_kernel_timed()


def test_negative_cold_call_inside_the_bound(tuning, monkeypatch):
    monkeypatch.setattr(autotuned.AutotunedOp, "_settle_tiles", lambda *a: None)
    must_fail(check_kernel_timed, "the cold first call timed inside the bound")
