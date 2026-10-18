"""EMA of 16-bit parameters converts them chunk by chunk: the update equals the
reference lerp and no foreach call sees more than `chunk_numel` converted elements."""

import pytest
import torch

from kohakumoss.training.ema import averagers
from kohakumoss.training.ema.averagers import ConstantEMA
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _named():
    g = torch.Generator().manual_seed(0)
    out = []
    for i, n in enumerate((100, 300, 50, 400, 10)):
        dtype = torch.float32 if i == 2 else torch.bfloat16
        out.append((f"p{i}", torch.nn.Parameter(torch.randn(n, generator=g).to(dtype))))
    return out


@pytest.fixture
def calls(monkeypatch):
    seen = []
    real = torch._foreach_lerp_

    def spy(bufs, src, weight):
        seen.append(sum(s.numel() for s in src))
        return real(bufs, src, weight)

    monkeypatch.setattr(averagers.torch, "_foreach_lerp_", spy)
    return seen


def _reference(named, decay: float) -> list[torch.Tensor]:
    bufs = [p.detach().float().clone() for _, p in named]
    for _, p in named:
        p.data.add_(1.0)
    return [b.lerp(p.detach().float(), 1 - decay) for b, (_, p) in zip(bufs, named)]


def _bounded(calls: list[int], limit: int) -> None:
    assert calls and max(calls) <= limit, calls


def test_chunked_update_matches_and_is_bounded(calls):
    named = _named()
    ema = ConstantEMA(named, (0.9,), warmup=False, chunk_numel=420)
    want = _reference(named, 0.9)
    calls.clear()
    ema.update(0)
    for got, ref in zip(ema.buffers["ema0.9"], want):
        torch.testing.assert_close(got, ref)
    assert calls[0] == 50  # the fp32 parameter, lerped in place
    _bounded(calls[1:], 420)
    one_shot = [sum(p.numel() for _, p in named if p.dtype != torch.float32)]
    must_fail(lambda: _bounded(one_shot, 420), "all 16-bit params converted at once")
