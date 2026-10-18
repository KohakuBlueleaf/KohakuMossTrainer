"""An EMA with several decays converts each 16-bit parameter once per update and
averages every decay from that copy (model-free: explicit parameters)."""

import pytest
import torch

from kohakumoss.training.ema import averagers
from kohakumoss.training.ema.averagers import ConstantEMA
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
DECAYS = (0.9, 0.99, 0.999)


def _named():
    g = torch.Generator().manual_seed(0)
    return [
        (f"p{i}", torch.nn.Parameter(torch.randn(n, generator=g).bfloat16()))
        for i, n in enumerate((30, 50, 20))
    ]


def _per_tag(self, weights) -> None:
    """The update before: every decay converts every parameter again."""
    for tag, w in zip(self.tags, weights):
        bufs = self.buffers[tag]
        for chunk in self.converted:
            src = [
                self.params[i].detach().to(bufs[i].device, torch.float32) for i in chunk
            ]
            torch._foreach_lerp_([bufs[i] for i in chunk], src, w)


def check_once(monkeypatch_ctx) -> None:
    named = _named()
    ema = ConstantEMA(named, DECAYS, warmup=False)
    for _, p in named:
        p.data.add_(1.0)
    want = {
        tag: [
            b.clone().lerp(p.detach().float(), 1 - d)
            for b, (_, p) in zip(ema.buffers[tag], named)
        ]
        for tag, d in zip(ema.tags, DECAYS)
    }
    seen = []
    real = torch.Tensor.to

    def spy(self, *args, **kwargs):
        if self.dtype == torch.bfloat16:
            seen.append(1)
        return real(self, *args, **kwargs)

    with monkeypatch_ctx() as m:
        m.setattr(torch.Tensor, "to", spy)
        ema.update(0)
    for tag in ema.tags:
        for got, ref in zip(ema.buffers[tag], want[tag]):
            torch.testing.assert_close(got, ref)
    assert len(seen) == len(named), f"{len(seen)} conversions for {len(named)}"


def test_one_conversion_for_every_decay(monkeypatch):
    check_once(monkeypatch.context)
    monkeypatch.setattr(averagers._BufferEMA, "_lerp_all", _per_tag)
    must_fail(lambda: check_once(monkeypatch.context), "a conversion per decay")
