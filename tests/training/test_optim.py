"""AdamW16 vs torch AdamW, rounding writers, Muon split + orthogonalisation."""

import pytest
import torch

from kohakumoss.training.optim import (
    AdamW16,
    MuonHybrid,
    newton_schulz,
    schedule_config,
)
from kohakumoss.training.optim.rounding import (
    write_kahan,
    write_nearest,
    write_stochastic,
)
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _train(opt_cls, steps=20, **kw):
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(16, 8))
    opt = opt_cls([p], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, **kw)
    g = torch.Generator().manual_seed(1)
    for _ in range(steps):
        p.grad = torch.randn(16, 8, generator=g)
        opt.step()
    return p.detach()


def test_adamw16_fp32_matches_torch_adamw():
    ref = _train(torch.optim.AdamW)
    got = _train(AdamW16)
    assert torch.allclose(got, ref, atol=1e-6, rtol=1e-5)


def test_muon_adam_groups_match_torch_adamw():
    ref = _train(torch.optim.AdamW)
    got = _train(MuonHybrid)
    assert torch.allclose(got, ref, atol=1e-6, rtol=1e-5)


def test_stochastic_rounding_is_unbiased_and_nearest_stalls():
    base = torch.ones(200_000, dtype=torch.bfloat16)
    target = torch.full((200_000,), 1.0 - 1e-3)
    p = base.clone()
    write_stochastic(p, target, None)
    assert abs(p.float().mean().item() - (1.0 - 1e-3)) < 1e-4
    q = base.clone()
    write_nearest(q, torch.full((200_000,), 1.0 - 1e-4), None)
    assert torch.equal(q, base)


def test_kahan_accumulates_tiny_updates():
    p = torch.ones(1024, dtype=torch.bfloat16)
    comp = torch.zeros_like(p)
    for _ in range(100):
        write_kahan(p, p.float() - 1e-4, comp)
    assert abs((p.float() + comp.float()).mean().item() - 0.99) < 1e-4


def test_newton_schulz_orthogonalises():
    g = torch.randn(64, 32, dtype=torch.float64)
    x = newton_schulz(g.float(), steps=5).double()
    sv = torch.linalg.svdvals(x)
    assert sv.min() > 0.5 and sv.max() < 1.3


def test_muon_splits_fused_matrix_per_chunk():
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.zeros(3 * 16, 16))
    grad = torch.randn(3 * 16, 16)
    opt = MuonHybrid(
        [{"params": [w], "use_muon": True, "splits": [3]}],
        lr=1.0,
        momentum=0.0,
        nesterov=False,
    )
    w.grad = grad.clone()
    opt.step()
    scale = 0.2 * (16**0.5)
    expect = -scale * torch.cat([newton_schulz(c) for c in grad.chunk(3, 0)], 0)
    assert torch.allclose(w.detach(), expect, atol=1e-5)

    def broken():
        unsplit = -0.2 * (48**0.5) * newton_schulz(grad)
        assert torch.allclose(w.detach(), unsplit, atol=1e-5)

    must_fail(broken, "fused matrix orthogonalised as one")


def test_schedule_config_fills_end():
    cfg = schedule_config({"mode": "cosine", "warmup": 10, "end": -1}, 500)
    assert cfg == {"lr": {"mode": "cosine", "warmup": 10, "end": 500, "value": 1.0}}
