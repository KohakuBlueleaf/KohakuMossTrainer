"""Flow noising / target and global token normalisation."""

import pytest
import torch
import torch.nn.functional as F

from kohakumoss.diffusion.objective import (
    HuberLoss,
    MSELoss,
    TokenNormalizer,
    noise_latents,
    x0_weighting,
)
from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss_kernels.backends.eager.loss import token_mse
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def test_noise_and_target_per_sample():
    seq = build_seqinfo(torch.tensor([[2, 2], [1, 3]]), torch.tensor([3, 2]))
    z0, eps = torch.randn(7, 4), torch.randn(7, 4)
    sigma = torch.tensor([0.0, 1.0])
    z, target = noise_latents(z0, eps, sigma, seq)
    assert torch.equal(z[:4], z0[:4]) and torch.allclose(z[4:], eps[4:])
    assert torch.allclose(target, eps - z0)
    sigma = torch.tensor([0.25, 0.75])
    z, _ = noise_latents(z0, eps, sigma, seq)
    assert torch.allclose(z[:4], 0.75 * z0[:4] + 0.25 * eps[:4])
    assert torch.allclose(z[4:], 0.25 * z0[4:] + 0.75 * eps[4:])


def _micro_losses(pred, target, seg, weight, splits, global_denom: bool):
    loss_fn = MSELoss(token_mse)
    norm = TokenNormalizer()
    total = torch.zeros(())
    for idx in splits:
        count = idx.numel() if not global_denom else seg.numel()
        denom = norm.denominator(count, "cpu")
        s = loss_fn(pred[idx], target[idx], seg[idx] * 0, weight[:1], None)
        total = total + norm(s, denom)
    return total


def test_global_denominator_equals_token_mean():
    pred, target = torch.randn(10, 4), torch.randn(10, 4)
    seg = torch.zeros(10, dtype=torch.int32)
    weight = torch.ones(1)
    ref = F.mse_loss(pred, target)
    splits = [torch.arange(0, 3), torch.arange(3, 10)]
    got = _micro_losses(pred, target, seg, weight, splits, global_denom=True)
    assert torch.allclose(got, ref, atol=1e-6)
    must_fail(
        lambda: _check(_micro_losses(pred, target, seg, weight, splits, False), ref),
        "local denominators",
    )


def _check(a, b):
    assert torch.allclose(a, b, atol=1e-6)


def test_world_factor_with_ddp_mean():
    pred, target = torch.randn(8, 4), torch.randn(8, 4)
    seg = torch.zeros(8, dtype=torch.int32)
    ref = F.mse_loss(pred, target)
    counts = torch.tensor([3, 5])
    norm = TokenNormalizer(lambda t: t * 0 + counts.sum(), world_factor=2)
    ranks = [slice(0, 3), slice(3, 8)]
    losses = [
        norm(
            token_mse(pred[r], target[r], seg[r], torch.ones(1), None),
            norm.denominator(1, "cpu"),
        )
        for r in ranks
    ]
    assert torch.allclose(sum(losses) / 2, ref, atol=1e-6)


def test_huber_small_error_is_half_mse_and_weighting():
    pred, target = torch.randn(6, 4) * 0.01, torch.zeros(6, 4)
    seg = torch.zeros(6, dtype=torch.int32)
    h = HuberLoss(token_mse, delta=1.0)(pred, target, seg, torch.ones(1), None)
    m = MSELoss(token_mse)(pred, target, seg, torch.ones(1), None)
    assert torch.allclose(h, 0.5 * m, rtol=1e-5)
    w = x0_weighting(torch.tensor([0.5, 2.0]), torch.ones(2))
    assert torch.allclose(w, torch.tensor([0.25, 4.0]))
