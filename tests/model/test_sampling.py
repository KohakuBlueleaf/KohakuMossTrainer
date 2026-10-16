"""Sampler: schedule equals the official grid snap; Euler + CFG on the tiny model."""

import pytest
import torch

from kohakumoss.diffusion.sampling import moss_sigmas, sample_latents
from kohakumoss.diffusion.sampling.sample import unpack

pytestmark = pytest.mark.cpu


def _official_sigmas(steps, shift, total=1000, training_shift=1.0):
    def grid(count, s):
        lower = s / (total + s - 1)
        values = torch.linspace(1, lower, count)
        return s * values / (1 + (s - 1) * values)

    sigmas = grid(steps, shift)
    training = grid(total, training_shift).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where(
        (sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(),
        lower,
        upper,
    )
    return torch.unique_consecutive(training[nearest])


@pytest.mark.parametrize("steps, shift", [(50, 5.0), (28, 3.0), (8, 1.0)])
def test_schedule_matches_official(steps, shift):
    ours = moss_sigmas(steps, shift)
    ref = _official_sigmas(steps, shift)
    assert ours[-1] == 0 and torch.equal(ours[:-1], ref)
    grid = torch.arange(1, 1001) / 1000
    assert all(((grid - s).abs().min() < 1e-6) for s in ref)


def test_sampler_runs_and_guidance_changes_result(tiny):
    cfg = tiny.cfg
    g = torch.Generator().manual_seed(0)
    pos = [
        torch.randn(5, cfg.text_dim, generator=g),
        torch.randn(9, cfg.text_dim, generator=g),
    ]
    neg = [torch.randn(2, cfg.text_dim, generator=g)] * 2
    kw = dict(size=(64, 48), steps=4, seeds=(1, 2), device="cpu")
    a = sample_latents(tiny, pos, neg, cfg=3.0, **kw)
    b = sample_latents(tiny, pos, neg, cfg=1.0, **kw)
    c = sample_latents(tiny, pos, neg, cfg=3.0, skip_layers=(), **kw)
    assert a.latents.shape == (2 * 4 * 3, cfg.latent_channels)
    assert torch.isfinite(a.latents).all()
    assert not torch.allclose(a.latents, b.latents)
    assert not torch.allclose(a.latents, c.latents)
    imgs = unpack(a.latents, a.hw)
    assert imgs[0].shape == (1, cfg.latent_channels, 4, 3)
    again = sample_latents(tiny, pos, neg, cfg=3.0, **kw)
    assert torch.equal(a.latents, again.latents)
