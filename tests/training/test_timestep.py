"""MOSS timestep sampler against a direct Monte-Carlo of its definition."""

import pytest
import torch

from kohakumoss.diffusion.timestep import MossTimestep, moss_pmf, shift_sigma
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def simulate(n, mu, s, cutoff, steps=1000, seed=0):
    """Definition: Z~N(mu, s^2), S = 1 - sigmoid(Z), reject S > a, j = ceil(T S / a)."""
    g = torch.Generator().manual_seed(seed)
    z = mu + s * torch.randn(4 * n, generator=g, dtype=torch.float64)
    big_s = 1.0 - torch.sigmoid(z)
    a = 1.0 if cutoff is None else cutoff
    big_s = big_s[big_s <= a][:n]
    j = torch.ceil(big_s / a * steps).clamp(1, steps).long()
    return torch.bincount(j - 1, minlength=steps).double() / j.numel()


def max_cdf_gap(p, q):
    return (p.cumsum(0) - q.cumsum(0)).abs().max().item()


@pytest.mark.parametrize(
    "mu,s,cutoff", [(-1.09, 0.95, 0.95), (0.0, 1.0, None), (-1.45, 1.05, None)]
)
def test_pmf_matches_definition(mu, s, cutoff):
    pmf = moss_pmf(1000, mu, s, cutoff, uniform_mix=0.0, p_full=0.0)
    assert abs(pmf.sum().item() - 1.0) < 1e-9
    assert max_cdf_gap(pmf, simulate(400_000, mu, s, cutoff)) < 4e-3


def test_pmf_wrong_sign_fails():
    def broken():
        pmf = moss_pmf(1000, 1.09, 0.95, 0.95, 0.0, 0.0)
        assert max_cdf_gap(pmf, simulate(400_000, -1.09, 0.95, 0.95)) < 4e-3

    must_fail(broken, "mu sign flipped")


def test_sampler_follows_pmf_and_median():
    sampler = MossTimestep(-1.09, 0.95, 1.0, 0.95, uniform_mix=0.0)
    x = sampler.sample(300_000, torch.Generator().manual_seed(1), 0)
    hist = (
        torch.bincount((x * 1000).round().long() - 1, minlength=1000).double()
        / x.numel()
    )
    assert max_cdf_gap(hist, moss_pmf(1000, -1.09, 0.95, 0.95, 0.0, 0.0)) < 4e-3
    assert 0.76 < x.median().item() < 0.80


def test_uniform_mix_and_full_noise_mass():
    pmf = moss_pmf(1000, 0.0, 1.0, None, uniform_mix=0.01, p_full=0.01)
    base = moss_pmf(1000, 0.0, 1.0, None, uniform_mix=0.0, p_full=0.0)
    expect = (0.99 * base + 0.01 / 1000) * 0.99
    expect[-1] += 0.01
    assert torch.allclose(pmf, expect / expect.sum(), atol=1e-12)


def test_shift_and_stage_switch():
    sampler = MossTimestep(schedule="tech_report")
    assert sampler.stage_at(0) is sampler.stages[0]
    assert sampler.stage_at(140_000) is sampler.stages[1]
    assert sampler.stage_at(300_000) is sampler.stages[3]
    grid = torch.arange(1, 1001, dtype=torch.float64) / 1000
    assert torch.allclose(
        sampler.stages[0].sigma.double(), shift_sigma(grid, 3.0), atol=1e-6
    )
    assert torch.allclose(sampler.stages[3].sigma.double(), grid, atol=1e-6)
