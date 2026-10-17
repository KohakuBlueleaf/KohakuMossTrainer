"""MOSS timestep sampler against a direct Monte-Carlo of its definition."""

import math

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
    assert pmf.dtype == torch.float32
    assert abs(pmf.double().sum().item() - 1.0) < 1e-6
    assert max_cdf_gap(pmf.double(), simulate(400_000, mu, s, cutoff)) < 4e-3


def _pmf64(steps, mu, s, cutoff, uniform_mix, p_full):
    """The fp64 reference table (test only)."""
    a = 1.0 if cutoff is None else float(cutoff)
    x = (a * torch.arange(steps + 1, dtype=torch.float64) / steps).clamp(0.0, 1.0)
    u = (torch.log1p(-x) - torch.log(x) - mu) / (s * math.sqrt(2.0))
    cdf = torch.where(x <= 0, 0.0, torch.where(x >= 1, 1.0, 0.5 * torch.erfc(u)))
    pmf = (1.0 - uniform_mix) * (cdf[1:] - cdf[:-1]) / cdf[-1] + uniform_mix / steps
    pmf = pmf * (1.0 - p_full)
    pmf[-1] += p_full
    return pmf / pmf.sum()


def _naive32(steps, mu, s, cutoff):
    """fp32 with plain cdf differences (bins near cdf ~ 1 lose their digits)."""
    a = 1.0 if cutoff is None else float(cutoff)
    x = (a * torch.arange(steps + 1, dtype=torch.float32) / steps).clamp(0.0, 1.0)
    u = (torch.log1p(-x) - torch.log(x) - mu) / (s * math.sqrt(2.0))
    cdf = torch.where(x <= 0, 0.0, torch.where(x >= 1, 1.0, 0.5 * torch.erfc(u)))
    pmf = (cdf[1:] - cdf[:-1]) / cdf[-1]
    return pmf / pmf.sum()


def _close_bins(got, ref) -> None:
    live = ref > 1e-6
    rel = ((got.double() - ref).abs() / ref)[live].max().item()
    assert rel < 3e-4, rel
    assert (got.double() - ref).abs().max().item() < 1e-6


@pytest.mark.parametrize(
    "mu,s,cutoff,mix,full",
    [
        (-1.09, 0.95, 0.95, 0.01, 0.0),
        (0.0, 1.0, None, 0.0, 0.01),
        (2.0, 0.6, None, 0, 0),
    ],
)
def test_fp32_table_matches_the_fp64_reference(mu, s, cutoff, mix, full):
    got = moss_pmf(1000, mu, s, cutoff, mix, full)
    _close_bins(got, _pmf64(1000, mu, s, cutoff, mix, full))


def test_naive_fp32_differences_fail_the_bar():
    """Mass at low sigma: the upper bins are differences of cdf values near 1."""
    ref = _pmf64(1000, 2.0, 0.6, None, 0.0, 0.0)
    _close_bins(moss_pmf(1000, 2.0, 0.6, None, 0.0, 0.0), ref)
    must_fail(lambda: _close_bins(_naive32(1000, 2.0, 0.6, None), ref), "naive fp32")


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
    assert torch.allclose(pmf, expect / expect.sum(), atol=1e-8)


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
