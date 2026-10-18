"""Conditioning: the official sinusoid formula in fp32."""

import pytest
import torch

from kohakumoss.model.components.embed import timestep_embedding
from tests._util.negative import must_fail


def _check(cond):
    assert cond


def _official_fp32(times, freq_dim, denom=None, order=("cos", "sin")):
    """The official MOSS sinusoid lines with float32 in place of float64."""
    half = freq_dim // 2
    freq = 10000 ** (
        -torch.arange(half, device=times.device, dtype=torch.float32) / (denom or half)
    )
    phase = times.float()[:, None] * freq
    parts = {"cos": phase.cos(), "sin": phase.sin()}
    return torch.cat([parts[k] for k in order], -1).float()


@pytest.mark.cpu
def test_sinusoid_is_official_formula():
    g = torch.Generator().manual_seed(0)
    sigma = torch.cat((torch.linspace(0, 1, 1001), torch.rand(4000, generator=g)))
    t = sigma.float() * 1000
    out = timestep_embedding(t, 256)
    assert out.dtype == torch.float32
    assert torch.equal(out, _official_fp32(t, 256))
    wrong_freq = _official_fp32(t, 256, denom=127)
    must_fail(lambda: _check(torch.equal(wrong_freq, out)), "wrong frequency")
    wrong_order = _official_fp32(t, 256, order=("sin", "cos"))
    must_fail(lambda: _check(torch.equal(wrong_order, out)), "wrong order")
