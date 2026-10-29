"""Weighted RMSNorm: exact for a D-shrunk stream at lam = 1/D^2, plain at lam = 1; the
anneal endpoints and the gain anneal."""

import pytest
import torch

from kohakumoss.heal.wnorm import GainAnneal, WeightedNorms, anneal_lam, anneal_t
from kohakumoss_kernels.backends.eager.norm import rms_norm, rms_norm_mod
from tests._util.negative import must_fail

pytestmark = [pytest.mark.cpu, pytest.mark.budget(10)]
DIM = 64


def _inputs():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(12, DIM, generator=g, dtype=torch.float64)
    x[:, 3] *= 200.0
    w = torch.rand(DIM, generator=g, dtype=torch.float64) + 0.5
    d = torch.ones(DIM, dtype=torch.float64)
    d[3] = 1 / 16
    shift = torch.randn(2, DIM, generator=g, dtype=torch.float64)
    scale = torch.randn(2, DIM, generator=g, dtype=torch.float64)
    seg = torch.tensor([0] * 6 + [1] * 6)
    return x, w, d, shift, scale, seg


def _exact_check(lam_of):
    """Shrunk stream through the weighted norm with gain w / D equals the original
    stream through the plain norm with gain w (rms_norm and rms_norm_mod)."""
    x, w, d, shift, scale, seg = _inputs()
    wn = WeightedNorms(DIM, "cpu")
    wn.lam = lam_of(d).float()
    got = wn.rms_norm(x * d, w / d, 1e-6)
    torch.testing.assert_close(got, rms_norm(x, w, 1e-6), rtol=1e-6, atol=1e-9)
    got = wn.rms_norm_mod(x * d, w / d, shift, scale, seg, 1e-6)
    torch.testing.assert_close(
        got, rms_norm_mod(x, w, shift, scale, seg, 1e-6), rtol=1e-6, atol=1e-9
    )


def test_weighted_norm_exact_at_t0():
    _exact_check(lambda d: anneal_lam(d, 0.0))


def test_weighted_norm_exact_negative():
    must_fail(
        lambda: _exact_check(lambda d: anneal_lam(d, 1.0)),
        "plain norm on a shrunk stream",
    )


def test_weighted_norm_plain_at_t1():
    x, w, d, shift, scale, seg = _inputs()
    wn = WeightedNorms(DIM, "cpu")
    wn.lam = anneal_lam(d, 1.0).float()
    torch.testing.assert_close(wn.rms_norm(x, w, 1e-6), rms_norm(x, w, 1e-6))
    torch.testing.assert_close(
        wn.rms_norm_mod(x, w, shift, scale, seg, 1e-6),
        rms_norm_mod(x, w, shift, scale, seg, 1e-6),
    )


def test_anneal_t():
    assert anneal_t(0, 3000) == 0.0
    assert anneal_t(1500, 3000) == 0.5
    assert anneal_t(4000, 3000) == 1.0
    assert anneal_t(0, 0) == 1.0


def test_gain_anneal():
    w = torch.full((4,), 2.0, dtype=torch.bfloat16)
    ga = GainAnneal()
    ga.add(w, 4.0)
    ga.set(1.0)
    torch.testing.assert_close(w.float(), torch.full((4,), 0.5))
    ga.set(0.5)
    torch.testing.assert_close(w.float(), torch.full((4,), 1.0))
    ga.check()
    ga.set(0.0)
    torch.testing.assert_close(w.float(), torch.full((4,), 2.0))
