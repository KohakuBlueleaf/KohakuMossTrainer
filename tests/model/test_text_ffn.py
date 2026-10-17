"""The text-block FFN runs SiTU on the two linears' outputs without concatenating them."""

import pytest
import torch

from kohakumoss.model.components.ffn import situ_pair
from kohakumoss_kernels.backends.eager.ffn import situ_glu
from tests._util.negative import must_fail
from tests.model.conftest import make_batch

pytestmark = pytest.mark.cpu


def _check(cond):
    assert cond


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_situ_pair_equals_spec(dtype):
    g = torch.Generator().manual_seed(0)
    u = (torch.randn(13, 40, generator=g) * 30).to(dtype)
    gate = (torch.randn(13, 40, generator=g) * 6).to(dtype)
    tol = 1e-12 if dtype == torch.float64 else 5e-3
    ref = situ_glu(torch.cat((u, gate), -1))
    assert _rel(situ_pair(u, gate), ref) < tol
    must_fail(lambda: _check(_rel(situ_pair(gate, u), ref) < tol), "up / gate swapped")


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def test_text_block_does_not_concatenate(tiny, monkeypatch):
    img, text, sigma, seq, _, _ = make_batch(tiny.cfg)
    ref = tiny(img, text, sigma, seq)
    calls = []
    real = torch.cat

    def spy(tensors, *args, **kwargs):
        if len(tensors) == 2 and tensors[0].shape[-1] == tiny.cfg.ffn_dim:
            calls.append(tensors[0].shape)
        return real(tensors, *args, **kwargs)

    monkeypatch.setattr(torch, "cat", spy)
    out = tiny(img, text, sigma, seq)
    assert torch.equal(out, ref) and not calls
