"""Packed batches through the device's default kernel selection: each sample's output
and input gradient equal those of the same sample run alone, with per-sample
modulation large enough that any cross-sample mix-up shows."""

import pytest
import torch

from kohakumoss.model import random_backbone
from kohakumoss.model.linear import MossLinear
from kohakumoss.types.seqinfo import build_seqinfo
from tests._util.negative import must_fail
from tests.model.conftest import HWS, TEXT_LENS, make_batch

pytestmark = pytest.mark.gpu
TOL = {torch.float32: 1e-4, torch.bfloat16: 2e-2}


def _model(device, dtype):
    """Tiny model whose time path has unit-gain weights, so every sample's shift /
    scale / gate differ at O(1)."""
    model = random_backbone("tiny", device=device, dtype=dtype, seed=3)
    conditioning = (
        *model.time_embedding.modules(),
        model.shared_block_modulation,
        model.video_head_modulation,
    )
    with torch.no_grad():
        for lin in conditioning:
            if isinstance(lin, MossLinear):
                lin.weight.normal_(0.0, lin.weight.shape[1] ** -0.5)
    return model


def _forward(model, img, text, sigma, seq):
    img = img.detach().clone().requires_grad_(True)
    out = model(img, text, sigma, seq)
    out.float().square().sum().backward()
    return out.detach().float(), img.grad.float()


def _per_sample(model, img, text, sigma, device):
    outs, grads, i0, t0 = [], [], 0, 0
    for i, ((h, w), n) in enumerate(zip(HWS, TEXT_LENS)):
        one = build_seqinfo(torch.tensor([[h, w]]), torch.tensor([n])).to(device)
        out, grad = _forward(
            model, img[i0 : i0 + h * w], text[t0 : t0 + n], sigma[i : i + 1], one
        )
        outs.append(out)
        grads.append(grad)
        i0, t0 = i0 + h * w, t0 + n
    return torch.cat(outs), torch.cat(grads)


def _assert_close(a, b, tol, what):
    assert torch.isfinite(a).all(), what
    rel = ((a.double() - b.double()).norm() / b.double().norm()).item()
    assert rel < tol, f"{what}: rel {rel:.3e} >= {tol:.0e}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_packed_equals_per_sample_default_kernels(device, dtype):
    model = _model(device, dtype)
    img, text, sigma, seq, _, _ = make_batch(model.cfg, dtype=dtype)
    img, text, sigma, seq = (
        img.to(device),
        text.to(device),
        sigma.to(device),
        seq.to(device),
    )
    ref_out, ref_grad = _per_sample(model, img, text, sigma, device)
    out, grad = _forward(model, img, text, sigma, seq)
    _assert_close(out, ref_out, TOL[dtype], "output")
    _assert_close(grad, ref_grad, TOL[dtype], "input grad")
    swapped = sigma.flip(0)

    def mixed():
        out, _ = _forward(model, img, text, swapped, seq)
        _assert_close(out, ref_out, TOL[dtype], "sigma of another sample")

    must_fail(mixed, "per-sample conditioning swapped")
