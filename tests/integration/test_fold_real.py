"""The residual fold on the real backbone (bf16, small token counts): a uniform gate fold
by a power of two changes the output by no more than another kernel choice does, and
unfolding restores the unfolded output exactly."""

import pytest
import torch

from kohakumoss.model import load_backbone
from kohakumoss.model.numerics import fold, unfold
from kohakumoss.types.seqinfo import build_seqinfo
from tests._util.negative import must_fail

pytestmark = [pytest.mark.real_weights, pytest.mark.cuda, pytest.mark.slow]
HWS = ((12, 16), (8, 20))
TEXT_LENS = (21, 33)
DTYPE = torch.bfloat16
NOISE = 0.12  # see docs/internals/fp16.md#real-model-structural-tests
EXACT = 1e-6


def _rel(a, b) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _close(out, ref, tol: float) -> None:
    assert torch.isfinite(out).all() and ref.abs().max() > 0
    assert _rel(out, ref) <= tol, _rel(out, ref)


@torch.inference_mode()
def test_fold_is_exact(weights_root):
    model, report = load_backbone(weights_root / "base", device="cuda", dtype=DTYPE)
    assert report.ok
    cfg = model.cfg
    g = torch.Generator().manual_seed(1)
    dev = torch.device("cuda")
    img = torch.randn(sum(h * w for h, w in HWS), cfg.latent_channels, generator=g)
    text = torch.randn(sum(TEXT_LENS), cfg.text_dim, generator=g)
    sigma = torch.tensor([0.3, 0.85], device=dev)
    seq = build_seqinfo(torch.tensor(HWS), torch.tensor(TEXT_LENS)).to(dev)
    img, text = img.to(dev, DTYPE), text.to(dev, DTYPE)
    ref = model(img, text, sigma, seq)
    fold(model, 2.0**-4, "gate")
    try:
        folded = model(img, text, sigma, seq)
    finally:
        unfold(model)
    _close(folded, ref, NOISE)
    _close(model(img, text, sigma, seq), ref, EXACT)
    fold(model, 2.0**-4, "gate")
    model.set_residual_scale((2.0**-4,) * cfg.depth)
    try:
        broken = model(img, text, sigma, seq)
    finally:
        unfold(model)
    must_fail(
        lambda: _close(broken, ref, NOISE), "a fold that also rescales the stream"
    )
