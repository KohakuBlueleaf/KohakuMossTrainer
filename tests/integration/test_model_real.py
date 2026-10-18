"""The real backbone's structural contracts (fp16, small token counts): a packed batch
equals each sample run alone, zero-padded extra samples leave the real rows unchanged,
the image positions follow each sample's text length, and the negative branch's skip
layers change the output. The fold is tests/integration/test_fold_real.py; parity with
the official forward is tests/model/test_real.py."""

import pytest
import torch

from kohakumoss.model import load_backbone
from kohakumoss.types.seqinfo import build_seqinfo
from tests._util.negative import must_fail

pytestmark = [pytest.mark.real_weights, pytest.mark.cuda, pytest.mark.slow]
HWS = ((12, 16), (16, 12), (8, 20))
TEXT_LENS = (21, 9, 33)
DTYPE = torch.float16  # see docs/internals/fp16.md#real-model-structural-tests
TOL = 2e-2


@pytest.fixture(scope="module")
def model(weights_root):
    m, report = load_backbone(weights_root / "base", device="cuda", dtype=DTYPE)
    assert report.ok
    return m.requires_grad_(False)


@pytest.fixture(scope="module")
def batch(model):
    cfg = model.cfg
    g = torch.Generator().manual_seed(1)
    n_img = sum(h * w for h, w in HWS)
    img = torch.randn(n_img, cfg.latent_channels, generator=g)
    text = torch.randn(sum(TEXT_LENS), cfg.text_dim, generator=g)
    sigma = torch.tensor([0.3, 0.6, 0.85])
    seq = build_seqinfo(torch.tensor(HWS), torch.tensor(TEXT_LENS))
    dev = torch.device("cuda")
    return (
        img.to(dev, DTYPE),
        text.to(dev, DTYPE),
        sigma.to(dev),
        seq.to(dev),
    )


def _rel(a, b) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _close(out, ref, tol=TOL) -> None:
    assert torch.isfinite(out).all() and ref.abs().max() > 0
    assert _rel(out, ref) < tol, _rel(out, ref)


@torch.inference_mode()
def test_packed_equals_per_sample(model, batch):
    img, text, sigma, seq = batch
    packed = model(img, text, sigma, seq)
    parts, i0, t0 = [], 0, 0
    for i, ((h, w), n) in enumerate(zip(HWS, TEXT_LENS)):
        one = build_seqinfo(torch.tensor([[h, w]]), torch.tensor([n])).to("cuda")
        parts.append(
            model(img[i0 : i0 + h * w], text[t0 : t0 + n], sigma[i : i + 1], one)
        )
        i0, t0 = i0 + h * w, t0 + n
    _close(packed, torch.cat(parts))
    wrong_lens = (TEXT_LENS[1], TEXT_LENS[0], TEXT_LENS[2])
    wrong = build_seqinfo(torch.tensor(HWS), torch.tensor(wrong_lens)).to("cuda")
    must_fail(
        lambda: _close(model(img, text, sigma, wrong), packed), "text lengths swapped"
    )


@torch.inference_mode()
def test_padding_samples_leave_real_rows(model, batch):
    img, text, sigma, seq = batch
    packed = model(img, text, sigma, seq)
    pad = 64
    seq_pad = build_seqinfo(
        torch.tensor(HWS + ((1, pad),)), torch.tensor(TEXT_LENS + (0,))
    ).to("cuda")
    img_pad = torch.cat((img, img.new_zeros(pad, img.shape[1])))
    out = model(img_pad, text, torch.cat((sigma, sigma[:1])), seq_pad)[: img.shape[0]]
    _close(out, packed)


@torch.inference_mode()
def test_skip_layers_change_the_output(model, batch):
    img, text, sigma, seq = batch
    full = model(img, text, sigma, seq)
    skipped = model.bind_forward(frozenset(model.cfg.negative_skip_layers))(
        img, text, sigma, seq
    )
    assert _rel(skipped, full) > 1e-3
    must_fail(lambda: _close(skipped, full, 1e-3), "skip layers ignored")
