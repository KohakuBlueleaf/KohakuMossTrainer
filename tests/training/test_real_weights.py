"""Real-weight training step on a packed batch (TRAIN-24): every sample's loss is that
of a working denoiser (at most 0.85 and below half the zero predictor's; the synthetic
images here are easier than real data, so there is no lower bound), the packed
forward equals each sample run alone, and the LoRA
backward is finite. A per-sample modulation read with the wrong layout gives samples
after the first another sample's shift / scale / gate, which breaks all three."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from kohakumoss.diffusion.objective import TokenNormalizer
from kohakumoss.encoders.text import QwenTextEncoder
from kohakumoss.training.build.loop import build_step
from kohakumoss.training.build.model import build_model
from kohakumoss.training.knobs import resolve_precision
from kohakumoss.training.precision import autocast_factory, build_policy
from kohakumoss.types.batch import DeviceBatch
from kohakumoss.types.seqinfo import build_seqinfo
from tests._util.negative import must_fail
from tests.training.test_knobs import runnable

pytestmark = [pytest.mark.real_weights, pytest.mark.cuda, pytest.mark.slow]
GRIDS = ((59, 38), (65, 34), (69, 32))  # latent h, w of the S4 failing batch
SIGMAS = (0.35, 0.55, 0.75)
PROMPTS = (
    "a red brick house beside a lake, morning light",
    "portrait of a woman in a blue coat, city street",
    "a bowl of oranges on a wooden table, still life",
)


class PerSample:
    """Sampler with one fixed sigma per sample slot."""

    def __init__(self, sigmas):
        self.sigmas = torch.tensor(sigmas, dtype=torch.float32)

    def sample(self, batch, generator, step):
        return self.sigmas[:batch].clone()


def _image(h: int, w: int, seed: int) -> np.ndarray:
    """A smooth structured image (gradients, discs, stripes) of h x w pixels."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    img = np.stack([xx, yy, 0.5 * (xx + yy)], -1)
    for _ in range(6):
        cy, cx, r = rng.random(3) * [1, 1, 0.3]
        disc = ((yy - cy) ** 2 + (xx - cx) ** 2) < r**2
        img[disc] = rng.random(3)
    img += 0.1 * np.sin(40 * xx)[..., None]
    return (img.clip(0, 1) * 255).astype(np.uint8)


def _text_embeddings(weights: Path, dev) -> list[torch.Tensor]:
    """PROMPTS through the real text encoder, alone on the GPU; or read from the
    safetensors file KOHAKUMOSS_TEST_TEXT_EMBEDS (keys "0", "1", ...) written by an
    earlier run, for GPUs too small to hold the encoder."""
    cached = os.environ.get("KOHAKUMOSS_TEST_TEXT_EMBEDS")
    if cached and Path(cached).exists():
        state = load_file(cached)
        return [state[str(i)] for i in range(len(PROMPTS))]
    encoder = QwenTextEncoder.from_pretrained(weights, device=dev)
    with torch.no_grad():
        parts = [p.cpu() for p in encoder.encode(list(PROMPTS))]
    del encoder
    torch.cuda.empty_cache()
    if cached:
        save_file({str(i): p.contiguous() for i, p in enumerate(parts)}, cached)
    return parts


@pytest.fixture(scope="module")
def setup():
    weights = Path(os.environ["KOHAKUMOSS_WEIGHTS"]) / "base"
    dev = torch.device("cuda")
    parts = _text_embeddings(weights, dev)
    k = runnable(
        WEIGHTS=str(weights),
        VARIANT="base",
        ALGO="lora",
        DIM=8,
        ALPHA=8.0,
        LATENT_CACHE="off",
        TEXT_CACHE="offline",
        SAMPLE_EVERY=0,
        PRECISION="auto",
        DEVICES=(0,),
    )
    k["PRECISION"] = resolve_precision(k)
    if k["PRECISION"] == "fp16":
        k.update(FP16_FIX="online", FP16_FIX_KWARGS={"c": 1 / 16})
    policy = build_policy(
        k["PRECISION"], k["KEEP_FP32"], k["FP16_FIX"], k["FP16_FIX_KWARGS"]
    )
    built = build_model(k, dev, policy)
    hw = torch.tensor(GRIDS, dtype=torch.int32)
    images = [_image(h * 16, w * 16, i) for i, (h, w) in enumerate(GRIDS)]
    with torch.no_grad():
        latents = built.vae.encode_images(images, hw).to(dev)
    text = torch.cat(parts).to(dev)
    text_len = torch.tensor([p.shape[0] for p in parts], dtype=torch.int32)
    built.vae = None
    torch.cuda.empty_cache()
    step = build_step(
        k,
        built,
        TokenNormalizer(),
        autocast_factory(policy, dev),
        dev,
        sampler=PerSample(SIGMAS),
    )
    return built, step, latents, text, hw, text_len, dev


def _batch(latents, text, hw, text_len, dev) -> DeviceBatch:
    n = len(hw)
    return DeviceBatch(
        ids=torch.arange(n),
        seq=build_seqinfo(hw, text_len).to(dev),
        latents=latents,
        text=text,
        text_drop=torch.zeros(n, dtype=torch.bool, device=dev),
        weight=torch.ones(n, device=dev),
        loss_mask=None,
        pixels=None,
        prompts=None,
    )


def _per_sample_mse(pred, target, seq) -> torch.Tensor:
    err = (pred.float() - target).square().mean(-1)
    sums = torch.zeros(seq.batch, device=err.device).index_add_(
        0, seq.seg_img.long(), err
    )
    counts = torch.bincount(seq.seg_img.long(), minlength=seq.batch).float()
    return sums / counts


def _forward(step, p, sigma=None):
    with step.autocast():
        return step.backbone(p.z, p.text, p.sigma if sigma is None else sigma, p.seq)


def _alone(step, p, hw, text_len, dev, sigma) -> list[torch.Tensor]:
    """Each sample run as a batch of one with the packed batch's inputs."""
    out, img0, txt0 = [], 0, 0
    for i in range(len(hw)):
        n_img, n_txt = int(hw[i, 0] * hw[i, 1]), int(text_len[i])
        seq = build_seqinfo(hw[i : i + 1], text_len[i : i + 1]).to(dev)
        with step.autocast():
            pred = step.backbone(
                p.z[img0 : img0 + n_img],
                p.text[txt0 : txt0 + n_txt],
                sigma[i : i + 1],
                seq,
            )
        out.append(pred.float())
        img0, txt0 = img0 + n_img, txt0 + n_txt
    return out


def _consistent(packed, alone, seq) -> None:
    for i, ref in enumerate(alone):
        got = packed[seq.cu_img[i] : seq.cu_img[i + 1]].float()
        rel = ((got - ref).norm() / ref.norm()).item()
        assert rel <= 2e-2, (i, rel)


def _below_half(mse, zero) -> None:
    assert (mse < 0.5 * zero).all(), (mse, zero)


def test_packed_step_is_a_working_denoiser(setup):
    built, step, latents, text, hw, text_len, dev = setup
    batch = _batch(latents, text, hw, text_len, dev)
    p = step.prepare(batch, 0)
    with torch.no_grad():
        pred = _forward(step, p)
        mse = _per_sample_mse(pred, p.target, p.seq)
        zero = _per_sample_mse(torch.zeros_like(p.target), p.target, p.seq)
        alone = _alone(step, p, hw, text_len, dev, p.sigma)
    for i in range(len(GRIDS)):
        assert torch.isfinite(mse[i]) and mse[i] < 0.5 * zero[i], (i, mse, zero)
        assert mse[i] <= 0.85, (i, mse.tolist())
    _consistent(pred, alone, p.seq)
    must_fail(lambda: _below_half(zero, zero), "zero predictor passes as a denoiser")

    swapped = p.sigma[[1, 0, 2]]  # samples 0 and 1 take each other's modulation
    with torch.no_grad():
        wrong = _alone(step, p, hw, text_len, dev, swapped)
    must_fail(lambda: _consistent(pred, wrong, p.seq), "cross-sample modulation")


def test_packed_lora_backward_is_finite(setup):
    built, step, latents, text, hw, text_len, dev = setup
    batch = _batch(latents, text, hw, text_len, dev)
    for prm in built.trainable.trainable_params():
        prm.grad = None
    out = step(batch, step.denominator(batch), 0)
    out.loss.backward()
    grads = [prm.grad for prm in built.trainable.trainable_params()]
    assert all(g is not None for g in grads)
    bad = sum(int(not torch.isfinite(g).all()) for g in grads)
    assert bad == 0, f"{bad} of {len(grads)} LoRA grads non-finite"
    assert any(g.abs().max() > 0 for g in grads)
