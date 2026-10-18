"""ComfyUI files on the real weights: the packaged tokenizer against MOSS-Image's
`tokenizer/` folder, ComfyUI's `flux2-vae_small` against MOSS-Image's `vae/` (same
decoder; MOSS-Image's encoder is the bf16 rounding of the file's) and `flux2-vae` (BFL
keys, the full FLUX.2 decoder) against the small one (same encoder, same latents), on
the host in fp32. KOHAKUMOSS_COMFY is the ComfyUI models folder; the latent difference
and the decode PSNRs are recorded as test properties."""

import math

import numpy as np
import pytest
import torch
from PIL import Image, ImageOps
from transformers import AutoTokenizer

from kohakumoss.encoders.text.comfy_files import packaged_tokenizer
from kohakumoss.encoders.text.tokens import MAX_LENGTH, chat_text, token_ids
from kohakumoss.encoders.vae import FluxVae
from tests._util.dataset import IMAGE_SUFFIXES
from tests._util.negative import must_fail

pytestmark = [pytest.mark.real_weights, pytest.mark.slow]
SIDE = 256
PROMPTS = (
    "a cat sitting on a red sofa, oil painting",
    "",
    "\\u65e5\\u672c\\u8a9e\\u306e\\u30d7\\u30ed\\u30f3\\u30d7\\u30c8 \U0001f338 and mixed English",
    "<|im_start|> literal special-looking text <|endoftext|>",
    "1girl, solo, long hair, looking at viewer, " * 40,
    "line one\nline two\n\n\ttabbed  double  spaces",
)


def check_tokenizer(tok, ref) -> None:
    assert token_ids(tok, PROMPTS, MAX_LENGTH) == token_ids(ref, PROMPTS, MAX_LENGTH)
    for prompt in PROMPTS:
        assert chat_text(tok, prompt) == chat_text(ref, prompt)
        assert tok(prompt)["input_ids"] == ref(prompt)["input_ids"]
    assert len(tok) == len(ref)
    assert (tok.eos_token_id, tok.pad_token_id) == (ref.eos_token_id, ref.pad_token_id)


def test_packaged_tokenizer_equals_the_hf_tokenizer(weights_root):
    ref = AutoTokenizer.from_pretrained(str(weights_root / "base" / "tokenizer"))
    check_tokenizer(packaged_tokenizer(), ref)
    broken = packaged_tokenizer()
    broken.add_tokens(["sofa"])
    must_fail(lambda: check_tokenizer(broken, ref), "a tokenizer with an extra token")


@pytest.fixture(scope="module")
def image(dataset_slice) -> np.ndarray:
    """The first dataset image center-fitted to SIDE x SIDE, uint8 HWC RGB."""
    path = sorted(
        p for p in dataset_slice.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
    )[0]
    with Image.open(path) as im:
        fitted = ImageOps.fit(im.convert("RGB"), (SIDE, SIDE))
    return np.asarray(fitted, dtype=np.uint8)


def _vae(path) -> FluxVae:
    return FluxVae.from_pretrained(path, "cpu", torch.float32)


def _hw() -> torch.Tensor:
    return torch.tensor([[SIDE // 16, SIDE // 16]])


def _psnr(vae: FluxVae, latents: torch.Tensor, image: np.ndarray) -> float:
    z = latents.transpose(0, 1).reshape(1, -1, SIDE // 16, SIDE // 16)
    out = (vae.decode(z)[0].permute(1, 2, 0) + 1) * 127.5
    ref = torch.from_numpy(image).float()
    mse = (out.clamp(0, 255) - ref).pow(2).mean().item()
    return 10 * math.log10(255**2 / mse)


def check_same(a: dict, b: dict) -> None:
    assert set(a) == set(b), sorted(set(a) ^ set(b))[:6]
    for key in a:
        assert torch.equal(a[key], b[key]), key


def _part(vae: FluxVae, prefixes: tuple[str, ...]) -> dict:
    """The floating tensors of `vae` under `prefixes` (bn.num_batches_tracked, an
    eval-unused counter, left out)."""
    state = vae.vae.state_dict()
    return {
        k: t
        for k, t in state.items()
        if k.startswith(prefixes) and t.is_floating_point()
    }


def _bf16(state: dict) -> dict:
    return {k: t.bfloat16().float() for k, t in state.items()}


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


ENCODER = ("encoder.", "quant_conv.", "bn.")
DECODER = ("decoder.", "post_quant_conv.")


@torch.no_grad()
def test_comfy_vae_files(weights_root, comfy_root, image, record_property):
    """Comfy small: MOSS-Image's decoder exactly, the fp32 encoder / quant conv / latent
    statistics whose bf16 rounding MOSS-Image's `vae/` holds. Comfy full: the small
    file's encoder exactly (same latents), the 512-wide decoder."""
    hf = _vae(weights_root / "base" / "vae")
    small = _vae(comfy_root / "vae" / "flux2-vae_small.safetensors")
    full = _vae(comfy_root / "vae" / "flux2-vae.safetensors")
    check_same(_part(small, DECODER), _part(hf, DECODER))
    check_same(_bf16(_part(small, ENCODER)), _part(hf, ENCODER))
    check_same(_part(full, ENCODER), _part(small, ENCODER))
    assert full.vae.config.decoder_block_out_channels[-1] == 512
    z = {
        name: v.encode_images([image], _hw())
        for name, v in (("hf", hf), ("small", small), ("full", full))
    }
    assert torch.equal(z["full"], z["small"])
    record_property("latent_rel_small_vs_hf_fp32", f"{_rel(z['small'], z['hf']):.3e}")
    for name, vae in (("hf", hf), ("small", small), ("full", full)):
        psnr = _psnr(vae, z[name], image)
        record_property(f"psnr_{name}_db", round(psnr, 3))
        assert math.isfinite(psnr)
    shifted = {
        k: t + 1e-3 if k == "bn.running_mean" else t
        for k, t in _part(small, ENCODER).items()
    }
    must_fail(lambda: check_same(_part(full, ENCODER), shifted), "a perturbed encoder")
    must_fail(
        lambda: check_same(_part(small, ENCODER), _part(hf, ENCODER)), "unrounded"
    )
