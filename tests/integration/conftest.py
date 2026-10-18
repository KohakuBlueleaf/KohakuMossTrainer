"""Whole-integration fixtures on the real model: the weights root (KOHAKUMOSS_WEIGHTS,
holding base/ and fp16fix/), the fp16-fix file, a small kohya-layout slice of the
KOHAKUMOSS_DATASET dataset, two of its captions, two of its images encoded by
the real VAE and text encoder, the train-script knobs of a real run, and a trained
adapter file. Every test releases the device cache it leaves behind."""

import gc
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageOps
from torch import Tensor

from kohakumoss.encoders.text.encoder import QwenTextEncoder
from kohakumoss.encoders.vae import FluxVae
from kohakumoss.offload.encoders.vae import place_vae
from kohakumoss.training.entry import run
from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss_kernels.hardware.arch import arch_of
from tests._util.dataset import IMAGE_SUFFIXES, kohya_slice, kohya_source
from tests.training.knobs.test_knobs import script_defaults

SLICE = 16
FIX_FILE = "fp16fix/moss_image_base_fp16fix_c0.0625.safetensors"
GRIDS = ((16, 16), (12, 20))  # latent grids of the encoded batch (256x256, 192x320 px)
SIGMAS = (0.35, 0.75)


def needs_precision(precision: str, device: str = "cuda") -> None:
    """Skip a case whose subject is `precision` on a card without it (bf16: sm80+)."""
    arch = arch_of(device)
    if precision == "bf16" and not arch.has_bf16:
        pytest.skip(f"bf16 needs a card with bf16; {arch.name} has none")


def card_precision(device: str = "cuda") -> dict:
    """PRECISION "auto" as the trainer resolves it on `device` (bf16 where the card has
    it, else fp16), with the online fold an fp16 run on the base weights needs."""
    if arch_of(device).has_bf16:
        return {"PRECISION": "bf16"}
    return {"PRECISION": "fp16", "FP16_FIX": "online", "FP16_FIX_KWARGS": {"c": 1 / 16}}


@pytest.fixture(autouse=True)
def _free_device():
    """After each test: unreachable tensors collected, the allocator's cache released."""
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def weights_root() -> Path:
    root = os.environ.get("KOHAKUMOSS_WEIGHTS")
    if not root:
        pytest.skip("KOHAKUMOSS_WEIGHTS unset")
    return Path(root)


@pytest.fixture(scope="session")
def comfy_root() -> Path:
    """A ComfyUI models folder with the Comfy-Org MOSS-Image files (KOHAKUMOSS_COMFY:
    diffusion_models/, text_encoders/, vae/)."""
    root = os.environ.get("KOHAKUMOSS_COMFY")
    if not root:
        pytest.skip("KOHAKUMOSS_COMFY unset")
    return Path(root)


@pytest.fixture(scope="session")
def fix_file(weights_root) -> Path | None:
    """The fp16-fix DiT file (KOHAKUMOSS_FP16_FIX_FILE, else <weights>/fp16fix/...)."""
    path = Path(os.environ.get("KOHAKUMOSS_FP16_FIX_FILE") or weights_root / FIX_FILE)
    return path if path.exists() else None


@pytest.fixture(scope="session")
def dataset_slice(tmp_path_factory) -> Path:
    root = os.environ.get("KOHAKUMOSS_DATASET")
    if not root:
        pytest.skip("KOHAKUMOSS_DATASET unset")
    return kohya_slice(Path(root), tmp_path_factory.mktemp("dataset"), SLICE)


def _captioned(dataset: Path, n: int) -> list[tuple[Path, str]]:
    """The first `n` images of `dataset` with a .txt caption, and that caption."""
    out = []
    images = sorted(p for p in dataset.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    for image in images:
        txt = image.with_suffix(".txt")
        caption = txt.read_text(encoding="utf-8").strip() if txt.is_file() else ""
        if caption:
            out.append((image, caption))
        if len(out) == n:
            return out
    pytest.skip(f"the dataset slice has fewer than {n} captioned images")


@pytest.fixture(scope="session")
def captions(dataset_slice) -> list[str]:
    """Two captions of the KOHAKUMOSS_DATASET dataset."""
    return [c for _, c in _captioned(dataset_slice, len(GRIDS))]


def _pixels(path: Path, hw: tuple[int, int]) -> np.ndarray:
    """The image center-fitted to 16 * hw pixels, uint8 HWC RGB."""
    with Image.open(path) as im:
        fitted = ImageOps.fit(im.convert("RGB"), (16 * hw[1], 16 * hw[0]))
    return np.asarray(fitted, dtype=np.uint8)


@dataclass
class RealBatch:
    """Two dataset images as packed fp32 latents [N_img, C] and their captions as packed
    fp32 text embeddings [N_txt, text_dim] (host tensors), one sigma per sample."""

    img: Tensor
    text: Tensor
    sigma: Tensor
    hw: tuple[tuple[int, int], ...]
    text_lens: tuple[int, ...]

    def on(self, device, dtype: torch.dtype) -> tuple:
        """(img, text, sigma, seq) for a backbone forward: latents and text in `dtype`,
        sigma fp32."""
        seq = build_seqinfo(torch.tensor(self.hw), torch.tensor(self.text_lens))
        return (
            self.img.to(device, dtype),
            self.text.to(device, dtype),
            self.sigma.to(device),
            seq.to(device),
        )


@pytest.fixture(scope="session")
def real_batch(weights_root, dataset_slice) -> RealBatch:
    """Two dataset images through the real VAE (offloaded) and their captions through
    the real text encoder (streamed), on the first CUDA device; both freed after."""
    pairs = _captioned(dataset_slice, len(GRIDS))
    device = torch.device("cuda")
    base = weights_root / "base"
    vae = place_vae(
        FluxVae.from_pretrained(base, "cpu", torch.bfloat16),
        {"memory": "offload"},
        device,
    )
    images = [_pixels(path, hw) for (path, _), hw in zip(pairs, GRIDS)]
    img = vae.encode_images(images, torch.tensor(GRIDS)).float().cpu()
    del vae
    te = QwenTextEncoder.from_pretrained(base, device, {"memory": "stream"})
    text = [t.float().cpu() for t in te.encode([c for _, c in pairs])]
    del te
    gc.collect()
    torch.cuda.empty_cache()
    return RealBatch(
        img,
        torch.cat(text),
        torch.tensor(SIGMAS),
        GRIDS,
        tuple(t.shape[0] for t in text),
    )


def real_knobs(weights_root: Path, dataset: Path, out: Path, **overrides) -> dict:
    """The train script's defaults for a short real run on `dataset` (kohya source,
    artist from the folder, online caches, the text encoder streamed) in the card's
    16-bit precision (`card_precision`)."""
    k = script_defaults()
    k.update(card_precision())
    k.update(
        WEIGHTS=str(weights_root / "base"),
        VARIANT="base",
        DATASETS=[kohya_source(dataset)],
        CACHE_DIR=str(out / "cache"),
        LATENT_CACHE="online",
        TEXT_CACHE="online",
        TEXT_ENCODE={"memory": "stream", "ahead": 8},
        TARGET_AREA=512 * 512,
        MAX_TOKENS=4352,
        STEPS=3,
        SCHEDULE={"mode": "constant", "warmup": 1, "end": -1},
        SAVE_EVERY=0,
        KEEP_LAST=0,
        SAMPLE_EVERY=0,
        LOG_EVERY=1,
        PROGRESS=False,
        TUNE="table",
        NUM_WORKERS=2,
        OUTPUT_DIR=str(out),
        RUN_NAME="run",
    )
    k.update(overrides)
    return k


@pytest.fixture(scope="session")
def adapter(weights_root, dataset_slice, tmp_path_factory) -> str:
    """KOHAKUMOSS_TEST_ADAPTER, else the LyCORIS export of a 2-step real LoRA run."""
    given = os.environ.get("KOHAKUMOSS_TEST_ADAPTER")
    if given:
        return given
    out = tmp_path_factory.mktemp("adapter")
    k = real_knobs(
        weights_root,
        dataset_slice,
        out,
        STEPS=2,
        ALGO="lora",
        DIM=8,
        ALPHA=8.0,
        LR=1e-2,
        EXPORT_FORMATS=("lycoris",),
    )
    run(k)
    return str(next((out / "run" / "export").glob("*-lycoris.safetensors")))


def fp16_fix_knobs(weights_root: Path, fix: Path) -> dict:
    """fp16 from the fix file: the DiT from `fix`, text encoder and VAE from base."""
    base = weights_root / "base"
    return {
        "PRECISION": "fp16",
        "FP16_FIX": "none",
        "WEIGHTS": str(fix),
        "TEXT_ENCODER": str(base / "text_encoder"),
        "VAE": str(base / "vae"),
    }
