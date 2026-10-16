"""MossPipeline: text-to-image with the official MOSS sampler over packed batches."""

from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from kohakumoss import adapters as adapter_api
from kohakumoss.diffusion.sampling import sample_latents, unpack
from kohakumoss.encoders import FluxVae, QwenTextEncoder
from kohakumoss.inference.residency import RESIDENCY
from kohakumoss.model.io import load_backbone
from kohakumoss.model.numerics import apply_fp16_fix, output_row_scales

Size = tuple[int, int]


@dataclass
class GenerateResult:
    """Images (PIL or [1, 3, H, W] tensors in [-1, 1]), latents, and every setting used."""

    images: list
    latents: list[Tensor]
    settings: dict = field(default_factory=dict)


def to_pil(image: Tensor) -> Image.Image:
    """[1, 3, H, W] in [-1, 1] -> RGB PIL image."""
    x = ((image[0].float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return Image.fromarray(np.ascontiguousarray(x.permute(1, 2, 0).cpu().numpy()))


def _as_latents(place, latents):
    return latents


def _as_tensors(place, latents):
    return place.decode(latents)


def _as_pil(place, latents):
    return [to_pil(im) for im in place.decode(latents)]


OUTPUTS = {"latent": _as_latents, "tensor": _as_tensors, "pil": _as_pil}


def _component_root(path: Path, variant: str | None, name: str) -> Path:
    for root in ((path / variant) if variant else None, path, path / "base"):
        if root is not None and (root / name).is_dir():
            return root
    raise FileNotFoundError(f"no {name}/ under {path}")


class MossPipeline:
    """Backbone + text encoder + VAE held explicitly; `residency` decides where the
    encoders live between uses ("gpu" | "offload" | "te_once")."""

    def __init__(
        self,
        backbone,
        text_encoder=None,
        vae=None,
        device: torch.device | str = "cuda",
        residency: str = "gpu",
        owned: bool = True,
    ):
        self.backbone = backbone
        self.device = torch.device(device)
        self.place = RESIDENCY[residency](text_encoder, vae, self.device)
        self.residency = residency
        self._grad_ctx = torch.inference_mode if owned else torch.no_grad

    @classmethod
    def from_components(
        cls, backbone, text_encoder=None, vae=None, device=None, residency="gpu"
    ) -> "MossPipeline":
        """Share already-built modules (the training model): no reload, no_grad only."""
        device = device or next(backbone.parameters()).device
        return cls(backbone, text_encoder, vae, device, residency, owned=False)

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        variant: str | None = None,
        transformer: str | Path | None = None,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cuda",
        adapters: Sequence[tuple[str, float]] = (),
        adapter_mode: str = "merge",
        residency: str = "gpu",
        fp16_fix: str = "none",
        fp16_fix_kwargs: dict | None = None,
        text_dtype: torch.dtype = torch.bfloat16,
        vae_dtype: torch.dtype = torch.float32,
        kernels: dict[str, str] | None = None,
    ) -> "MossPipeline":
        """`path`: weights root (base/ sft/ subdirs or one pipeline dir). `transformer`
        overrides the DiT weights (single file, HF dir, our checkpoint or an fp16-fix
        file). `fp16_fix`: none | weights | online | fp32_residual (+ kwargs)."""
        path = Path(path)
        fix_kw = dict(fp16_fix_kwargs or {})
        dit = (
            Path(transformer)
            if transformer
            else _component_root(path, variant, "transformer")
        )
        backbone, _ = load_backbone(
            dit,
            device=device,
            dtype=dtype,
            kernels=kernels,
            fp16_fix_eps=fix_kw.get("eps", "stock"),
        )
        backbone.eval().requires_grad_(False)
        if adapter_mode == "merge":
            for file, mult in adapters:
                adapter_api.load(backbone, file, multiplier=mult, mode="merge")
            apply_fp16_fix(backbone, fp16_fix, **fix_kw)
        else:
            apply_fp16_fix(backbone, fp16_fix, **fix_kw)
            scales = output_row_scales(backbone)
            for file, mult in adapters:
                adapter_api.load(
                    backbone, file, multiplier=mult, mode="bypass", row_scales=scales
                )
            backbone.bind_ops()
        te_root = _component_root(path, variant, "text_encoder")
        offload = "cpu" if residency == "offload" else "none"
        te = QwenTextEncoder.from_pretrained(te_root, device, text_dtype, offload)
        vae = FluxVae.from_pretrained(
            _component_root(path, variant, "vae"), "cpu", vae_dtype
        )
        return cls(backbone, te, vae, device, residency, owned=True)

    @contextmanager
    def _eval(self):
        was = self.backbone.training
        self.backbone.eval()
        try:
            with self._grad_ctx():
                yield
        finally:
            self.backbone.train(was)

    def encode(self, prompts: Sequence[str]) -> list[Tensor]:
        return self.place.encode(prompts)

    def generate(
        self,
        prompts: Sequence[str] | None = None,
        negative_prompts: Sequence[str] | None = None,
        sizes: Size | Sequence[Size] = (1024, 1024),
        steps: int = 50,
        cfg: float = 3.0,
        shift: float = 5.0,
        skip_layers: Sequence[int] | None = None,
        seeds: Sequence[int] | None = None,
        batch: int = 4,
        cfg_mode: str = "packed",
        output: str = "pil",
        prompt_embeds: Sequence[Tensor] | None = None,
        negative_embeds: Sequence[Tensor] | None = None,
        step_callback: Callable[[int, Tensor], None] | None = None,
    ) -> GenerateResult:
        """One image per prompt; sample i depends only on its prompt, size and seed."""
        with self._eval():
            pos, neg = self._embeddings(
                prompts, negative_prompts, prompt_embeds, negative_embeds, cfg
            )
            n = len(pos)
            seeds = list(seeds) if seeds is not None else list(range(n))
            size_list = [sizes] * n if isinstance(sizes[0], int) else list(sizes)
            starts = range(0, n, batch)
            chunks = [slice(s, s + batch) for s in starts]
            negs = [None] * len(chunks) if neg is None else [neg[c] for c in chunks]
            convert = OUTPUTS[output]
            latents = []
            for chunk, chunk_neg in zip(chunks, negs):
                res = sample_latents(
                    self.backbone,
                    pos[chunk],
                    chunk_neg,
                    size_list[chunk],
                    steps,
                    cfg,
                    shift,
                    skip_layers,
                    seeds[chunk],
                    self.device,
                    cfg_mode,
                    step_callback,
                )
                latents += unpack(res.latents, res.hw)
            images = convert(self.place, latents)
        settings = dict(
            prompts=list(prompts) if prompts is not None else None,
            negative_prompts=list(negative_prompts) if negative_prompts else None,
            sizes=size_list,
            steps=steps,
            cfg=cfg,
            shift=shift,
            skip_layers=list(
                self.backbone.cfg.negative_skip_layers
                if skip_layers is None
                else skip_layers
            ),
            seeds=seeds,
            cfg_mode=cfg_mode,
            batch=batch,
            dtype=str(next(self.backbone.parameters()).dtype),
            fp16_fix=self.backbone.fp16_fix,
            fold=(
                None
                if self.backbone.fold_scales is None
                else self.backbone.fold_scales[0]
            ),
        )
        return GenerateResult(images, latents, settings)

    def _embeddings(
        self, prompts, negative_prompts, prompt_embeds, negative_embeds, cfg
    ):
        if prompt_embeds is not None:
            pos = list(prompt_embeds)
        else:
            texts = list(prompts)
            if cfg != 1.0 and negative_prompts:
                texts += [p for p in negative_prompts if p]
            enc = self.place.encode(texts)
            pos, extra = enc[: len(prompts)], enc[len(prompts) :]
            if cfg != 1.0 and negative_prompts and negative_embeds is None:
                it = iter(extra)
                negative_embeds = [
                    next(it) if p else self.place.empty() for p in negative_prompts
                ]
        if cfg == 1.0:
            return pos, None
        neg = list(negative_embeds) if negative_embeds is not None else None
        if neg is None:
            neg = [self.place.empty()] * len(pos)
        if len(neg) != len(pos):
            raise ValueError("one negative per prompt")
        return pos, neg
