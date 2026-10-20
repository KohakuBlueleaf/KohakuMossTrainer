"""MossPipeline: text-to-image with the official MOSS sampler over packed batches."""

import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from kohakumoss.adapters.lycoris.tuning import set_lycoris_tuning
from kohakumoss.diffusion.sampling import sample_latents, unpack
from kohakumoss.diffusion.sampling.schedule import moss_sigmas
from kohakumoss.inference import peaks
from kohakumoss.inference import placement_report as report
from kohakumoss.inference.components import Components
from kohakumoss.inference.loading import (
    component_paths,
    load_dit,
    load_text_encoder,
    load_vae,
)
from kohakumoss.inference.steplog import StepClock
from kohakumoss.logging.log import bar as log_bar
from kohakumoss.logging.log import bars, get_logger, hl, log_table, timed
from kohakumoss.model.backbone import compute_dtype
from kohakumoss.offload.accel import release_cache
from kohakumoss.offload.encoders.report import encoder_lines
from kohakumoss.precision import autocast_for
from kohakumoss.types.plans import MemoryPlan, TextEncoderPlan
from kohakumoss_common.calls import nothing
from kohakumoss_common.units import GIB
from kohakumoss_data.resolution.geometry import LATENT_FACTOR
from kohakumoss_kernels.backends.compile.choice import retune_pending
from kohakumoss_kernels.hardware.memory import close_window, open_window
from kohakumoss_kernels.tuning import decisions, tuner

Size = tuple[int, int]
DEFAULT_CFG = 3.0
log = get_logger(__name__)


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


def _as_latents(place, latents, wrap=None):
    return latents


def _as_tensors(place, latents, wrap=None):
    return place.decode(latents, wrap) if wrap else place.decode(latents)


def _as_pil(place, latents, wrap=None):
    return [to_pil(im) for im in _as_tensors(place, latents, wrap)]


def sampling_autocast(backbone, device) -> Callable:
    """The context the sampler's DiT forwards run under: training's autocast to the
    backbone's compute dtype (the KEEP_FP32 conditioning linears take 16-bit inputs)."""
    return autocast_for(compute_dtype(backbone), device)


def apply_tune(tune: str) -> None:
    """The process's TUNE and LyCORIS's tile tuning (on under "table") for an owned
    pipeline."""
    tuner.set_mode(tune)
    set_lycoris_tuning(tune == "table")


def _step_report(pbar, sigmas: list[float], user, clock, i: int, x: Tensor) -> None:
    """Sampler step `i` done: time it on `clock`, advance `pbar` (next sigma in the
    postfix), then `user`."""
    clock.tick(i)
    pbar.update(1)
    pbar.set_postfix_str(f"sigma={sigmas[i + 1]:.3f}", refresh=False)
    if user is not None:
        user(i, x)


def _decode_bar(progress: bool, n: int):
    """Latent-iteration wrapper of the VAE decode: a bar over more than one image."""
    if not progress or n < 2:
        return None
    return partial(log_bar, desc="decode", unit="img", enabled=progress)


OUTPUTS = {"latent": _as_latents, "tensor": _as_tensors, "pil": _as_pil}


@contextmanager
def cusolver_linalg():
    """cuSOLVER for torch.linalg inside a generate call; the previous choice after."""
    previous = torch.backends.cuda.preferred_linalg_library()
    torch.backends.cuda.preferred_linalg_library("cusolver")
    try:
        yield
    finally:
        torch.backends.cuda.preferred_linalg_library(previous)


class MossPipeline:
    """Backbone + text encoder + VAE held explicitly; the text encoder places itself by
    its TextEncoderPlan, the VAE by its VaePlan (a bare FluxVae: resident). An owned
    model runs under inference_mode and applies `tune` as the process's TUNE: "off"
    binds stored decisions or defaults; "table" also tunes a missing one on first use
    and measures queued kernel-vs-compile problems after each generate. Nothing is
    prewarmed."""

    def __init__(
        self,
        backbone,
        text_encoder=None,
        vae=None,
        device: torch.device | str = "cuda",
        owned: bool = True,
        tune: str = "off",
        adapters: list | None = None,
        sources: dict | None = None,
    ):
        self.backbone = backbone
        self.device = torch.device(device)
        self.place = Components(text_encoder, vae, self.device)
        self.adapters = adapters or []
        self.sources = dict(sources or {})
        self._grad_ctx = torch.inference_mode if owned else torch.no_grad
        if owned:
            apply_tune(tune)
        self.retune = retune_pending if owned and tune == "table" else nothing
        cusolver = owned and self.device.type == "cuda"
        self._linalg = cusolver_linalg if cusolver else nullcontext
        self.memory_plan = MemoryPlan()
        self.weight_stream = None

    @classmethod
    def from_components(
        cls,
        backbone,
        text_encoder=None,
        vae=None,
        device=None,
        memory_plan: MemoryPlan | None = None,
        weight_stream=None,
    ) -> "MossPipeline":
        """Share already-built modules (the training model under its `memory_plan` and
        `weight_stream`, its placed VAE): no reload, no_grad only, TUNE left to the
        trainer (it owns the kernels)."""
        device = device or next(backbone.parameters()).device
        pipe = cls(backbone, text_encoder, vae, device, owned=False)
        pipe.memory_plan = memory_plan or MemoryPlan()
        pipe.weight_stream = weight_stream
        return pipe

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        variant: str | None = "base",
        transformer: str | Path | None = None,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cuda",
        adapters: Sequence[tuple[str, float]] = (),
        adapter_mode: str = "merge",
        fp16_fix: str = "none",
        fp16_fix_kwargs: dict | None = None,
        vae_dtype: torch.dtype | str = "auto",
        kernels: dict[str, str] | None = None,
        tune: str = "off",
        text_encoder: bool = True,
        vae: bool = True,
        text_device: torch.device | str | None = None,
        text_plan: TextEncoderPlan | dict | None = None,
        vae_plan: dict | None = None,
        memory: str | None = None,
        memory_overrides: dict | None = None,
        reserve_gib: float | None = None,
        forward_tokens: int | None = None,
        text_encoder_file: str = "",
        vae_file: str = "",
    ) -> "MossPipeline":
        """`path`: a ComfyUI models folder (one file per component: `transformer`,
        `text_encoder_file`, `vae_file` name them, "" the defaults, `variant` picks the
        DiT), a weights root (base/ sft/ subdirs, `variant` picks one) or one pipeline
        dir. `transformer` overrides the DiT weights (single file, HF dir, our
        checkpoint or an fp16-fix file), `text_encoder_file` / `vae_file` the text
        encoder and VAE. `fp16_fix`: none | online | fp32_residual (+
        kwargs); an fp16-fix file carries its fold. `text_plan` (TextEncoderPlan or its
        fields, `free_after_use` included) places the text encoder, `vae_plan` (VaePlan
        fields) the VAE; an unset `memory` of either is 'auto': the fastest placement
        that fits next to the DiT, keeping `reserve_gib` free (default: the DiT
        forward's working set at `forward_tokens`, the largest packed forward the caller
        will run, placement_report.forward_tokens; default one 1024x1024 CFG sample);
        `memory='auto'` sizes for the same forward. `text_device` runs the text
        encoder elsewhere (e.g. "cpu" = memory 'cpu'). `memory` (a memory profile, +
        `memory_overrides`) places the DiT under an inference MemoryPlan (quantized /
        streamed blocks). `text_encoder=False` / `vae=False` skip loading them
        (embeddings / latents in and out). `tune`, the process's TUNE: "off" (stored
        per-card decisions, else defaults; nothing timed) | "table" (a missing decision
        is also tuned on its first use and stored; no prewarm)."""
        apply_tune(tune)
        path = Path(path)
        dit, te_path, vae_path = component_paths(
            path, variant, transformer or "", text_encoder_file, vae_file
        )
        sources = {"transformer": str(dit)}
        tokens = forward_tokens or report.REFERENCE_TOKENS
        loaded = load_dit(
            dit,
            dtype,
            device,
            adapters,
            adapter_mode,
            fp16_fix,
            fp16_fix_kwargs,
            kernels,
            memory,
            memory_overrides,
            tokens,
        )
        if reserve_gib is None:
            reserve = peaks.working_set(loaded.backbone, loaded.plan, tokens)
        else:
            reserve = int(reserve_gib * GIB)
        te = decoder = None
        if text_encoder:
            te_root = te_path()
            sources["text_encoder"] = str(te_root)
            te = load_text_encoder(te_root, device, text_device, text_plan, reserve)
        if vae:
            vae_root = vae_path()
            sources["vae"] = str(vae_root)
            decoder = load_vae(vae_root, vae_dtype, device, vae_plan, reserve)
        pipe = cls(
            loaded.backbone,
            te,
            decoder,
            device,
            True,
            tune,
            adapters=loaded.stacks,
            sources=sources,
        )
        pipe.memory_plan = loaded.plan
        pipe.weight_stream = loaded.weight_stream
        pipe.log_placement()
        return pipe

    def log_placement(self) -> None:
        """One line per component (where its weights live and compute, sizes) and the
        placement summary."""
        device, src = self.device, self.sources
        lines = [
            report.dit_line(
                self.backbone, src.get("transformer", "-"), device, self.memory_plan
            )
        ]
        encoders = encoder_lines(
            self.place.text_encoder,
            self.place.vae,
            src.get("text_encoder", "-"),
            src.get("vae", "-"),
            device,
        )
        lines += [line for line in encoders if line is not None]
        for message, fields in lines:
            log.info(message, **fields)
        title, rows = report.summary_rows(self)
        log_table(log, title, report.SUMMARY_HEADERS, rows)

    @contextmanager
    def _eval(self):
        was = self.backbone.training
        self.backbone.eval()
        try:
            with self._grad_ctx(), self._linalg():
                yield
        finally:
            self.backbone.train(was)

    def encode(self, prompts: Sequence[str]) -> list[Tensor]:
        """EncodeText through the pipeline's text encoder, without autograd."""
        with self._grad_ctx():
            return self.place.encode(prompts)

    def embeddings(
        self,
        prompts: Sequence[str],
        negative_prompts: Sequence[str] | None = None,
        cfg: float = DEFAULT_CFG,
    ) -> tuple[list[Tensor], list[Tensor] | None]:
        """(prompt embeddings, negative embeddings or None at cfg == 1) as `generate`
        encodes them, in one text-encoder pass; pass them back as `prompt_embeds` /
        `negative_embeds` to sample several times without encoding again."""
        with self._grad_ctx():
            return self._embeddings(prompts, negative_prompts, None, None, cfg)

    def decode(self, latents: Sequence[Tensor]) -> list[Tensor]:
        """Latents [1, C, h, w] -> pixels [1, 3, H, W] in [-1, 1], without autograd."""
        with self._grad_ctx():
            return self.place.decode(list(latents))

    def generate(
        self,
        prompts: Sequence[str] | None = None,
        negative_prompts: Sequence[str] | None = None,
        sizes: Size | Sequence[Size] = (1024, 1024),
        steps: int = 50,
        cfg: float = DEFAULT_CFG,
        shift: float = 5.0,
        skip_layers: Sequence[int] | None = None,
        seeds: Sequence[int] | None = None,
        batch: int = 4,
        cfg_mode: str = "packed",
        output: str = "pil",
        prompt_embeds: Sequence[Tensor] | None = None,
        negative_embeds: Sequence[Tensor] | None = None,
        step_callback: Callable[[int, Tensor], None] | None = None,
        progress: bool = True,
    ) -> GenerateResult:
        """One image per prompt. Sample i's noise and conditioning depend only on its
        prompt, size and seed (no other sample's content reaches it, its slot does
        not matter); bitwise it also depends on the batch's shapes, which set the
        kernels' problem shapes. `progress` shows tqdm bars (sampler steps per batch,
        decode) at INFO level."""
        start = time.perf_counter()
        te = self.place.text_encoder
        if te is not None and hasattr(te, "progress"):
            te.progress = progress
        vram = open_window(self.device)
        with self._grad_ctx():
            pos, neg = self._embeddings(
                prompts, negative_prompts, prompt_embeds, negative_embeds, cfg
            )
        release_cache(self.device)
        n = len(pos)
        seeds = list(seeds) if seeds is not None else list(range(n))
        size_list = [sizes] * n if isinstance(sizes[0], int) else list(sizes)
        chunks = [slice(s, s + batch) for s in range(0, n, batch)]
        negs = [None] * len(chunks) if neg is None else [neg[c] for c in chunks]
        batch_max = self._largest_batch(pos, neg, size_list, chunks, cfg_mode)
        est = self._log_estimate(batch_max, max(h * w for h, w in size_list))
        call = dict(
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
        )
        log.info(
            f"sampling {hl(n)} image(s) in {hl(len(chunks))} batch(es)",
            steps=steps,
            cfg=cfg,
            shift=shift,
            cfg_mode=cfg_mode,
            sizes=sorted({tuple(s) for s in size_list}),
        )
        sample = partial(
            self._sample_chunk,
            steps=steps,
            cfg=cfg,
            shift=shift,
            skip_layers=skip_layers,
            cfg_mode=cfg_mode,
            step_callback=step_callback,
            progress=progress,
        )
        convert = OUTPUTS[output]
        step_stats, latents = [], []
        with self._eval(), bars():
            for b, (chunk, chunk_neg) in enumerate(zip(chunks, negs), start=1):
                desc = f"batch {b}/{len(chunks)}"
                res, stats = sample(
                    pos[chunk], chunk_neg, size_list[chunk], seeds[chunk], desc
                )
                step_stats.append(stats)
                latents += unpack(res.latents, res.hw)
            release_cache(self.device)
            if output != "latent":
                with timed(log, f"decoding {n} image(s)"):
                    images = convert(self.place, latents, _decode_bar(progress, n))
            else:
                images = convert(self.place, latents)
        self.retune()
        decisions.flush("kernel bindings of this generate")
        release_cache(self.device)
        took = time.perf_counter() - start
        log.info(f"generate: {hl(n)} image(s) in {hl(f'{took:.2f} s')}")
        measured = peaks.vram_summary(self.device, *close_window(vram))
        _log_vram(measured, est)
        settings = dict(vram=measured, step_stats=step_stats, **call)
        return GenerateResult(images, latents, {**settings, **self._model_settings()})

    def _log_estimate(self, batch_max: int, pixels: int) -> dict:
        """The per-phase device estimate of a call whose largest forward is
        `batch_max` tokens and largest image `pixels`, logged as a table."""
        est = peaks.estimated(
            self.backbone, self.memory_plan, batch_max, self.place, pixels
        )
        peak_gib = f"{est['peak']:.2f} GiB"
        log_table(
            log,
            f"estimated peak {hl(peak_gib)} (the largest phase)",
            ("phase", "GiB", "on the device"),
            [
                (
                    "text encode",
                    est["encode"],
                    "DiT weights + text-encoder pass + resident VAE",
                ),
                ("sampling", est["sampling"], f"DiT forward of {batch_max} tokens"),
                (
                    "decode",
                    est["decode"],
                    "DiT weights + resident text encoder + VAE pass",
                ),
            ],
        )
        return est

    def _sample_chunk(
        self,
        pos,
        neg,
        sizes,
        seeds,
        desc: str,
        *,
        steps,
        cfg,
        shift,
        skip_layers,
        cfg_mode,
        step_callback,
        progress,
    ):
        """One packed batch through the sampler under the sampling autocast, timed per
        step; returns (sample result, step stats)."""
        sigmas = moss_sigmas(steps, shift).tolist()
        clock = StepClock(self.device, len(sigmas) - 1, self.weight_stream)
        clock.start()
        autocast = sampling_autocast(self.backbone, self.device)
        with (
            autocast(),
            log_bar(
                total=len(sigmas) - 1, desc=desc, unit="step", enabled=progress
            ) as pbar,
        ):
            res = sample_latents(
                self.backbone,
                pos,
                neg,
                sizes,
                steps,
                cfg,
                shift,
                skip_layers,
                seeds,
                self.device,
                cfg_mode,
                partial(_step_report, pbar, sigmas, step_callback, clock),
            )
        return res, {"rows": clock.rows, **clock.summary()}

    def _model_settings(self) -> dict:
        """The model side of a generate's settings: dtype, fp16 fix, fold, sources."""
        fold = self.backbone.fold_scales
        return dict(
            dtype=str(next(self.backbone.parameters()).dtype),
            fp16_fix=self.backbone.fp16_fix,
            fold=None if fold is None else fold[0],
            sources=dict(self.sources),
        )

    @staticmethod
    def _largest_batch(pos, neg, sizes, chunks, cfg_mode) -> int:
        """The largest packed forward of this call, in tokens."""
        neg = neg or pos
        lf = LATENT_FACTOR
        joint = [
            (h // lf) * (w // lf) + max(p.shape[0], q.shape[0])
            for (h, w), p, q in zip(sizes, pos, neg)
        ]
        factor = 2 if cfg_mode == "packed" else 1
        return max(factor * sum(joint[c]) for c in chunks)

    def _embeddings(
        self, prompts, negative_prompts, prompt_embeds, negative_embeds, cfg
    ):
        texts = [] if prompt_embeds is not None else list(prompts)
        encode_neg = cfg != 1.0 and negative_prompts and negative_embeds is None
        if encode_neg:
            texts += [p for p in negative_prompts if p]
        enc = []
        if texts:
            with timed(log, f"encoding {len(texts)} prompt(s)") as done:
                enc = self.place.encode(texts)
                done["tokens"] = sum(e.shape[0] for e in enc)
        n_pos = 0 if prompt_embeds is not None else len(prompts)
        pos = list(prompt_embeds) if prompt_embeds is not None else enc[:n_pos]
        if encode_neg:
            it = iter(enc[n_pos:])
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


def _log_vram(measured: dict | None, est: dict) -> None:
    """The measured peak of a generate call next to its estimate (CUDA devices)."""
    if not measured:
        return
    peak = f"{measured['peak_alloc_gib']:.2f} GiB"
    log.info(
        f"peak VRAM {hl(peak)} allocated",
        peak_reserved_gib=measured["peak_reserved_gib"],
        device_used_gib=measured["device_used_gib"],
        estimated_gib=round(est["peak"], 2),
    )
