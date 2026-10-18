"""TrainStep (driver-independent) and the HostBatch -> DeviceBatch transfer.

An online cache hit arrives with `latents` / `text` set and is used as is; a miss
arrives with `pixels` / `prompts` and is encoded.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import torch
from torch import Tensor, nn

from kohakumoss.diffusion.objective import TokenNormalizer, noise_latents
from kohakumoss.types.batch import DeviceBatch
from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss.types.step import StepOut
from kohakumoss_common.calls import nothing
from kohakumoss_data.seeds import seed_of


def _move(t: Tensor, device) -> Tensor:
    return t.to(device, non_blocking=True)


def _mask_move(hb, device) -> Tensor:
    return hb.loss_mask.to(device, non_blocking=True)


def _cached_text(hb, device) -> tuple[Tensor, Tensor]:
    return _move(hb.text, device), hb.text_len


def _cached_latents(hb, device) -> tuple[Tensor, None]:
    return _move(hb.latents, device), None


def cap_text(parts: list[Tensor], limits: list[int]) -> list[Tensor]:
    """Keep the first `limits[i]` rows of sample i (0 or no limits = all rows)."""
    if not limits:
        return parts
    return [p[: limit or None] for p, limit in zip(parts, limits)]


def _as_is(x, *args):
    return x


def _encode_prompts(encode: Callable, hb) -> list[Tensor]:
    return encode(hb.prompts)


class OnlineText:
    """A cache hit's `hb.text`, else `hb.prompts` encoded, rounded as the cache stores it
    (`as_cached`, an online cache's `texts.as_cached`; identity with no cache) and
    capped to `hb.text_limit`; the kept lengths become `text_len` and fill
    `hb.text_len`. `source(hb)` gives the encodings (default: EncodeText on
    `hb.prompts`; encode-ahead hands over a pass made earlier)."""

    def __init__(
        self,
        encode: Callable,
        as_cached: Callable = _as_is,
        source: Callable | None = None,
    ):
        self.encode = encode
        self.as_cached = as_cached
        self.source = source or partial(_encode_prompts, encode)

    @torch.no_grad()
    def __call__(self, hb, device) -> tuple[Tensor, Tensor]:
        if hb.text is not None:
            return _cached_text(hb, device)
        parts = cap_text(self.as_cached(self.source(hb)), hb.text_limit)
        lens = torch.tensor([p.shape[0] for p in parts], dtype=torch.int32)
        hb.text_len = lens
        return _move(torch.cat(parts, 0), device), lens


def gather_kept(latents: Tensor, hw: Tensor, img_keep) -> Tensor:
    """Packed full-grid latents -> only the kept tokens of each sample (`img_keep`)."""
    if img_keep is None:
        return latents
    sizes = (hw[:, 0] * hw[:, 1]).tolist()
    parts = latents.split(sizes, 0)
    return torch.cat(
        [
            p if keep is None else p.index_select(0, _move(keep, p.device))
            for p, keep in zip(parts, img_keep)
        ],
        0,
    )


class OnlineLatents:
    """A cache hit's `hb.latents` (already the kept tokens), else `hb.pixels` encoded at
    the planned latent grid `hb.hw` (EncodeImages), rounded as the cache stores it
    (`as_cached(full, hw)`, an online cache's `latents.as_cached`; identity with no
    cache) and gathered to the token-drop subset `hb.img_keep`. Returns (kept latents,
    the raw full-grid encoding or None for a hit). `source(pixels, hw)` gives the
    encoding (default: `encode`; encode-ahead hands over a pass made earlier)."""

    def __init__(
        self,
        encode: Callable,
        as_cached: Callable = _as_is,
        source: Callable | None = None,
    ):
        self.encode = encode
        self.as_cached = as_cached
        self.source = source or encode

    @torch.no_grad()
    def __call__(self, hb, device) -> tuple[Tensor, Tensor | None]:
        if hb.latents is not None:
            return _cached_latents(hb, device)
        full = _move(self.source(hb.pixels, hb.hw), device)
        kept = gather_kept(self.as_cached(full, hb.hw), hb.hw, hb.img_keep)
        return kept, full


class LayoutTransform:
    """Loader transform (runs in the worker): sets the batch's `layout` field to its
    SeqInfo when its text lengths are final (text already encoded: offline cache or
    online hit); batches whose prompts are encoded on the device keep None."""

    def __init__(self, rope_axes: int = 4):
        self.rope_axes = rope_axes

    def __call__(self, hb):
        if hb.prompts is None:
            hb.layout = build_seqinfo(
                hb.hw, hb.text_len, self.rope_axes, img_keep=hb.img_keep
            )
        return hb


class DeviceBatcher:
    """HostBatch -> DeviceBatch: text / latent strategy, mask transfer and cache
    write-back are bound at construction. `on_encoded(hb, full_latents, text,
    text_len)` sees the full-grid latents of an encoded batch (None for a cache hit)
    and the kept text rows. A batch carrying `layout` (LayoutTransform) uses it; the
    others get their SeqInfo built here."""

    def __init__(
        self,
        device: torch.device,
        rope_axes: int = 4,
        text: Callable = _cached_text,
        latents: Callable = _cached_latents,
        on_encoded: Callable = nothing,
        masked: bool = False,
    ):
        self.device = device
        self.rope_axes = rope_axes
        self.text = text
        self.latents = latents
        self.on_encoded = on_encoded
        self.mask = _mask_move if masked else nothing

    def __call__(self, hb) -> DeviceBatch:
        text, text_len = self.text(hb, self.device)
        latents, full = self.latents(hb, self.device)
        self.on_encoded(hb, full, text, text_len)
        layout = hb.layout
        if layout is None:
            layout = build_seqinfo(
                hb.hw, text_len, self.rope_axes, img_keep=hb.img_keep
            )
        seq = layout.to(self.device)
        return DeviceBatch(
            ids=hb.ids,
            seq=seq,
            latents=latents,
            text=text,
            text_drop=_move(hb.text_drop, self.device),
            weight=_move(hb.weight, self.device),
            loss_mask=self.mask(hb, self.device),
            pixels=hb.pixels,
            prompts=hb.prompts,
        )


def image_tokens_per_sample(hb) -> list[int]:
    """Image tokens of each sample after token drop (`img_keep`)."""
    full = (hb.hw[:, 0] * hb.hw[:, 1]).tolist()
    keeps = hb.img_keep or [None] * len(full)
    return [n if k is None else int(k.numel()) for n, k in zip(full, keeps)]


def image_token_count(hb) -> int:
    """Image tokens of a host batch (kept tokens only)."""
    return sum(image_tokens_per_sample(hb))


def mask_token_count(hb) -> int:
    """Loss-mask weight of a host batch (masked losses)."""
    return int(hb.loss_mask.sum())


def token_counter(masked: bool) -> Callable:
    """The host-side loss-token count for the configured loss."""
    return mask_token_count if masked else image_token_count


def _batch_mask(batch: DeviceBatch) -> Tensor:
    return batch.loss_mask


def _image_count(batch: DeviceBatch) -> int:
    return batch.seq.n_img


def _mask_count(batch: DeviceBatch) -> Tensor:
    return batch.loss_mask.sum()


@dataclass
class Prepared:
    """A noised batch: model inputs (z, text, sigma, seq), target, weight, mask."""

    z: Tensor
    text: Tensor
    sigma: Tensor
    seq: object
    target: Tensor
    weight: Tensor
    mask: Tensor | None


class TrainStep:
    """sigma ~ sampler; z_sigma = (1-sigma) z0 + sigma eps; v = backbone; token loss.

    `__call__(batch, denom, step)` takes the global token denominator of the optimiser
    step; `denominator(batch)` gives the one of a lone batch (validation, tests). The
    sigma and noise generators are seeded from `(seed, rank)`. `keep(prepared, denom)`
    sees every prepared micro-batch (the non-finite recorder's last one).
    """

    def __init__(
        self,
        backbone: nn.Module,
        loss_fn: Callable,
        weighting: Callable,
        sampler,
        normalizer: TokenNormalizer,
        autocast: Callable,
        device: torch.device,
        seed: int = 0,
        masked: bool = False,
        pre_forward: Callable | None = None,
        rank: int = 0,
        keep: Callable | None = None,
    ):
        self.pre_forward = pre_forward or nothing
        self.keep = keep or nothing
        self.backbone = backbone
        self.loss_fn = loss_fn
        self.weighting = weighting
        self.sample_sigma = sampler.sample
        self.normalizer = normalizer
        self.autocast = autocast
        self.device = device
        self.mask_of = _batch_mask if masked else nothing
        self.count_of = _mask_count if masked else _image_count
        self.sigma_gen = torch.Generator().manual_seed(seed_of("sigma", seed, rank))
        self.noise_gen = torch.Generator(device=device).manual_seed(
            seed_of("noise", seed, rank)
        )

    def denominator(self, batch: DeviceBatch) -> Tensor:
        return self.normalizer.denominator(self.count_of(batch), self.device)

    def prepare(self, batch: DeviceBatch, step: int) -> Prepared:
        """Noise the batch (`noise`) and install its adapter step state
        (`pre_forward`) for the forward that follows."""
        p = self.noise(batch, step)
        self.pre_forward(p.sigma, p.seq, step)
        return p

    def noise(self, batch: DeviceBatch, step: int) -> Prepared:
        """Model inputs, velocity target, per-sample loss weight; the adapter step
        state is left to the caller (pipelines install it per micro-batch)."""
        seq = batch.seq
        z0 = batch.latents.float()
        sigma = _move(self.sample_sigma(seq.batch, self.sigma_gen, step), self.device)
        eps = torch.randn(
            z0.shape, generator=self.noise_gen, device=self.device, dtype=torch.float32
        )
        z_sigma, target = noise_latents(z0, eps, sigma, seq)
        weight = self.weighting(sigma, batch.weight.float())
        return Prepared(
            z_sigma, batch.text, sigma, seq, target, weight, self.mask_of(batch)
        )

    def loss(self, pred: Tensor, p: Prepared, denom: Tensor) -> Tensor:
        """Token loss of `pred` for prepared batch `p`, over the step's denominator."""
        loss_sum = self.loss_fn(pred, p.target, p.seq.seg_img, p.weight, p.mask)
        return self.normalizer(loss_sum, denom)

    def __call__(self, batch: DeviceBatch, denom: Tensor, step: int = 0) -> StepOut:
        p = self.prepare(batch, step)
        self.keep(p, denom)
        with self.autocast():
            pred = self.backbone(p.z, p.text, p.sigma, p.seq)
        return StepOut(self.loss(pred, p, denom), denom, {"sigma_mean": p.sigma.mean()})

    def rng_state(self) -> dict:
        return {
            "sigma_gen": self.sigma_gen.get_state(),
            "noise_gen": self.noise_gen.get_state(),
        }

    def load_rng_state(self, state: dict) -> None:
        self.sigma_gen.set_state(state["sigma_gen"])
        self.noise_gen.set_state(state["noise_gen"])
