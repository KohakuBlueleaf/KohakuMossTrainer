"""TrainStep (driver-independent) and the HostBatch -> DeviceBatch transfer.

Every build-time condition (cached vs online encoders, loss mask on/off) is bound to a
callable at construction; the per-step paths carry no configuration branches. The one
per-batch choice is data: an online cache hit arrives with `latents` / `text` set and
is used as is, a miss arrives with `pixels` / `prompts` and is encoded.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from kohakumoss.diffusion.objective import TokenNormalizer, noise_latents
from kohakumoss.types.batch import DeviceBatch
from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss.types.step import StepOut


def _move(t: Tensor, device) -> Tensor:
    return t.to(device, non_blocking=True)


def _no_mask_move(hb, device) -> None:
    return None


def _mask_move(hb, device) -> Tensor:
    return hb.loss_mask.to(device, non_blocking=True)


def _cached_text(hb, device) -> tuple[Tensor, Tensor]:
    return _move(hb.text, device), hb.text_len


def _cached_latents(hb, device) -> tuple[Tensor, None]:
    return _move(hb.latents, device), None


def _no_write(hb, latents, text, text_len) -> None:
    return None


def cap_text(parts: list[Tensor], limits: list[int]) -> list[Tensor]:
    """Keep the first `limits[i]` rows of sample i (0 or no limits = all rows)."""
    if not limits:
        return parts
    return [p[: limit or None] for p, limit in zip(parts, limits)]


class OnlineText:
    """A cache hit's `hb.text`, else `hb.prompts` encoded (EncodeText) and capped to
    `hb.text_limit`; the kept lengths become `text_len`."""

    def __init__(self, encode: Callable):
        self.encode = encode

    @torch.no_grad()
    def __call__(self, hb, device) -> tuple[Tensor, Tensor]:
        if hb.text is not None:
            return _cached_text(hb, device)
        parts = cap_text(self.encode(hb.prompts), hb.text_limit)
        lens = torch.tensor([p.shape[0] for p in parts], dtype=torch.int32)
        return torch.cat(parts, 0).to(device), lens


def gather_kept(latents: Tensor, hw: Tensor, img_keep) -> Tensor:
    """Packed full-grid latents -> only the kept tokens of each sample (`img_keep`)."""
    if img_keep is None:
        return latents
    sizes = (hw[:, 0] * hw[:, 1]).tolist()
    parts = latents.split(sizes, 0)
    return torch.cat(
        [
            p if keep is None else p.index_select(0, keep.to(p.device))
            for p, keep in zip(parts, img_keep)
        ],
        0,
    )


class OnlineLatents:
    """A cache hit's `hb.latents` (already the kept tokens), else `hb.pixels` encoded at
    the planned latent grid `hb.hw` (EncodeImages) and gathered to the token-drop subset
    `hb.img_keep`. Returns (kept latents, full-grid encoding or None for a hit)."""

    def __init__(self, encode: Callable):
        self.encode = encode

    @torch.no_grad()
    def __call__(self, hb, device) -> tuple[Tensor, Tensor | None]:
        if hb.latents is not None:
            return _cached_latents(hb, device)
        full = self.encode(hb.pixels, hb.hw).to(device)
        return gather_kept(full, hb.hw, hb.img_keep), full


class DeviceBatcher:
    """HostBatch -> DeviceBatch: text / latent strategy, mask transfer and cache
    write-back are bound at construction. `on_encoded(hb, full_latents, text,
    text_len)` sees the full-grid latents of an encoded batch (None for a cache hit)
    and the kept text rows."""

    def __init__(
        self,
        device: torch.device,
        rope_axes: int = 4,
        text: Callable = _cached_text,
        latents: Callable = _cached_latents,
        on_encoded: Callable = _no_write,
        masked: bool = False,
    ):
        self.device = device
        self.rope_axes = rope_axes
        self.text = text
        self.latents = latents
        self.on_encoded = on_encoded
        self.mask = _mask_move if masked else _no_mask_move

    def __call__(self, hb) -> DeviceBatch:
        text, text_len = self.text(hb, self.device)
        latents, full = self.latents(hb, self.device)
        self.on_encoded(hb, full, text, text_len)
        seq = build_seqinfo(hb.hw, text_len, self.rope_axes, img_keep=hb.img_keep).to(
            self.device
        )
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


def _no_pre_forward(sigma: Tensor, seq) -> None:
    return None


class TokenMaskHook:
    """Per-step T-LoRA rank mask from the batch's sigma (`set_token_mask`)."""

    def __init__(self, set_token_mask: Callable, max_rank: int, **kwargs):
        self.set_token_mask = set_token_mask
        self.max_rank = max_rank
        self.kwargs = kwargs

    def __call__(self, sigma: Tensor, seq) -> None:
        self.set_token_mask(sigma, seq.seg_joint, self.max_rank, **self.kwargs)


def _batch_mask(batch: DeviceBatch) -> Tensor:
    return batch.loss_mask


def _no_mask(batch: DeviceBatch) -> None:
    return None


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
    step; `denominator(batch)` gives the one of a lone batch (validation, tests).
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
        pre_forward: Callable = None,
    ):
        self.pre_forward = pre_forward or _no_pre_forward
        self.backbone = backbone
        self.loss_fn = loss_fn
        self.weighting = weighting
        self.sample_sigma = sampler.sample
        self.normalizer = normalizer
        self.autocast = autocast
        self.device = device
        self.mask_of = _batch_mask if masked else _no_mask
        self.count_of = _mask_count if masked else _image_count
        self.sigma_gen = torch.Generator().manual_seed(seed)
        self.noise_gen = torch.Generator(device=device).manual_seed(seed + 1)

    def denominator(self, batch: DeviceBatch) -> Tensor:
        return self.normalizer.denominator(self.count_of(batch), self.device)

    def prepare(self, batch: DeviceBatch, step: int) -> Prepared:
        """Noise the batch: model inputs, velocity target, per-sample loss weight."""
        seq = batch.seq
        z0 = batch.latents.float()
        sigma = self.sample_sigma(seq.batch, self.sigma_gen, step).to(self.device)
        eps = torch.randn(
            z0.shape, generator=self.noise_gen, device=self.device, dtype=torch.float32
        )
        z_sigma, target = noise_latents(z0, eps, sigma, seq)
        self.pre_forward(sigma, seq)
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
