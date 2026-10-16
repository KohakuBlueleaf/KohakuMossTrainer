"""Euler sampling over packed batches: fp32 noise / state / CFG, per-sample seeds and sizes.

CFG modes: `packed` runs cond and uncond samples in one packed forward and skips the
negative-branch blocks for the uncond segments only (zeroed gates); `two_pass` runs them
as two forwards (half the tokens per forward).
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

import torch
from torch import Tensor

from kohakumoss.diffusion.sampling.schedule import moss_sigmas
from kohakumoss.types.seqinfo import build_seqinfo

Size = tuple[int, int]


@dataclass
class SampleResult:
    """Packed final latents [N_img, C] and each sample's latent grid `hw` [B, 2]."""

    latents: Tensor
    hw: Tensor


def latent_grid(sizes: Size | Sequence[Size], count: int) -> Tensor:
    """Pixel size(s) -> latent grid [count, 2]; one size broadcasts to every sample."""
    if isinstance(sizes[0], int):
        sizes = [sizes] * count
    if len(sizes) != count:
        raise ValueError(f"{len(sizes)} sizes for {count} samples")
    for h, w in sizes:
        if h <= 0 or w <= 0 or h % 16 or w % 16:
            raise ValueError(f"size {(h, w)} must be positive multiples of 16")
    return torch.tensor([[h // 16, w // 16] for h, w in sizes])


def initial_noise(seeds: Sequence[int], hw: Tensor, channels: int, device) -> Tensor:
    """Packed fp32 noise; sample i depends only on (seeds[i], hw[i])."""
    out = []
    for seed, (h, w) in zip(seeds, hw.tolist()):
        g = torch.Generator(device).manual_seed(int(seed))
        z = torch.randn((1, channels, h, w), generator=g, device=device)
        out.append(z[0].flatten(1).transpose(0, 1))
    return torch.cat(out)


def unpack(latents: Tensor, hw: Tensor) -> list[Tensor]:
    """Packed [N_img, C] -> list of [1, C, h, w]."""
    out, start = [], 0
    for h, w in hw.tolist():
        n = h * w
        out.append(latents[start : start + n].transpose(0, 1).reshape(1, -1, h, w))
        start += n
    return out


def _cat(text: Sequence[Tensor], device) -> tuple[Tensor, Tensor]:
    lens = torch.tensor([t.shape[0] for t in text])
    return torch.cat([t.to(device) for t in text]), lens


def _single(backbone, hw, positive, negative, cfg, skip, device) -> Callable:
    text, lens = _cat(positive, device)
    seq = build_seqinfo(hw, lens).to(device)
    forward = backbone.bind_forward()

    def velocity(x, s):
        return forward(x, text, s, seq).float()

    return velocity


def _two_pass(backbone, hw, positive, negative, cfg, skip, device) -> Callable:
    pos_text, pos_len = _cat(positive, device)
    neg_text, neg_len = _cat(negative, device)
    pos_seq = build_seqinfo(hw, pos_len).to(device)
    neg_seq = build_seqinfo(hw, neg_len).to(device)
    cond, uncond = backbone.bind_forward(), backbone.bind_forward(skip)

    def velocity(x, s):
        v = cond(x, pos_text, s, pos_seq).float()
        u = uncond(x, neg_text, s, neg_seq).float()
        return u + cfg * (v - u)

    return velocity


def _packed(backbone, hw, positive, negative, cfg, skip, device) -> Callable:
    b = hw.shape[0]
    pos_text, pos_len = _cat(positive, device)
    neg_text, neg_len = _cat(negative, device)
    text = torch.cat((pos_text, neg_text))
    seq = build_seqinfo(torch.cat((hw, hw)), torch.cat((pos_len, neg_len))).to(device)
    uncond = torch.arange(2 * b, device=device) >= b
    forward = backbone.bind_forward(segment_skip_layers=tuple(sorted(skip)))
    if not skip:
        plain = backbone.bind_forward()
        forward = partial(_drop_mask, plain)

    def velocity(x, s):
        n = x.shape[0]
        out = forward(torch.cat((x, x)), text, torch.cat((s, s)), seq, uncond).float()
        v, u = out[:n], out[n:]
        return u + cfg * (v - u)

    return velocity


def _drop_mask(forward, img, text, sigma, seq, mask):
    return forward(img, text, sigma, seq)


def _noop(i: int, x: Tensor) -> None:
    return None


CFG_MODES = {"packed": _packed, "two_pass": _two_pass}


def sample_latents(
    backbone,
    positive: Sequence[Tensor],
    negative: Sequence[Tensor] | None,
    size: Size | Sequence[Size] = (1024, 1024),
    steps: int = 50,
    cfg: float = 3.0,
    shift: float = 5.0,
    skip_layers: Sequence[int] | None = None,
    seeds: Sequence[int] = (0,),
    device: torch.device | str = "cuda",
    cfg_mode: str = "packed",
    callback: Callable[[int, Tensor], None] | None = None,
) -> SampleResult:
    """Positive / negative: per-sample unpadded embeddings [L_i, text_dim]; `cfg == 1`
    runs the positive branch only. `skip_layers=None` uses the config's negative skip.
    """
    b = len(positive)
    if len(seeds) != b:
        raise ValueError("one seed per sample")
    hw = latent_grid(size, b)
    if skip_layers is None:
        skip_layers = backbone.cfg.negative_skip_layers
    skip = frozenset(skip_layers)
    build = _single if cfg == 1.0 else CFG_MODES[cfg_mode]
    velocity = build(backbone, hw, positive, negative, cfg, skip, device)
    report = callback or _noop
    x = initial_noise(seeds, hw, backbone.cfg.latent_channels, device)
    sigmas = moss_sigmas(steps, shift).to(device)
    for i in range(len(sigmas) - 1):
        v = velocity(x, sigmas[i].expand(b))
        x = x + (sigmas[i + 1] - sigmas[i]) * v
        report(i, x)
    return SampleResult(x, hw)
