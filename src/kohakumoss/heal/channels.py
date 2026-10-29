"""Channel statistics and the shrink of massive channels (pure functions).

The shrink scales stream channel c by D_c in every writer (embedders, text-refiner proj /
ffn_down, joint o / ffn_down) and every reader norm gain by 1 / D_c (times 1 / r for the
calibrated divisor ratio r of that norm). See docs/concepts/massive-channels.md."""

import numpy as np
import torch
from torch import Tensor

from kohakumoss.heal.sites import norm_path


def peak_ratio(mean_square: dict[str, np.ndarray]) -> np.ndarray:
    """M_c: max over sites of channel c's RMS over the median channel's RMS."""
    rel = [np.sqrt(v) / np.median(np.sqrt(v)) for v in mean_square.values()]
    return np.stack(rel).max(0)


def shrink_scales(peak: np.ndarray, tau: float) -> Tensor:
    """D_c = min(1, tau / M_c) (fp32)."""
    return torch.tensor(np.minimum(1.0, tau / peak), dtype=torch.float32)


def top_channels(peak: np.ndarray, n: int = 32) -> list[list]:
    """[[channel, M_c], ...] of the n largest M_c."""
    order = np.argsort(-peak)[:n]
    return [[int(c), round(float(peak[c]), 2)] for c in order]


def writer_names(text_blocks: int, depth: int) -> list[str]:
    """Every parameter that writes the residual stream (weights and their biases)."""
    names = []
    for name in ("video_embedder", "text_embedder.proj"):
        names += [f"{name}.weight", f"{name}.bias"]
    for t in range(text_blocks):
        for lin in ("proj", "ffn_down"):
            names += [
                f"text_embedder.blocks.{t}.{lin}.weight",
                f"text_embedder.blocks.{t}.{lin}.bias",
            ]
    names += [
        f"layers.{i}.proj.{w}.weight" for i in range(depth) for w in ("o", "ffn_down")
    ]
    return names


def shrink_table(
    text_blocks: int,
    depth: int,
    sites: list[str],
    d: Tensor,
    rbar: dict[str, float] | None = None,
) -> dict[str, Tensor]:
    """Parameter name -> per-output-row multiplier of the shrink: writers times d, each
    reader norm gain times 1 / (d r) (r = rbar[site], 1 when `rbar` is None)."""
    out = {name: d for name in writer_names(text_blocks, depth)}
    inv = d.reciprocal()
    for site in sites:
        r = 1.0 if rbar is None else float(rbar[site])
        out[f"{norm_path(site)}.weight"] = inv / r
    return out


def scaled(t: Tensor, m: Tensor) -> Tensor:
    """`t` times the per-output-row multiplier `m` in fp32, rounded once to t's dtype."""
    m = m.to(device=t.device, dtype=torch.float32)
    return (t.float() * m.reshape(-1, *([1] * (t.dim() - 1)))).to(t.dtype)


def stream_stats(x: Tensor, massive: Tensor | None = None) -> list[float]:
    """[rms, max |x|, top-8 channel energy share, energy share of `massive` channels]
    of a stream [N, D] (fp32 reductions; the last entry 0 without `massive`)."""
    xf = x.float()
    e = xf.square().sum(0)
    tot = float(e.sum()) + 1e-30
    shrunk = (
        float(e[massive].sum()) / tot
        if massive is not None and massive.numel()
        else 0.0
    )
    return [
        float(xf.square().mean().sqrt()),
        float(xf.abs().max()),
        float(e.topk(8).values.sum()) / tot,
        shrunk,
    ]


def ratio_summary(img: Tensor, txt: Tensor) -> dict:
    """Per-token divisor ratios of one site: `rbar` (median over all tokens) and, per
    token group, the 5 / 50 / 95 % quantiles of ratio / rbar."""
    both = torch.cat([img, txt])
    med = float(both.median())
    out = {"rbar": med}
    for name, v in (("img", img), ("txt", txt)):
        if v.numel():
            q = torch.quantile(v.float() / med, torch.tensor([0.05, 0.5, 0.95]))
            out[name] = [round(float(x), 4) for x in q]
    return out
