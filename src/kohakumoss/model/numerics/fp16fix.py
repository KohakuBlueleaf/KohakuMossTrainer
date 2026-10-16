"""fp16 overflow fixes, one knob: FP16_FIX = none | weights | online | fp32_residual.

- weights: the uniform gate fold (+ pre-gate scale) written into the weights (`fold`).
- online: the original weights; the same factors, per block if wanted, applied at run
  time through the StreamPolicy the blocks bind (gate multipliers, writer input scales,
  per-block residual rescale, exact eps).
- fp32_residual: residual streams kept in fp32; norm outputs cast to the compute dtype.
Any mode can add `clamp` (saturating clamp of the joint stream with on-device counters).
"""

from collections.abc import Sequence

import torch

from kohakumoss.model.backbone import MossBackbone
from kohakumoss.model.numerics.fold import (
    fold,
    norm_eps_scales,
    residual_ratios,
    set_norm_eps,
)
from kohakumoss.model.stream import ClampCounter, StreamPolicy

FP16_MAX = 65504.0
MODES = ("none", "weights", "online", "fp32_residual")


def _per_block(v: float | Sequence[float], depth: int) -> tuple[float, ...]:
    if isinstance(v, (int, float)):
        return (float(v),) * depth
    v = tuple(float(x) for x in v)
    if len(v) != depth:
        raise ValueError(f"need {depth} per-block factors, got {len(v)}")
    return v


def online_policy(
    model: MossBackbone,
    c: float | Sequence[float],
    s_attn: float | Sequence[float] = 1.0,
    s_ffn: float | Sequence[float] = 1.0,
    clamp: float | None = None,
) -> StreamPolicy:
    """Stream scale c_l per block, pre-gate output scale s per block (o, ffn_down)."""
    depth = model.cfg.depth
    cs, sa, sf = (_per_block(v, depth) for v in (c, s_attn, s_ffn))
    return StreamPolicy(
        text_scale=cs[0],
        embed_scale=cs[0],
        gate_attn=tuple(cl / s for cl, s in zip(cs, sa)),
        gate_ffn=tuple(cl / s for cl, s in zip(cs, sf)),
        in_attn=sa,
        in_ffn=sf,
        clamp=clamp,
    )


@torch.no_grad()
def apply_fp16_fix(model: MossBackbone, mode: str = "none", **kw) -> None:
    """Apply one FP16_FIX mode in place and re-bind the model's ops.

    weights: c, s_attn, s_ffn (uniform), eps ("exact" | "stock"), clamp.
    online: c, s_attn, s_ffn (float or per block), clamp.
    fp32_residual: clamp. `clamp=True` clamps at the fp16 max."""
    clamp = kw.get("clamp")
    clamp = FP16_MAX if clamp is True else (clamp or None)
    match mode:
        case "none":
            policy = StreamPolicy(clamp=clamp)
        case "weights":
            if model.fold_scales is None:
                pregate = (float(kw.get("s_attn", 1.0)), float(kw.get("s_ffn", 1.0)))
                fold(model, float(kw["c"]), "gate", pregate, kw.get("eps", "exact"))
            policy = StreamPolicy(clamp=clamp)
        case "online":
            policy = online_policy(
                model, kw["c"], kw.get("s_attn", 1.0), kw.get("s_ffn", 1.0), clamp
            )
            cs = _per_block(kw["c"], model.cfg.depth)
            model._online_eps = set_norm_eps(
                model, norm_eps_scales(model, cs, per_block=True)
            )
            model.set_residual_scale(residual_ratios(cs))
        case "fp32_residual":
            policy = StreamPolicy(stream_dtype=torch.float32, clamp=clamp)
        case _:
            raise ValueError(f"FP16_FIX={mode!r} not in {MODES}")
    model.fp16_fix = mode
    model.set_stream_policy(policy)


def clamp_counts(model: MossBackbone) -> dict[int, int]:
    """Clamped-value count per joint block (empty when clamping is off); one sync."""
    return {
        i: int(b._post.count)
        for i, b in enumerate(model.layers)
        if isinstance(b._post, ClampCounter)
    }


def choose_factors(probe: dict, target: float = 2.0**12) -> dict[str, float]:
    """Powers of two from an activation-probe report: stream max * c <= target and
    pre-gate max * s <= target."""

    def pow2(peak: float) -> float:
        k = 0
        while peak * 2.0**-k > target:
            k += 1
        return 2.0**-k

    summary = probe["summary_max"]
    stream = max(summary.get("stream", 0.0), summary.get("text_stream", 0.0))
    return {
        "c": pow2(stream),
        "s_attn": pow2(summary.get("o", 0.0)),
        "s_ffn": pow2(summary.get("ffn_down", 0.0)),
    }
