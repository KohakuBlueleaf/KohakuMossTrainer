"""fp16 overflow fixes. The FP16_FIX knob of the train and sample scripts is none |
online | fp32_residual; `weights` is the mode a model carries after `fold` (an fp16-fix
weights file records it), never a knob value.

- weights: the uniform gate fold (+ pre-gate scale) written into the weights (`fold`).
- online: the original weights; the same factors, per block if wanted, applied at run
  time through the StreamConfig the blocks bind (gate multipliers, writer input scales,
  per-block residual rescale, exact eps).
- fp32_residual: residual streams kept in fp32; norm outputs cast to the compute dtype.
Any mode can add `clamp` (saturating clamp of the joint stream with on-device counters).
"""

from collections.abc import Sequence
from dataclasses import replace

import torch

from kohakumoss.model.backbone import MossBackbone
from kohakumoss.model.numerics.fold import (
    fold,
    fold_record,
    norm_eps_scales,
    residual_ratios,
    same_fold,
    set_norm_eps,
    unfold,
)
from kohakumoss.model.stream import ClampCounter, StreamConfig

FP16_MAX = 65504.0
MODES = ("none", "weights", "online", "fp32_residual")


def _per_block(v: float | Sequence[float], depth: int) -> tuple[float, ...]:
    if isinstance(v, (int, float)):
        return (float(v),) * depth
    v = tuple(float(x) for x in v)
    if len(v) != depth:
        raise ValueError(f"need {depth} per-block factors, got {len(v)}")
    return v


def online_stream(
    model: MossBackbone,
    c: float | Sequence[float],
    s_attn: float | Sequence[float] = 1.0,
    s_ffn: float | Sequence[float] = 1.0,
    clamp: float | None = None,
) -> StreamConfig:
    """Stream scale c_l per block, pre-gate output scale s per block (o, ffn_down)."""
    depth = model.cfg.depth
    cs, sa, sf = (_per_block(v, depth) for v in (c, s_attn, s_ffn))
    return StreamConfig(
        text_scale=cs[0],
        embed_scale=cs[0],
        gate_attn=tuple(cl / s for cl, s in zip(cs, sa)),
        gate_ffn=tuple(cl / s for cl, s in zip(cs, sf)),
        in_attn=sa,
        in_ffn=sf,
        clamp=clamp,
    )


def _factors(mode: str, model: MossBackbone, kw: dict) -> dict:
    """The factors a mode is (or would be) applied with, normalised for comparison."""
    depth = model.cfg.depth
    match mode:
        case "weights":
            return {
                "c": float(kw["c"]),
                "s_attn": float(kw.get("s_attn", 1.0)),
                "s_ffn": float(kw.get("s_ffn", 1.0)),
            }
        case "online":
            return {
                "c": _per_block(kw["c"], depth),
                "s_attn": _per_block(kw.get("s_attn", 1.0), depth),
                "s_ffn": _per_block(kw.get("s_ffn", 1.0), depth),
            }
    return {}


def current_factors(model: MossBackbone) -> dict:
    """The factors of the fix the model carries now ({} for none / fp32_residual)."""
    if model.fp16_fix == "weights":
        return {
            "c": model.fold_scales[0],
            "s_attn": model.fold_pregate[0],
            "s_ffn": model.fold_pregate[1],
        }
    return dict(getattr(model, "_online_factors", {}))


@torch.no_grad()
def apply_fp16_fix(model: MossBackbone, mode: str = "none", **kw) -> None:
    """Apply one FP16_FIX mode in place and re-bind the model's ops.

    weights: c, s_attn, s_ffn (uniform), eps ("exact" | "stock"), clamp.
    online: c, s_attn, s_ffn (float or per block), clamp.
    fp32_residual: clamp. `clamp=True` clamps at the fp16 max.
    "none" keeps a fix the model already carries (a loaded fp16-fix file); asking
    for the fix it carries is a no-op; any other change needs `clear_fp16_fix` first."""
    if mode not in MODES:
        raise ValueError(f"FP16_FIX={mode!r} not in {MODES}")
    clamp = kw.get("clamp")
    clamp = FP16_MAX if clamp is True else (clamp or None)
    current = model.fp16_fix
    if mode == "none" or mode == current:
        asks = mode != "none" and "c" in kw
        if asks and _factors(mode, model, kw) != current_factors(model):
            raise ValueError(
                f"model carries FP16_FIX={current} with {current_factors(model)}; "
                f"requested {_factors(mode, model, kw)}; clear_fp16_fix first"
            )
        model.set_stream_config(replace(model.stream, clamp=clamp))
        return
    if current != "none":
        raise ValueError(
            f"model carries FP16_FIX={current}; clear_fp16_fix before applying {mode}"
        )
    match mode:
        case "weights":
            f = _factors(mode, model, kw)
            pregate = (f["s_attn"], f["s_ffn"])
            fold(model, f["c"], "gate", pregate, kw.get("eps", "exact"))
            stream = StreamConfig(clamp=clamp)
        case "online":
            f = _factors(mode, model, kw)
            stream = online_stream(model, f["c"], f["s_attn"], f["s_ffn"], clamp)
            model._online_eps = set_norm_eps(
                model, norm_eps_scales(model, f["c"], per_block=True)
            )
            model._online_factors = f
            model.set_residual_scale(residual_ratios(f["c"]))
        case "fp32_residual":
            stream = StreamConfig(stream_dtype=torch.float32, clamp=clamp)
    model.fp16_fix = mode
    model.set_stream_config(stream)


@torch.no_grad()
def clear_fp16_fix(model: MossBackbone) -> None:
    """Undo whatever fix the model carries: unfold weights, restore online eps and
    stream scales, reset the stream config."""
    match model.fp16_fix:
        case "weights":
            unfold(model)
        case "online":
            for name, eps in model._online_eps.items():
                model.get_submodule(name).eps = eps
            model._online_factors = {}
            model.set_residual_scale((1.0,) * model.cfg.depth)
    model.fp16_fix = "none"
    model.set_stream_config(StreamConfig())


@torch.no_grad()
def carry_fold(model: MossBackbone, record: dict | None) -> None:
    """Make `model` carry exactly the weight fold of `record` (a `fold_record`; None:
    unfolded); when the folds differ, whatever fix the model carries is cleared first.
    Exact for power-of-two factors."""
    if same_fold(fold_record(model), record):
        return
    clear_fp16_fix(model)
    unfold(model)
    if record is None:
        return
    pregate = tuple(float(s) for s in record["pregate"])
    fold(model, record["scales"], record["via"], pregate, record.get("eps", "exact"))
    model.fp16_fix = "weights"


def fix_problems(
    fix: str, kwargs: dict, fp16: bool, carried: bool, subject: str, file_knob: str
) -> list[str]:
    """The FP16_FIX knob problems every script shares: 'online' needs `c`; fp16 compute
    (`subject`, the knob text naming it) needs a fix unless the weights file
    (`file_knob`) carries one (`carried`)."""
    p = []
    if fix == "online" and "c" not in kwargs:
        p.append(
            "FP16_FIX='online' needs FP16_FIX_KWARGS['c'] (e.g. 1/16 for the base)"
        )
    if fp16 and fix == "none" and not carried:
        p.append(
            f"{subject} needs FP16_FIX ('online' with FP16_FIX_KWARGS={{'c': 1/16}} or "
            f"'fp32_residual') or an fp16-fix {file_knob} file: the residual stream "
            "exceeds the fp16 range"
        )
    return p


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
