"""Per-layer tensor sets for each export format: `native` is what LyCORIS saves,
`comfy` what ComfyUI's weight adapters evaluate to the same dW (see
docs/guides/export-to-comfy.md)."""

import torch
from torch import Tensor

from kohakumoss.adapters.kinds.common import diff_set
from kohakumoss.adapters.target import NormTarget, Target


def native(t: Target) -> dict[str, Tensor]:
    """The LyCORIS module's own state dict (scalar folded, LyCORIS alpha rules)."""
    return {k: v.detach() for k, v in t.lyco.state_dict().items()}


def norm_native(n: NormTarget) -> dict[str, Tensor]:
    w, b = n.delta()
    out = {"w_norm": w}
    if b is not None:
        out["b_norm"] = b
    return out


def row_norm(w: Tensor) -> Tensor:
    """Per-output-row L2 norm, [O, 1]."""
    return w.reshape(w.shape[0], -1).norm(dim=1, keepdim=True)


def _comfy_dora_scale(t: Target, dora_scale: Tensor) -> Tensor:
    """ComfyUI normalises output-axis DoRA by ||W|| instead of ||W + dW||; fold that in."""
    if not t.wd_on_out:
        return dora_scale
    w = t.base_weight(torch.float32)
    dw = t.unit_fn().to(w.device)
    eps = torch.finfo(torch.float32).eps
    ratio = (row_norm(w) + eps) / (row_norm(w + dw) + eps)
    return dora_scale.float() * ratio.to(dora_scale.device)


def comfy(t: Target) -> dict[str, Tensor]:
    """Tensors ComfyUI evaluates to this layer's dW at strength 1."""
    if t.record.comfy is not None:
        return t.record.comfy(t)
    out = native(t)
    if t.wd and "dora_scale" in out:
        out["dora_scale"] = _comfy_dora_scale(t, out["dora_scale"])
    return out


def reciprocal(scale: Tensor | float) -> Tensor | float:
    return 1.0 / scale.float() if isinstance(scale, Tensor) else 1.0 / float(scale)


def _per_row(scale) -> bool:
    return (
        isinstance(scale, Tensor)
        and scale.numel() > 1
        and not bool((scale == scale.flatten()[0]).all())
    )


def _scaled_diff(t: Target, scale) -> dict[str, Tensor]:
    dw, db = t.delta(1.0)
    s = scale.float() if isinstance(scale, Tensor) else torch.tensor(float(scale))
    s = s.to(dw.device).reshape(-1, *[1] * (dw.dim() - 1))
    out = {"diff": dw * s}
    if db is not None:
        out["diff_b"] = db * s.flatten()
    return out


def apply_row_scale(t: Target, tensors: dict, scale: Tensor | float) -> dict:
    """Multiply this layer's dW rows by `scale`: one factor per term for a uniform
    scale, an exact `diff` for a per-row one or a set with no such factor."""
    if _per_row(scale):
        return _scaled_diff(t, scale)
    if t.record.stacking == "commuting" and "diff" not in tensors:
        return dict(tensors)
    names = ("diff", "diff_b") if "diff" in tensors else t.record.row_names(tensors)
    if not names:
        return _scaled_diff(t, scale)
    c = float(scale.flatten()[0]) if isinstance(scale, Tensor) else float(scale)
    out = dict(tensors)
    for name in names:
        out[name] = out[name] * c
    if "dora_scale" in out:
        out["dora_scale"] = out["dora_scale"] * c
    return out


__all__ = ["apply_row_scale", "comfy", "diff_set", "native", "reciprocal", "row_norm"]
