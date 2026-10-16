"""Per-layer tensor sets for each export format.

`native` is exactly what LyCORIS saves. `comfy` is what ComfyUI's weight adapters
evaluate to the same ΔW (see docs/guides/export-to-comfy.md for every conversion).
"""

import torch
from torch import Tensor

from kohakumoss.adapters.delta import delta, unit_delta
from kohakumoss.adapters.target import NormTarget, Target

AS_DIFF = ("ia3",)
UNIFORM_SCALE = {
    "locon": ("lora_up.weight",),
    "loha": ("hada_w1_a",),
    "glora": ("b1.weight",),
    "full": ("diff", "diff_b"),
}


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
    """ComfyUI normalises output-axis DoRA by ||W|| instead of ||W + ΔW||; fold that in."""
    if not t.wd_on_out:
        return dora_scale
    w = t.base_weight(torch.float32)
    dw = unit_delta(t).to(w.device)
    eps = torch.finfo(torch.float32).eps
    ratio = (row_norm(w) + eps) / (row_norm(w + dw) + eps)
    return dora_scale.float() * ratio.to(dora_scale.device)


def _diff_set(t: Target, multiplier: float = 1.0) -> dict[str, Tensor]:
    dw, db = delta(t, multiplier)
    out = {"diff": dw}
    if db is not None:
        out["diff_b"] = db
    return out


def _tlora_as_lora(t: Target) -> dict[str, Tensor]:
    """T-LoRA (full mask) as one rank-2r LoRA: [P·λ·s, -P0·λ0·s] @ [Q ; Q0]."""
    lyco = t.lyco
    s = lyco.scale
    up = torch.cat(
        (
            lyco.p_layer.weight * lyco.lambda_layer * s,
            -lyco.base_p * lyco.base_lambda * s,
        ),
        dim=1,
    )
    down = torch.cat((lyco.q_layer.weight, lyco.base_q), dim=0)
    rank = down.shape[0]
    return {
        "lora_up.weight": up.detach().float(),
        "lora_down.weight": down.detach().float(),
        "alpha": torch.tensor(float(rank)),
    }


def _dylora_as_lora(t: Target) -> dict[str, Tensor]:
    down, up, scale = t.lyco.get_weight(t.lyco.lora_dim)
    rank = down.shape[0]
    return {
        "lora_up.weight": (up * scale).detach().float(),
        "lora_down.weight": down.detach().float(),
        "alpha": torch.tensor(float(rank)),
    }


def comfy(t: Target) -> dict[str, Tensor]:
    """Tensors ComfyUI evaluates to this layer's ΔW at strength 1."""
    lyco = t.lyco
    match t.kind:
        case kind if kind in AS_DIFF:
            return _diff_set(t)
        case "oft" if lyco.rescaled:
            return _diff_set(t)
        case "oft" | "boft":
            out = {"oft_blocks": lyco.oft_blocks.detach()}
            if lyco.rescaled:
                out["rescale"] = lyco.rescale.detach()
            out["alpha"] = torch.tensor(float(lyco.constraint))
            return out
        case "tlora":
            return _tlora_as_lora(t)
        case "dylora":
            return _dylora_as_lora(t)
    out = native(t)
    if t.wd and "dora_scale" in out:
        out["dora_scale"] = _comfy_dora_scale(t, out["dora_scale"])
    return out


def reciprocal(scale: Tensor | float) -> Tensor | float:
    return 1.0 / scale.float() if isinstance(scale, Tensor) else 1.0 / float(scale)


def apply_row_scale(t: Target, tensors: dict, scale: Tensor | float) -> dict:
    """Multiply this layer's ΔW rows by `scale` (exact: per-row scales export as diff)."""
    if (
        isinstance(scale, Tensor)
        and scale.numel() > 1
        and not bool((scale == scale.flatten()[0]).all())
    ):
        dw, db = delta(t)
        s = scale.float().to(dw.device).reshape(-1, *[1] * (dw.dim() - 1))
        out = {"diff": dw * s}
        if db is not None:
            out["diff_b"] = db * s.flatten()
        return out
    c = float(scale.flatten()[0]) if isinstance(scale, Tensor) else float(scale)
    out = dict(tensors)
    if "diff" in out:
        names = ("diff", "diff_b")
    elif t.kind == "lokr":
        names = ("lokr_w1",) if "lokr_w1" in out else ("lokr_w1_a",)
    elif "lora_up.weight" in out:
        names = ("lora_up.weight",)
    else:
        names = UNIFORM_SCALE.get(t.kind, ())
    for name in names:
        if name in out:
            out[name] = out[name] * c
    if "dora_scale" in out:
        out["dora_scale"] = out["dora_scale"] * c
    return out
