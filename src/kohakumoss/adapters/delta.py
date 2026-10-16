"""Forward-consistent ΔW per adapted layer, as builders resolved once per Target.

`resolve(t)` binds `t.unit_fn` (ΔW at multiplier 1, before any DoRA epilogue) from a
table keyed by kind; `bind_delta(t, m)` returns a zero-argument callable specialised for
the multiplier and the DoRA flag. The base weight is always read through the Target's
bound source (dense, quantized or streamed storage alike). See docs/concepts/adapters.md.
"""

from collections.abc import Callable
from functools import partial

import torch
from lycoris.functional import boft as f_boft
from lycoris.functional import diag_oft as f_oft
from lycoris.functional.general import weight_decompose
from lycoris.modules.glora import _glora_weight
from lycoris.modules.ia3 import _ia3_weight
from torch import Tensor

from kohakumoss.adapters.target import Target

F32 = torch.float32


def _unit_locon(t: Target) -> Tensor:
    lyco = t.lyco
    return (lyco.make_weight(lyco.dtype_tensor.device) * lyco.scale).reshape(lyco.shape)


def _unit_kron_hada(t: Target) -> Tensor:
    lyco = t.lyco
    return (lyco.get_weight(lyco.shape) * lyco.scalar).float().reshape(lyco.shape)


def _unit_dylora(t: Target) -> Tensor:
    lyco = t.lyco
    return lyco.get_diff_weight(1.0, rank=lyco.lora_dim)[0].float().reshape(lyco.shape)


def _unit_diff(t: Target) -> Tensor:
    return t.lyco.get_diff_weight(1.0)[0].float().reshape(t.lyco.shape)


def _unit_oft_plain(t: Target) -> Tensor:
    lyco = t.lyco
    w = t.base_fn(F32)
    return f_oft.diff_weight(
        w, lyco.oft_blocks.float(), None, constraint=lyco.constraint
    )


def _unit_oft_rescaled(t: Target) -> Tensor:
    lyco = t.lyco
    w = t.base_fn(F32)
    return f_oft.diff_weight(
        w, lyco.oft_blocks.float(), lyco.rescale.float(), constraint=lyco.constraint
    )


def _boft_plain(t: Target, multiplier: float) -> Tensor:
    lyco = t.lyco
    w = t.base_fn(F32)
    return f_boft.diff_weight(
        w, lyco.oft_blocks.float(), None, constraint=lyco.constraint, scale=multiplier
    )


def _boft_rescaled(t: Target, multiplier: float) -> Tensor:
    lyco = t.lyco
    w = t.base_fn(F32)
    return f_boft.diff_weight(
        w,
        lyco.oft_blocks.float(),
        lyco.rescale.float(),
        constraint=lyco.constraint,
        scale=multiplier,
    )


def _unit_ia3(t: Target) -> Tensor:
    lyco = t.lyco
    return _ia3_weight(t.base_fn(F32), lyco.weight.float(), lyco.train_input, 1.0, True)


def _unit_glora(t: Target) -> Tensor:
    lyco = t.lyco
    parts = (lyco.a1.weight, lyco.a2.weight, lyco.b1.weight, lyco.b2.weight)
    dw = _glora_weight(t.base_fn(F32), *(p.float() for p in parts), None, lyco.scale)
    return (dw * lyco.scalar.float()).reshape(lyco.shape)


UNIT: dict[str, Callable[[Target], Tensor]] = {
    "locon": _unit_locon,
    "loha": _unit_kron_hada,
    "lokr": _unit_kron_hada,
    "dylora": _unit_dylora,
    "full": _unit_diff,
    "tlora": _unit_diff,
    "ia3": _unit_ia3,
    "glora": _unit_glora,
}


def resolve(t: Target) -> None:
    """Bind `t.unit_fn` for the layer's kind (and its OFT/BOFT rescale variant)."""
    match t.kind:
        case "oft":
            fn = _unit_oft_rescaled if t.lyco.rescaled else _unit_oft_plain
            t.unit_fn = partial(fn, t)
        case "boft":
            fn = _boft_rescaled if t.lyco.rescaled else _boft_plain
            t.unit_fn = partial(fn, t, 1.0)
        case _:
            t.unit_fn = partial(UNIT[t.kind], t)


def _rotated(lyco, w: Tensor, multiplier: float) -> tuple[Tensor, Tensor]:
    """(R_m W blockwise, W) with R_m = m·R + (1−m)·I."""
    r = lyco.get_r().float()
    eye = torch.eye(r.shape[-1], device=r.device, dtype=r.dtype)
    r = multiplier * r + (1.0 - multiplier) * eye
    w = w.to(device=r.device, dtype=r.dtype)
    blocks = w.reshape(lyco.block_num, lyco.block_size, -1)
    return torch.einsum("knm,kni->kmi", r, blocks).reshape(w.shape), w


def oft_delta_plain(lyco, w: Tensor, multiplier: float) -> Tensor:
    """Diag-OFT ΔW at a multiplier without rescale."""
    wp, w = _rotated(lyco, w, multiplier)
    return wp - w


def oft_delta_rescaled(lyco, w: Tensor, multiplier: float) -> Tensor:
    """Diag-OFT ΔW at a multiplier, output rows rescaled."""
    wp, w = _rotated(lyco, w, multiplier)
    return lyco.rescale.float().reshape(-1, *[1] * (w.dim() - 1)) * wp - w


def _eps_norm(norm: Tensor, dtype: torch.dtype) -> Tensor:
    return (norm + torch.finfo(dtype).eps).detach()


def detached_out(merged: Tensor, dora_scale: Tensor) -> Tensor:
    """Output-axis DoRA with the row norm treated as a constant in backward."""
    norm = merged.reshape(merged.shape[0], -1).norm(dim=1, keepdim=True)
    return merged * (dora_scale.to(merged.dtype) / _eps_norm(norm, merged.dtype))


def detached_in(merged: Tensor, dora_scale: Tensor) -> Tensor:
    """Input-axis DoRA with the column norm treated as a constant in backward."""
    norm = merged.norm(dim=0, keepdim=True)
    return merged * (dora_scale.to(merged.dtype) / _eps_norm(norm, merged.dtype))


def detached(fn: Callable, lyco, merged: Tensor) -> Tensor:
    """A detached DoRA epilogue with the layer's magnitude."""
    return fn(merged, lyco.dora_scale.float())


def dora_merged(t: Target, merged: Tensor) -> Tensor:
    """LyCORIS weight_decompose on an already-merged weight (multiplier 1)."""
    return weight_decompose(merged, t.lyco.dora_scale.float(), 1.0, t.wd_on_out)


def _unit_only(t: Target) -> tuple[Tensor, None]:
    return t.unit_fn(), None


def _unit_times(t: Target, multiplier: float) -> tuple[Tensor, None]:
    return t.unit_fn() * multiplier, None


def _full(t: Target, multiplier: float) -> tuple[Tensor, Tensor]:
    dw, db = t.lyco.get_diff_weight(multiplier)
    return dw.float(), db.float()


def _full_nobias(t: Target, multiplier: float) -> tuple[Tensor, None]:
    return t.lyco.get_diff_weight(multiplier)[0].float(), None


def _oft_at(t: Target, multiplier: float, fn: Callable) -> tuple[Tensor, None]:
    return fn(t.lyco, t.base_fn(F32), multiplier), None


def _boft_at(t: Target, multiplier: float, fn: Callable) -> tuple[Tensor, None]:
    return fn(t, multiplier), None


def _dora(t: Target, scale: Callable) -> tuple[Tensor, None]:
    w = t.base_fn(F32)
    dw = t.unit_fn().to(w.device)
    return scale(dora_merged(t, w + dw) - w), None


def _times(multiplier: float, x: Tensor) -> Tensor:
    return x * multiplier


def _same(x: Tensor) -> Tensor:
    return x


def bind_delta(
    t: Target, multiplier: float
) -> Callable[[], tuple[Tensor, Tensor | None]]:
    """(ΔW, Δbias) builder for `t` at a fixed multiplier, fp32 on the adapter device."""
    one = multiplier == 1.0
    if t.wd:
        return partial(_dora, t, _same if one else partial(_times, multiplier))
    match t.kind:
        case "full":
            has_bias = t.lyco.bias is not None and t.lyco.org_bias is not None
            return partial(_full if has_bias else _full_nobias, t, multiplier)
        case "oft" if not one:
            fn = oft_delta_rescaled if t.lyco.rescaled else oft_delta_plain
            return partial(_oft_at, t, multiplier, fn)
        case "boft" if not one:
            fn = _boft_rescaled if t.lyco.rescaled else _boft_plain
            return partial(_boft_at, t, multiplier, fn)
    return partial(_unit_only, t) if one else partial(_unit_times, t, multiplier)


def delta(t: Target, multiplier: float = 1.0) -> tuple[Tensor, Tensor | None]:
    """Export / merge convenience: the bound builder evaluated once."""
    return bind_delta(t, multiplier)()


def unit_delta(t: Target) -> Tensor:
    """ΔW at multiplier 1 before any DoRA epilogue (the bound unit builder)."""
    return t.unit_fn()
