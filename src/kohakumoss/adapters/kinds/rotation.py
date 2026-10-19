"""Records of the rotation kinds: diagonal OFT and butterfly OFT (commute with a
uniform row scale of the base)."""

import torch
from lycoris.functional import boft as f_boft
from lycoris.functional import diag_oft as f_oft
from lycoris.modules import ButterflyOFTModule, DiagOFTModule
from torch import Tensor

from kohakumoss.adapters.bind import compose, eager
from kohakumoss.adapters.kinds.common import F32, diff_set
from kohakumoss.adapters.kinds.record import Kind, Stacking, register


def _rescale(lyco) -> Tensor | None:
    return lyco.rescale if lyco.rescaled else None


def _oft_rotation(t, rescale) -> Tensor:
    lyco = t.lyco
    return f_oft.diff_weight(
        t.base_fn(F32),
        lyco.oft_blocks,
        rescale,
        constraint=lyco.constraint,
        backend=t.backend,
    )


def _oft_unit(t) -> Tensor:
    return _oft_rotation(t, _rescale(t.lyco))


def _oft_at_builder(t, multiplier: float):
    """dW at `multiplier`: s m (R - I) W + (s - 1) W (s the rescale, else 1)."""

    def build() -> tuple[Tensor, None]:
        rotation = _oft_rotation(t, None)
        rescale = _rescale(t.lyco)
        if rescale is None:
            return rotation * multiplier, None
        w = t.base_fn(F32)
        return rescale * (rotation * multiplier) + (rescale - 1) * w, None

    return build


def _boft_diff(t, multiplier: float) -> Tensor:
    lyco = t.lyco
    return f_boft.diff_weight(
        t.base_fn(F32),
        lyco.oft_blocks,
        _rescale(lyco),
        constraint=lyco.constraint,
        scale=multiplier,
        backend=t.backend,
    )


def _boft_unit(t) -> Tensor:
    return _boft_diff(t, 1.0)


def _boft_at_builder(t, multiplier: float):
    def build() -> tuple[Tensor, None]:
        return _boft_diff(t, multiplier), None

    return build


def _oft_at(t, multiplier: float):
    if multiplier == 1.0:
        return None
    return _oft_at_builder(t, multiplier)


def _boft_at(t, multiplier: float):
    if multiplier == 1.0:
        return None
    return _boft_at_builder(t, multiplier)


def _oft_bypass(t, ctx):
    return ctx.ops.oft(t.layer, t.lyco) if ctx.m == 1.0 else None


def _oft_merge(t, ctx):
    if ctx.m != 1.0:
        return None
    ops = ctx.ops
    unit = ops.oft_weight(t.lyco)
    merge = ops.oft_merged_weight(t.lyco) if unit is not None else None
    return compose.merged(t, unit, 1.0, merge)


def _oft_eager(t, ctx):
    if ctx.m == 1.0 or t.stored:
        return eager.module_rotation(t.lyco, ctx.m)
    dense = compose.dense_of(t.linear)
    return eager.delta_bypass(t.layer, t.lyco.shape, t.delta_at(ctx.m), dense)


def _boft_eager(t, ctx):
    return eager.module_rotation(t.lyco, ctx.m)


def _boft_bypass(t, ctx):
    return ctx.ops.boft(t.layer, t.lyco, ctx.m)


def _boft_merge(t, ctx):
    return compose.merged(t, ctx.ops.boft_weight(t.lyco, ctx.m), 1.0)


def _blocks_set(t) -> dict:
    lyco = t.lyco
    out = {"oft_blocks": lyco.oft_blocks.detach()}
    if lyco.rescaled:
        out["rescale"] = lyco.rescale.detach()
    out["alpha"] = torch.tensor(float(lyco.constraint))
    return out


def _oft_comfy(t) -> dict:
    """ComfyUI's OFT merge ignores `rescale`, so a rescaled OFT is written as diff."""
    return diff_set(t) if t.lyco.rescaled else _blocks_set(t)


def _at_one(m: float) -> bool:
    return m == 1.0


def _constraint_from_alpha(lyco, parts: dict) -> None:
    """Our comfy files carry the OFT constraint as `alpha`."""
    if "alpha" in parts:
        lyco.constraint = float(parts["alpha"])


OFT = register(
    Kind(
        name="oft",
        module=DiagOFTModule,
        algos={"oft": "diag-oft", "diag-oft": "diag-oft"},
        diff_weight=_oft_unit,
        diff_weight_at=_oft_at,
        kernel_bypass=_oft_bypass,
        kernel_merge=_oft_merge,
        module_forward=_oft_eager,
        comfy_tensors=_oft_comfy,
        stacking=Stacking.COMMUTING,
        bypass_exact_at=_at_one,
        undo_comfy=_constraint_from_alpha,
    )
)
BOFT = register(
    Kind(
        name="boft",
        module=ButterflyOFTModule,
        algos={"boft": "boft"},
        diff_weight=_boft_unit,
        diff_weight_at=_boft_at,
        kernel_bypass=_boft_bypass,
        kernel_merge=_boft_merge,
        module_forward=_boft_eager,
        comfy_tensors=_blocks_set,
        stacking=Stacking.COMMUTING,
        undo_comfy=_constraint_from_alpha,
    )
)
