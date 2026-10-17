"""Records of the rotation kinds: diagonal OFT and butterfly OFT (commute with a
uniform row scale of the base)."""

from functools import partial

import torch
from lycoris.functional import boft as f_boft
from lycoris.functional import diag_oft as f_oft
from lycoris.modules import ButterflyOFTModule, DiagOFTModule
from torch import Tensor

from kohakumoss.adapters.bind import compose, eager
from kohakumoss.adapters.kinds.common import F32, diff_set
from kohakumoss.adapters.kinds.record import Kind, register


def _rescale(lyco):
    return lyco.rescale.float()


def _no_rescale(lyco):
    return None


def _rescale_of(lyco):
    return _rescale if lyco.rescaled else _no_rescale


def _oft_unit(t, rescale) -> Tensor:
    lyco = t.lyco
    return f_oft.diff_weight(
        t.base_fn(F32),
        lyco.oft_blocks.float(),
        rescale(lyco),
        constraint=lyco.constraint,
        backend=t.backend,
    )


def _oft_unit_at(t, rescale, multiplier: float) -> tuple[Tensor, None]:
    lyco = t.lyco
    dw = f_oft.diff_weight(
        t.base_fn(F32),
        lyco.oft_blocks.float(),
        rescale(lyco),
        constraint=lyco.constraint,
        scale=multiplier,
        backend=t.backend,
    )
    return dw, None


def _boft_unit(t, rescale, multiplier: float = 1.0) -> Tensor:
    lyco = t.lyco
    return f_boft.diff_weight(
        t.base_fn(F32),
        lyco.oft_blocks.float(),
        rescale(lyco),
        constraint=lyco.constraint,
        scale=multiplier,
        backend=t.backend,
    )


def _boft_at_builder(t, rescale, multiplier: float) -> tuple[Tensor, None]:
    return _boft_unit(t, rescale, multiplier), None


def _oft_bind_unit(t):
    return partial(_oft_unit, t, _rescale_of(t.lyco))


def _boft_bind_unit(t):
    return partial(_boft_unit, t, _rescale_of(t.lyco))


def _oft_at(t, multiplier: float):
    if multiplier == 1.0:
        return None
    return partial(_oft_unit_at, t, _rescale_of(t.lyco), multiplier)


def _boft_at(t, multiplier: float):
    if multiplier == 1.0:
        return None
    return partial(_boft_at_builder, t, _rescale_of(t.lyco), multiplier)


def _oft_bypass(t, ctx):
    return ctx.kit.oft(t.linear, t.lyco) if ctx.m == 1.0 else None


def _oft_merge(t, ctx):
    if ctx.m != 1.0:
        return None
    return compose.merged(t, ctx.kit.oft_unit(t.lyco), 1.0)


def _oft_eager(t, ctx):
    if ctx.m == 1.0:
        return partial(eager.module_bypass, t.lyco, 1.0)
    dense = compose.dense_of(t.linear)
    shape = t.lyco.shape
    return partial(eager.delta_bypass, t.linear, shape, t.delta_at(ctx.m), dense)


def _boft_bypass(t, ctx):
    return ctx.kit.boft(t.linear, t.lyco, ctx.m)


def _boft_merge(t, ctx):
    return compose.merged(t, ctx.kit.boft_unit(t.lyco, ctx.m), 1.0)


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
        unit=_oft_bind_unit,
        at=_oft_at,
        bypass=_oft_bypass,
        merge=_oft_merge,
        eager=_oft_eager,
        comfy=_oft_comfy,
        stacking="commuting",
        exact_at=_at_one,
        from_comfy=_constraint_from_alpha,
    )
)
BOFT = register(
    Kind(
        name="boft",
        module=ButterflyOFTModule,
        algos={"boft": "boft"},
        unit=_boft_bind_unit,
        at=_boft_at,
        bypass=_boft_bypass,
        merge=_boft_merge,
        comfy=_blocks_set,
        stacking="commuting",
        from_comfy=_constraint_from_alpha,
    )
)
