"""Records of GLoRA and IA3."""

from functools import partial

from lycoris.modules import GLoRAModule, IA3Module

from kohakumoss.adapters.bind import compose, eager, forwards
from kohakumoss.adapters.bind.bodies import glora_delta, ia3_delta
from kohakumoss.adapters.kinds.common import F32, diff_set, names
from kohakumoss.adapters.kinds.record import Kind, Stacking, register


def _glora_unit(t):
    lyco = t.lyco
    a1 = lyco.a1.weight
    w = t.base_fn(a1.dtype)
    dw = glora_delta(w, a1, lyco.a2.weight, lyco.b1.weight, lyco.b2.weight, lyco.scale)
    return (dw * lyco.scalar).float().reshape(lyco.shape)


def _glora_kernel_unit(t, ops):
    return ops.glora_weight(t.lyco)


def _glora_eager(t, ctx):
    return eager.glora_bypass(t.layer, t.lyco, ctx.m)


def _ia3_unit(t):
    lyco = t.lyco
    return ia3_delta(t.base_fn(F32), lyco.weight.float(), lyco.train_input, 1.0)


def _ia3_bypass(t, ctx):
    scale = ctx.ops.ia3_scale(t.lyco, ctx.m)
    if scale is None:
        return None
    wrap = forwards.scaled_input if t.lyco.train_input else forwards.scaled_output
    return wrap(t.layer, scale)


def _ia3_merge(t, ctx):
    return compose.merged(t, ctx.ops.ia3_weight(t.lyco, ctx.m), 1.0)


def _ia3_eager(t, ctx):
    fn = eager.ia3_in if t.lyco.train_input else eager.ia3_out
    return fn(t.layer, t.lyco, ctx.m)


GLORA = register(
    Kind(
        name="glora",
        module=GLoRAModule,
        algos={"glora": "glora"},
        diff_weight=_glora_unit,
        kernel_diff_weight=_glora_kernel_unit,
        module_forward=_glora_eager,
        uniform_scale_tensors=partial(names, ("b1.weight",)),
        row_scale_tensors=partial(names, ("b1.weight",)),
        stacking=Stacking.CHAINED,
        loraplus_tensors=("a2.weight", "b2.weight"),
    )
)
IA3 = register(
    Kind(
        name="ia3",
        module=IA3Module,
        algos={"ia3": "ia3"},
        diff_weight=_ia3_unit,
        kernel_bypass=_ia3_bypass,
        kernel_merge=_ia3_merge,
        module_forward=_ia3_eager,
        comfy_tensors=diff_set,
        stacking=Stacking.COMMUTING,
        rankless=True,
    )
)
