"""Records of GLoRA, IA3 and the full finetune module."""

from functools import partial

from lycoris.modules import FullModule, GLoRAModule, IA3Module
from lycoris.modules.glora import glora_weight
from lycoris.modules.ia3 import ia3_weight

from kohakumoss.adapters.bind import compose, eager, fused
from kohakumoss.adapters.kinds.common import F32, bound, diff_set, names, unit_diff
from kohakumoss.adapters.kinds.record import Kind, register


def _glora_unit(t):
    lyco = t.lyco
    parts = (lyco.a1.weight, lyco.a2.weight, lyco.b1.weight, lyco.b2.weight)
    dw = glora_weight(t.base_fn(F32), *(p.float() for p in parts), None, lyco.scale)
    return (dw * lyco.scalar.float()).reshape(lyco.shape)


def _glora_fused_unit(t, kit):
    return partial(fused.glora_unit, t.lyco, kit.backend)


def _ia3_unit(t):
    lyco = t.lyco
    return ia3_weight(t.base_fn(F32), lyco.weight.float(), lyco.train_input, 1.0, True)


def _ia3_bypass(t, ctx):
    fn = fused.ia3_bypass_in if t.lyco.train_input else fused.ia3_bypass_out
    return partial(fn, t.linear, t.lyco, ctx.m, ctx.kit.backend)


def _ia3_merge(t, ctx):
    unit = partial(fused.ia3_unit, t.lyco, ctx.m, ctx.kit.backend)
    return compose.merged(t, unit, 1.0)


def _ia3_eager(t, ctx):
    fn = eager.ia3_in if t.lyco.train_input else eager.ia3_out
    return partial(fn, t.linear, t.lyco, ctx.m)


def _full_with_bias(t, multiplier: float):
    dw, db = t.lyco.get_diff_weight(multiplier)
    return dw.float(), db.float()


def _full_without_bias(t, multiplier: float):
    return t.lyco.get_diff_weight(multiplier)[0].float(), None


def _full_at(t, multiplier: float):
    has_bias = t.lyco.bias is not None and t.lyco.org_bias is not None
    return partial(_full_with_bias if has_bias else _full_without_bias, t, multiplier)


def _full_forward(t, ctx):
    return partial(eager.module_forward, t.lyco)


def _diff_rows(out: dict) -> tuple:
    return names(("diff", "diff_b"), out)


GLORA = register(
    Kind(
        name="glora",
        module=GLoRAModule,
        algos={"glora": "glora"},
        unit=partial(bound, _glora_unit),
        fused_unit=_glora_fused_unit,
        row_names=partial(names, ("b1.weight",)),
        stacking="chained",
        up_side=("a2.weight", "b2.weight"),
    )
)
IA3 = register(
    Kind(
        name="ia3",
        module=IA3Module,
        algos={"ia3": "ia3"},
        unit=partial(bound, _ia3_unit),
        bypass=_ia3_bypass,
        merge=_ia3_merge,
        eager=_ia3_eager,
        comfy=diff_set,
        stacking="commuting",
        dimless=True,
    )
)
FULL = register(
    Kind(
        name="full",
        module=FullModule,
        algos={"full": "full"},
        unit=partial(bound, unit_diff),
        at=_full_at,
        special=_full_forward,
        comfy=diff_set,
        row_names=_diff_rows,
        stacking="dense",
        owns_weight=True,
        dimless=True,
    )
)
