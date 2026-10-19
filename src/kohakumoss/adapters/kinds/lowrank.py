"""Records of the additive low-rank kinds: LoCon/LoRA, LoHa, LoKr, T-LoRA, DyLoRA."""

from functools import partial

import torch
from lycoris.modules import (
    DyLoraModule,
    LoConModule,
    LohaModule,
    LokrModule,
    TLoraModule,
)

from kohakumoss.adapters.bind import compose, eager
from kohakumoss.adapters.bind.context import Draws
from kohakumoss.adapters.kinds.common import names, unit_diff
from kohakumoss.adapters.kinds.record import Kind, Stacking, register


def _locon_unit(t):
    lyco = t.lyco
    return (lyco.make_weight(lyco.dtype_tensor.device) * lyco.scale).reshape(lyco.shape)


def _kron_hada_unit(t):
    lyco = t.lyco
    return (lyco.get_weight(lyco.shape) * lyco.scalar).float().reshape(lyco.shape)


def _locon_bypass(t, ctx):
    gamma = t.lyco.scale * ctx.m
    ops = ctx.ops
    diff, with_base = ops.lora(t.lyco, gamma), ops.lora_with_base(t.lyco, gamma)
    return compose.plus_base(t, diff, with_base)


def _loha_bypass(t, ctx):
    gamma = t.lyco.scale * ctx.m
    ops = ctx.ops
    diff, with_base = ops.loha(t.lyco, gamma), ops.loha_with_base(t.lyco, gamma)
    return compose.plus_base(t, diff, with_base)


def _lokr_bypass(t, ctx):
    if t.lyco.tucker:
        return None
    scale = t.lyco.scale * ctx.m
    ops = ctx.ops
    diff, with_base = ops.lokr(t.lyco, scale), ops.lokr_with_base(t.lyco, scale)
    return compose.plus_base(t, diff, with_base)


def _locon_kernel_unit(t, ops):
    return ops.lora_weight(t.lyco)


def _loha_kernel_unit(t, ops):
    return ops.loha_weight(t.lyco)


def _lokr_kernel_unit(t, ops):
    return None if t.lyco.tucker else ops.lokr_weight(t.lyco)


def _locon_kernel_merged(t, ops):
    return ops.lora_merged_weight(t.lyco)


def _loha_kernel_merged(t, ops):
    return ops.loha_merged_weight(t.lyco)


def _lokr_kernel_merged(t, ops):
    return None if t.lyco.tucker else ops.lokr_merged_weight(t.lyco)


def _lokr_rows(out: dict) -> tuple:
    return ("lokr_w1",) if "lokr_w1" in out else names(("lokr_w1_a",), out)


def _tlora_as_lora(t) -> dict:
    """T-LoRA (no mask) as one rank-2r LoRA: [P*lam*s, -P0*lam0*s] @ [Q ; Q0]."""
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
    return {
        "lora_up.weight": up.detach().float(),
        "lora_down.weight": down.detach().float(),
        "alpha": torch.tensor(float(down.shape[0])),
    }


def _dylora_unit(t):
    lyco = t.lyco
    return lyco.get_diff_weight(1.0, rank=lyco.lora_dim)[0].float().reshape(lyco.shape)


def _dylora_as_lora(t) -> dict:
    down, up, scale = t.lyco.get_weight(t.lyco.lora_dim)
    return {
        "lora_up.weight": (up * scale).detach().float(),
        "lora_down.weight": down.detach().float(),
        "alpha": torch.tensor(float(down.shape[0])),
    }


def _dylora_draw(t, gen: torch.Generator) -> None:
    """This step's rank: a uniform number of leading blocks (LyCORIS's rule)."""
    lyco = t.lyco
    n = int(torch.randint(1, lyco.block_count + 1, (), generator=gen))
    t.draws.rank = n * lyco.block_size


def _dylora_forward(t, ctx):
    """Training: the step's drawn rank prefix; eval: every block, whatever a step
    draws (both as bypass products, no dW)."""
    if ctx.training:
        return eager.dylora_train(t.layer, t.lyco, ctx.m, ctx.draws)
    return eager.dylora_train(t.layer, t.lyco, ctx.m, Draws(t.lyco.lora_dim))


LOCON = register(
    Kind(
        name="locon",
        module=LoConModule,
        algos={"lora": "lora", "locon": "locon"},
        diff_weight=_locon_unit,
        kernel_diff_weight=_locon_kernel_unit,
        kernel_merged_weight=_locon_kernel_merged,
        kernel_bypass=_locon_bypass,
        uniform_scale_tensors=partial(names, ("lora_up.weight",)),
        row_scale_tensors=partial(names, ("lora_up.weight",)),
        stacking=Stacking.ADDITIVE,
        loraplus_tensors=("lora_up.weight",),
        supports_dora=True,
    )
)
LOHA = register(
    Kind(
        name="loha",
        module=LohaModule,
        algos={"loha": "loha"},
        diff_weight=_kron_hada_unit,
        kernel_diff_weight=_loha_kernel_unit,
        kernel_merged_weight=_loha_kernel_merged,
        kernel_bypass=_loha_bypass,
        uniform_scale_tensors=partial(names, ("hada_w1_a",)),
        row_scale_tensors=partial(names, ("hada_w1_a",)),
        stacking=Stacking.ADDITIVE,
        loraplus_tensors=("hada_w2_a",),
        supports_dora=True,
    )
)
LOKR = register(
    Kind(
        name="lokr",
        module=LokrModule,
        algos={"lokr": "lokr"},
        diff_weight=_kron_hada_unit,
        kernel_diff_weight=_lokr_kernel_unit,
        kernel_merged_weight=_lokr_kernel_merged,
        kernel_bypass=_lokr_bypass,
        uniform_scale_tensors=_lokr_rows,
        stacking=Stacking.ADDITIVE,
        loraplus_tensors=("lokr_w2", "lokr_w2_b"),
        supports_dora=True,
    )
)
TLORA = register(
    Kind(
        name="tlora",
        module=TLoraModule,
        algos={"tlora": "tlora"},
        diff_weight=unit_diff,
        comfy_tensors=_tlora_as_lora,
        uniform_scale_tensors=partial(
            names, ("lora_up.weight", "lambda_layer", "base_lambda")
        ),
        row_scale_tensors=partial(
            names, ("lora_up.weight", "p_layer.weight", "base_p")
        ),
        stacking=Stacking.ADDITIVE,
        loraplus_tensors=("p_layer.weight",),
        per_token_mask=True,
    )
)
DYLORA = register(
    Kind(
        name="dylora",
        module=DyLoraModule,
        algos={"dylora": "dylora"},
        diff_weight=_dylora_unit,
        own_forward=_dylora_forward,
        comfy_tensors=_dylora_as_lora,
        uniform_scale_tensors=partial(names, ("lora_up.weight",)),
        row_scale_tensors=partial(names, ("lora_up.weight",)),
        stacking=Stacking.DENSE,
        loraplus_tensors=("up_list",),
        training_dependent=True,
        draw_step=_dylora_draw,
    )
)
