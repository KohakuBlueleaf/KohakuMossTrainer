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
from kohakumoss.adapters.kinds.common import bound, names, unit_diff
from kohakumoss.adapters.kinds.record import Kind, register


def _locon_unit(t):
    lyco = t.lyco
    return (lyco.make_weight(lyco.dtype_tensor.device) * lyco.scale).reshape(lyco.shape)


def _kron_hada_unit(t):
    lyco = t.lyco
    return (lyco.get_weight(lyco.shape) * lyco.scalar).float().reshape(lyco.shape)


def _locon_bypass(t, ctx):
    return compose.added(t, ctx.kit.lora(t.lyco, t.lyco.scale * ctx.m))


def _loha_bypass(t, ctx):
    return compose.added(t, ctx.kit.loha(t.lyco, t.lyco.scale * ctx.m))


def _lokr_bypass(t, ctx):
    if t.lyco.tucker:
        return None
    return compose.added(t, ctx.kit.lokr(t.lyco, t.lyco.scale * ctx.m))


def _locon_fused_unit(t, kit):
    return kit.lora_unit(t.lyco)


def _loha_fused_unit(t, kit):
    return kit.loha_unit(t.lyco)


def _lokr_fused_unit(t, kit):
    return None if t.lyco.tucker else kit.lokr_unit(t.lyco)


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
    """Training: the step's drawn rank prefix; eval: every block."""
    dense = compose.dense_of(t.linear)
    if ctx.training:
        return partial(eager.dylora_train, t.linear, t.lyco, ctx.m, ctx.draws, dense)
    shape = t.lyco.shape
    return partial(eager.delta_bypass, t.linear, shape, t.delta_at(ctx.m), dense)


LOCON = register(
    Kind(
        name="locon",
        module=LoConModule,
        algos={"lora": "lora", "locon": "locon"},
        unit=partial(bound, _locon_unit),
        fused_unit=_locon_fused_unit,
        bypass=_locon_bypass,
        row_names=partial(names, ("lora_up.weight",)),
        stacking="additive",
        up_side=("lora_up.weight",),
        dora=True,
    )
)
LOHA = register(
    Kind(
        name="loha",
        module=LohaModule,
        algos={"loha": "loha"},
        unit=partial(bound, _kron_hada_unit),
        fused_unit=_loha_fused_unit,
        bypass=_loha_bypass,
        row_names=partial(names, ("hada_w1_a",)),
        stacking="additive",
        up_side=("hada_w2_a",),
        dora=True,
    )
)
LOKR = register(
    Kind(
        name="lokr",
        module=LokrModule,
        algos={"lokr": "lokr"},
        unit=partial(bound, _kron_hada_unit),
        fused_unit=_lokr_fused_unit,
        bypass=_lokr_bypass,
        row_names=_lokr_rows,
        stacking="additive",
        up_side=("lokr_w2", "lokr_w2_b"),
        dora=True,
    )
)
TLORA = register(
    Kind(
        name="tlora",
        module=TLoraModule,
        algos={"tlora": "tlora"},
        unit=partial(bound, unit_diff),
        comfy=_tlora_as_lora,
        row_names=partial(names, ("lora_up.weight", "lambda_layer", "base_lambda")),
        stacking="additive",
        up_side=("p_layer.weight",),
        token_axis=True,
    )
)
DYLORA = register(
    Kind(
        name="dylora",
        module=DyLoraModule,
        algos={"dylora": "dylora"},
        unit=partial(bound, _dylora_unit),
        special=_dylora_forward,
        comfy=_dylora_as_lora,
        row_names=partial(names, ("lora_up.weight",)),
        stacking="dense",
        up_side=("up_list",),
        training_dependent=True,
        draw=_dylora_draw,
    )
)
