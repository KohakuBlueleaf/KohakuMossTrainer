"""The LyCORIS tier ops of each adapter kind, as `race.Op`s over one layer's module.

Each mirrors the `tier.run` call of its `lycoris.functional` entry: the same op name,
operand tuple, scope test, fused kernel, eager body and exact extras, so the race and
its recorded decision are the ones LyCORIS makes for that call. `operands(t)` builds
the operand objects once per call and every arm reads those same objects (the race's
fp64 gate maps gradients by operand identity). Bypass ops take the (tokens, channels)
activation — the input for the low-rank kinds, the base output for the rotations —
with factors cast to its dtype as the modules do; merge ops take the base weight.
"""

from functools import partial

from lycoris.functional import boft as f_boft
from lycoris.functional import diag_oft as f_oft
from lycoris.functional import general as f_general
from lycoris.functional import locon as f_locon
from lycoris.functional import loha as f_loha
from lycoris.functional import lokr as f_lokr
from lycoris.functional.diag_oft import MAX_BLOCK
from lycoris.kernels.autograd import (
    apply_dora,
    boft_bypass_diff,
    boft_diff_weight,
    diag_oft_bypass_diff,
    diag_oft_diff_weight,
    locon_bypass_diff,
    locon_diff_weight,
    loha_bypass_diff,
    loha_diff_weight,
    lokr_kron_bypass,
    lokr_kron_weight,
)
from lycoris.kernels.select import static_scale
from torch import Tensor

from kohakumoss.adapters.bind.fused import kron_factors, kron_getters
from kohakumoss.adapters.bind.race import Op


def _then(tail: tuple, ops: tuple) -> tuple:
    """Eager-body arguments: the operands followed by the bound constants."""
    return (*ops, *tail)


def _rescale(lyco):
    return lyco.rescale


def _no_rescale(lyco):
    return None


def rescale_getter(lyco):
    return _rescale if lyco.rescaled else _no_rescale


# Low-rank bypass: Δy of the flattened input.


def _lora_operands(lyco, x: Tensor) -> tuple:
    dt = x.dtype
    return (x, lyco.lora_down.weight.to(dt), lyco.lora_up.weight.to(dt))


def _lora_fused(gamma, backend: str, ops: tuple) -> Tensor:
    return locon_bypass_diff(*ops, gamma, backend)


def lora_bypass(lyco, gamma: float) -> Op:
    return Op(
        "locon.bypass",
        partial(_lora_operands, lyco),
        static_scale(gamma),
        partial(_lora_fused, gamma),
        f_locon._bypass_diff,
        partial(_then, (None, gamma, {})),
    )


def _loha_operands(lyco, x: Tensor) -> tuple:
    dt = x.dtype
    return (
        x,
        lyco.hada_w1_b.to(dt),
        lyco.hada_w1_a.to(dt),
        lyco.hada_w2_b.to(dt),
        lyco.hada_w2_a.to(dt),
    )


def _loha_fused(gamma, backend: str, ops: tuple) -> Tensor:
    return loha_bypass_diff(*ops, gamma, backend)


def _rank_product(gamma, ops: tuple) -> Tensor:
    return f_loha.rank_product_bypass(*ops, gamma)


def loha_bypass(lyco, gamma: float) -> Op:
    extras = {}
    if lyco.hada_w1_b.shape[0] * lyco.hada_w2_b.shape[0] <= f_loha.RANK_PRODUCT_MAX:
        extras["rank_product"] = partial(_rank_product, gamma)
    return Op(
        "loha.bypass",
        partial(_loha_operands, lyco),
        static_scale(gamma),
        partial(_loha_fused, gamma),
        f_loha._bypass_diff,
        partial(_then, (None, None, gamma, {})),
        extras,
    )


def _kron_operands(lyco, getters, x: Tensor) -> tuple:
    return (x, *kron_factors(lyco, getters, x.dtype))


def _kron_fused(scale, backend: str, ops: tuple) -> Tensor:
    return lokr_kron_bypass(*ops, None, scale, backend)


def lokr_bypass(lyco, scale: float) -> Op:
    raw = kron_factors(lyco, kron_getters(lyco, cast=False))
    return Op(
        "lokr.bypass",
        partial(_kron_operands, lyco, kron_getters(lyco)),
        f_lokr._apply_supported(*raw, None) and static_scale(scale),
        partial(_kron_fused, scale),
        f_lokr._kron_bypass,
        partial(_then, (None, scale, {})),
    )


# Rotation bypass: Δy of the flattened base output.


def _oft_operands(lyco, rescale, t: Tensor) -> tuple:
    return (t, lyco.oft_blocks, rescale(lyco))


def _oft_fused(lyco, backend: str, ops: tuple) -> Tensor:
    return diag_oft_bypass_diff(*ops, lyco.constraint, False, backend)


def _oft_args(lyco, ops: tuple) -> tuple:
    return (ops[0], *ops, lyco.constraint, False)


def oft_bypass(lyco) -> Op:
    return Op(
        "diag_oft.bypass",
        partial(_oft_operands, lyco, rescale_getter(lyco)),
        lyco.oft_blocks.shape[-1] <= MAX_BLOCK,
        partial(_oft_fused, lyco),
        f_oft._bypass_diff,
        partial(_oft_args, lyco),
    )


def _boft_fused(lyco, m, backend: str, ops: tuple) -> Tensor:
    return boft_bypass_diff(*ops, lyco.constraint, m, False, backend)


def _boft_args(lyco, m, ops: tuple) -> tuple:
    return (*ops, lyco.constraint, False, m)


def boft_bypass(lyco, m: float) -> Op:
    return Op(
        "boft.bypass",
        partial(_oft_operands, lyco, rescale_getter(lyco)),
        True,
        partial(_boft_fused, lyco, m),
        f_boft._bypass_diff,
        partial(_boft_args, lyco, m),
    )


# Merge: ΔW at multiplier 1 (BOFT: at the bound multiplier) from the base weight.


def _scaled(lyco, ops: tuple) -> tuple:
    return (*ops, lyco.scale)


def _lora_weights(lyco, w: Tensor) -> tuple:
    return (lyco.lora_down.weight, lyco.lora_up.weight, None)


def _lora_weight_fused(lyco, backend: str, ops: tuple) -> Tensor:
    return locon_diff_weight(*ops, lyco.scale, backend).reshape(lyco.shape)


def lora_merge(lyco) -> Op:
    return Op(
        "locon.merge",
        partial(_lora_weights, lyco),
        static_scale(lyco.scale),
        partial(_lora_weight_fused, lyco),
        f_locon._diff_weight,
        partial(_scaled, lyco),
    )


def _loha_weights(lyco, w: Tensor) -> tuple:
    return (lyco.hada_w1_b, lyco.hada_w1_a, lyco.hada_w2_b, lyco.hada_w2_a, None, None)


def _loha_weight_fused(lyco, backend: str, ops: tuple) -> Tensor:
    return loha_diff_weight(*ops, lyco.scale, backend)


def loha_merge(lyco) -> Op:
    return Op(
        "loha.merge",
        partial(_loha_weights, lyco),
        static_scale(lyco.scale),
        partial(_loha_weight_fused, lyco),
        f_loha._diff_weight,
        partial(_scaled, lyco),
    )


def _kron_weights(lyco, getters, w: Tensor) -> tuple:
    return (*kron_factors(lyco, getters), None)


def _kron_weight_fused(lyco, backend: str, ops: tuple) -> Tensor:
    return lokr_kron_weight(*ops, lyco.scale, backend)


def lokr_merge(lyco) -> Op:
    getters = kron_getters(lyco, cast=False)
    w2, w2b = kron_factors(lyco, getters)[3::2]
    return Op(
        "lokr.merge",
        partial(_kron_weights, lyco, getters),
        f_lokr._plain_w2(w2, w2b, None) and static_scale(lyco.scale),
        partial(_kron_weight_fused, lyco),
        f_lokr._kron_weight,
        partial(_scaled, lyco),
    )


def _oft_weight_fused(lyco, backend: str, ops: tuple) -> Tensor:
    return diag_oft_diff_weight(*ops, lyco.constraint, backend)


def _constrained(lyco, ops: tuple) -> tuple:
    return (*ops, lyco.constraint)


def oft_merge(lyco) -> Op:
    return Op(
        "diag_oft.merge",
        partial(_oft_operands, lyco, rescale_getter(lyco)),
        lyco.oft_blocks.shape[-1] <= MAX_BLOCK,
        partial(_oft_weight_fused, lyco),
        f_oft._diff_weight,
        partial(_constrained, lyco),
    )


def _boft_weight_fused(lyco, m, backend: str, ops: tuple) -> Tensor:
    return boft_diff_weight(*ops, lyco.constraint, m, backend)


def _boft_weight_args(lyco, m, ops: tuple) -> tuple:
    return (*ops, lyco.constraint, m)


def boft_merge(lyco, m: float) -> Op:
    return Op(
        "boft.merge",
        partial(_oft_operands, lyco, rescale_getter(lyco)),
        True,
        partial(_boft_weight_fused, lyco, m),
        f_boft._diff_weight,
        partial(_boft_weight_args, lyco, m),
    )


# DoRA epilogue on the merged weight.


def _dora_operands(lyco, merged: Tensor) -> tuple:
    return (merged, lyco.dora_scale)


def _dora_fused(on_out: bool, backend: str, ops: tuple) -> Tensor:
    return apply_dora(*ops, 1.0, on_out, backend)


def decompose(lyco, on_out: bool) -> Op:
    return Op(
        "weight_decompose",
        partial(_dora_operands, lyco),
        True,
        partial(_dora_fused, on_out),
        f_general._weight_decompose,
        partial(_then, (1.0, on_out)),
    )
