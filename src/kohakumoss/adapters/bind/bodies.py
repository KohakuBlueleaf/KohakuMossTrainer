"""Eager formulas of the adapter kernel ops, `body(ops, *consts)`: the public
`lycoris.functional` entries on backend "torch" (IA3 and GLoRA, which have none, as
their formula). Each is the `eager` candidate and what the `compile` candidate
compiles."""

import torch
from lycoris.functional import boft as f_boft
from lycoris.functional import diag_oft as f_oft
from lycoris.functional import general as f_general
from lycoris.functional import locon as f_locon
from lycoris.functional import loha as f_loha
from lycoris.functional import lokr as f_lokr
from torch import Tensor


def lora(ops: tuple, gamma: float) -> Tensor:
    x, d, u = ops
    return f_locon.bypass_forward_diff(
        x, None, d, u, None, gamma=gamma, backend="torch"
    )


def lora_with_base(ops: tuple, gamma: float) -> Tensor:
    x, d, u, base = ops
    return base + lora((x, d, u), gamma)


def hada_scale(gamma: float | Tensor) -> Tensor:
    """LoHa's scale as the 0-d tensor `lycoris.functional.loha` saves for backward."""
    return gamma if isinstance(gamma, Tensor) else torch.tensor(float(gamma))


def loha(ops: tuple, gamma: Tensor) -> Tensor:
    x, w1d, w1u, w2d, w2u = ops
    return f_loha.bypass_forward_diff(
        x, None, w1d, w1u, w2d, w2u, None, None, gamma=gamma, backend="torch"
    )


def loha_with_base(ops: tuple, gamma: Tensor) -> Tensor:
    return ops[-1] + loha(ops[:-1], gamma)


def lokr(ops: tuple, scale: float) -> Tensor:
    return f_lokr.kron_bypass(*ops, None, scale=scale, backend="torch")


def lokr_with_base(ops: tuple, scale: float) -> Tensor:
    return ops[-1] + lokr(ops[:-1], scale)


def oft(ops: tuple, constraint: float) -> Tensor:
    y, blocks, rescale = ops
    return f_oft.bypass_forward_diff(
        None, y, blocks, rescale, constraint=constraint, backend="torch"
    )


def oft_rotated(ops: tuple, constraint: float) -> Tensor:
    return ops[0] + oft(ops, constraint)


def boft(ops: tuple, constraint: float, m: float) -> Tensor:
    y, blocks, rescale = ops
    return f_boft.bypass_forward_diff(
        y, blocks, rescale, constraint=constraint, scale=m, backend="torch"
    )


def boft_rotated(ops: tuple, constraint: float, m: float) -> Tensor:
    return ops[0] + boft(ops, constraint, m)


def ia3_scale(x: Tensor, w: Tensor, m: float) -> Tensor:
    """x * (w m + 1) on x's last (channel) axis."""
    return x * (w * m + 1)


def ia3(ops: tuple, m: float) -> Tensor:
    return ia3_scale(*ops, m)


def ia3_delta(w: Tensor, scale: Tensor, on_input: bool, m: float) -> Tensor:
    """IA3's dW = W * (scale m), scale over W's input columns or output rows."""
    s = scale * m
    return w * s if on_input else w * s.reshape(-1, *[1] * (w.dim() - 1))


def ia3_weight(ops: tuple, on_input: bool, m: float) -> Tensor:
    return ia3_delta(*ops, on_input, m)


def glora_delta(w, a1, a2, b1, b2, scale: float) -> Tensor:
    """GLoRA's dW = scale (W a1 a2 + b1 b2) of a linear layer."""
    return (w @ a1 @ a2 + b1 @ b2) * scale


def glora_weight(ops: tuple, scale: float) -> Tensor:
    return glora_delta(*ops, scale)


def lora_weight(ops: tuple, scale: float, shape: tuple) -> Tensor:
    d, u = ops
    return f_locon.diff_weight(d, u, None, gamma=scale, backend="torch").reshape(shape)


def lora_merged(ops: tuple, scale: float, shape: tuple) -> Tensor:
    return ops[-1] + lora_weight(ops[:-1], scale, shape)


def loha_weight(ops: tuple, scale: Tensor) -> Tensor:
    return f_loha.diff_weight(*ops, None, None, gamma=scale, backend="torch")


def loha_merged(ops: tuple, scale: Tensor) -> Tensor:
    return ops[-1] + loha_weight(ops[:-1], scale).reshape(ops[-1].shape)


def lokr_weight(ops: tuple, scale: float) -> Tensor:
    return f_lokr.kron_weight(*ops, None, scale=scale, backend="torch")


def lokr_merged(ops: tuple, scale: float) -> Tensor:
    return ops[-1] + lokr_weight(ops[:-1], scale).reshape(ops[-1].shape)


def oft_weight(ops: tuple, constraint: float) -> Tensor:
    w, blocks, rescale = ops
    return f_oft.diff_weight(w, blocks, rescale, constraint=constraint, backend="torch")


def oft_merged(ops: tuple, constraint: float) -> Tensor:
    return ops[0] + oft_weight(ops, constraint)


def boft_weight(ops: tuple, constraint: float, m: float) -> Tensor:
    w, blocks, rescale = ops
    return f_boft.diff_weight(
        w, blocks, rescale, constraint=constraint, scale=m, backend="torch"
    )


def decompose(ops: tuple, on_out: bool) -> Tensor:
    merged, magnitude = ops
    return f_general.weight_decompose(merged, magnitude, 1.0, on_out, backend="torch")


def decompose_with_delta(ops: tuple, on_out: bool) -> Tensor:
    w, magnitude, dw = ops
    return f_general.weight_decompose(w + dw, magnitude, 1.0, on_out, backend="torch")


# Bodies that run through a custom torch.autograd.Function (LoHa's HadaWeight).
AUTOGRAD_FUNCTION_BODIES = frozenset({loha, loha_with_base, loha_weight, loha_merged})
