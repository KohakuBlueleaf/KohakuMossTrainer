"""The weight-side kernel ops of each adapter kind as `autotuned.Op`s over one layer's
module: dW (or W + dW with the add inside the op) on the fp32 base weight, and the DoRA
decompose epilogue. See docs/internals/adapter-binding.md#kernel-ops."""

from collections.abc import Callable

from lycoris.kernels.autograd import (
    apply_dora,
    boft_diff_weight,
    diag_oft_diff_weight,
    glora_diff_weight,
    ia3_diff_weight,
    locon_diff_weight,
    loha_diff_weight,
    lokr_kron_weight,
)
from lycoris.kernels.select import static_scale
from torch import Tensor

from kohakumoss.adapters.bind import bodies
from kohakumoss.adapters.bind.autotuned import Op, Weight, WeightWithDelta
from kohakumoss.adapters.bind.ops import (
    fixed_scope,
    kron_plain,
    kron_reader,
    loha_scope,
    oft_scope,
    rotation_operands,
)

# Weight ops on the fp32 base weight W of `lyco.shape`: dW at multiplier 1 (BOFT and
# IA3: at the bound multiplier), or W + dW with the add inside the op.


def lora_weight(lyco) -> Op:
    scale, shape = lyco.scale, tuple(lyco.shape)

    def operands(w: Tensor) -> tuple:
        return (lyco.lora_down.weight, lyco.lora_up.weight)

    def kernel(backend: str, ops: tuple) -> Tensor:
        return locon_diff_weight(*ops, None, scale, backend).reshape(shape)

    return Op(
        "locon.weight",
        Weight(shape),
        operands,
        kernel,
        bodies.lora_weight,
        (scale, shape),
        fixed_scope(static_scale(scale)),
    )


def lora_merged_weight(lyco) -> Op:
    scale, shape = lyco.scale, tuple(lyco.shape)

    def operands(w: Tensor) -> tuple:
        return (lyco.lora_down.weight, lyco.lora_up.weight, w)

    def kernel(backend: str, ops: tuple) -> Tensor:
        d, u, w = ops
        return locon_diff_weight(d, u, None, scale, backend, w).reshape(shape)

    return Op(
        "locon.merged_weight",
        Weight(shape),
        operands,
        kernel,
        bodies.lora_merged,
        (scale, shape),
        fixed_scope(static_scale(scale)),
    )


def _loha_weights(lyco) -> tuple:
    return (lyco.hada_w1_b, lyco.hada_w1_a, lyco.hada_w2_b, lyco.hada_w2_a)


def loha_weight(lyco) -> Op:
    scale = lyco.scale

    def operands(w: Tensor) -> tuple:
        return _loha_weights(lyco)

    def kernel(backend: str, ops: tuple) -> Tensor:
        return loha_diff_weight(*ops, None, None, scale, backend)

    return Op(
        "loha.weight",
        Weight(lyco.shape),
        operands,
        kernel,
        bodies.loha_weight,
        (bodies.hada_scale(scale),),
        loha_scope(lyco, scale),
    )


def loha_merged_weight(lyco) -> Op:
    scale = lyco.scale

    def operands(w: Tensor) -> tuple:
        return (*_loha_weights(lyco), w)

    def kernel(backend: str, ops: tuple) -> Tensor:
        return loha_diff_weight(*ops[:-1], None, None, scale, backend, ops[-1])

    return Op(
        "loha.merged_weight",
        Weight(lyco.shape),
        operands,
        kernel,
        bodies.loha_merged,
        (bodies.hada_scale(scale),),
        loha_scope(lyco, scale),
    )


def _kron_weight_scope(lyco) -> Callable[[str], bool]:
    return fixed_scope(kron_plain(lyco) and static_scale(lyco.scale))


def lokr_weight(lyco) -> Op:
    read, scale = kron_reader(lyco), lyco.scale

    def operands(w: Tensor) -> tuple:
        return read()

    def kernel(backend: str, ops: tuple) -> Tensor:
        return lokr_kron_weight(*ops, None, scale, backend)

    return Op(
        "lokr.weight",
        Weight(lyco.shape),
        operands,
        kernel,
        bodies.lokr_weight,
        (scale,),
        _kron_weight_scope(lyco),
    )


def lokr_merged_weight(lyco) -> Op:
    read, scale = kron_reader(lyco), lyco.scale

    def operands(w: Tensor) -> tuple:
        return (*read(), w)

    def kernel(backend: str, ops: tuple) -> Tensor:
        return lokr_kron_weight(*ops[:-1], None, scale, backend, ops[-1])

    return Op(
        "lokr.merged_weight",
        Weight(lyco.shape),
        operands,
        kernel,
        bodies.lokr_merged,
        (scale,),
        _kron_weight_scope(lyco),
    )


def oft_weight(lyco) -> Op:
    constraint = lyco.constraint

    def kernel(backend: str, ops: tuple) -> Tensor:
        return diag_oft_diff_weight(*ops, constraint, backend)

    return Op(
        "diag_oft.weight",
        Weight(lyco.shape),
        rotation_operands(lyco),
        kernel,
        bodies.oft_weight,
        (constraint,),
        oft_scope(lyco),
    )


def oft_merged_weight(lyco) -> Op:
    constraint = lyco.constraint

    def kernel(backend: str, ops: tuple) -> Tensor:
        return diag_oft_diff_weight(*ops, constraint, backend, diff=False)

    return Op(
        "diag_oft.merged_weight",
        Weight(lyco.shape),
        rotation_operands(lyco),
        kernel,
        bodies.oft_merged,
        (constraint,),
        oft_scope(lyco),
    )


def boft_weight(lyco, m: float) -> Op:
    constraint = lyco.constraint

    def kernel(backend: str, ops: tuple) -> Tensor:
        return boft_diff_weight(*ops, constraint, m, backend)

    return Op(
        "boft.weight",
        Weight(lyco.shape),
        rotation_operands(lyco),
        kernel,
        bodies.boft_weight,
        (constraint, m),
    )


def ia3_weight(lyco, m: float) -> Op:
    on_input = bool(lyco.train_input)

    def operands(w: Tensor) -> tuple:
        return (w, lyco.weight.to(w.dtype))

    def kernel(backend: str, ops: tuple) -> Tensor:
        return ia3_diff_weight(*ops, on_input, m, True, backend)

    return Op(
        "ia3.weight",
        Weight(lyco.shape),
        operands,
        kernel,
        bodies.ia3_weight,
        (on_input, m),
    )


def glora_weight(lyco) -> Op:
    scale = lyco.scale

    def operands(w: Tensor) -> tuple:
        a1 = lyco.a1.weight
        return (w.to(a1.dtype), a1, lyco.a2.weight, lyco.b1.weight, lyco.b2.weight)

    def kernel(backend: str, ops: tuple) -> Tensor:
        return glora_diff_weight(*ops, scale, backend)

    return Op(
        "glora.weight",
        Weight(lyco.shape),
        operands,
        kernel,
        bodies.glora_weight,
        (scale,),
    )


# DoRA epilogue at multiplier 1: on the merged weight, or on the (W, dW) pair with the
# add read inside the op.


def decompose(lyco, on_out: bool) -> Op:
    def operands(merged: Tensor) -> tuple:
        return (merged, lyco.dora_scale)

    def kernel(backend: str, ops: tuple) -> Tensor:
        return apply_dora(*ops, 1.0, on_out, backend)

    return Op(
        "weight_decompose",
        Weight(lyco.shape),
        operands,
        kernel,
        bodies.decompose,
        (on_out,),
    )


def decompose_with_delta(lyco, on_out: bool) -> Op:
    def operands(t: tuple) -> tuple:
        w, dw = t
        return (w, lyco.dora_scale, dw)

    def kernel(backend: str, ops: tuple) -> Tensor:
        w, magnitude, dw = ops
        return apply_dora(w, magnitude, 1.0, on_out, backend, delta=dw)

    return Op(
        "weight_decompose_with_delta",
        WeightWithDelta(lyco.shape),
        operands,
        kernel,
        bodies.decompose_with_delta,
        (on_out,),
    )
