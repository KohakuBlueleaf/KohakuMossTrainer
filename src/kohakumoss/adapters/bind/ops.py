"""The activation-side kernel ops of each adapter kind as `autotuned.Op`s over one
layer's module: the LyCORIS fused op (`lycoris.kernels.autograd`, by `backend=`), its
eager formula (`bind.bodies`) and the fused kernel's scope on the layer; plus the
per-layer readers and scopes `bind.weight_ops` shares. See
docs/internals/adapter-binding.md#kernel-ops."""

import importlib
import math
from collections.abc import Callable
from functools import cache

import torch
from lycoris.functional.diag_oft import MAX_BLOCK
from lycoris.kernels.autograd import (
    boft_bypass_diff,
    diag_oft_bypass_diff,
    ia3_bypass,
    locon_bypass_diff,
    loha_bypass_diff,
    lokr_kron_bypass,
)
from lycoris.kernels.limits import MAX_HADA_RANK
from lycoris.kernels.select import static_scale
from torch import Tensor

from kohakumoss.adapters.bind import bodies
from kohakumoss.adapters.bind.autotuned import Op, Tokens, TokensWithBase

KRON_SIDE1 = {True: ("lokr_w1", None, None), False: (None, "lokr_w1_a", "lokr_w1_b")}
KRON_SIDE2 = {True: ("lokr_w2", None, None), False: (None, "lokr_w2_a", "lokr_w2_b")}
TOKEN = (0,)
TOKEN_AND_BASE = (0, -1)


def in_features(lyco) -> int:
    return int(lyco.shape[1])


def out_features(lyco) -> int:
    return int(lyco.shape[0])


def _param(lyco, name: str | None, dtype: torch.dtype | None):
    if name is None:
        return None
    p = getattr(lyco, name)
    return p if dtype is None else p.to(dtype)


def kron_reader(lyco) -> Callable[..., tuple]:
    """dtype -> (w1, w1a, w1b, w2, w2a, w2b) of a LoKr module (None for the absent half
    of each side; cast to dtype unless None), the sides chosen once per layer."""
    names = KRON_SIDE1[bool(lyco.use_w1)] + KRON_SIDE2[bool(lyco.use_w2)]

    def read(dtype: torch.dtype | None = None) -> tuple:
        return tuple(_param(lyco, n, dtype) for n in names)

    return read


def hada_rank(lyco) -> int:
    """The larger hadamard rank of a LoHa module."""
    return max(lyco.hada_w1_b.shape[0], lyco.hada_w2_b.shape[0])


@cache
def _backend_limit(backend: str, module: str, name: str, default: float) -> float:
    try:
        mod = importlib.import_module(f"lycoris.kernels.{backend}.{module}")
    except ImportError:
        return default
    return getattr(mod, name, default)


def oft_block_limit(backend: str) -> int:
    """The largest diag-OFT block `backend`'s fused kernels take."""
    return _backend_limit(backend, "oft", "MAX_BLOCK", MAX_BLOCK)


def kron_factor_limit(backend: str) -> float:
    """The largest kron factor dim `backend`'s fused kron apply takes."""
    return _backend_limit(backend, "lokr", "MAX_KRON_FACTOR", math.inf)


def kron_plain(lyco) -> bool:
    """The kron's second half a plain matrix (the fused layout): no tucker core."""
    w2 = lyco.lokr_w2 if lyco.use_w2 else lyco.lokr_w2_b
    return not lyco.tucker and w2.dim() == 2


def kron_dims(lyco) -> tuple[int, int, int, int]:
    """(a, b, c, d) of kron(w1 (a, b), w2 (c, d)), off the parameters."""
    w1, w1a, w1b, w2, w2a, w2b = kron_reader(lyco)()
    a, b = w1.shape if w1 is not None else (w1a.shape[0], w1b.shape[1])
    c, d = w2.shape if w2 is not None else (w2a.shape[0], w2b.shape[1])
    return a, b, c, d


def fixed_scope(value: bool) -> Callable[[str], bool]:
    """A scope that is `value` on every backend."""

    def in_scope(backend: str) -> bool:
        return value

    return in_scope


def loha_scope(lyco, scale) -> Callable[[str], bool]:
    """A static scale and the hadamard rank within the fused kernels' bound."""
    return fixed_scope(static_scale(scale) and hada_rank(lyco) <= MAX_HADA_RANK)


def oft_scope(lyco) -> Callable[[str], bool]:
    """The diag-OFT block within the backend's bound."""
    size = lyco.oft_blocks.shape[-1]

    def in_scope(backend: str) -> bool:
        return size <= oft_block_limit(backend)

    return in_scope


def _rescale_reader(lyco) -> Callable:
    """() -> the rotation's rescale (None when not rescaled), chosen once per layer."""
    if lyco.rescaled:
        return lambda: lyco.rescale
    return lambda: None


def rotation_operands(lyco) -> Callable[[Tensor], tuple]:
    """t -> (t, the rotation blocks, the rescale or None)."""
    rescale = _rescale_reader(lyco)

    def operands(t: Tensor) -> tuple:
        return (t, lyco.oft_blocks, rescale())

    return operands


# Low-rank bypass: dy of a (tokens, in) activation; with base: base + dy, the add
# inside the op.


def lora_bypass(lyco, gamma: float) -> Op:
    def operands(x: Tensor) -> tuple:
        dt = x.dtype
        return (x, lyco.lora_down.weight.to(dt), lyco.lora_up.weight.to(dt))

    def kernel(backend: str, ops: tuple) -> Tensor:
        return locon_bypass_diff(*ops, gamma, backend)

    return Op(
        "locon.bypass",
        Tokens(in_features(lyco)),
        operands,
        kernel,
        bodies.lora,
        (gamma,),
        fixed_scope(static_scale(gamma)),
        TOKEN,
    )


def lora_bypass_with_base(lyco, gamma: float) -> Op:
    def operands(t: tuple) -> tuple:
        x, base = t
        dt = x.dtype
        return (x, lyco.lora_down.weight.to(dt), lyco.lora_up.weight.to(dt), base)

    def kernel(backend: str, ops: tuple) -> Tensor:
        x, d, u, base = ops
        return locon_bypass_diff(x, d, u, gamma, backend, base=base)

    return Op(
        "locon.bypass_with_base",
        TokensWithBase(in_features(lyco), out_features(lyco)),
        operands,
        kernel,
        bodies.lora_with_base,
        (gamma,),
        fixed_scope(static_scale(gamma)),
        TOKEN_AND_BASE,
    )


def _loha_factors(lyco, dt: torch.dtype) -> tuple:
    return (
        lyco.hada_w1_b.to(dt),
        lyco.hada_w1_a.to(dt),
        lyco.hada_w2_b.to(dt),
        lyco.hada_w2_a.to(dt),
    )


def loha_bypass(lyco, gamma: float) -> Op:
    def operands(x: Tensor) -> tuple:
        return (x, *_loha_factors(lyco, x.dtype))

    def kernel(backend: str, ops: tuple) -> Tensor:
        return loha_bypass_diff(*ops, gamma, backend)

    return Op(
        "loha.bypass",
        Tokens(in_features(lyco)),
        operands,
        kernel,
        bodies.loha,
        (bodies.hada_scale(gamma),),
        loha_scope(lyco, gamma),
        TOKEN,
    )


def loha_bypass_with_base(lyco, gamma: float) -> Op:
    def operands(t: tuple) -> tuple:
        x, base = t
        return (x, *_loha_factors(lyco, x.dtype), base)

    def kernel(backend: str, ops: tuple) -> Tensor:
        return loha_bypass_diff(*ops[:-1], gamma, backend, base=ops[-1])

    return Op(
        "loha.bypass_with_base",
        TokensWithBase(in_features(lyco), out_features(lyco)),
        operands,
        kernel,
        bodies.loha_with_base,
        (bodies.hada_scale(gamma),),
        loha_scope(lyco, gamma),
        TOKEN_AND_BASE,
    )


def _kron_apply_scope(lyco, scale) -> Callable[[str], bool]:
    plain = kron_plain(lyco) and static_scale(scale)
    widest = max(kron_dims(lyco))

    def in_scope(backend: str) -> bool:
        return plain and widest <= kron_factor_limit(backend)

    return in_scope


def lokr_bypass(lyco, scale: float) -> Op:
    read = kron_reader(lyco)

    def operands(x: Tensor) -> tuple:
        return (x, *read(x.dtype))

    def kernel(backend: str, ops: tuple) -> Tensor:
        return lokr_kron_bypass(*ops, None, scale, backend)

    return Op(
        "lokr.bypass",
        Tokens(in_features(lyco)),
        operands,
        kernel,
        bodies.lokr,
        (scale,),
        _kron_apply_scope(lyco, scale),
        TOKEN,
    )


def lokr_bypass_with_base(lyco, scale: float) -> Op:
    read = kron_reader(lyco)

    def operands(t: tuple) -> tuple:
        x, base = t
        return (x, *read(x.dtype), base)

    def kernel(backend: str, ops: tuple) -> Tensor:
        return lokr_kron_bypass(*ops[:-1], None, scale, backend, base=ops[-1])

    return Op(
        "lokr.bypass_with_base",
        TokensWithBase(in_features(lyco), out_features(lyco)),
        operands,
        kernel,
        bodies.lokr_with_base,
        (scale,),
        _kron_apply_scope(lyco, scale),
        TOKEN_AND_BASE,
    )


# Rotations on a (tokens, out) base output y: dy, or the rotated y itself (y + dy, the
# add inside the op).


def oft_bypass(lyco) -> Op:
    constraint = lyco.constraint

    def kernel(backend: str, ops: tuple) -> Tensor:
        return diag_oft_bypass_diff(*ops, constraint, False, backend)

    return Op(
        "diag_oft.bypass",
        Tokens(out_features(lyco)),
        rotation_operands(lyco),
        kernel,
        bodies.oft,
        (constraint,),
        oft_scope(lyco),
        TOKEN,
    )


def oft_rotate(lyco) -> Op:
    constraint = lyco.constraint

    def kernel(backend: str, ops: tuple) -> Tensor:
        return diag_oft_bypass_diff(*ops, constraint, False, backend, diff=False)

    return Op(
        "diag_oft.rotate",
        Tokens(out_features(lyco)),
        rotation_operands(lyco),
        kernel,
        bodies.oft_rotated,
        (constraint,),
        oft_scope(lyco),
        TOKEN,
    )


def boft_bypass(lyco, m: float) -> Op:
    constraint = lyco.constraint

    def kernel(backend: str, ops: tuple) -> Tensor:
        return boft_bypass_diff(*ops, constraint, m, False, backend)

    return Op(
        "boft.bypass",
        Tokens(out_features(lyco)),
        rotation_operands(lyco),
        kernel,
        bodies.boft,
        (constraint, m),
        tokens=TOKEN,
    )


def boft_rotate(lyco, m: float) -> Op:
    constraint = lyco.constraint

    def kernel(backend: str, ops: tuple) -> Tensor:
        return boft_bypass_diff(*ops, constraint, m, False, backend, diff=False)

    return Op(
        "boft.rotate",
        Tokens(out_features(lyco)),
        rotation_operands(lyco),
        kernel,
        bodies.boft_rotated,
        (constraint, m),
        tokens=TOKEN,
    )


def ia3_bypass_scale(lyco, m: float) -> Op:
    """IA3's channel scale x (w m + 1) of the input (train_input) or the base output."""
    width = in_features(lyco) if lyco.train_input else out_features(lyco)

    def operands(x: Tensor) -> tuple:
        return (x, lyco.weight.to(x.dtype))

    def kernel(backend: str, ops: tuple) -> Tensor:
        x, w = ops
        return ia3_bypass(x, w, -1, m, False, backend)

    return Op(
        "ia3.scale", Tokens(width), operands, kernel, bodies.ia3, (m,), tokens=TOKEN
    )
