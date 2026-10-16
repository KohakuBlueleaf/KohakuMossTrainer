"""torch.compile of the eager specs, one compiled callable per op for the process."""

from collections.abc import Callable
from functools import cache

import torch
import torch.nn.functional as F
from torch import Tensor

from kohakumoss_kernels.backends.eager import ffn, loss, norm, rope


def _linear_situ(y: Tensor, w: Tensor, bias: Tensor | None) -> Tensor:
    return ffn.situ_glu(
        F.linear(y, w.to(y.dtype), None if bias is None else bias.to(y.dtype))
    )


def _linear_gated_residual(a, w, bias, x, gate, seg, r):
    o = F.linear(a, w.to(a.dtype), None if bias is None else bias.to(a.dtype))
    return norm.gated_residual(x, o, gate, seg, r)


def _linear_qk_rope(y, wq, wk, q_weight, k_weight, pos, bands, eps, heads):
    q = F.linear(y, wq.to(y.dtype)).unflatten(-1, (heads, -1))
    k = F.linear(y, wk.to(y.dtype)).unflatten(-1, (heads, -1))
    return rope.qk_norm_rope(q, k, q_weight, k_weight, pos, bands, eps)


EAGER: dict[str, Callable] = {
    "rms_norm": norm.rms_norm,
    "rms_norm_mod": norm.rms_norm_mod,
    "gated_residual": norm.gated_residual,
    "qk_norm_rope": rope.qk_norm_rope,
    "situ_glu": ffn.situ_glu,
    "token_mse": loss.token_mse,
    "linear_situ": _linear_situ,
    "linear_gated_residual": _linear_gated_residual,
    "linear_qk_rope": _linear_qk_rope,
}
COMPILED = tuple(EAGER)


@cache
def compiled_op(name: str) -> Callable:
    """torch.compile(eager spec of `name`), dynamic shapes, built lazily on first call.

    Inside an outer torch.compile the eager spec itself is traced (no nested compile).
    """
    eager = EAGER[name]
    holder: list = []

    def call(*args):
        if torch.compiler.is_compiling():
            return eager(*args)
        if not holder:
            holder.append(torch.compile(eager, dynamic=True))
        return holder[0](*args)

    call.__name__ = f"compiled_{name}"
    return call
