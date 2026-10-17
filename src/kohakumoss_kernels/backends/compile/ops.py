"""torch.compile of the eager specs, one compiled callable per op for the process."""

from collections.abc import Callable
from functools import cache

import torch
import torch.nn.functional as F
from torch import Tensor

from kohakumoss_kernels.backends.eager import ffn, loss, norm, rope
from kohakumoss_kernels.errors import unavailable


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


class CompiledOp:
    """torch.compile of one eager spec, built on the first call; a build failure of
    that call raises `BackendUnavailable`, later calls run the compiled callable
    directly. Inside an outer torch.compile the eager spec is traced. See
    docs/internals/fp16.md."""

    def __init__(self, name: str, dynamic: bool | None, backend: str | Callable):
        self.eager = EAGER[name]
        self.dynamic = dynamic
        self.backend = backend
        self.__name__ = f"compiled_{name}"
        self.run = self._first

    def __call__(self, *args):
        if torch.compiler.is_compiling():
            return self.eager(*args)
        return self.run(*args)

    def _first(self, *args):
        compiled = torch.compile(self.eager, dynamic=self.dynamic, backend=self.backend)
        try:
            out = compiled(*args)
        except Exception as err:  # noqa: BLE001
            raise unavailable(self.__name__, err) from err
        self.run = compiled
        return out


@cache
def compiled_op(
    name: str, dynamic: bool | None = True, backend: str | Callable = "inductor"
) -> CompiledOp:
    """The process's `CompiledOp` of `name` for (`dynamic`, `backend`)."""
    return CompiledOp(name, dynamic, backend)
