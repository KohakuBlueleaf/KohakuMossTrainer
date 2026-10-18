"""OpSet: the ops a model build uses, resolved once from the kernel registry.

Primitive ops are resolved per (device, dtype, backend choice). Fused linear+epilogue
callables are bound per layer by the `bind_*` methods: the fused kernel when one is
registered and the layer is dense, else the composition of primitives. The
backbone re-binds after any structural change (adapter attach, base storage swap), never
per call.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import torch
from torch import Tensor

from kohakumoss.model.linear import MossLinear
from kohakumoss.types.seqinfo import AttnPlan
from kohakumoss_kernels import registry

PRIMITIVES = (
    "rms_norm",
    "rms_norm_mod",
    "qk_norm_rope",
    "attention_varlen",
    "situ_glu",
    "gated_residual",
    "token_mse",
)
FUSED = ("linear_qk_rope", "linear_situ", "linear_gated_residual")


@dataclass
class OpSet:
    """Resolved op callables for one (device, dtype, backend choice)."""

    rms_norm: Callable
    rms_norm_mod: Callable
    qk_norm_rope: Callable
    attention_varlen: Callable
    situ_glu: Callable
    gated_residual: Callable
    token_mse: Callable
    fused: dict[str, Callable] = field(default_factory=dict)

    @classmethod
    def resolve(
        cls,
        device: torch.device | str,
        dtype: torch.dtype,
        choices: dict[str, str] | None = None,
    ) -> "OpSet":
        """`choices` maps op name -> backend; unspecified ops follow the arch priority."""
        choices = choices or {}
        fns = {
            name: registry.resolve(name, device, dtype, backend=choices.get(name))
            for name in PRIMITIVES
        }
        fused = {}
        for name in FUSED:
            try:
                fused[name] = registry.resolve(
                    name, device, dtype, backend=choices.get(name)
                )
            except registry.NoImplementationError:
                continue
        return cls(**fns, fused=fused)

    def attention(self, q: Tensor, k: Tensor, v: Tensor, plan: AttnPlan) -> Tensor:
        """Packed attention over the segments of `plan`; softmax scale Dh^-0.5."""
        return self.attention_varlen(
            q, k, v, plan.cu_seqlens, plan.max_seqlen, q.shape[-1] ** -0.5
        )

    def bind_qk_rope(
        self, lin_q: MossLinear, lin_k: MossLinear, heads: int
    ) -> Callable:
        """(y, q_weight, k_weight, pos, bands, eps) -> (q, k) [N, H, Dh] for this layer."""
        fused = self.fused.get("linear_qk_rope")
        if fused is not None and lin_q.is_plain and lin_k.is_plain:
            return partial(_fused_qk_rope, fused, lin_q, lin_k, heads)
        return partial(_composed_qk_rope, self.qk_norm_rope, lin_q, lin_k, heads)

    def bind_situ(self, lin: MossLinear) -> Callable:
        """y -> situ_glu(lin(y)) for this layer."""
        fused = self.fused.get("linear_situ")
        if fused is not None and lin.is_plain:
            return partial(_fused_situ, fused, lin)
        return partial(_composed_situ, self.situ_glu, lin)

    def bind_gated_residual(self, lin: MossLinear) -> Callable:
        """(a, x, gate, seg, r) -> x * r + lin(a) * gate[seg] for this layer."""
        fused = self.fused.get("linear_gated_residual")
        if fused is not None and lin.is_plain:
            return partial(_fused_gated_residual, fused, lin)
        return partial(_composed_gated_residual, self.gated_residual, lin)


def _fused_qk_rope(fused, lin_q, lin_k, heads, y, q_weight, k_weight, pos, bands, eps):
    return fused(
        y, lin_q.weight, lin_k.weight, q_weight, k_weight, pos, bands, eps, heads
    )


def _composed_qk_rope(op, lin_q, lin_k, heads, y, q_weight, k_weight, pos, bands, eps):
    q = lin_q(y).unflatten(-1, (heads, -1))
    k = lin_k(y).unflatten(-1, (heads, -1))
    return op(q, k, q_weight, k_weight, pos, bands, eps)


def _fused_situ(fused, lin, y):
    return fused(y, lin.weight, lin.bias)


def _composed_situ(op, lin, y):
    return op(lin(y))


def _fused_gated_residual(fused, lin, a, x, gate, seg, r):
    return fused(a, lin.weight, lin.bias, x, gate, seg, r)


def _composed_gated_residual(op, lin, a, x, gate, seg, r):
    return op(x, lin(a), gate, seg, r)
