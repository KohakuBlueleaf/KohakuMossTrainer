"""Pre-quantized frozen base weights for fp8 / MXFP8 compute, as MossLinear base storage.

`q` [N, K] e4m3 and `scale` in the recipe's layout (`kohakumoss_kernels.fp8.formats`)
are quantized once; `aux` holds the vendor-layout scale (mx: swizzled e8m0 bytes).
With a dgrad copy, `q_t` [K, N] is W^T quantized along N (the input-gradient GEMM's
contraction axis). `linear` runs the `fp8_linear` op bound once per (device, dtype);
activations are quantized per call. See docs/internals/quant-kernels.md.
"""

from functools import partial

import torch
from torch import Tensor, nn

from kohakumoss_kernels import registry
from kohakumoss_kernels.fp8 import RECIPES, dequantize, quantize, swizzle_mx

COMPUTE = ("fp8_compute", "mxfp8")
FP8_RECIPES = ("tensor", "row", "block")
DGRAD = ("fp8", "16bit")
ALIGN = 128
PARTS = ("q", "scale", "aux", "q_t", "scale_t", "aux_t")


def recipe_of(precision: str, fp8_recipe: str = "row") -> str:
    """The kernel recipe of a compute precision: mxfp8 -> mx, fp8_compute -> the
    fp8 recipe (tensor / row / block)."""
    return "mx" if precision == "mxfp8" else fp8_recipe


def eligible(out_features: int, in_features: int) -> str | None:
    """Why a [N, K] weight cannot take fp8 compute, or None."""
    if in_features % ALIGN or out_features % ALIGN:
        return f"[{out_features}, {in_features}] is not a multiple of {ALIGN}"
    return None


def quantize_pair(w: Tensor, recipe: str) -> tuple[Tensor, Tensor, Tensor | None]:
    """(q, scale, aux) of a 2-D weight under `recipe`."""
    q, scale = quantize(w, recipe, "weight")
    aux = swizzle_mx(scale) if recipe == "mx" else None
    return q, scale, aux


class ComputeWeight(nn.Module):
    """BaseWeight for fp8 compute; every part is a buffer (follows `.to(device)`)."""

    def __init__(self, recipe: str, parts: dict[str, Tensor | None]):
        super().__init__()
        if recipe not in RECIPES:
            raise ValueError(f"fp8 recipe {recipe!r} not in {RECIPES}")
        self.recipe = recipe
        for name in PARTS:
            self.register_buffer(name, parts.get(name))
        self.shape = self.q.shape
        self._linear = ComputeWeight._unbound

    @classmethod
    def from_weight(cls, w: Tensor, recipe: str, dgrad: bool = False):
        """Quantize `w` [N, K] once (and W^T for the fp8 input gradient)."""
        q, scale, aux = quantize_pair(w, recipe)
        parts = {"q": q, "scale": scale, "aux": aux}
        if dgrad:
            parts["q_t"], parts["scale_t"], parts["aux_t"] = quantize_pair(
                w.detach().t(), recipe
            )
        return cls(recipe, parts)

    @classmethod
    def from_stored(cls, recipe: str, q: Tensor, scale: Tensor):
        """The forward-only base of a stored (pre-quantized) `q` and `scale`."""
        aux = swizzle_mx(scale) if recipe == "mx" else None
        return cls(recipe, {"q": q, "scale": scale, "aux": aux})

    @property
    def has_dgrad(self) -> bool:
        return self.q_t is not None

    @torch.no_grad()
    def add_dgrad(self) -> None:
        """Store the W^T copy, quantized from the dequantized forward copy (a base
        loaded pre-quantized has no original weight)."""
        if self.has_dgrad:
            return
        w_t = self.weight(torch.float32).t()
        self.q_t, self.scale_t, self.aux_t = quantize_pair(w_t, self.recipe)

    def tensors(self) -> dict[str, Tensor]:
        """The stored parts that are present."""
        return {n: getattr(self, n) for n in PARTS if getattr(self, n) is not None}

    def weight(self, dtype: torch.dtype) -> Tensor:
        return dequantize(self.q, self.scale, self.recipe, dtype)

    def materialize(self) -> Tensor:
        return self.weight(torch.float32)

    def bind(
        self,
        dtype: torch.dtype,
        device=None,
        backend: str | None = None,
        autograd: bool = True,
    ) -> None:
        """Bind `linear` to the `fp8_linear` op for (device, dtype) once (`backend`
        pins one, e.g. "eager" for the eager reference); `autograd=False` binds the gradient-free
        `fp8_linear_infer` (a frozen model that never runs backward)."""
        device = self.q.device if device is None else torch.device(device)
        name = "fp8_linear" if autograd else "fp8_linear_infer"
        op = registry.resolve(name, device, dtype, backend, recipe=self.recipe)
        self._linear = partial(compute_linear, op)

    def linear(self, x: Tensor, bias: Tensor | None) -> Tensor:
        """x W^T (+ bias) through the bound `fp8_linear` op."""
        return self._linear(self, x, bias)

    def _unbound(self, x: Tensor, bias: Tensor | None) -> Tensor:
        self.bind(x.dtype, x.device)
        return self.linear(x, bias)

    def nbytes(self) -> int:
        return sum(t.nbytes for t in self.tensors().values())


def compute_linear(op, w, x: Tensor, bias: Tensor | None) -> Tensor:
    """`op` over the parts of `w` (a ComputeWeight or a streamed view of one)."""
    return op(x, w.q, w.scale, w.aux, w.q_t, w.scale_t, w.aux_t, bias, w.recipe)


class ComputeCache:
    """fp8 copies of a trainable weight (the forward copy and, with `dgrad`, W^T),
    rebuilt from the weight by `refresh` after every optimizer step; plain attributes
    (never module state, never moved by `.to`). `linear(x, w, bias)` runs the
    `fp8_linear_train` op: fp8 forward and input gradient, 16-bit weight gradient."""

    def __init__(self, recipe: str, dgrad: bool = True):
        if recipe not in RECIPES:
            raise ValueError(f"fp8 recipe {recipe!r} not in {RECIPES}")
        self.recipe, self.dgrad = recipe, dgrad
        self.q = self.scale = self.aux = None
        self.q_t = self.scale_t = self.aux_t = None
        self.op = None

    @torch.no_grad()
    def refresh(self, w: Tensor) -> None:
        """Requantize from the current weight."""
        self.q, self.scale, self.aux = quantize_pair(w, self.recipe)
        if self.dgrad:
            self.q_t, self.scale_t, self.aux_t = quantize_pair(w.t(), self.recipe)

    def bind(self, dtype: torch.dtype, device) -> None:
        self.op = registry.resolve(
            "fp8_linear_train", torch.device(device), dtype, recipe=self.recipe
        )

    def linear(self, x: Tensor, w: Tensor, bias: Tensor | None) -> Tensor:
        return self.op(
            x, w, self.q, self.scale, self.aux, self.q_t, self.scale_t, self.aux_t,
            bias, self.recipe,
        )  # fmt: skip

    def nbytes(self) -> int:
        parts = (self.q, self.scale, self.aux, self.q_t, self.scale_t, self.aux_t)
        return sum(t.nbytes for t in parts if t is not None)
