"""fp8 test references: the exact fp64 value of a quantized operand, and a weight-only
binding of fp8-compute bases (the reference output of an fp8-compute model: its
quantized weights, unquantized activations)."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kohakumoss.quant.compute import ComputeWeight


def deq64(q: Tensor, s: Tensor, recipe: str) -> Tensor:
    """q * scale per element in fp64 (exact); a block scale with fewer rows than q is a
    128x128 weight-block scale."""
    rows, k = q.shape
    r = torch.arange(rows, device=q.device)[:, None]
    c = torch.arange(k, device=q.device)[None, :]
    v = q.double()
    match recipe:
        case "tensor":
            return v * s.double().reshape(())
        case "row":
            return v * s.double()[:, None]
        case "mx":
            return v * torch.pow(2.0, s.double()[r, c // 32] - 127)
    if s.shape[0] == rows:
        return v * s.double()[r, c // 128]
    return v * s.double()[r // 128, c // 128]


def _weight_only(w: ComputeWeight, x: Tensor, bias: Tensor | None) -> Tensor:
    weight = deq64(w.q, w.scale, w.recipe).to(x.dtype)
    return F.linear(x, weight, None if bias is None else bias.to(x.dtype))


def double_params(model: nn.Module) -> nn.Module:
    """fp64 parameters and floating buffers, ComputeWeight parts untouched."""
    for mod in model.modules():
        for p in mod.parameters(recurse=False):
            p.data = p.data.double()
        if isinstance(mod, ComputeWeight):
            continue
        for name, b in list(mod.named_buffers(recurse=False)):
            if b is not None and b.is_floating_point():
                setattr(mod, name, b.double())
    return model


def bind_weight_only(model: nn.Module) -> None:
    """Every ComputeWeight runs x @ deq(W)^T with no activation quantization."""
    for m in model.modules():
        if isinstance(m, ComputeWeight):
            m.linear = lambda x, b, m=m: _weight_only(m, x, b)
