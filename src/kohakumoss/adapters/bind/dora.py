"""DoRA epilogues on an already merged weight W + dW (multiplier 1), and DoRA as a
bypass over a quantized / streamed base. See
docs/internals/adapter-binding.md#quantized-and-streamed-bases."""

from collections.abc import Callable

import torch
from lycoris.functional.general import weight_decompose
from torch import Tensor

F32 = torch.float32


def _eps_norm(norm: Tensor, dtype: torch.dtype) -> Tensor:
    return (norm + torch.finfo(dtype).eps).detach()


def detached_out(merged: Tensor, dora_scale: Tensor) -> Tensor:
    """Output-axis DoRA with the row norm treated as a constant in backward."""
    norm = merged.reshape(merged.shape[0], -1).norm(dim=1, keepdim=True)
    return merged * (dora_scale.to(merged.dtype) / _eps_norm(norm, merged.dtype))


def detached_in(merged: Tensor, dora_scale: Tensor) -> Tensor:
    """Input-axis DoRA with the column norm treated as a constant in backward."""
    norm = merged.norm(dim=0, keepdim=True)
    return merged * (dora_scale.to(merged.dtype) / _eps_norm(norm, merged.dtype))


def detached(fn: Callable, lyco) -> Callable[[Tensor], Tensor]:
    """merged -> a detached DoRA epilogue (`fn`) with the layer's magnitude."""

    def decompose(merged: Tensor) -> Tensor:
        return fn(merged, lyco.dora_scale.float())

    return decompose


def decomposed(t, backend: str | None, merged: Tensor) -> Tensor:
    """LyCORIS weight_decompose on an already-merged weight (multiplier 1)."""
    scale = t.lyco.dora_scale.float()
    return weight_decompose(merged, scale, 1.0, t.wd_on_out, backend=backend)


def decomposer(t, backend: str | None) -> Callable[[Tensor], Tensor]:
    """merged -> `decomposed(t, backend, merged)`."""

    def decompose(merged: Tensor) -> Tensor:
        return decomposed(t, backend, merged)

    return decompose


def magnitude_ratio(base_fn, unit_fn, lyco, dim: int, detach: bool) -> Tensor:
    """m / ||W + dW|| over `dim` (fp32, flattened): W read once, transiently, from its
    storage and dropped before the forward runs."""
    w = base_fn(F32)
    norm = (w + unit_fn().to(w.device)).norm(dim=dim) + torch.finfo(F32).eps
    del w
    if detach:
        norm = norm.detach()
    return lyco.dora_scale.float().reshape(-1) / norm


def _out(y: Tensor, wx: Tensor, s: Tensor, lyco, m: float, x) -> Tensor:
    s = s.to(device=y.device, dtype=y.dtype)
    dy = lyco.bypass_forward_diff(x, 1.0).to(y.dtype)
    return y + m * ((s - 1) * wx + s * dy)


def bypass_out(layer, base_fn, unit_fn, lyco, m: float, detach: bool, x):
    """Output-axis DoRA without a merged weight on a bias-less layer: y = (1 - m) Wx +
    m s (Wx + dWx) with s = mag / ||W + dW||_row, Wx from the base's own forward;
    `layer` is a weak reference to the layer."""
    s = magnitude_ratio(base_fn, unit_fn, lyco, 1, detach)
    lin = layer()
    y = lin._impl(lin, x)
    return _out(y, y, s, lyco, m, x)


def bypass_out_biased(layer, base_fn, unit_fn, lyco, m: float, detach: bool, x):
    """`bypass_out` on a layer with a bias b: y = b + (1 - m) Wx + m s (Wx + dWx)."""
    s = magnitude_ratio(base_fn, unit_fn, lyco, 1, detach)
    lin = layer()
    y = lin._impl(lin, x)
    return _out(y, y - lin.bias.to(y.dtype), s, lyco, m, x)


def bypass_in(layer, base_fn, unit_fn, lyco, m: float, detach: bool, x):
    """Input-axis DoRA without a merged weight: W'x = (W + dW)(c x) with
    c = mag / ||W + dW||_col, both products through the base forward and the bypass."""
    c = magnitude_ratio(base_fn, unit_fn, lyco, 0, detach)
    xc = x * c.to(device=x.device, dtype=x.dtype)
    lin = layer()
    y = lin._impl(lin, x)
    yc = lin._impl(lin, xc) + lyco.bypass_forward_diff(xc, 1.0).to(y.dtype)
    return y + m * (yc - y)
