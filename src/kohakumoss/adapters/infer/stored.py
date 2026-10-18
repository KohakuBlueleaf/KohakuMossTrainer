"""Inference adapters on a quantized / streamed / fp8-compute base: LyCORIS modules
built from a file without reading the base weight, and bypass forwards that add each
adapter's delta to the base's own output. No dense weight and no dense dW is held; a
DoRA ratio is computed once from one transient copy of the layer's effective weight.
See docs/internals/adapter-binding.md#quantized-and-streamed-bases.
"""

from functools import partial

import torch
from lycoris.modules import (
    ButterflyOFTModule,
    DiagOFTModule,
    GLoRAModule,
    IA3Module,
    TLoraModule,
    make_module,
)
from torch import Tensor, nn

from kohakumoss.adapters.build.network import svd_init, transient_base

ROTATIONS = (DiagOFTModule, ButterflyOFTModule)


def _tlora(name: str, linear: nn.Module, params: list):
    """A T-LoRA from its file tensors (q, p, lambda, alpha[, base q, p, lambda]); the
    frozen base factors come from the file, else from the SVD of the transient base."""
    q, p, lam, alpha, *base = list(params) + [None] * (7 - len(params))
    lyco = TLoraModule(
        name,
        linear,
        multiplier=1.0,
        lora_dim=q.shape[0],
        alpha=float(alpha),
        use_data_init=False,
    )
    if base[0] is None:
        svd_init(lyco, transient_base(linear))
    with torch.no_grad():
        lyco.q_layer.weight.copy_(q)
        lyco.p_layer.weight.copy_(p)
        lyco.lambda_layer.copy_(lam)
        for buffer, value in zip(("base_q", "base_p", "base_lambda"), base):
            if value is not None:
                getattr(lyco, buffer).copy_(value)
    return lyco


def build(lyco_type, params: list, name: str, linear: nn.Module) -> nn.Module:
    """`make_module` for a stored base: a rotation's rescale and a T-LoRA's frozen
    factors, which LyCORIS would size or derive from the base weight, set here."""
    if lyco_type is TLoraModule:
        return _tlora(name, linear, params)
    if lyco_type in ROTATIONS and params[1] is not None:
        rescale = params[1]
        lyco = make_module(lyco_type, [params[0], None, *params[2:]], name, linear)
        lyco.rescaled = True
        lyco.rescale = nn.Parameter(rescale.float().reshape(-1, 1))
        return lyco
    return make_module(lyco_type, params, name, linear)


def _unfold(prev, s: Tensor, x):
    return prev(x) / s.to(x.dtype)


def bypass(prev, lyco, m: float, x):
    """prev(x) + dy: dy the module's bypass delta."""
    y = prev(x)
    return y + lyco.bypass_forward_diff(x, m).to(y.dtype)


def bypass_less_bias(prev, lyco, m: float, bias_u: Tensor, x):
    """prev(x) + dy - b: less the base bias the delta adds once more."""
    y = prev(x)
    dy = lyco.bypass_forward_diff(x, m).to(y.dtype)
    return y + (dy - bias_u.to(y.dtype))


def bypass_folded(prev, lyco, m: float, s: Tensor, x):
    """prev(x) + s dy: dy in unfolded space, `s` the fold's row scale."""
    y = prev(x)
    dy = lyco.bypass_forward_diff(x, m).to(y.dtype)
    return y + dy * s.to(y.dtype)


def bypass_folded_less_bias(prev, lyco, m: float, s: Tensor, bias_u: Tensor, x):
    """prev(x) + s (dy - b_u): both terms in unfolded space."""
    y = prev(x)
    dy = lyco.bypass_forward_diff(x, m).to(y.dtype)
    return y + (dy - bias_u.to(y.dtype)) * s.to(y.dtype)


def wire(prev, lyco, m: float, s: Tensor | None, bias_u: Tensor | None):
    """The chained forward for one module (fold scale `s` and doubled bias `bias_u`
    chosen here); a rotation / channel kind reads the unfolded base output through
    `org_forward`."""
    lyco.org_forward = prev if s is None else partial(_unfold, prev, s)
    match (s is None, bias_u is None):
        case (True, True):
            return partial(bypass, prev, lyco, m)
        case (True, False):
            return partial(bypass_less_bias, prev, lyco, m, bias_u)
        case (False, True):
            return partial(bypass_folded, prev, lyco, m, s)
    return partial(bypass_folded_less_bias, prev, lyco, m, s, bias_u)


def diff_bias(lyco, linear: nn.Module, s: Tensor | None) -> Tensor | None:
    """The unfolded base bias a delta that runs the base forward again counts twice:
    GLoRA's W A x and input-axis IA3's W (w x); None for the other kinds."""
    reruns = isinstance(lyco, GLoRAModule) or (
        isinstance(lyco, IA3Module) and lyco.train_input
    )
    if not reruns or linear.bias is None:
        return None
    b = linear.bias.float()
    return b if s is None else b / s.to(b.device)


def dora_ratio(lyco, w: Tensor, dw: Tensor) -> Tensor:
    """mag / ||W + dW|| per output row (output-axis DoRA) or per input column."""
    dim = 1 if lyco.wd_on_out else 0
    norm = (w + dw.to(w.device)).norm(dim=dim) + torch.finfo(torch.float32).eps
    return (lyco.dora_scale.float().reshape(-1).to(norm.device) / norm).float()


def dora_out(prev, lyco, m: float, r: Tensor, x):
    """Output-axis DoRA: y + m ((r - 1) y + r dy), y the base output."""
    y = prev(x)
    dy = lyco.bypass_forward_diff(x, 1.0).to(y.dtype)
    r = r.to(y.dtype)
    return y + m * ((r - 1) * y + r * dy)


def dora_out_less_bias(prev, lyco, m: float, r: Tensor, bias: Tensor, x):
    """Output-axis DoRA: y + m ((r - 1)(y - b) + r dy)."""
    y = prev(x)
    wx = y - bias.to(y.dtype)
    dy = lyco.bypass_forward_diff(x, 1.0).to(y.dtype)
    r = r.to(y.dtype)
    return y + m * ((r - 1) * wx + r * dy)


def dora_out_folded(prev, lyco, m: float, r: Tensor, s: Tensor, x):
    """Output-axis DoRA: y + m ((r - 1) y + r s dy), y the folded base output."""
    y = prev(x)
    dy = lyco.bypass_forward_diff(x, 1.0).to(y.dtype) * s.to(y.dtype)
    r = r.to(y.dtype)
    return y + m * ((r - 1) * y + r * dy)


def dora_out_folded_less_bias(prev, lyco, m: float, r: Tensor, s: Tensor, bias, x):
    """Output-axis DoRA: y + m ((r - 1)(y - b) + r s dy), y the folded base output."""
    y = prev(x)
    wx = y - bias.to(y.dtype)
    dy = lyco.bypass_forward_diff(x, 1.0).to(y.dtype) * s.to(y.dtype)
    r = r.to(y.dtype)
    return y + m * ((r - 1) * wx + r * dy)


def dora_in(prev, lyco, m: float, c: Tensor, x):
    """Input-axis DoRA: y + m (prev(c x) + dy(c x) - y)."""
    y = prev(x)
    xc = x * c.to(x.dtype)
    dy = lyco.bypass_forward_diff(xc, 1.0).to(y.dtype)
    return y + m * (prev(xc) + dy - y)


def dora_in_folded(prev, lyco, m: float, c: Tensor, s: Tensor, x):
    """Input-axis DoRA: y + m (prev(c x) + s dy(c x) - y)."""
    y = prev(x)
    xc = x * c.to(x.dtype)
    dy = lyco.bypass_forward_diff(xc, 1.0).to(y.dtype) * s.to(y.dtype)
    return y + m * (prev(xc) + dy - y)


def wire_dora(prev, lyco, m: float, ratio: Tensor, s: Tensor | None, bias):
    """The DoRA chained forward for one module (axis, fold scale `s` and base bias
    `bias`, output axis only, chosen here)."""
    if not lyco.wd_on_out:
        if s is None:
            return partial(dora_in, prev, lyco, m, ratio)
        return partial(dora_in_folded, prev, lyco, m, ratio, s)
    match (s is None, bias is None):
        case (True, True):
            return partial(dora_out, prev, lyco, m, ratio)
        case (True, False):
            return partial(dora_out_less_bias, prev, lyco, m, ratio, bias)
        case (False, True):
            return partial(dora_out_folded, prev, lyco, m, ratio, s)
    return partial(dora_out_folded_less_bias, prev, lyco, m, ratio, s, bias)
