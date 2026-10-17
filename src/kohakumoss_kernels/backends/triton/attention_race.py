"""The `attention_varlen` registry entry on sm80+ / XPU: a per-shape race.

Candidates are the Triton kernels, the TileLang kernels (when the backend loads) and
torch: its flash-based `varlen_attn` where the build has it (left out under
`KOHAKUMOSS_DETERMINISTIC=1`), else the per-segment SDPA loop (the eager spec; not used
inside torch.compile). The tuner times forward + backward on its own leaf copies once
per problem key and binds the winner; later calls only look it up.
"""

import importlib

import torch
from torch import Tensor

from kohakumoss_kernels import tuning
from kohakumoss_kernels.backends.eager.attention import attention_varlen as sdpa_loop
from kohakumoss_kernels.backends.triton import attention as tri
from kohakumoss_kernels.errors import BUILD_ERRORS, unavailable
from kohakumoss_kernels.registry import register
from kohakumoss_kernels.tuning.tuner import on_reset

try:
    _TILELANG = importlib.import_module(
        "kohakumoss_kernels.backends.tilelang.attention"
    )
except ImportError:
    _TILELANG = None
try:
    from torch.nn.attention.varlen import varlen_attn as _varlen_attn
except ImportError:
    _varlen_attn = None

KERNEL = "attn_varlen_impl_triton"
_BOUND: dict[tuple, object] = {}
on_reset(_BOUND.clear)


def _torch_varlen(q, k, v, cu, max_len, scale):
    return _varlen_attn(q, k, v, cu, cu, max_len, max_len, scale=scale)


def _impls(device: torch.device) -> dict:
    impls = {"triton": tri.attention_varlen}
    if _TILELANG is not None and device.type == "cuda":
        impls["tilelang"] = _TILELANG.attention_varlen
    flash = device.type == "cuda" and torch.backends.cuda.is_flash_attention_available()
    if _varlen_attn is not None and flash and not tri.DETERMINISTIC:
        impls["torch_varlen"] = _torch_varlen
    else:
        impls["sdpa"] = sdpa_loop
    return impls


def _launch(fn, q, k, v, cu, max_len, scale):
    leaves = [t.detach().clone().requires_grad_() for t in (q, k, v)]
    grad = torch.randn_like(q)

    def run():
        for t in leaves:
            t.grad = None
        fn(*leaves, cu, max_len, scale).backward(grad)

    return run


def _build(q, k, v, cu, max_len, scale, key):
    impls = _impls(q.device)
    names = list(impls)
    choice = tuning.select(
        KERNEL,
        q.device,
        key,
        lambda: [{"impl": name} for name in names],
        lambda c: _launch(impls[c["impl"]], q, k, v, cu, max_len, scale),
        shortlist=len(names),
        errors=BUILD_ERRORS,
        autograd=True,
    )
    return impls[choice["impl"]]


def attention_varlen(
    q: Tensor, k: Tensor, v: Tensor, cu_seqlens: Tensor, max_seqlen: int, scale: float
) -> Tensor:
    """Packed varlen attention through the fastest implementation for this shape."""
    max_seqlen, scale = int(max_seqlen), float(scale)
    key = tri._key(q, cu_seqlens.shape[0] - 1, max_seqlen)
    ident = (q.device, key)
    if torch.compiler.is_compiling():
        fn = _BOUND.get(ident, tri.attention_varlen)
        fn = tri.attention_varlen if fn is sdpa_loop else fn
        return fn(q, k, v, cu_seqlens, max_seqlen, scale)
    try:
        fn = _BOUND.get(ident)
        if fn is None:
            fn = _build(q, k, v, cu_seqlens, max_seqlen, scale, key)
            if tuning.decided(KERNEL, q.device, key):
                _BOUND[ident] = fn
        return fn(q, k, v, cu_seqlens, max_seqlen, scale)
    except Exception as err:
        raise unavailable("attention_varlen", err, "triton race") from err


register("attention_varlen", "triton", attention_varlen, tri.SUPPORTS)
