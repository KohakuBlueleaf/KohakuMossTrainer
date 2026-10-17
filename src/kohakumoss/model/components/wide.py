"""16-bit-input matmul with fp32 accumulation and an fp32 output, forward and backward."""

from collections.abc import Callable

import torch
from torch import Tensor


def _mm_native(a: Tensor, b: Tensor) -> Tensor:
    return torch.mm(a, b, out_dtype=torch.float32)


def _mm_rounded(a: Tensor, b: Tensor) -> Tensor:
    return torch.mm(a.float(), b.float())


_WIDE_MM: dict[tuple[torch.device, torch.dtype], Callable] = {}


def wide_mm(device: torch.device, dtype: torch.dtype) -> Callable:
    """(a, b) -> fp32 a @ b for `dtype` operands: torch.mm(out_dtype=fp32) where the
    backend has it, else an fp32 GEMM of the 16-bit operands. Resolved once per
    (device, dtype). See docs/concepts/numerics.md."""
    key = (device, dtype)
    if key not in _WIDE_MM:
        probe = torch.ones(1, 1, device=device, dtype=dtype)
        try:
            _mm_native(probe, probe)
            _WIDE_MM[key] = _mm_native
        except (NotImplementedError, RuntimeError):
            _WIDE_MM[key] = _mm_rounded
    return _WIDE_MM[key]


class _WideLinear(torch.autograd.Function):
    """y = x16 @ w16^T in fp32; grads are 16-bit-input fp32-output GEMMs in the inputs'
    dtypes. Saves only the 16-bit operands the requested grads need."""

    @staticmethod
    def forward(ctx, x: Tensor, w: Tensor, dtype: torch.dtype) -> Tensor:
        x16, w16 = x.to(dtype), w.to(dtype)
        mm = wide_mm(x.device, dtype)
        need_x, need_w = ctx.needs_input_grad[:2]
        ctx.save_for_backward(w16 if need_x else None, x16 if need_w else None)
        ctx.mm, ctx.dtype = mm, dtype
        ctx.dtypes = (x.dtype, w.dtype)
        return mm(x16, w16.t())

    @staticmethod
    def backward(ctx, grad: Tensor):
        w16, x16 = ctx.saved_tensors
        g16 = grad.to(ctx.dtype)
        gx = None if w16 is None else ctx.mm(g16, w16).to(ctx.dtypes[0])
        gw = None if x16 is None else ctx.mm(g16.t(), x16).to(ctx.dtypes[1])
        return gx, gw, None


def wide_linear(x: Tensor, w: Tensor, dtype: torch.dtype) -> Tensor:
    """x [..., in] @ w [out, in]^T with both cast to `dtype` (fp16 / bf16), accumulated in
    fp32 -> fp32 [..., out]."""
    y = _WideLinear.apply(x.reshape(-1, x.shape[-1]), w, dtype)
    return y.reshape(*x.shape[:-1], w.shape[0])
