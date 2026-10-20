"""16-bit GEMMs on a host without native 16-bit GEMM support: every matmul-class op
(mm, addmm, bmm, baddbmm, mv, dot, convolution, attention; and the composite linear,
matmul, einsum, convN / conv_transposeN, scaled_dot_product_attention an inference-mode
call reaches the mode as) on bf16 / fp16 host tensors runs on its inputs upcast to
fp32 and rounds its output back to the input dtype once. Chosen once per dtype
(`host_gemm`). See docs/internals/text-encoder.md#host-gemm."""

from contextlib import nullcontext

import torch
from torch.utils._python_dispatch import TorchDispatchMode

aten = torch.ops.aten
HALF = (torch.bfloat16, torch.float16)
GEMMS = frozenset(
    {
        aten.mm.default,
        aten.addmm.default,
        aten.bmm.default,
        aten.baddbmm.default,
        aten.mv.default,
        aten.addmv.default,
        aten.dot.default,
        aten.convolution.default,
        aten._scaled_dot_product_flash_attention_for_cpu.default,
        aten.linear.default,
        aten.matmul.default,
        aten.einsum.default,
        aten.conv1d.default,
        aten.conv2d.default,
        aten.conv3d.default,
        aten.conv_transpose1d.default,
        aten.conv_transpose2d.input,
        aten.conv_transpose3d.input,
        aten.scaled_dot_product_attention.default,
    }
)


def native_half(dtype: torch.dtype) -> bool:
    """Whether the host has native GEMMs in `dtype` (torch's oneDNN capability)."""
    match dtype:
        case torch.bfloat16:
            return bool(torch.ops.mkldnn._is_mkldnn_bf16_supported())
        case torch.float16:
            return bool(torch.ops.mkldnn._is_mkldnn_fp16_supported())
    return True


def _half_cpu(x) -> bool:
    return isinstance(x, torch.Tensor) and x.device.type == "cpu" and x.dtype in HALF


def _halves(args):
    """The 16-bit host tensors among `args` (one level of lists included)."""
    for a in args:
        if isinstance(a, (list, tuple)):
            yield from (t for t in a if _half_cpu(t))
        elif _half_cpu(a):
            yield a


def _up(x):
    if isinstance(x, (list, tuple)):
        return type(x)(_up(t) for t in x)
    return x.float() if _half_cpu(x) else x


def _down(out, dtype: torch.dtype):
    if isinstance(out, torch.Tensor) and out.dtype == torch.float32:
        return out.to(dtype)
    if isinstance(out, tuple):
        return (_down(out[0], dtype), *out[1:])
    return out


class HostGemm(TorchDispatchMode):
    """Inside the mode, a matmul-class op (`GEMMS`) on a 16-bit host tensor runs in
    fp32 on its upcast inputs, its first output rounded back to that dtype."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func not in GEMMS:
            return func(*args, **kwargs)
        half = next((a.dtype for a in _halves(args)), None)
        if half is None:
            return func(*args, **kwargs)
        up_args = [_up(a) for a in args]
        up_kwargs = {k: _up(v) for k, v in kwargs.items()}
        return _down(func(*up_args, **up_kwargs), half)


def host_gemm(dtype: torch.dtype):
    """The context a host pass in `dtype` runs under: HostGemm where the host has no
    native GEMMs in a 16-bit `dtype`, else nothing."""
    return nullcontext() if native_half(dtype) else HostGemm()
