"""Triton SiTU-GLU: a = (4 tanh(g/4) sigmoid(g)) * (25 tanh(u/25)) of x = [u | g].

Shared elementwise bodies (`situ`, `situ_grad`) are also the epilogues of the fused
linear kernels in `fused_linear.py`.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op
from triton.language.extra import libdevice

from kohakumoss_kernels.backends.compile.choice import kernel_or_compile, shape_key
from kohakumoss_kernels.backends.triton._common import Launcher, eager_function
from kohakumoss_kernels.constraints import requires
from kohakumoss_kernels.registry import register
from kohakumoss_kernels.tuning.buckets import cdiv
from kohakumoss_kernels.tuning.tuner import select

__all__ = ["situ_glu"]

DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def situ(u, g):
    """fp32 SiTU-GLU of up `u` and gate `g`."""
    sig = 1.0 / (1.0 + libdevice.exp(-g))
    return (4.0 * libdevice.tanh(g * 0.25) * sig) * (25.0 * libdevice.tanh(u * 0.04))


@triton.jit
def situ_grad(u, g, gr):
    """(d up, d gate) of SiTU-GLU for upstream gradient `gr`, fp32."""
    sig = 1.0 / (1.0 + libdevice.exp(-g))
    tg = libdevice.tanh(g * 0.25)
    tu = libdevice.tanh(u * 0.04)
    act = 4.0 * tg * sig
    up = 25.0 * tu
    d_act = (1.0 - tg * tg) * sig + 4.0 * tg * sig * (1.0 - sig)
    d_up = 1.0 - tu * tu
    return gr * act * d_up, gr * up * d_act


@triton.jit
def _fwd_kernel(x_ptr, a_ptr, N, F, BR: tl.constexpr, BF: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cols = tl.program_id(1) * BF + tl.arange(0, BF)
    mask = (rows < N)[:, None] & (cols < F)[None, :]
    r64 = rows.to(tl.int64)[:, None]
    src = x_ptr + r64 * (2 * F) + cols[None, :]
    u = tl.load(src, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(src + F, mask=mask, other=0.0).to(tl.float32)
    out = situ(u, g)
    tl.store(a_ptr + r64 * F + cols[None, :], out.to(a_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _bwd_kernel(gr_ptr, x_ptr, dx_ptr, N, F, BR: tl.constexpr, BF: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cols = tl.program_id(1) * BF + tl.arange(0, BF)
    mask = (rows < N)[:, None] & (cols < F)[None, :]
    r64 = rows.to(tl.int64)[:, None]
    src = x_ptr + r64 * (2 * F) + cols[None, :]
    u = tl.load(src, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(src + F, mask=mask, other=0.0).to(tl.float32)
    gr = tl.load(gr_ptr + r64 * F + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    du, dg = situ_grad(u, g, gr)
    dst = dx_ptr + r64 * (2 * F) + cols[None, :]
    tl.store(dst, du.to(dx_ptr.dtype.element_ty), mask=mask)
    tl.store(dst + F, dg.to(dx_ptr.dtype.element_ty), mask=mask)


_fwd = Launcher(_fwd_kernel)
_bwd = Launcher(_bwd_kernel)


def candidates() -> list[dict]:
    return [
        {"BR": br, "BF": bf, "warps": w}
        for br, bf, w in (
            (8, 256, 4),
            (4, 512, 4),
            (16, 128, 4),
            (8, 512, 8),
            (2, 1024, 4),
        )
    ]


def launch_fwd(cfg, x, a):
    n, f = a.shape
    grid = (cdiv(n, cfg["BR"]), cdiv(f, cfg["BF"]))
    _fwd[grid](x, a, n, f, BR=cfg["BR"], BF=cfg["BF"], num_warps=cfg["warps"])


def launch_bwd(cfg, gr, x, dx):
    n, f = gr.shape
    grid = (cdiv(n, cfg["BR"]), cdiv(f, cfg["BF"]))
    _bwd[grid](gr, x, dx, n, f, BR=cfg["BR"], BF=cfg["BF"], num_warps=cfg["warps"])


def situ_forward(x: Tensor) -> Tensor:
    """[N, 2F] -> [N, F] launch (no autograd); shared with fused_linear."""
    n, f2 = x.shape
    a = torch.empty(n, f2 // 2, device=x.device, dtype=x.dtype)
    key = f"f{f2 // 2}-{x.dtype}"
    cfg = select(
        "triton_situ_fwd",
        x.device,
        key,
        candidates,
        lambda c: lambda: launch_fwd(c, x, a),
    )
    launch_fwd(cfg, x, a)
    return a


def situ_backward(gr: Tensor, x: Tensor) -> Tensor:
    """d[u | g] [N, 2F] from the gradient of a [N, F] (no autograd)."""
    dx = torch.empty_like(x)
    key = f"f{gr.shape[1]}-{x.dtype}"
    cfg = select(
        "triton_situ_bwd",
        x.device,
        key,
        candidates,
        lambda c: lambda: launch_bwd(c, gr, x, dx),
        in_backward=True,
    )
    launch_bwd(cfg, gr, x, dx)
    return dx


@triton_op("kohakumoss::triton_situ_fwd", mutates_args=())
def _fwd_op(x: Tensor) -> Tensor:
    return situ_forward(x)


@triton_op("kohakumoss::triton_situ_bwd", mutates_args=())
def _bwd_op(gr: Tensor, x: Tensor) -> Tensor:
    return situ_backward(gr, x)


def _setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[0])


def _backward_with(bwd):
    """The autograd backward over `bwd` (the backward op, or its body in eager)."""

    def backward(ctx, gr):
        (x,) = ctx.saved_tensors
        return bwd(gr.contiguous(), x)

    return backward


_fwd_op.register_autograd(_backward_with(_bwd_op), setup_context=_setup)
_Eager = eager_function("SituGlu", situ_forward, _setup, _backward_with(situ_backward))


def situ_glu(x: Tensor) -> Tensor:
    """x [..., 2F] = [up | gate] -> SiTU-GLU [..., F], as the eager reference computes."""
    shape = x.shape
    fwd = _fwd_op if torch.compiler.is_compiling() else _Eager.apply
    a = fwd(x.reshape(-1, shape[-1]).contiguous())
    return a.view(*shape[:-1], shape[-1] // 2)


register(
    "situ_glu",
    "triton",
    kernel_or_compile("triton", "situ_glu", situ_glu, shape_key),
    requires(kinds=("cuda", "xpu"), min_sm=70, dtypes=DTYPES),
)
