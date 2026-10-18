"""TileLang SiTU-GLU: a = (4 tanh(g/4) sigmoid(g)) * (25 tanh(u/25)) of x = [u | g]."""

import tilelang.language as T
import torch
from torch import Tensor
from torch.library import custom_op

from kohakumoss_kernels.backends.compile.choice import kernel_or_compile, shape_key
from kohakumoss_kernels.backends.eager import ffn as eager_ffn
from kohakumoss_kernels.backends.tilelang.blocks._blockops import (
    DTYPE,
    F32,
    FP16_SUPPORTS,
    CompileError,
    compiled,
    tuned_run,
)
from kohakumoss_kernels.registry import register

__all__ = ["situ_glu"]

CFGS = [
    {"br": br, "bf": bf, "threads": th}
    for br, bf, th in ((8, 256, 256), (4, 512, 256), (2, 1024, 256), (8, 128, 128), (16, 256, 256), (4, 256, 128))
]  # fmt: skip


def _fwd_prim(f, BR, BF, THREADS, dtype):
    n = T.dynamic("n")

    @T.prim_func
    def main(x: T.Tensor((n, 2 * f), dtype), a: T.Tensor((n, f), dtype)):
        with T.Kernel(T.ceildiv(n, BR), T.ceildiv(f, BF), threads=THREADS) as (bx, by):
            for i, j in T.Parallel(BR, BF):
                r = bx * BR + i
                c = by * BF + j
                rr = T.min(r, n - 1)
                cc = T.min(c, f - 1)
                u = T.cast(x[rr, cc], F32)
                g = T.cast(x[rr, cc + f], F32)
                v = (4.0 * T.tanh(g * 0.25) * T.sigmoid(g)) * (25.0 * T.tanh(u * 0.04))
                if r < n and c < f:
                    a[r, c] = T.cast(v, dtype)

    return main


def _bwd_prim(f, BR, BF, THREADS, dtype):
    n = T.dynamic("n")

    @T.prim_func
    def main(
        gr: T.Tensor((n, f), dtype),
        x: T.Tensor((n, 2 * f), dtype),
        dx: T.Tensor((n, 2 * f), dtype),
    ):
        with T.Kernel(T.ceildiv(n, BR), T.ceildiv(f, BF), threads=THREADS) as (bx, by):
            for i, j in T.Parallel(BR, BF):
                r = bx * BR + i
                c = by * BF + j
                rr = T.min(r, n - 1)
                cc = T.min(c, f - 1)
                u = T.cast(x[rr, cc], F32)
                g = T.cast(x[rr, cc + f], F32)
                go = T.cast(gr[rr, cc], F32)
                sig = T.sigmoid(g)
                tg = T.tanh(g * 0.25)
                tu = T.tanh(u * 0.04)
                d_act = (1.0 - tg * tg) * sig + 4.0 * tg * sig * (1.0 - sig)
                if r < n and c < f:
                    dx[r, c] = T.cast(go * 4.0 * tg * sig * (1.0 - tu * tu), dtype)
                    dx[r, c + f] = T.cast(go * 25.0 * tu * d_act, dtype)

    return main


def _launch(prim, kind, x, f, *args):
    def run(c):
        compiled(prim, x.device, f, c["br"], c["bf"], c["threads"], DTYPE[x.dtype])(
            *args
        )

    tuned_run(f"tilelang_situ_{kind}", x.device, f"f{f}-{x.dtype}", CFGS, run)


def situ_forward(x: Tensor) -> Tensor:
    """[N, 2F] -> [N, F] (no autograd); shared with fused_linear."""
    n, f2 = x.shape
    a = torch.empty(n, f2 // 2, device=x.device, dtype=x.dtype)
    _launch(_fwd_prim, "fwd", x, f2 // 2, x, a)
    return a


def situ_backward(gr: Tensor, x: Tensor) -> Tensor:
    """d[u | g] [N, 2F] from the gradient of a [N, F] (no autograd)."""
    dx = torch.empty_like(x)
    _launch(_bwd_prim, "bwd", x, gr.shape[1], gr, x, dx)
    return dx


@custom_op("kohakumoss::tilelang_situ_fwd", mutates_args=())
def _fwd_op(x: Tensor) -> Tensor:
    return situ_forward(x)


@_fwd_op.register_fake
def _(x):
    return x.new_empty(x.shape[0], x.shape[1] // 2)


@custom_op("kohakumoss::tilelang_situ_bwd", mutates_args=())
def _bwd_op(gr: Tensor, x: Tensor) -> Tensor:
    return situ_backward(gr, x)


@_bwd_op.register_fake
def _(gr, x):
    return torch.empty_like(x)


def _setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[0])


def _backward(ctx, gr):
    (x,) = ctx.saved_tensors
    return _bwd_op(gr.contiguous(), x)


_fwd_op.register_autograd(_backward, setup_context=_setup)


def situ_glu(x: Tensor) -> Tensor:
    """x [..., 2F] = [up | gate] -> SiTU-GLU [..., F], as the eager reference computes."""
    shape = x.shape
    try:
        a = _fwd_op(x.reshape(-1, shape[-1]).contiguous())
    except CompileError:
        return eager_ffn.situ_glu(x)
    return a.view(*shape[:-1], shape[-1] // 2)


register(
    "situ_glu",
    "tilelang",
    kernel_or_compile("tilelang", "situ_glu", situ_glu, shape_key),
    FP16_SUPPORTS,
)
