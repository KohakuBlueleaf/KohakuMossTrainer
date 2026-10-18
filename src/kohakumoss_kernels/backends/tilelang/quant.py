"""TileLang weight-only int8 linear (the sm70 / sm75 tensor-core path).

Forward: y = (x @ q^T) * scale + bias, the int8 tile converted to the activation
dtype in shared memory and multiplied on the MMA units, the scale applied to the
fp32 accumulator. Dgrad: dx = dy @ (q * scale[:, None]), the scaled tile rounded
once to the activation dtype. The token count is symbolic (one compile per
weight shape); outputs are caller-allocated.
"""

import tilelang.language as T

from kohakumoss_kernels import registry
from kohakumoss_kernels.backends.tilelang.blocks._blockops import DTYPE, compiled, tuned
from kohakumoss_kernels.constraints import requires
from kohakumoss_kernels.quant import (
    PYTORCH_PATHS,
    make_quant_linear,
    pytorch_path,
    register_launchers,
)
from kohakumoss_kernels.tuning import decided, tuner
from kohakumoss_kernels.tuning.buckets import pow2

F32 = "float32"
I8 = "int8"


def _fwd_prim(n, k, bm, bn, bk, threads, has_bias, dtype):
    m = T.dynamic("m")

    @T.prim_func
    def main(
        x: T.Tensor((m, k), dtype),
        q: T.Tensor((n, k), I8),
        s: T.Tensor((n,), F32),
        b: T.Tensor((n,), F32),
        y: T.Tensor((m, n), dtype),
    ):
        with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm), threads=threads) as (bx, by):
            x_s = T.alloc_shared((bm, bk), dtype)
            q_s = T.alloc_shared((bn, bk), I8)
            w_s = T.alloc_shared((bn, bk), dtype)
            acc = T.alloc_fragment((bm, bn), F32)
            T.clear(acc)
            for kk in T.Pipelined(T.ceildiv(k, bk), num_stages=1):
                T.copy(x[by * bm, kk * bk], x_s)
                T.copy(q[bx * bn, kk * bk], q_s)
                for i, j in T.Parallel(bn, bk):
                    w_s[i, j] = T.cast(q_s[i, j], dtype)
                T.gemm(x_s, w_s, acc, transpose_B=True)
            for i, j in T.Parallel(bm, bn):
                row = by * bm + i
                col = bx * bn + j
                cc = T.min(col, n - 1)
                v = acc[i, j] * s[cc]
                if has_bias:
                    v = v + b[cc]
                if row < m and col < n:
                    y[row, col] = T.cast(v, dtype)

    return main


def _dgrad_prim(n, k, bm, bk, bn, threads, dtype):
    m = T.dynamic("m")

    @T.prim_func
    def main(
        g: T.Tensor((m, n), dtype),
        q: T.Tensor((n, k), I8),
        s: T.Tensor((n,), F32),
        dx: T.Tensor((m, k), dtype),
    ):
        with T.Kernel(T.ceildiv(k, bk), T.ceildiv(m, bm), threads=threads) as (bx, by):
            g_s = T.alloc_shared((bm, bn), dtype)
            q_s = T.alloc_shared((bn, bk), I8)
            w_s = T.alloc_shared((bn, bk), dtype)
            acc = T.alloc_fragment((bm, bk), F32)
            T.clear(acc)
            for nn in T.Pipelined(T.ceildiv(n, bn), num_stages=1):
                T.copy(g[by * bm, nn * bn], g_s)
                T.copy(q[nn * bn, bx * bk], q_s)
                for i, j in T.Parallel(bn, bk):
                    w_s[i, j] = T.cast(
                        T.cast(q_s[i, j], F32) * s[T.min(nn * bn + i, n - 1)], dtype
                    )
                T.gemm(g_s, w_s, acc)
            for i, j in T.Parallel(bm, bk):
                if by * bm + i < m and bx * bk + j < k:
                    dx[by * bm + i, bx * bk + j] = T.cast(acc[i, j], dtype)

    return main


_CFGS = [
    {"bm": bm, "bn": bn, "bk": bk, "threads": th}
    for bm, bn, bk, th in (
        (128, 128, 32, 256),
        (128, 128, 32, 128),
        (64, 128, 32, 128),
        (128, 64, 32, 128),
        (64, 64, 32, 128),
        (32, 64, 32, 128),
    )
]
_RESOLVED: dict[tuple, object] = {}
tuner.on_reset(_RESOLVED.clear)


def _fwd_runner(c, dev, n, k, has_bias, dtype):
    """(x, q, scale, bias) -> y for one config: the kernel or a PyTorch path."""
    ref = pytorch_path(c, "fwd")
    if ref is not None:
        return ref
    kernel = compiled(
        _fwd_prim,
        dev,
        n,
        k,
        c["bm"],
        c["bn"],
        c["bk"],
        c["threads"],
        has_bias,
        DTYPE[dtype],
    )

    def run(x, q, scale, bias):
        y = x.new_empty(x.shape[0], n)
        kernel(x, q, scale, scale if bias is None else bias.float().contiguous(), y)
        return y

    return run


def _dgrad_runner(c, dev, n, k, dtype):
    ref = pytorch_path(c, "dgrad")
    if ref is not None:
        return ref
    kernel = compiled(
        _dgrad_prim, dev, n, k, c["bm"], c["bn"], c["bk"], c["threads"], DTYPE[dtype]
    )

    def run(g, q, scale):
        dx = g.new_empty(g.shape[0], k)
        kernel(g, q, scale, dx)
        return dx

    return run


def fwd(x, q, scale, bias):
    m, k = x.shape
    n = q.shape[0]
    key = ("fwd", x.device.index, pow2(m, 32), n, k, x.dtype, bias is not None)
    run = _RESOLVED.get(key)
    if run is None:
        dev = x.device
        args = (dev, n, k, bias is not None, x.dtype)
        table_key = f"m{key[2]}_n{n}_k{k}_{x.dtype}_b{key[6]}"
        c = tuned(
            "quant_fwd_tl",
            dev,
            table_key,
            [*_CFGS, *PYTORCH_PATHS],
            lambda c: _fwd_runner(c, *args)(x, q, scale, bias),
        )
        run = _fwd_runner(c, *args)
        # bind only a settled pick
        if decided("quant_fwd_tl", dev, table_key):
            _RESOLVED[key] = run
    return run(x, q, scale, bias)


def dgrad(g, q, scale):
    m, n = g.shape
    k = q.shape[1]
    key = ("dgrad", g.device.index, pow2(m, 32), n, k, g.dtype)
    run = _RESOLVED.get(key)
    if run is None:
        dev = g.device
        args = (dev, n, k, g.dtype)
        table_key = f"m{key[2]}_n{n}_k{k}_{g.dtype}"
        c = tuned(
            "quant_dgrad_tl",
            dev,
            table_key,
            [*_CFGS, *PYTORCH_PATHS],
            lambda c: _dgrad_runner(c, *args)(g, q, scale),
        )
        run = _dgrad_runner(c, *args)
        if decided("quant_dgrad_tl", dev, table_key):
            _RESOLVED[key] = run
    return run(g, q, scale)


def _int8_only(query):
    fmt = query.get("fmt", "int8")
    return None if fmt == "int8" else f"tilelang quant serves int8 only, not {fmt}"


register_launchers("tilelang", fwd, dgrad)
registry.register(
    "quant_linear",
    "tilelang",
    make_quant_linear("tilelang"),
    supports=requires(kinds=("cuda",), min_sm=70, extra=_int8_only),
)
