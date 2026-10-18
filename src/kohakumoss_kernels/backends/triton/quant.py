"""Triton weight-only quantized linear (CUDA sm80+ and XPU; fp8 storage from sm89).

Forward: y = (x @ q^T) * scale + bias; the int8 / e4m3 tile converts exactly to
the activation dtype in registers, the scale is applied to the fp32 accumulator.
Dgrad: dx = dy @ (q * scale[:, None]); the scaled tile is rounded once to the
activation dtype, the same rounding as the dequantize-then-matmul reference.
Configs come from the tuner, which also times the dequantize+matmul path.
"""

import triton
import triton.language as tl

from kohakumoss_kernels import registry, tuning
from kohakumoss_kernels.constraints import requires
from kohakumoss_kernels.hardware.current import on_device
from kohakumoss_kernels.quant import (
    PYTORCH_PATHS,
    fmt_supported,
    make_quant_linear,
    pytorch_path,
    register_launchers,
)
from kohakumoss_kernels.tuning import tuner
from kohakumoss_kernels.tuning.buckets import pow2


@triton.jit
def _fwd_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    b_ptr,
    y_ptr,
    M,
    N,
    K,
    sxm,
    sxk,
    sqn,
    sqk,
    sym,
    syn,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    per_group = GROUP_M * num_n
    first_m = (pid // per_group) * GROUP_M
    group_m = min(num_m - first_m, GROUP_M)
    pid_m = first_m + ((pid % per_group) % group_m)
    pid_n = (pid % per_group) // group_m
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    x_ptrs = x_ptr + rm[:, None] * sxm + rk[None, :] * sxk
    q_ptrs = q_ptr + rn[None, :] * sqn + rk[:, None] * sqk
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        kmask = rk < K - k * BK
        a = tl.load(x_ptrs, mask=(rm[:, None] < M) & kmask[None, :], other=0.0)
        w = tl.load(q_ptrs, mask=kmask[:, None] & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, w.to(a.dtype), acc)
        x_ptrs += BK * sxk
        q_ptrs += BK * sqk
    nmask = rn < N
    acc = acc * tl.load(s_ptr + rn, mask=nmask, other=0.0)[None, :]
    if HAS_BIAS:
        acc += tl.load(b_ptr + rn, mask=nmask, other=0.0).to(tl.float32)[None, :]
    y_ptrs = y_ptr + rm[:, None] * sym + rn[None, :] * syn
    tl.store(
        y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=(rm[:, None] < M) & nmask[None, :]
    )


@triton.jit
def _dgrad_kernel(
    g_ptr,
    q_ptr,
    s_ptr,
    dx_ptr,
    M,
    N,
    K,
    sgm,
    sgn,
    sqn,
    sqk,
    sdm,
    sdk,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_k = tl.cdiv(K, BK)
    per_group = GROUP_M * num_k
    first_m = (pid // per_group) * GROUP_M
    group_m = min(num_m - first_m, GROUP_M)
    pid_m = first_m + ((pid % per_group) % group_m)
    pid_k = (pid % per_group) // group_m
    rm = pid_m * BM + tl.arange(0, BM)
    rk = pid_k * BK + tl.arange(0, BK)
    rn = tl.arange(0, BN)
    g_ptrs = g_ptr + rm[:, None] * sgm + rn[None, :] * sgn
    q_ptrs = q_ptr + rn[:, None] * sqn + rk[None, :] * sqk
    acc = tl.zeros((BM, BK), dtype=tl.float32)
    for n in range(0, tl.cdiv(N, BN)):
        nn = n * BN + rn
        nmask = nn < N
        g = tl.load(g_ptrs, mask=(rm[:, None] < M) & nmask[None, :], other=0.0)
        w = tl.load(q_ptrs, mask=nmask[:, None] & (rk[None, :] < K), other=0.0)
        s = tl.load(s_ptr + nn, mask=nmask, other=0.0)
        w = (w.to(tl.float32) * s[:, None]).to(g.dtype)
        acc = tl.dot(g, w, acc)
        g_ptrs += BN * sgn
        q_ptrs += BN * sqn
    dx_ptrs = dx_ptr + rm[:, None] * sdm + rk[None, :] * sdk
    tl.store(
        dx_ptrs,
        acc.to(dx_ptr.dtype.element_ty),
        mask=(rm[:, None] < M) & (rk[None, :] < K),
    )


_TILES = (
    (128, 128, 64, 8, 3),
    (128, 256, 64, 8, 3),
    (256, 128, 64, 8, 3),
    (128, 128, 32, 4, 4),
    (64, 128, 64, 4, 4),
    (128, 64, 64, 4, 4),
    (64, 64, 64, 4, 3),
    (32, 64, 64, 4, 3),
)


def _candidates(m: int) -> list[dict]:
    """Tiles that fit the problem first, then the dequantize+matmul path."""
    fits = [t for t in _TILES if t[0] <= max(32, pow2(m, 32))]
    ranked = sorted(fits, key=lambda t: -(t[0] * t[1]) if m >= 1024 else t[0])
    out = [dict(zip(("bm", "bn", "bk", "warps", "stages"), t)) for t in ranked]
    return [*out, *PYTORCH_PATHS]


def _launch_fwd(cfg, x, q, scale, bias, y):
    m, k = x.shape
    n = q.shape[0]
    grid = (triton.cdiv(m, cfg["bm"]) * triton.cdiv(n, cfg["bn"]),)
    with on_device(x.device):
        _fwd_kernel[grid](
            x, q, scale, bias if bias is not None else scale, y, m, n, k,
            *x.stride(), *q.stride(), *y.stride(),
            HAS_BIAS=bias is not None, BM=cfg["bm"], BN=cfg["bn"], BK=cfg["bk"],
            GROUP_M=8, num_warps=cfg["warps"], num_stages=cfg["stages"],
        )  # fmt: skip


def _launch_dgrad(cfg, g, q, scale, dx):
    m, n = g.shape
    k = q.shape[1]
    # output columns (K) tile by `bn`, reduction (N) by `bk`
    grid = (triton.cdiv(m, cfg["bm"]) * triton.cdiv(k, cfg["bn"]),)
    with on_device(g.device):
        _dgrad_kernel[grid](
            g, q, scale, dx, m, n, k, *g.stride(), *q.stride(), *dx.stride(),
            BM=cfg["bm"], BK=cfg["bn"], BN=cfg["bk"],
            GROUP_M=8, num_warps=cfg["warps"], num_stages=cfg["stages"],
        )  # fmt: skip


def _key(m, n, k, x, q, bias) -> str:
    return f"m{pow2(m, 32)}_n{n}_k{k}_{x.dtype}_{q.dtype}_b{bias is not None}"


_RESOLVED: dict[tuple, object] = {}
tuner.on_reset(_RESOLVED.clear)


def _fwd_runner(cfg):
    """(x, q, scale, bias) -> y for one config: the kernel or a PyTorch path."""
    ref = pytorch_path(cfg, "fwd")
    if ref is not None:
        return ref

    def run(x, q, scale, bias):
        y = x.new_empty(x.shape[0], q.shape[0])
        _launch_fwd(cfg, x, q, scale, bias, y)
        return y

    return run


def _dgrad_runner(cfg):
    ref = pytorch_path(cfg, "dgrad")
    if ref is not None:
        return ref

    def run(g, q, scale):
        dx = g.new_empty(g.shape[0], q.shape[1])
        _launch_dgrad(cfg, g, q, scale, dx)
        return dx

    return run


def fwd(x, q, scale, bias):
    m = x.shape[0]
    n, k = q.shape
    key = (
        "fwd",
        x.device.index,
        pow2(m, 32),
        n,
        k,
        x.dtype,
        q.dtype,
        bias is None,
    )
    run = _RESOLVED.get(key)
    if run is None:
        table_key = _key(m, n, k, x, q, bias)
        cfg = tuning.select(
            "quant_fwd",
            x.device,
            table_key,
            lambda: _candidates(m),
            lambda c: lambda: _fwd_runner(c)(x, q, scale, bias),
            shortlist=len(_TILES) + len(PYTORCH_PATHS),
        )
        run = _fwd_runner(cfg)
        # bind only a settled pick
        if tuning.decided("quant_fwd", x.device, table_key):
            _RESOLVED[key] = run
    return run(x, q, scale, bias)


def dgrad(g, q, scale):
    m = g.shape[0]
    n, k = q.shape
    key = ("dgrad", g.device.index, pow2(m, 32), n, k, g.dtype, q.dtype)
    run = _RESOLVED.get(key)
    if run is None:
        table_key = _key(m, n, k, g, q, None)
        cfg = tuning.select(
            "quant_dgrad",
            g.device,
            table_key,
            lambda: _candidates(m),
            lambda c: lambda: _dgrad_runner(c)(g, q, scale),
            shortlist=len(_TILES) + len(PYTORCH_PATHS),
            in_backward=True,
        )
        run = _dgrad_runner(cfg)
        if tuning.decided("quant_dgrad", g.device, table_key):
            _RESOLVED[key] = run
    return run(g, q, scale)


def _format_supported(query):
    return fmt_supported(query.get("fmt", "int8"), query.arch.sm)


register_launchers("triton", fwd, dgrad)
registry.register(
    "quant_linear",
    "triton",
    make_quant_linear("triton"),
    supports=requires(kinds=("cuda", "xpu"), min_sm=80, extra=_format_supported),
)
