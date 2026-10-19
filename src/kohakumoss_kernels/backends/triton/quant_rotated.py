"""Triton rotated weight-only quantized linear (ConvRot): y = (x R) (q * scale)^T + b.

Per shape the tuner chooses between `separate` (the Triton rotation pass, then the
plain quant kernel; its own tuning picks kernel tile or PyTorch path) and the fused
kernels:
- forward: each [BM, BK] tile of x (BK a multiple of the group) is rotated in fp32
  registers and rounded once to the activation dtype in the prologue -- the bytes the
  separate pass writes -- before it meets the int8 tile on the MMA units;
- dgrad: the fp32 accumulator tile [BM, BK] of dy (q * scale) is rotated in the
  epilogue and rounded once (the separate path rounds before and after the rotation).
See docs/internals/quant-kernels.md#rotated-int8-linear.
"""

import triton
import triton.language as tl

from kohakumoss_kernels import registry, tuning
from kohakumoss_kernels.backends.triton import quant as plain
from kohakumoss_kernels.backends.triton.hadamard import hadamard_tile, rotate_2d
from kohakumoss_kernels.constraints import requires
from kohakumoss_kernels.hadamard import stages
from kohakumoss_kernels.hardware.current import on_device
from kohakumoss_kernels.quant import fmt_supported
from kohakumoss_kernels.quant.rotated import (
    compose,
    group_supported,
    make_quant_linear_rotated,
    register_rotated_launchers,
)
from kohakumoss_kernels.tuning import tuner
from kohakumoss_kernels.tuning.buckets import pow2

SEPARATE = {"impl": "separate"}
_FWD = (
    (64, 64, 4, 2),
    (32, 128, 4, 2),
    (64, 128, 8, 1),
    (128, 64, 8, 1),
    (32, 64, 4, 3),
)
_DGRAD = (
    (64, 32, 8, 2),
    (32, 64, 4, 2),
    (128, 32, 8, 1),
    (64, 64, 8, 1),
    (32, 32, 4, 3),
)


@triton.jit
def _fwd_kernel(
    x_ptr, q_ptr, s_ptr, b_ptr, y_ptr, M, N, K,
    sxm, sxk, sqn, sqk, sym, syn, hscale,
    HAS_BIAS: tl.constexpr, G: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
):  # fmt: skip
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
    dt = x_ptr.dtype.element_ty
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K // BK):
        a = tl.load(x_ptrs, mask=rm[:, None] < M, other=0.0).to(tl.float32)
        a = (hadamard_tile(a, BM, BK, G) * hscale).to(dt)
        w = tl.load(q_ptrs, mask=rn[None, :] < N, other=0.0)
        acc = tl.dot(a, w.to(dt), acc)
        x_ptrs += BK * sxk
        q_ptrs += BK * sqk
    nmask = rn < N
    acc = acc * tl.load(s_ptr + rn, mask=nmask, other=0.0)[None, :]
    if HAS_BIAS:
        acc += tl.load(b_ptr + rn, mask=nmask, other=0.0).to(tl.float32)[None, :]
    y_ptrs = y_ptr + rm[:, None] * sym + rn[None, :] * syn
    tl.store(y_ptrs, acc.to(dt), mask=(rm[:, None] < M) & nmask[None, :])


@triton.jit
def _dgrad_kernel(
    g_ptr, q_ptr, s_ptr, dx_ptr, M, N, K,
    sgm, sgn, sqn, sqk, sdm, sdk, hscale,
    G: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
    GROUP_M: tl.constexpr,
):  # fmt: skip
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_k = K // BK
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
        w = tl.load(q_ptrs, mask=nmask[:, None], other=0.0)
        s = tl.load(s_ptr + nn, mask=nmask, other=0.0)
        w = (w.to(tl.float32) * s[:, None]).to(g.dtype)
        acc = tl.dot(g, w, acc)
        g_ptrs += BN * sgn
        q_ptrs += BN * sqn
    acc = hadamard_tile(acc, BM, BK, G) * hscale
    dx_ptrs = dx_ptr + rm[:, None] * sdm + rk[None, :] * sdk
    tl.store(dx_ptrs, acc.to(dx_ptr.dtype.element_ty), mask=rm[:, None] < M)


def fwd_candidates(k: int, group: int) -> list[dict]:
    """`separate` first, then the fused tiles (K tile = one group) when K allows."""
    out = [SEPARATE]
    if k % group == 0 and group >= 16:
        out += [
            {"impl": "fused", "bm": bm, "bn": bn, "warps": w, "stages": s}
            for bm, bn, w, s in _FWD
        ]
    return out


def dgrad_candidates(k: int, group: int) -> list[dict]:
    """`separate` first, then the fused tiles (output K tile = one group)."""
    out = [SEPARATE]
    if k % group == 0 and group >= 16:
        out += [
            {"impl": "fused", "bm": bm, "bn": bn, "warps": w, "stages": s}
            for bm, bn, w, s in _DGRAD
        ]
    return out


def launch_fwd(cfg, x, q, scale, bias, y, group):
    m, k = x.shape
    n = q.shape[0]
    grid = (triton.cdiv(m, cfg["bm"]) * triton.cdiv(n, cfg["bn"]),)
    with on_device(x.device):
        _fwd_kernel[grid](
            x, q, scale, bias if bias is not None else scale, y, m, n, k,
            *x.stride(), *q.stride(), *y.stride(), 2.0 ** -stages(group),
            HAS_BIAS=bias is not None, G=group, BM=cfg["bm"], BN=cfg["bn"],
            BK=group, GROUP_M=8, num_warps=cfg["warps"], num_stages=cfg["stages"],
        )  # fmt: skip


def launch_dgrad(cfg, g, q, scale, dx, group):
    m, n = g.shape
    k = q.shape[1]
    grid = (triton.cdiv(m, cfg["bm"]) * (k // group),)
    with on_device(g.device):
        _dgrad_kernel[grid](
            g, q, scale, dx, m, n, k, *g.stride(), *q.stride(), *dx.stride(),
            2.0 ** -stages(group), G=group, BM=cfg["bm"], BK=group, BN=cfg["bn"],
            GROUP_M=8, num_warps=cfg["warps"], num_stages=cfg["stages"],
        )  # fmt: skip


separate_fwd, separate_dgrad = compose(plain.fwd, plain.dgrad, rotate_2d)


def fwd_runner(cfg):
    """(x, q, scale, bias, group) -> y for one config: separate or a fused tile."""
    if cfg["impl"] == "separate":
        return separate_fwd

    def run(x, q, scale, bias, group):
        y = x.new_empty(x.shape[0], q.shape[0])
        launch_fwd(cfg, x, q, scale, bias, y, group)
        return y

    return run


def dgrad_runner(cfg):
    """(dy, q, scale, group) -> dx for one config: separate or a fused tile."""
    if cfg["impl"] == "separate":
        return separate_dgrad

    def run(g, q, scale, group):
        dx = g.new_empty(g.shape[0], q.shape[1])
        launch_dgrad(cfg, g, q, scale, dx, group)
        return dx

    return run


_RESOLVED: dict[tuple, object] = {}
tuner.on_reset(_RESOLVED.clear)


def _key(m, n, k, group, x, q, bias) -> str:
    return f"m{pow2(m, 32)}_n{n}_k{k}_g{group}_{x.dtype}_{q.dtype}_b{bias is not None}"


def fwd(x, q, scale, bias, group):
    m = x.shape[0]
    n, k = q.shape
    key = (
        "fwd",
        x.device.index,
        pow2(m, 32),
        n,
        k,
        group,
        x.dtype,
        q.dtype,
        bias is None,
    )
    run = _RESOLVED.get(key)
    if run is None:
        table_key = _key(m, n, k, group, x, q, bias)
        cands = fwd_candidates(k, group)
        cfg = tuning.select(
            "quant_fwd_rot",
            x.device,
            table_key,
            lambda: cands,
            lambda c: lambda: fwd_runner(c)(x, q, scale, bias, group),
            shortlist=len(cands),
        )
        run = fwd_runner(cfg)
        if tuning.decided("quant_fwd_rot", x.device, table_key):
            _RESOLVED[key] = run
    return run(x, q, scale, bias, group)


def dgrad(g, q, scale, group):
    m = g.shape[0]
    n, k = q.shape
    key = ("dgrad", g.device.index, pow2(m, 32), n, k, group, g.dtype, q.dtype)
    run = _RESOLVED.get(key)
    if run is None:
        table_key = _key(m, n, k, group, g, q, None)
        cands = dgrad_candidates(k, group)
        cfg = tuning.select(
            "quant_dgrad_rot",
            g.device,
            table_key,
            lambda: cands,
            lambda c: lambda: dgrad_runner(c)(g, q, scale, group),
            shortlist=len(cands),
            in_backward=True,
        )
        run = dgrad_runner(cfg)
        if tuning.decided("quant_dgrad_rot", g.device, table_key):
            _RESOLVED[key] = run
    return run(g, q, scale, group)


def _supported(query):
    return fmt_supported(query.get("fmt", "int8"), query.arch.sm) or group_supported(
        query
    )


register_rotated_launchers("triton", fwd, dgrad)
registry.register(
    "quant_linear_rotated",
    "triton",
    make_quant_linear_rotated("triton"),
    supports=requires(kinds=("cuda", "xpu"), min_sm=80, extra=_supported),
)
