"""Triton int8 compute linear: y = Q(x R) q^T * s_x s_w + b, and its input gradient.

Three kernels around the int8 GEMM:
- `_quant_rows`: per row, the ConvRot rotation (fp32 registers, `G` > 0), an optional
  column scale, the row amax, then the int8 codes (div_rn + rint, the eager rounding);
- `_epilogue`: the int32 accumulator times the row scales and the optional column
  scales, plus the bias, the optional rotation of the output rows, one rounding;
- `_fused_fwd` (sm80+): the forward GEMM with the epilogue in registers.
The GEMM itself is `torch._int_mm` (cuBLAS) or, in the forward on sm80+, the fused tile;
the tuner picks per shape. See docs/internals/quant-kernels.md#int8-compute.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

from kohakumoss_kernels import registry, tuning
from kohakumoss_kernels.backends.triton.hadamard import hadamard_tile, rotate_2d
from kohakumoss_kernels.constraints import requires
from kohakumoss_kernels.hadamard import stages
from kohakumoss_kernels.hardware.current import on_device
from kohakumoss_kernels.int8.formats import col_major, int_mm
from kohakumoss_kernels.int8.op import (
    dgrad_reference,
    fwd_reference,
    int8_supported,
    make_int8_linear,
    register_launchers,
)
from kohakumoss_kernels.quant import dequantize
from kohakumoss_kernels.tuning import tuner
from kohakumoss_kernels.tuning.buckets import pow2

CUBLAS = {"impl": "cublas"}
_CHUNKS = (256, 128, 64, 32, 16)
_FUSED = (
    (128, 128, 64, 4, 4),
    (128, 256, 64, 8, 3),
    (256, 128, 64, 8, 3),
    (128, 128, 128, 8, 3),
    (64, 128, 128, 4, 4),
)


@triton.jit
def _quant_rows(
    x_ptr, cs_ptr, q_ptr, s_ptr, R, K, hscale,
    HAS_CS: tl.constexpr, G: tl.constexpr, BR: tl.constexpr, C: tl.constexpr,
):  # fmt: skip
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    rmask = (rows < R)[:, None]
    base = rows.to(tl.int64)[:, None] * K
    amax = tl.zeros((BR,), dtype=tl.float32)
    for k0 in range(0, K, C):
        cols = k0 + tl.arange(0, C)
        x = tl.load(x_ptr + base + cols[None, :], mask=rmask, other=0.0).to(tl.float32)
        if G > 0:
            x = hadamard_tile(x, BR, C, G) * hscale
        if HAS_CS:
            x = x * tl.load(cs_ptr + cols)[None, :]
        amax = tl.maximum(amax, tl.max(tl.abs(x), axis=1))
    s = tl.div_rn(tl.maximum(amax, 1e-12), 127.0)
    tl.store(s_ptr + rows, s, mask=rows < R)
    for k0 in range(0, K, C):
        cols = k0 + tl.arange(0, C)
        x = tl.load(x_ptr + base + cols[None, :], mask=rmask, other=0.0).to(tl.float32)
        if G > 0:
            x = hadamard_tile(x, BR, C, G) * hscale
        if HAS_CS:
            x = x * tl.load(cs_ptr + cols)[None, :]
        code = libdevice.rint(tl.div_rn(x, s[:, None]))
        code = tl.minimum(tl.maximum(code, -127.0), 127.0)
        tl.store(q_ptr + base + cols[None, :], code.to(tl.int8), mask=rmask)


@triton.jit
def _epilogue(
    a_ptr, rs_ptr, cs_ptr, b_ptr, y_ptr, R, N, hscale,
    HAS_CS: tl.constexpr, HAS_BIAS: tl.constexpr, G: tl.constexpr,
    BR: tl.constexpr, C: tl.constexpr,
):  # fmt: skip
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cols = tl.program_id(1) * C + tl.arange(0, C)
    mask = (rows < R)[:, None]
    offs = rows.to(tl.int64)[:, None] * N + cols[None, :]
    y = tl.load(a_ptr + offs, mask=mask, other=0).to(tl.float32)
    y = y * tl.load(rs_ptr + rows, mask=rows < R, other=0.0)[:, None]
    if HAS_CS:
        y = y * tl.load(cs_ptr + cols)[None, :]
    if HAS_BIAS:
        y = y + tl.load(b_ptr + cols).to(tl.float32)[None, :]
    if G > 0:
        y = hadamard_tile(y, BR, C, G) * hscale
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _fused_fwd(
    a_ptr, q_ptr, rs_ptr, cs_ptr, b_ptr, y_ptr, M, N, K,
    HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr,
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
    a_ptrs = a_ptr + rm.to(tl.int64)[:, None] * K + rk[None, :]
    q_ptrs = q_ptr + rn.to(tl.int64)[None, :] * K + rk[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for _ in range(0, K // BK):
        a = tl.load(a_ptrs, mask=rm[:, None] < M, other=0)
        w = tl.load(q_ptrs, mask=rn[None, :] < N, other=0)
        acc = tl.dot(a, w, acc)
        a_ptrs += BK
        q_ptrs += BK
    nmask = rn < N
    y = acc.to(tl.float32) * tl.load(rs_ptr + rm, mask=rm < M, other=0.0)[:, None]
    y = y * tl.load(cs_ptr + rn, mask=nmask, other=0.0)[None, :]
    if HAS_BIAS:
        y = y + tl.load(b_ptr + rn, mask=nmask, other=0.0).to(tl.float32)[None, :]
    y_ptrs = y_ptr + rm.to(tl.int64)[:, None] * N + rn[None, :]
    tl.store(
        y_ptrs, y.to(y_ptr.dtype.element_ty), mask=(rm[:, None] < M) & nmask[None, :]
    )


def chunk(k: int, group: int) -> int:
    """The widest column chunk (at most 256) dividing `k` and divisible by `group`, 0
    when none does."""
    for c in _CHUNKS:
        if k % c == 0 and (not group or c % group == 0):
            return c
    return 0


def _hscale(group: int) -> float:
    return 2.0 ** -stages(group) if group else 1.0


def quant_rows(
    x: Tensor, col_scale: Tensor | None, group: int
) -> tuple[Tensor, Tensor]:
    """(q int8 [R, K], s fp32 [R]) of contiguous x [R, K]: rotated when `group`, its
    columns times `col_scale` when given, then quantized per row."""
    r, k = x.shape
    c = chunk(k, group)
    if not c:
        raise ValueError(f"int8 rows of width {k} do not split into chunks of {group}")
    q = torch.empty(r, k, dtype=torch.int8, device=x.device)
    s = torch.empty(r, dtype=torch.float32, device=x.device)
    if r:
        br = 4 if c >= 256 else 8
        _quant_rows[(triton.cdiv(r, br),)](
            x, x if col_scale is None else col_scale, q, s, r, k, _hscale(group),
            HAS_CS=col_scale is not None, G=group, BR=br, C=c, num_warps=4,
        )  # fmt: skip
    return q, s


def epilogue(acc, row_scale, col_scale, bias, group, dtype) -> Tensor:
    """acc int32 [R, N] * row_scale[:, None] (* col_scale) (+ bias), the output rows
    rotated when `group`, in `dtype`."""
    r, n = acc.shape
    c = chunk(n, group)
    if not c:
        raise ValueError(f"int8 output of width {n} does not split into {group}")
    y = torch.empty(r, n, dtype=dtype, device=acc.device)
    if r:
        br = 8 if c >= 256 else 16
        _epilogue[(triton.cdiv(r, br), n // c)](
            acc, row_scale, row_scale if col_scale is None else col_scale,
            row_scale if bias is None else bias, y, r, n, _hscale(group),
            HAS_CS=col_scale is not None, HAS_BIAS=bias is not None, G=group, BR=br,
            C=c, num_warps=4,
        )  # fmt: skip
    return y


def fused_candidates(device: torch.device, k: int) -> list[dict]:
    """`cublas` first, then the fused forward tiles where the card has int8 MMAs that
    Triton lowers (sm80+) and the K tile divides `k`."""
    out = [CUBLAS]
    if torch.cuda.get_device_capability(device)[0] >= 8:
        out += [
            {"impl": "fused", "bm": bm, "bn": bn, "bk": bk, "warps": w, "stages": s}
            for bm, bn, bk, w, s in _FUSED
            if k % bk == 0
        ]
    return out


def gemm_runner(cfg: dict):
    """(xq, xs, q, scale, bias, dtype) -> y for one forward GEMM config."""
    if cfg["impl"] == "cublas":

        def cublas(xq, xs, q, scale, bias, dtype):
            return epilogue(int_mm(xq, q.t()), xs, scale, bias, 0, dtype)

        return cublas

    def fused(xq, xs, q, scale, bias, dtype):
        m, k = xq.shape
        n = q.shape[0]
        y = torch.empty(m, n, dtype=dtype, device=xq.device)
        grid = (triton.cdiv(m, cfg["bm"]) * triton.cdiv(n, cfg["bn"]),)
        _fused_fwd[grid](
            xq, q, xs, scale, scale if bias is None else bias, y, m, n, k,
            HAS_BIAS=bias is not None, BM=cfg["bm"], BN=cfg["bn"], BK=cfg["bk"],
            GROUP_M=8, num_warps=cfg["warps"], num_stages=cfg["stages"],
        )  # fmt: skip
        return y

    return fused


_RESOLVED: dict[tuple, object] = {}
tuner.on_reset(_RESOLVED.clear)


def _dense(t: Tensor | None) -> Tensor | None:
    """`t` with real strides (a weightless stride-0 view is copied out)."""
    return None if t is None else t.contiguous()


def fwd(x, q, scale, bias, group):
    m, k = x.shape
    n = q.shape[0]
    if not (chunk(k, group) and chunk(n, 0)):
        return fwd_reference(x, q, scale, bias, group)
    q, scale, bias = _dense(q), _dense(scale), _dense(bias)
    with on_device(x.device):
        xq, xs = quant_rows(x, None, group)
        key = ("fwd", x.device.index, pow2(m, 32), n, k, x.dtype, bias is None)
        run = _RESOLVED.get(key)
        if run is None:
            table_key = f"m{pow2(m, 32)}_n{n}_k{k}_{x.dtype}_b{bias is not None}"
            cands = fused_candidates(x.device, k)
            cfg = tuning.select(
                "int8_fwd",
                x.device,
                table_key,
                lambda: cands,
                lambda c: lambda: gemm_runner(c)(xq, xs, q, scale, bias, x.dtype),
                shortlist=len(cands),
            )
            run = gemm_runner(cfg)
            if tuning.decided("int8_fwd", x.device, table_key):
                _RESOLVED[key] = run
        return run(xq, xs, q, scale, bias, x.dtype)


def dgrad(dy, q, scale, group, mode):
    n, k = q.shape
    if not (chunk(n, 0) and chunk(k, group)):
        return dgrad_reference(dy, q, scale, group, mode)
    scale = _dense(scale)
    with on_device(dy.device):
        if mode == "16bit":
            dx = (dy @ dequantize(q, scale, dy.dtype)).contiguous()
            return rotate_2d(dx, group) if group else dx
        gq, gs = quant_rows(dy, scale, 0)
        acc = int_mm(gq, col_major(q))
        return epilogue(acc, gs, None, None, group, dy.dtype)


register_launchers("triton", fwd, dgrad)
registry.register(
    "int8_linear",
    "triton",
    make_int8_linear("triton"),
    supports=requires(kinds=("cuda",), min_sm=75, extra=int8_supported),
)
