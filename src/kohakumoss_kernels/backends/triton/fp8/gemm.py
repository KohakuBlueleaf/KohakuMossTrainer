"""fp8 GEMMs C = A B^T over e4m3 operands A [M, K], B [N, K] (K-contiguous rows),
fp32 accumulation, scales applied in fp32, one rounding at the store (+ bias).

    scaled_mm      tensor / row recipes: `acc="promote"` adds each BK-deep fp8 MMA
                   result, times the activation scale, to the fp32 accumulator; "mma"
                   chains the MMA accumulator across K; "upcast" runs the fp16 MMA on
                   the exactly converted operands; the weight scale in the epilogue
    promote_mm     per-K-block scales (block 128, mx 32): each K block's MMA result is
                   scaled and added to the fp32 accumulator; "upcast" runs the fp16
                   MMA on the exactly converted operands
    dot_scaled_mm  mx on the native block-scaled MMA (`tl.dot_scaled`, e8m0 scales)

Tile configs come from the caller (the tuner). See docs/internals/quant-kernels.md.
"""

import torch
import triton
import triton.language as tl

from kohakumoss_kernels.backends.triton._common import Launcher


@triton.jit
def _tile(M, N, BM: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr):
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
    rmp = tl.max_contiguous(tl.multiple_of(rm % M, BM), BM)
    rnp = tl.max_contiguous(tl.multiple_of(rn % N, BN), BN)
    return rm, rn, rmp, rnp


@triton.jit
def _store(acc, bias_ptr, c_ptr, rm, rn, M, N, scm, scn, HAS_BIAS: tl.constexpr):
    nmask = rn < N
    if HAS_BIAS:
        acc += tl.load(bias_ptr + rn, mask=nmask, other=0.0).to(tl.float32)[None, :]
    tl.store(
        c_ptr + rm[:, None] * scm + rn[None, :] * scn,
        acc.to(c_ptr.dtype.element_ty),
        mask=(rm < M)[:, None] & nmask[None, :],
    )


@triton.jit
def _mma(a, b, UPCAST: tl.constexpr):
    """a @ b of e4m3 tiles into a fresh fp32 tile: on the fp8 MMA, or (UPCAST) on the
    fp16 MMA of the exactly converted operands."""
    if UPCAST:
        return tl.dot(a.to(tl.float16), b.to(tl.float16))
    return tl.dot(a, b)


@triton.jit
def _pow2(biased):
    return (biased.to(tl.int32) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _scaled_kernel(
    a_ptr, b_ptr, sa_ptr, sb_ptr, bias_ptr, c_ptr, M, N, K,
    sam, sak, sbn, sbk, scm, scn,
    SA_ROW: tl.constexpr, SB_ROW: tl.constexpr, HAS_BIAS: tl.constexpr,
    EVEN_K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr, PROMOTE: tl.constexpr, UPCAST: tl.constexpr,
):  # fmt: skip
    rm, rn, rmp, rnp = _tile(M, N, BM, BN, GROUP_M)
    rk = tl.arange(0, BK)
    a_ptrs = a_ptr + rmp[:, None] * sam + rk[None, :] * sak
    b_ptrs = b_ptr + rnp[:, None] * sbn + rk[None, :] * sbk
    if SA_ROW:
        sa = tl.load(sa_ptr + rmp)
    else:
        sa = tl.zeros((BM,), dtype=tl.float32) + tl.load(sa_ptr)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            kmask = (rk < K - k * BK)[None, :]
            a = tl.load(a_ptrs, mask=kmask, other=0.0)
            b = tl.load(b_ptrs, mask=kmask, other=0.0)
        if UPCAST:
            acc = tl.dot(a.to(tl.float16), tl.trans(b).to(tl.float16), acc)
        elif PROMOTE:
            acc += tl.dot(a, tl.trans(b)) * sa[:, None]
        else:
            acc = tl.dot(a, tl.trans(b), acc)
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    if not PROMOTE or UPCAST:
        acc = acc * sa[:, None]
    if SB_ROW:
        acc = acc * tl.load(sb_ptr + rnp)[None, :]
    else:
        acc = acc * tl.load(sb_ptr)
    _store(acc, bias_ptr, c_ptr, rm, rn, M, N, scm, scn, HAS_BIAS)


@triton.jit
def _promote_kernel(
    a_ptr, b_ptr, sa_ptr, sb_ptr, bias_ptr, c_ptr, M, N, K,
    sam, sak, sbn, sbk, ssam, ssak, ssbn, ssbk, scm, scn,
    SUB: tl.constexpr, A_MX: tl.constexpr, B_MX: tl.constexpr, B_ROWS: tl.constexpr,
    HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr,
    UPCAST: tl.constexpr,
):  # fmt: skip
    rm, rn, rmp, rnp = _tile(M, N, BM, BN, GROUP_M)
    rk = tl.arange(0, SUB)
    a_ptrs = a_ptr + rmp[:, None] * sam + rk[None, :] * sak
    b_ptrs = b_ptr + rnp[:, None] * sbn + rk[None, :] * sbk
    sa_ptrs = sa_ptr + rmp * ssam
    sb_ptrs = sb_ptr + (rnp // B_ROWS) * ssbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K // SUB):
        part = _mma(tl.load(a_ptrs), tl.trans(tl.load(b_ptrs)), UPCAST)
        sa = tl.load(sa_ptrs + k * ssak)
        sb = tl.load(sb_ptrs + k * ssbk)
        if A_MX:
            sa = _pow2(sa)
        if B_MX:
            sb = _pow2(sb)
        acc += part * sa[:, None] * sb[None, :]
        a_ptrs += SUB * sak
        b_ptrs += SUB * sbk
    _store(acc, bias_ptr, c_ptr, rm, rn, M, N, scm, scn, HAS_BIAS)


@triton.jit
def _dot_scaled_kernel(
    a_ptr, as_ptr, b_ptr, bs_ptr, bias_ptr, c_ptr, M, N, K,
    sam, sak, sasm, sask, sbn, sbk, sbsn, sbsk, scm, scn,
    HAS_BIAS: tl.constexpr, EVEN_K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
    BK: tl.constexpr, GROUP_M: tl.constexpr,
):  # fmt: skip
    rm, rn, rmp, rnp = _tile(M, N, BM, BN, GROUP_M)
    rk = tl.arange(0, BK)
    groups: tl.constexpr = BK // 32
    rg = tl.arange(0, groups)
    a_ptrs = a_ptr + rmp[:, None] * sam + rk[None, :] * sak
    b_ptrs = b_ptr + rnp[:, None] * sbn + rk[None, :] * sbk
    as_ptrs = as_ptr + rmp[:, None] * sasm + rg[None, :] * sask
    bs_ptrs = bs_ptr + rnp[:, None] * sbsn + rg[None, :] * sbsk
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            a_s = tl.load(as_ptrs)
            b_s = tl.load(bs_ptrs)
        else:
            kmask = (rk < K - k * BK)[None, :]
            gmask = (rg < (K - k * BK) // 32)[None, :]
            a = tl.load(a_ptrs, mask=kmask, other=0.0)
            b = tl.load(b_ptrs, mask=kmask, other=0.0)
            a_s = tl.load(as_ptrs, mask=gmask, other=0)
            b_s = tl.load(bs_ptrs, mask=gmask, other=0)
        acc = tl.dot_scaled(a, a_s, "e4m3", tl.trans(b), b_s, "e4m3", acc=acc)
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
        as_ptrs += groups * sask
        bs_ptrs += groups * sbsk
    _store(acc, bias_ptr, c_ptr, rm, rn, M, N, scm, scn, HAS_BIAS)


_SCALED = Launcher(_scaled_kernel)
_PROMOTE = Launcher(_promote_kernel)
_DOT_SCALED = Launcher(_dot_scaled_kernel)


def _grid(m: int, n: int, cfg: dict) -> tuple[int]:
    return (triton.cdiv(m, cfg["bm"]) * triton.cdiv(n, cfg["bn"]),)


def _launch_opts(cfg: dict) -> dict:
    return {"GROUP_M": 8, "num_warps": cfg["warps"], "num_stages": cfg["stages"]}


def scaled_mm(a, b, sa, sb, bias, out_dtype, cfg) -> torch.Tensor:
    """a [M, K] e4m3, b [N, K] e4m3; sa [M] or [1], sb [N] or [1] fp32."""
    m, k = a.shape
    n = b.shape[0]
    c = torch.empty(m, n, device=a.device, dtype=out_dtype)
    _SCALED[_grid(m, n, cfg)](
        a, b, sa, sb, bias if bias is not None else sa, c, m, n, k,
        *a.stride(), *b.stride(), *c.stride(),
        SA_ROW=sa.numel() != 1, SB_ROW=sb.numel() != 1,
        HAS_BIAS=bias is not None, EVEN_K=k % cfg["bk"] == 0,
        BM=cfg["bm"], BN=cfg["bn"], BK=cfg["bk"], PROMOTE=cfg.get("acc") == "promote",
        UPCAST=cfg.get("acc") == "upcast", **_launch_opts(cfg),
    )  # fmt: skip
    return c


def promote_mm(a, b, sa, sb, bias, out_dtype, cfg, sub: int, b_rows: int):
    """Per-K-block scales: sa [M, K/sub], sb [N / b_rows, K/sub]; uint8 scales are
    biased e8m0, fp32 scales multiply as they are."""
    m, k = a.shape
    n = b.shape[0]
    c = torch.empty(m, n, device=a.device, dtype=out_dtype)
    _PROMOTE[_grid(m, n, cfg)](
        a, b, sa, sb, bias if bias is not None else a, c, m, n, k,
        *a.stride(), *b.stride(), *sa.stride(), *sb.stride(), *c.stride(),
        SUB=sub, A_MX=sa.dtype == torch.uint8, B_MX=sb.dtype == torch.uint8,
        B_ROWS=b_rows, HAS_BIAS=bias is not None,
        BM=cfg["bm"], BN=cfg["bn"], UPCAST=cfg.get("acc") == "upcast",
        **_launch_opts(cfg),
    )  # fmt: skip
    return c


def dot_scaled_mm(a, a_s, b, b_s, bias, out_dtype, cfg) -> torch.Tensor:
    """mx: a_s [M, K/32], b_s [N, K/32] uint8 biased e8m0."""
    m, k = a.shape
    n = b.shape[0]
    c = torch.empty(m, n, device=a.device, dtype=out_dtype)
    _DOT_SCALED[_grid(m, n, cfg)](
        a, a_s, b, b_s, bias if bias is not None else a, c, m, n, k,
        *a.stride(), *a_s.stride(), *b.stride(), *b_s.stride(), *c.stride(),
        HAS_BIAS=bias is not None, EVEN_K=k % cfg["bk"] == 0,
        BM=cfg["bm"], BN=cfg["bn"], BK=cfg["bk"], **_launch_opts(cfg),
    )  # fmt: skip
    return c
