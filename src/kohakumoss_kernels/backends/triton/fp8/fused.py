"""One-launch fp8 GEMMs that quantize the 16-bit activation inside the tile loop:
C = Q(a) B^T with a [M, K] fp16 / bf16 and B [N, K] e4m3 pre-quantized.

    fused_row_mm    row recipe: a K prologue takes each row's amax, the main loop
                    quantizes A tiles with that scale (div_rn, as the eager reference)
    fused_block_mm  block recipe: each 128-deep K tile is quantized and its MMA result
                    scaled into the fp32 accumulator (128x128 weight blocks)
    fused_mx_mm     mx recipe: each 32-column group quantized to e4m3 + e8m0, the
                    native block-scaled MMA (`tl.dot_scaled`)

Every program quantizes its own A rows. See docs/internals/quant-kernels.md.
"""

import torch
import triton
import triton.language as tl

from kohakumoss_kernels.backends.triton._common import Launcher
from kohakumoss_kernels.backends.triton.fp8.gemm import (
    _grid,
    _launch_opts,
    _mma,
    _store,
    _tile,
)
from kohakumoss_kernels.backends.triton.fp8.quantize import (
    _FLOOR,
    _MAX,
    _e4m3,
    _fp32_cast,
    _mx_scale,
)


@triton.jit
def _fused_row_kernel(
    a_ptr, b_ptr, sb_ptr, bias_ptr, c_ptr, M, N, K,
    sam, sak, sbn, sbk, scm, scn,
    HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr, PROMOTE: tl.constexpr, UPCAST: tl.constexpr,
):  # fmt: skip
    rm, rn, rmp, rnp = _tile(M, N, BM, BN, GROUP_M)
    rk = tl.arange(0, BK)
    amax = tl.zeros((BM,), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kmask = (k0 + rk < K)[None, :]
        a = tl.load(a_ptr + rmp[:, None] * sam + (k0 + rk)[None, :] * sak, mask=kmask, other=0.0)  # fmt: skip
        amax = tl.maximum(amax, tl.max(tl.abs(a.to(tl.float32)), axis=1))
    sa = tl.div_rn(tl.maximum(amax, _FLOOR), _MAX)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kmask = (k0 + rk < K)[None, :]
        a = tl.load(a_ptr + rmp[:, None] * sam + (k0 + rk)[None, :] * sak, mask=kmask, other=0.0)  # fmt: skip
        b = tl.load(b_ptr + rnp[:, None] * sbn + (k0 + rk)[None, :] * sbk, mask=kmask, other=0.0)  # fmt: skip
        aq = _fp32_cast(a.to(tl.float32), sa[:, None])
        if PROMOTE or UPCAST:
            acc += _mma(aq, tl.trans(b), UPCAST) * sa[:, None]
        else:
            acc = tl.dot(aq, tl.trans(b), acc)
    if not (PROMOTE or UPCAST):
        acc = acc * sa[:, None]
    acc = acc * tl.load(sb_ptr + rnp)[None, :]
    _store(acc, bias_ptr, c_ptr, rm, rn, M, N, scm, scn, HAS_BIAS)


@triton.jit
def _fused_block_kernel(
    a_ptr, b_ptr, sb_ptr, bias_ptr, c_ptr, M, N, K,
    sam, sak, sbn, sbk, ssbn, ssbk, scm, scn,
    HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr,
    UPCAST: tl.constexpr,
):  # fmt: skip
    rm, rn, rmp, rnp = _tile(M, N, BM, BN, GROUP_M)
    rk = tl.arange(0, 128)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K // 128):
        a = tl.load(a_ptr + rmp[:, None] * sam + (k * 128 + rk)[None, :] * sak).to(tl.float32)  # fmt: skip
        b = tl.load(b_ptr + rnp[:, None] * sbn + (k * 128 + rk)[None, :] * sbk)
        sa = tl.div_rn(tl.maximum(tl.max(tl.abs(a), axis=1), _FLOOR), _MAX)
        part = _mma(_fp32_cast(a, sa[:, None]), tl.trans(b), UPCAST)
        sb = tl.load(sb_ptr + (rnp // 128) * ssbn + k * ssbk)
        acc += part * sa[:, None] * sb[None, :]
    _store(acc, bias_ptr, c_ptr, rm, rn, M, N, scm, scn, HAS_BIAS)


@triton.jit
def _fused_mx_kernel(
    a_ptr, b_ptr, bs_ptr, bias_ptr, c_ptr, M, N, K,
    sam, sak, sbn, sbk, sbsn, sbsk, scm, scn,
    HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):  # fmt: skip
    rm, rn, rmp, rnp = _tile(M, N, BM, BN, GROUP_M)
    rk = tl.arange(0, BK)
    groups: tl.constexpr = BK // 32
    rg = tl.arange(0, groups)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K // BK):
        a = tl.load(a_ptr + rmp[:, None] * sam + (k * BK + rk)[None, :] * sak)
        g = tl.reshape(a.to(tl.float32), (BM, groups, 32))
        biased, inv = _mx_scale(tl.max(tl.abs(g), axis=2))
        qg = _e4m3(tl.minimum(tl.maximum(g * inv[:, :, None], -_MAX), _MAX))
        aq = tl.reshape(qg, (BM, BK))
        b = tl.load(b_ptr + rnp[:, None] * sbn + (k * BK + rk)[None, :] * sbk)
        b_s = tl.load(bs_ptr + rnp[:, None] * sbsn + (k * groups + rg)[None, :] * sbsk)
        acc = tl.dot_scaled(aq, biased.to(tl.uint8), "e4m3", tl.trans(b), b_s, "e4m3", acc=acc)  # fmt: skip
    _store(acc, bias_ptr, c_ptr, rm, rn, M, N, scm, scn, HAS_BIAS)


_ROW = Launcher(_fused_row_kernel)
_BLOCK = Launcher(_fused_block_kernel)
_MX = Launcher(_fused_mx_kernel)


def fused_row_mm(a, bq, sb, bias, cfg) -> torch.Tensor:
    """a [M, K] 16-bit, bq [N, K] e4m3, sb [N] fp32."""
    m, k = a.shape
    n = bq.shape[0]
    c = torch.empty(m, n, device=a.device, dtype=a.dtype)
    _ROW[_grid(m, n, cfg)](
        a, bq, sb, bias if bias is not None else sb, c, m, n, k,
        *a.stride(), *bq.stride(), *c.stride(),
        HAS_BIAS=bias is not None, BM=cfg["bm"], BN=cfg["bn"], BK=cfg["bk"],
        PROMOTE=cfg.get("acc") == "promote", UPCAST=cfg.get("acc") == "upcast",
        **_launch_opts(cfg),
    )  # fmt: skip
    return c


def fused_block_mm(a, bq, sb, bias, cfg) -> torch.Tensor:
    """`K % 128 == 0`; sb [ceil(N/128), K/128] fp32."""
    m, k = a.shape
    n = bq.shape[0]
    c = torch.empty(m, n, device=a.device, dtype=a.dtype)
    _BLOCK[_grid(m, n, cfg)](
        a, bq, sb, bias if bias is not None else sb, c, m, n, k,
        *a.stride(), *bq.stride(), *sb.stride(), *c.stride(),
        HAS_BIAS=bias is not None, BM=cfg["bm"], BN=cfg["bn"],
        UPCAST=cfg.get("acc") == "upcast", **_launch_opts(cfg),
    )  # fmt: skip
    return c


def fused_mx_mm(a, bq, bs, bias, cfg) -> torch.Tensor:
    """`K % BK == 0`; bs [N, K/32] uint8 biased e8m0."""
    m, k = a.shape
    n = bq.shape[0]
    c = torch.empty(m, n, device=a.device, dtype=a.dtype)
    _MX[_grid(m, n, cfg)](
        a, bq, bs, bias if bias is not None else bs, c, m, n, k,
        *a.stride(), *bq.stride(), *bs.stride(), *c.stride(),
        HAS_BIAS=bias is not None, BM=cfg["bm"], BN=cfg["bn"], BK=cfg["bk"],
        **_launch_opts(cfg),
    )  # fmt: skip
    return c


FUSED = {"row": fused_row_mm, "block": fused_block_mm, "mx": fused_mx_mm}
