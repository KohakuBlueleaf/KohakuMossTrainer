"""Activation quantizers (e4m3 values) for every fp8 recipe, bit-equal to the eager reference
`kohakumoss_kernels.fp8.formats.quantize(x, recipe, "act")`:

    quantize_tensor(x) -> (q, s[1])            one fp32 scale
    quantize_rows(x)   -> (q, s[M])            fp32 scale per row
    quantize_blocks(x, mx) -> (q, s[M, K/sub]) fp32 per 128 (block) or e8m0 per
                                               32 (mx) along K
    quantize_mx_swizzled(x) -> (q, s flat)     e8m0 per 32, vendor 128x4 tile layout

Tiles are fixed, not tuned per shape (docs/internals/quant-kernels.md).
"""

import torch
import triton
import triton.language as tl

from kohakumoss_kernels.backends.triton._common import Launcher
from kohakumoss_kernels.fp8.formats import (
    AMAX_FLOOR,
    BLOCK,
    E4M3,
    E4M3_MAX,
    MX_AMAX_FLOOR,
    MX_BLOCK,
    fp32_scale,
)

_MAX = tl.constexpr(E4M3_MAX)
_FLOOR = tl.constexpr(AMAX_FLOOR)
_MX_FLOOR = tl.constexpr(MX_AMAX_FLOOR)
ROWS_TILE = {"BLOCK_M": 4, "BLOCK_K": 1024}
BLOCKS_TILE = {"BLOCK_M": 8, "BLOCK_K": 512}
CAST_TILE = 4096


@triton.jit
def _e4m3(v):
    """fp32 values in [-448, 448] -> e4m3, rounded once to nearest even on the e4m3 grid
    in fp32 (the conversion that follows is exact)."""
    bits = v.to(tl.int32, bitcast=True)
    ulp = tl.maximum((bits >> 23) & 0xFF, 121) - 3
    magic = (((ulp + 23) << 23) | 0x400000).to(tl.float32, bitcast=True)
    r = (tl.abs(v) + magic) - magic
    return tl.where(bits < 0, -r, r).to(tl.float8e4nv)


@triton.jit
def _fp32_cast(x, s):
    q = tl.div_rn(x, s)
    return _e4m3(tl.minimum(tl.maximum(q, -_MAX), _MAX))


@triton.jit
def _mx_scale(amax):
    """(biased exponent int32, exact fp32 2^-e) of the block scale 2^e."""
    bits = tl.maximum(amax, _MX_FLOOR).to(tl.int32, bitcast=True)
    up = ((bits & 0x7FFFFF) > 0x600000).to(tl.int32)
    biased = tl.maximum(((bits >> 23) & 0xFF) - 8 + up, 0)
    inv = ((254 - biased) << 23).to(tl.float32, bitcast=True)
    return biased, inv


@triton.jit
def _rows_kernel(
    x_ptr, q_ptr, s_ptr, M, K, sxm, sxk, sqm, sqk,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):  # fmt: skip
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mmask = rm < M
    amax = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        mask = mmask[:, None] & (rk < K)[None, :]
        x = tl.load(x_ptr + rm[:, None] * sxm + rk[None, :] * sxk, mask=mask, other=0.0)
        amax = tl.maximum(amax, tl.max(tl.abs(x.to(tl.float32)), axis=1))
    s = tl.div_rn(tl.maximum(amax, _FLOOR), _MAX)
    tl.store(s_ptr + rm, s, mask=mmask)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        mask = mmask[:, None] & (rk < K)[None, :]
        x = tl.load(x_ptr + rm[:, None] * sxm + rk[None, :] * sxk, mask=mask, other=0.0)
        q = _fp32_cast(x.to(tl.float32), s[:, None])
        tl.store(q_ptr + rm[:, None] * sqm + rk[None, :] * sqk, q, mask=mask)


@triton.jit
def _scalar_cast_kernel(x_ptr, s_ptr, q_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(q_ptr + offs, _fp32_cast(x, tl.load(s_ptr)), mask=mask)


@triton.jit
def _blocks_kernel(
    x_ptr, q_ptr, s_ptr, M, K, sxm, sxk, sqm, sqk, ssm, ssk,
    SUB: tl.constexpr, MX: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):  # fmt: skip
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = (rm < M)[:, None] & (rk < K)[None, :]
    x = tl.load(x_ptr + rm[:, None] * sxm + rk[None, :] * sxk, mask=mask, other=0.0)
    groups: tl.constexpr = BLOCK_K // SUB
    g = tl.reshape(x.to(tl.float32), (BLOCK_M, groups, SUB))
    amax = tl.max(tl.abs(g), axis=2)
    if MX:
        biased, inv = _mx_scale(amax)
        qg = g * inv[:, :, None]
        qg = _e4m3(tl.minimum(tl.maximum(qg, -_MAX), _MAX))
        s = biased.to(tl.uint8)
    else:
        s = tl.div_rn(tl.maximum(amax, _FLOOR), _MAX)
        qg = _fp32_cast(g, s[:, :, None])
    q = tl.reshape(qg, (BLOCK_M, BLOCK_K))
    tl.store(q_ptr + rm[:, None] * sqm + rk[None, :] * sqk, q, mask=mask)
    rg = tl.program_id(1) * groups + tl.arange(0, groups)
    smask = (rm < M)[:, None] & (rg < K // SUB)[None, :]
    tl.store(s_ptr + rm[:, None] * ssm + rg[None, :] * ssk, s, mask=smask)


@triton.jit
def _mx_swizzled_kernel(
    x_ptr, q_ptr, s_ptr, M, K, col_tiles, sxm, sxk, sqm, sqk,
):  # fmt: skip
    rows = tl.arange(0, 128)
    cols = tl.arange(0, 128)
    rm = tl.program_id(0) * 128 + rows
    rk = tl.program_id(1) * 128 + cols
    mask = (rm < M)[:, None] & (rk < K)[None, :]
    x = tl.load(x_ptr + rm[:, None] * sxm + rk[None, :] * sxk, mask=mask, other=0.0)
    g = tl.reshape(x.to(tl.float32), (128, 4, 32))
    biased, inv = _mx_scale(tl.max(tl.abs(g), axis=2))
    qg = _e4m3(tl.minimum(tl.maximum(g * inv[:, :, None], -_MAX), _MAX))
    tl.store(
        q_ptr + rm[:, None] * sqm + rk[None, :] * sqk,
        tl.reshape(qg, (128, 128)),
        mask=mask,
    )
    base = (tl.program_id(0) * col_tiles + tl.program_id(1)) * 512
    sub = tl.arange(0, 4)
    offs = base + (rows % 32)[:, None] * 16 + (rows // 32)[:, None] * 4 + sub[None, :]
    tl.store(s_ptr + offs, biased.to(tl.uint8), mask=(rm < M)[:, None])


_ROWS = Launcher(_rows_kernel)
_CAST = Launcher(_scalar_cast_kernel)
_BLOCKS = Launcher(_blocks_kernel)
_SWIZZLED = Launcher(_mx_swizzled_kernel)


def quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x [M, K] (any strides) -> (q e4m3 [M, K], s fp32 [M])."""
    m, k = x.shape
    q = torch.empty(m, k, device=x.device, dtype=E4M3)
    s = torch.empty(m, device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(m, ROWS_TILE["BLOCK_M"]),)
    _ROWS[grid](x, q, s, m, k, *x.stride(), *q.stride(), num_warps=4, **ROWS_TILE)
    return q, s


def quantize_tensor(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x [M, K] -> (q e4m3 [M, K], s fp32 [1])."""
    s = fp32_scale(x.abs().amax().reshape(1))
    xc = x.contiguous()
    q = torch.empty(x.shape, device=x.device, dtype=E4M3)
    n = xc.numel()
    _CAST[(triton.cdiv(n, CAST_TILE),)](xc, s, q, n, BLOCK=CAST_TILE, num_warps=8)
    return q, s


def quantize_blocks(x: torch.Tensor, mx: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """x [M, K] -> (q e4m3 [M, K], s [M, K/sub]): fp32 per 128 columns, or (`mx`)
    biased e8m0 uint8 per 32."""
    m, k = x.shape
    sub = MX_BLOCK if mx else BLOCK
    q = torch.empty(m, k, device=x.device, dtype=E4M3)
    s = torch.empty(
        m, k // sub, device=x.device, dtype=torch.uint8 if mx else torch.float32
    )
    bm = BLOCKS_TILE["BLOCK_M"]
    bk = min(BLOCKS_TILE["BLOCK_K"], triton.next_power_of_2(k))
    grid = (triton.cdiv(m, bm), triton.cdiv(k, bk))
    _BLOCKS[grid](
        x, q, s, m, k, *x.stride(), *q.stride(), *s.stride(),
        SUB=sub, MX=mx, BLOCK_M=bm, BLOCK_K=bk, num_warps=4,
    )  # fmt: skip
    return q, s


def quantize_mx_swizzled(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`K % 128 == 0`; the scale buffer is zero-padded to whole 128-row tiles."""
    m, k = x.shape
    col_tiles = k // (4 * MX_BLOCK)
    q = torch.empty(m, k, device=x.device, dtype=E4M3)
    s = torch.zeros(triton.cdiv(m, 128) * 128 * col_tiles * 4, device=x.device, dtype=torch.uint8)  # fmt: skip
    grid = (triton.cdiv(m, 128), col_tiles)
    _SWIZZLED[grid](x, q, s, m, k, col_tiles, *x.stride(), *q.stride(), num_warps=8)
    return q, s


def quantize_act(x: torch.Tensor, recipe: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The natural-layout activation pair of `recipe` (the eager reference's layout)."""
    match recipe:
        case "tensor":
            return quantize_tensor(x)
        case "row":
            return quantize_rows(x)
        case "block":
            return quantize_blocks(x, mx=False)
        case "mx":
            return quantize_blocks(x, mx=True)
    raise ValueError(f"unknown fp8 recipe {recipe!r}")
