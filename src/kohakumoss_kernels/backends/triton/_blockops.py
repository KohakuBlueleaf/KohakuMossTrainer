"""Shared pieces of the Triton block ops: segment tile plans and their reductions.

Rows of a packed batch belong to samples through a nondecreasing `seg [N]`. Backward
kernels that reduce over a sample's rows run one program per *segment tile*: a run
of at most CHUNK rows inside one sample. `segment_bounds` writes the sample row
bounds `cu [B+1]` on the device (no host sync); a program finds its tile from `cu`
with a vector prefix sum, so partials are per tile and sums are deterministic.
Partials are fp32 pairs (hi, lo planes) summed in double-float. See
docs/internals/fp16.md.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

from kohakumoss_kernels.backends.triton._common import Launcher, cdiv
from kohakumoss_kernels.hardware.device import probe


@triton.jit
def kahan(s, c, x):
    """(s, c) after adding x with compensation; the sum is s - c."""
    y = x - c
    t = s + y
    return t, (t - s) - y


@triton.jit
def two_sum(a, b):
    """(s, e) with s + e == a + b exactly (Knuth)."""
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


@triton.jit
def two_prod(a, b):
    """(p, e) with p + e == a * b exactly (FMA)."""
    p = a * b
    return p, tl.fma(a, b, -p)


@triton.jit
def df_add(ah, al, bh, bl):
    """Double-float a + b (each an unevaluated pair hi + lo of fp32)."""
    s, e = two_sum(ah, bh)
    e = e + (al + bl)
    hi = s + e
    return hi, e - (hi - s)


@triton.jit
def df_mul(ah, al, bh, bl):
    """Double-float a * b."""
    p, e = two_prod(ah, bh)
    e = e + (ah * bl + al * bh)
    hi = p + e
    return hi, e - (hi - p)


@triton.jit
def df_sum(hi, lo, axis: tl.constexpr):
    """Double-float sum of pairs along `axis` (a tree of `df_add`)."""
    return tl.reduce((hi, lo), axis, df_add)


@triton.jit
def df_scale(a, bh, bl):
    """Double-float a * (bh + bl) for an fp32 a."""
    p, e = two_prod(a, bh)
    e = e + a * bl
    hi = p + e
    return hi, e - (hi - p)


@triton.jit
def df_div(h, lo, n):
    """Double-float (h + lo) / n for an fp32 n."""
    q = tl.div_rn(h, n)
    ql = tl.div_rn(tl.fma(-q, n, h) + lo, n)
    s = q + ql
    return s, ql - (s - q)


@triton.jit
def df_rstd(h, lo, eps):
    """Double-float 1 / sqrt(h + lo + eps): fp32 seed and one Newton step in pairs."""
    q, ql = df_add(h, lo, eps + 0.0 * h, 0.0 * h)
    r0 = tl.div_rn(1.0, tl.sqrt_rn(q))
    r2, r2l = two_prod(r0, r0)
    t, tlo = df_mul(q, ql, r2, r2l)
    e, el = df_add(1.0 - t, 0.0 * t, -tlo, 0.0 * t)
    return two_sum(r0, 0.5 * r0 * (e + el))


@triton.jit
def df_mean_sq(x, axis: tl.constexpr, n):
    """Double-float mean of x * x along `axis` (exact squares, pair sum)."""
    sh, sl = two_prod(x, x)
    sh, sl = df_sum(sh, sl, axis)
    return df_div(sh, sl, n)


@triton.jit
def store_pair(ptr, stride, hi, lo, mask):
    """Store a pair as two fp32 planes `stride` elements apart."""
    tl.store(ptr, hi, mask=mask)
    tl.store(ptr + stride, lo, mask=mask)


@triton.jit
def segment_bounds_kernel(seg_ptr, cu_ptr, N, LOG2N: tl.constexpr):
    """cu[b] = first row with seg >= b (binary search); one program per b."""
    b = tl.program_id(0)
    lo = 0
    hi = N
    for _ in range(LOG2N):
        mid = (lo + hi) // 2
        v = tl.load(seg_ptr + tl.minimum(mid, N - 1))
        active = lo < hi
        less = v < b
        new_lo = tl.where(active & less, mid + 1, lo)
        hi = tl.where(active & (v >= b), mid, hi)
        lo = new_lo
    tl.store(cu_ptr + b, lo)


@triton.jit
def tile_range(t, cu_ptr, N, B, CHUNK, HAS_SEG: tl.constexpr, BLOCK_B: tl.constexpr):
    """(sample, first row, end row) of segment tile `t`; empty range past the last."""
    if HAS_SEG:
        bb = tl.arange(0, BLOCK_B)
        ok = bb < B
        lo = tl.load(cu_ptr + bb, mask=ok, other=0)
        hi = tl.load(cu_ptr + bb + 1, mask=ok, other=0)
        nt = tl.where(ok, (hi - lo + CHUNK - 1) // CHUNK, 0)
        end = tl.cumsum(nt, 0)
        beg = end - nt
        sel = (t >= beg) & (t < end)
        b = tl.sum(tl.where(sel, bb, 0))
        start = tl.sum(tl.where(sel, lo + (t - beg) * CHUNK, 0))
        stop = tl.sum(tl.where(sel, hi, 0))
        stop = tl.where(
            tl.sum(sel.to(tl.int32)) > 0, tl.minimum(start + CHUNK, stop), start
        )
    else:
        b = t * 0
        start = t * CHUNK
        stop = tl.minimum(start + CHUNK, N)
    return b, start, stop


@triton.jit
def segment_tiles(b, cu_ptr, B, CHUNK, T, HAS_SEG: tl.constexpr, BLOCK_B: tl.constexpr):
    """[first tile, end tile) of sample `b` (all T tiles without segments)."""
    if HAS_SEG:
        bb = tl.arange(0, BLOCK_B)
        ok = bb < B
        lo = tl.load(cu_ptr + bb, mask=ok, other=0)
        hi = tl.load(cu_ptr + bb + 1, mask=ok, other=0)
        nt = tl.where(ok, (hi - lo + CHUNK - 1) // CHUNK, 0)
        end = tl.cumsum(nt, 0)
        sel = bb == b
        first = tl.sum(tl.where(sel, end - nt, 0))
        last = tl.sum(tl.where(sel, end, 0))
    else:
        first = b * 0
        last = b * 0 + T
    return first, last


@triton.jit
def segment_sum_kernel(
    part_ptr,
    out_ptr,
    cu_ptr,
    D,
    B,
    CHUNK,
    T,
    SLOTS: tl.constexpr,
    HAS_SEG: tl.constexpr,
    ROUND: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    """Pair sums over the tiles of sample b of part [T, SLOTS, 2, D] (fp32 pairs), BT
    tiles per step: out [SLOTS, 2, B, D] pairs, or [SLOTS, B, D] rounded to fp32 with
    ROUND."""
    b = tl.program_id(0)
    cols = tl.program_id(1) * BD + tl.arange(0, BD)
    ok = cols < D
    beg, end = segment_tiles(b, cu_ptr, B, CHUNK, T, HAS_SEG, BLOCK_B)
    for s in tl.static_range(SLOTS):
        hi = tl.zeros([BD], tl.float32)
        lo = tl.zeros([BD], tl.float32)
        for t0 in range(beg, end, BT):
            ts = t0 + tl.arange(0, BT)
            m = (ts < end)[:, None] & ok[None, :]
            src = (
                part_ptr
                + (ts.to(tl.int64)[:, None] * SLOTS + s) * 2 * D
                + cols[None, :]
            )
            ph, pl = df_sum(
                tl.load(src, mask=m, other=0.0), tl.load(src + D, mask=m, other=0.0), 0
            )
            hi, lo = df_add(hi, lo, ph, pl)
        if ROUND:
            tl.store(out_ptr + (s * B + b) * D + cols, hi + lo, mask=ok)
        else:
            store_pair(out_ptr + (s * 2 * B + b) * D + cols, B * D, hi, lo, ok)


_bounds = Launcher(segment_bounds_kernel)
_segsum = Launcher(segment_sum_kernel)


def num_sms(device: torch.device) -> int:
    return max(1, probe(device).sms)


def log2_ceil(n: int) -> int:
    k = 1
    while (1 << k) <= n:
        k += 1
    return k


class SegPlan:
    """Segment tiles of N rows over B samples (or plain CHUNK tiles without seg)."""

    def __init__(
        self, seg: Tensor | None, n: int, b: int, rows: int, device: torch.device
    ):
        self.n = n
        self.has_seg = seg is not None
        self.b = b if self.has_seg else 1
        target = 2 * num_sms(device)
        chunk = max(rows, cdiv(cdiv(n, target), rows) * rows)
        self.chunk = chunk
        self.tiles = cdiv(n, chunk) + (self.b if self.has_seg else 0)
        self.block_b = triton.next_power_of_2(max(self.b, 1))
        if self.has_seg:
            self.cu = torch.empty(self.b + 1, device=device, dtype=torch.int32)
            _bounds[(self.b + 1,)](seg, self.cu, n, LOG2N=log2_ceil(max(n, 1)))
        else:
            self.cu = torch.zeros(1, device=device, dtype=torch.int32)

    def reduce(self, part: Tensor, slots: int, d: int, rounded: bool) -> Tensor:
        """Per-sample sums of the tile pair partials [tiles, slots, 2, D]: fp32 pairs
        [slots, 2, B, D], or [slots, B, D] rounded to fp32 when `rounded`."""
        return pair_sums(part, self.cu, self.b, self.chunk, self.has_seg, rounded)


def pair_sums(
    part: Tensor, cu: Tensor, b: int, chunk: int, has_seg: bool, rounded: bool
) -> Tensor:
    """Pair sums of part [T, slots, 2, D] over each sample's tiles (all T tiles as one
    sample without segments); see `segment_sum_kernel`."""
    tiles, slots, _, d = part.shape
    shape = (slots, b, d) if rounded else (slots, 2, b, d)
    out = torch.empty(shape, device=part.device, dtype=torch.float32)
    bd = min(128, triton.next_power_of_2(d))
    bt = max(8, 2048 // bd)
    _segsum[(b, cdiv(d, bd))](
        part,
        out,
        cu,
        d,
        b,
        chunk,
        tiles,
        SLOTS=slots,
        HAS_SEG=has_seg,
        ROUND=rounded,
        BLOCK_B=triton.next_power_of_2(max(b, 1)),
        BT=bt,
        BD=bd,
        num_warps=4,
    )
    return out


def total(part: Tensor) -> Tensor:
    """fp32 [slots, D] sums of every row of part [T, slots, 2, D] (pairs)."""
    cu = torch.empty(1, device=part.device, dtype=torch.int32)
    return pair_sums(part, cu, 1, 1, False, True)[:, 0]


def empty() -> Tensor:
    return torch.empty(0)


def strides2(t: Tensor | None) -> tuple[int, int]:
    """(row, column) strides of a per-sample [B, D] tensor ((0, 0) when absent)."""
    return (0, 0) if t is None else (t.stride(0), t.stride(1))


def opt(t: Tensor) -> Tensor | None:
    return t if t.numel() else None


def row_warps(block_d: int) -> int:
    if block_d <= 512:
        return 2
    if block_d <= 2048:
        return 4
    if block_d <= 4096:
        return 8
    return 16


def row_candidates(block_d: int) -> list[dict]:
    """(ROWS, warps) configs keeping about 2k-8k fp32 elements per live tile."""
    base = row_warps(block_d)
    out = []
    for rows in (1, 2, 4, 8, 16, 32):
        if rows * block_d > 16384 or rows * block_d < 512 and rows < 32:
            continue
        for warps in sorted({base, min(16, base * 2), max(1, base // 2)}):
            out.append({"ROWS": rows, "warps": warps})
    if not out:
        out.append({"ROWS": 1, "warps": base})
    out.sort(key=lambda c: (abs(c["ROWS"] * block_d - 4096), c["warps"] != base))
    return out
