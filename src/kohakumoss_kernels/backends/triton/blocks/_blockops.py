"""Shared pieces of the Triton block ops: segment tile plans and their reductions.

Rows of a packed batch belong to samples through a nondecreasing `seg [N]`. Backward
kernels that reduce over a sample's rows run one program per *segment tile*: a run
of at most CHUNK rows inside one sample. `segment_bounds` writes the sample row
bounds `cu [B+1]` on the device (no host sync); a program finds its tile from `cu`
with a vector prefix sum, so partials are per tile and sums are deterministic.
Partials are plain fp32 [tiles, slots, D].
See docs/internals/fp16.md.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

from kohakumoss_kernels.backends._memo import TensorMemo
from kohakumoss_kernels.backends.triton._common import PLAIN, Launcher
from kohakumoss_kernels.hardware.device import measure_rates
from kohakumoss_kernels.tuning.buckets import cdiv, chunk_rows


@triton.jit
def segment_bounds_kernel(seg_ptr, cu_ptr, N, LOG2N: tl.constexpr):
    """cu[b] = first row with seg >= b (binary search); one program per b."""
    b = tl.program_id(0)
    lo = 0
    hi = N
    for _ in range(LOG2N):
        mid = (lo + hi) // 2
        active = lo < hi
        v = tl.load(seg_ptr + tl.minimum(mid, N - 1), mask=active, other=0)
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
    BLOCK_B: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    """fp32 sums over the tiles of sample b of part [T, SLOTS, D], BT tiles per step,
    into out [SLOTS, B, D]."""
    b = tl.program_id(0)
    cols = tl.program_id(1) * BD + tl.arange(0, BD)
    ok = cols < D
    beg, end = segment_tiles(b, cu_ptr, B, CHUNK, T, HAS_SEG, BLOCK_B)
    for s in tl.static_range(SLOTS):
        acc = tl.zeros([BD], tl.float32)
        for t0 in range(beg, end, BT):
            ts = t0 + tl.arange(0, BT)
            m = (ts < end)[:, None] & ok[None, :]
            src = part_ptr + (ts.to(tl.int64)[:, None] * SLOTS + s) * D + cols[None, :]
            acc += tl.sum(tl.load(src, mask=m, other=0.0), axis=0)
        tl.store(out_ptr + (s * B + b) * D + cols, acc, mask=ok)


_bounds = Launcher(segment_bounds_kernel)
_segsum = Launcher(segment_sum_kernel)


def num_sms(device: torch.device) -> int:
    return max(1, measure_rates(device).sms)


def log2_ceil(n: int) -> int:
    k = 1
    while (1 << k) <= n:
        k += 1
    return k


_CU = TensorMemo()


def _bounds_of(seg: Tensor, n: int, b: int) -> Tensor:
    cu = torch.empty(b + 1, device=seg.device, dtype=torch.int32)
    _bounds[(b + 1,)](seg, cu, n, LOG2N=log2_ceil(max(n, 1)))
    return cu


def segment_bounds(seg: Tensor, n: int, b: int) -> Tensor:
    """cu [B+1] int32 of `seg` (device, no host sync); computed once while the same
    eager tensor at the same version is alive (every op of a step shares it)."""
    if type(seg) not in PLAIN:
        return _bounds_of(seg, n, b)
    return _CU.get(seg, (n, b), lambda: _bounds_of(seg, n, b))


class SegPlan:
    """Segment tiles of N rows over B samples (or plain CHUNK tiles without seg):
    about `per_sm` tiles per SM, or tiles of `rows` rows with `per_sm` None."""

    def __init__(
        self,
        seg: Tensor | None,
        n: int,
        b: int,
        rows: int,
        device: torch.device,
        per_sm: int | None = 2,
    ):
        self.n = n
        self.has_seg = seg is not None
        self.b = b if self.has_seg else 1
        if per_sm is None:
            chunk = rows
        else:
            chunk = chunk_rows(n, rows, num_sms(device), per_sm)
        self.chunk = chunk
        self.tiles = cdiv(n, chunk) + (self.b if self.has_seg else 0)
        self.block_b = triton.next_power_of_2(max(self.b, 1))
        if self.has_seg:
            self.cu = segment_bounds(seg, n, self.b)
        else:
            self.cu = torch.empty(1, device=device, dtype=torch.int32)

    def reduce(self, part: Tensor) -> Tensor:
        """Per-sample fp32 sums [slots, B, D] of the tile partials [tiles, slots, D]."""
        return segment_sums(part, self.cu, self.b, self.chunk, self.has_seg)


def segment_sums(part: Tensor, cu: Tensor, b: int, chunk: int, has_seg: bool) -> Tensor:
    """fp32 sums [slots, B, D] of part [T, slots, D] over each sample's tiles (all T
    tiles as one sample without segments); see `segment_sum_kernel`."""
    tiles, slots, d = part.shape
    out = torch.empty(slots, b, d, device=part.device, dtype=torch.float32)
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
        BLOCK_B=triton.next_power_of_2(max(b, 1)),
        BT=bt,
        BD=bd,
        num_warps=4,
    )
    return out


def total(part: Tensor) -> Tensor:
    """fp32 [slots, D] sums of every row of part [T, slots, D]."""
    cu = torch.empty(1, device=part.device, dtype=torch.int32)
    return segment_sums(part, cu, 1, 1, False)[:, 0]


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


def reduce_candidates(block_d: int) -> list[dict]:
    """`row_candidates` crossed with segment tiles per SM (`per_sm`) for kernels that
    write per-tile partials: more tiles, more parallelism and more partials."""
    return [
        {**c, "per_sm": ps} for ps in (2, 8, 32) for c in row_candidates(block_d)[:4]
    ]


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
