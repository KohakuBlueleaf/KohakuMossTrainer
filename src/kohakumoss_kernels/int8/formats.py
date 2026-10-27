"""int8 compute formats: the eager reference of the per-row int8 quantizer and the int8
GEMM (int32 accumulation) the `int8_linear` op runs.

An operand is 2-D [rows, K] with K the contraction axis. A row's scale is
s = max(amax, 1e-12) / 127 (a correctly rounded fp32 division) and its codes
q = round_half_even(x / s) clamped to [-127, 127]; `col_scale` multiplies the columns
first (in fp32). See docs/internals/quant-kernels.md#int8-compute.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

INT8_MAX = 127.0
AMAX_FLOOR = 1e-12
MIN_ROWS = 17
ROW_ALIGN = 8


def row_scale(amax: Tensor) -> Tensor:
    """max(amax, 1e-12) / 127 per row, fp32."""
    amax = amax.float().clamp_min(AMAX_FLOOR)
    return amax / torch.full_like(amax, INT8_MAX)


def quantize_rows(x: Tensor, col_scale: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """(q int8 [rows, K], s fp32 [rows]) of a 2-D operand, its columns times
    `col_scale` [K] first when given."""
    if x.dim() != 2:
        raise ValueError(f"int8 operand must be 2-D, got {tuple(x.shape)}")
    xf = x.detach().float()
    if col_scale is not None:
        xf = xf * col_scale.float()[None, :]
    s = row_scale(xf.abs().amax(1))
    q = torch.round(xf / s[:, None]).clamp_(-INT8_MAX, INT8_MAX)
    return q.to(torch.int8), s


def int_mm(a: Tensor, b: Tensor) -> Tensor:
    """a [M, K] int8 @ b [K, N] int8 -> int32 [M, N] (`torch._int_mm`; the rows are
    zero-padded to at least MIN_ROWS and a multiple of ROW_ALIGN, as cuBLASLt's int8
    GEMM needs; a stride-0 operand, a weightless view, is copied out first)."""
    if 0 in b.stride():
        b = b.t().contiguous().t()
    if 0 in a.stride():
        a = a.contiguous()
    m = a.shape[0]
    rows = max(MIN_ROWS, m + (-m) % ROW_ALIGN)
    rows += (-rows) % ROW_ALIGN
    if rows == m:
        return torch._int_mm(a, b)
    return torch._int_mm(F.pad(a, (0, 0, 0, rows - m)), b)[:m]


def col_major(q: Tensor) -> Tensor:
    """q [N, K] as a column-major [N, K] view (one transposed copy): the layout the
    int8 GEMM reads fast as its right operand."""
    return q.t().contiguous().t()
