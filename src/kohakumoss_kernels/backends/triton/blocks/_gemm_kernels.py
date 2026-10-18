"""Triton GEMMs `A [M, K] @ W[rows, K]^T` with the MOSS block epilogues fused in.

fp32 accumulation; the epilogue reads the fp32 accumulator and rounds once at the
store. Grid is 1-D with GROUP_M swizzling for L2 reuse.
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice

from kohakumoss_kernels.backends.triton.blocks.ffn import situ


@triton.jit
def _pid_mn(M, N, BM: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    group = GROUP_M * num_n
    first_m = (pid // group) * GROUP_M
    size_m = tl.minimum(num_m - first_m, GROUP_M)
    return first_m + (pid % group) % size_m, (pid % group) // size_m


@triton.jit
def _dot_rows(a_ptr, w_ptr, offs_m, w_rows, M, K, acc, BK: tl.constexpr):
    """acc += A[offs_m, :] @ W[w_rows, :]^T over the full K."""
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m.to(tl.int64)[:, None] * K + offs_k[None, :]
    w_ptrs = w_ptr + w_rows.to(tl.int64)[None, :] * K + offs_k[:, None]
    m_ok = (offs_m < M)[:, None]
    for k0 in range(0, K, BK):
        k_ok = (k0 + offs_k) < K
        a = tl.load(a_ptrs, mask=m_ok & k_ok[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_ok[:, None], other=0.0)
        acc = tl.dot(a, w, acc)
        a_ptrs += BK
        w_ptrs += BK
    return acc


@triton.jit
def situ_gemm_kernel(
    a_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    ug_ptr,
    M,
    F,
    K,
    HAS_BIAS: tl.constexpr,
    SAVE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """out [M, F] = situ(A W_up^T + b_up, A W_gate^T + b_gate); ug [M, 2F] saved when SAVE."""
    pm, pn = _pid_mn(M, F, BM, BN, GROUP_M)
    offs_m = pm * BM + tl.arange(0, BM)
    offs_n = pn * BN + tl.arange(0, BN)
    rows_u = tl.minimum(offs_n, F - 1)
    acc_u = _dot_rows(
        a_ptr, w_ptr, offs_m, rows_u, M, K, tl.zeros([BM, BN], tl.float32), BK
    )
    acc_g = _dot_rows(
        a_ptr, w_ptr, offs_m, rows_u + F, M, K, tl.zeros([BM, BN], tl.float32), BK
    )
    n_ok = offs_n < F
    if HAS_BIAS:
        acc_u += tl.load(b_ptr + offs_n, mask=n_ok, other=0.0).to(tl.float32)[None, :]
        acc_g += tl.load(b_ptr + F + offs_n, mask=n_ok, other=0.0).to(tl.float32)[
            None, :
        ]
    mask = (offs_m < M)[:, None] & n_ok[None, :]
    r64 = offs_m.to(tl.int64)[:, None]
    if SAVE:
        ug = ug_ptr + r64 * (2 * F) + offs_n[None, :]
        tl.store(ug, acc_u.to(ug_ptr.dtype.element_ty), mask=mask)
        tl.store(ug + F, acc_g.to(ug_ptr.dtype.element_ty), mask=mask)
    out = situ(acc_u, acc_g)
    tl.store(
        out_ptr + r64 * F + offs_n[None, :], out.to(out_ptr.dtype.element_ty), mask=mask
    )


@triton.jit
def residual_gemm_kernel(
    a_ptr,
    w_ptr,
    b_ptr,
    x_ptr,
    gate_ptr,
    seg_ptr,
    out_ptr,
    o_ptr,
    M,
    N,
    K,
    r,
    g_row,
    g_col,
    HAS_BIAS: tl.constexpr,
    HAS_GATE: tl.constexpr,
    SCALED: tl.constexpr,
    SAVE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """out = x * r + (A W^T + b) * gate[seg] (gate read through its strides); o = A W^T
    + b saved when SAVE."""
    pm, pn = _pid_mn(M, N, BM, BN, GROUP_M)
    offs_m = pm * BM + tl.arange(0, BM)
    offs_n = pn * BN + tl.arange(0, BN)
    rows_w = tl.minimum(offs_n, N - 1)
    acc = _dot_rows(
        a_ptr, w_ptr, offs_m, rows_w, M, K, tl.zeros([BM, BN], tl.float32), BK
    )
    m_ok = offs_m < M
    n_ok = offs_n < N
    mask = m_ok[:, None] & n_ok[None, :]
    if HAS_BIAS:
        acc += tl.load(b_ptr + offs_n, mask=n_ok, other=0.0).to(tl.float32)[None, :]
    offs = offs_m.to(tl.int64)[:, None] * N + offs_n[None, :]
    if SAVE:
        tl.store(o_ptr + offs, acc.to(o_ptr.dtype.element_ty), mask=mask)
    if HAS_GATE:
        s = tl.load(seg_ptr + offs_m, mask=m_ok, other=0).to(tl.int64)
        goff = s[:, None] * g_row + offs_n[None, :] * g_col
        g = tl.load(gate_ptr + goff, mask=mask, other=0.0)
        acc = acc * g.to(tl.float32)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if SCALED:
        x = x * r
    tl.store(out_ptr + offs, (x + acc).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def qk_rope_gemm_kernel(
    a_ptr,
    wq_ptr,
    wk_ptr,
    qw_ptr,
    kw_ptr,
    pos_ptr,
    bands_ptr,
    qo_ptr,
    ko_ptr,
    qp_ptr,
    kp_ptr,
    qr_ptr,
    kr_ptr,
    M,
    K,
    H,
    A,
    eps,
    NB: tl.constexpr,
    HALF: tl.constexpr,
    SAVE: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """One program = BM tokens x one head of q (pn < H) or k (pn >= H): GEMM, then
    per-head RMSNorm with weight and split-half RoPE from positions."""
    pm, pn = _pid_mn(M, 2 * H, BM, 1, GROUP_M)
    is_k = pn >= H
    head = tl.where(is_k, pn - H, pn)
    offs_m = pm * BM + tl.arange(0, BM)
    j = tl.arange(0, HALF)
    rows = head * (2 * HALF) + j
    zero = tl.zeros([BM, HALF], tl.float32)
    if is_k:
        u = _dot_rows(a_ptr, wk_ptr, offs_m, rows, M, K, zero, BK)
        v = _dot_rows(a_ptr, wk_ptr, offs_m, rows + HALF, M, K, zero, BK)
        nw = kw_ptr
        out_ptr = ko_ptr
        pre_ptr = kp_ptr
        rstd_ptr = kr_ptr
    else:
        u = _dot_rows(a_ptr, wq_ptr, offs_m, rows, M, K, zero, BK)
        v = _dot_rows(a_ptr, wq_ptr, offs_m, rows + HALF, M, K, zero, BK)
        nw = qw_ptr
        out_ptr = qo_ptr
        pre_ptr = qp_ptr
        rstd_ptr = qr_ptr
    m_ok = offs_m < M
    mask = m_ok[:, None]
    offs = offs_m.to(tl.int64)[:, None] * (H * 2 * HALF) + rows[None, :]
    if SAVE:
        tl.store(pre_ptr + offs, u.to(pre_ptr.dtype.element_ty), mask=mask)
        tl.store(pre_ptr + offs + HALF, v.to(pre_ptr.dtype.element_ty), mask=mask)
    ms = tl.div_rn(tl.sum(u * u, axis=1) + tl.sum(v * v, axis=1), 2.0 * HALF)
    rstd = tl.div_rn(1.0, tl.sqrt_rn(ms + eps.to(tl.float32)))
    un = u * rstd[:, None] * tl.load(nw + j).to(tl.float32)[None, :]
    vn = v * rstd[:, None] * tl.load(nw + HALF + j).to(tl.float32)[None, :]
    p = tl.load(
        pos_ptr + offs_m.to(tl.int64)[:, None] * A + (j // NB)[None, :],
        mask=mask,
        other=0.0,
    )
    phase = p.to(tl.float32) * tl.load(bands_ptr + (j % NB)).to(tl.float32)[None, :]
    s = libdevice.sin(phase)
    c = libdevice.cos(phase)
    tl.store(out_ptr + offs, (un * c - vn * s).to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(
        out_ptr + offs + HALF, (vn * c + un * s).to(out_ptr.dtype.element_ty), mask=mask
    )
    if SAVE:
        tl.store(rstd_ptr + offs_m.to(tl.int64) * H + head, rstd, mask=m_ok)
