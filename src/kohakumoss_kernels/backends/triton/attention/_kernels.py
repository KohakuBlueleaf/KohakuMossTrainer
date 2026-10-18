"""Varlen flash attention in Triton: forward, delta, dK/dV and dQ kernels.

Packed `[N, H, D]` tensors (last dim contiguous), segments from `cu_seqlens`; grid
axis 0 covers the longest segment, programs past their segment's end exit. Row
statistics are two fp32 planes `M` (raw score max) and `L` (log2 of the row sum),
probabilities are `exp2((s - m) * c - l)`. Every MMA takes 16-bit operands (P and dS
rounded once) with fp32 accumulation. Token offsets are int64 (addresses past 2^31
elements). See docs/internals/kernel-dev.md.
"""

import triton
import triton.language as tl


@triton.jit
def _mma(a, b, acc):
    """acc + a @ b for fp32 `a` (rounded to `b`'s 16-bit dtype) and 16-bit `b`."""
    return tl.dot(a.to(b.dtype), b, acc)


@triton.jit
def _fwd_tile(acc, m_i, l_i, q, k_ptrs, v_ptrs, cols, c, MASKED: tl.constexpr):
    if MASKED:
        k = tl.load(k_ptrs, mask=cols[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=cols[:, None], other=0.0)
    else:
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
    s = tl.dot(q, tl.trans(k))
    if MASKED:
        s = tl.where(cols[None, :], s, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(s, 1))
    alpha = tl.exp2((m_i - m_new) * c)
    p = tl.exp2((s - m_new[:, None]) * c)
    acc = _mma(p, v, acc * alpha[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    return acc, m_new, l_i


@triton.jit
def fwd_kernel(
    Q, K, V, OUT, M, L, CU, c, N,
    sq_n, sq_h, sk_n, sk_h, sv_n, sv_h, so_n, so_h,
    D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
):  # fmt: skip
    """Grid (blocks of the longest segment, heads, segments)."""
    pid, head, seg = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start = tl.load(CU + seg).to(tl.int64)
    seqlen = tl.load(CU + seg + 1) - start
    if pid * BM >= seqlen:
        return
    c = c.to(tl.float32)
    offs_m = pid * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    rows = offs_m < seqlen
    q = tl.load(
        Q + (start + offs_m)[:, None] * sq_n + head * sq_h + offs_d[None, :],
        mask=rows[:, None],
        other=0.0,
    )
    acc = tl.zeros([BM, D], tl.float32)
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    k_base = K + start * sk_n + head * sk_h
    v_base = V + start * sv_n + head * sv_h
    full = seqlen // BN * BN
    for n0 in range(0, full, BN):
        offs_n = n0 + tl.arange(0, BN)
        acc, m_i, l_i = _fwd_tile(
            acc, m_i, l_i, q,
            k_base + offs_n[:, None] * sk_n + offs_d[None, :],
            v_base + offs_n[:, None] * sv_n + offs_d[None, :],
            offs_n < seqlen, c, False,
        )  # fmt: skip
    if full < seqlen:
        offs_n = full + tl.arange(0, BN)
        acc, m_i, l_i = _fwd_tile(
            acc, m_i, l_i, q,
            k_base + offs_n[:, None] * sk_n + offs_d[None, :],
            v_base + offs_n[:, None] * sv_n + offs_d[None, :],
            offs_n < seqlen, c, True,
        )  # fmt: skip
    tok = start + offs_m
    tl.store(M + head * N + tok, m_i, mask=rows)
    tl.store(L + head * N + tok, tl.log2(l_i), mask=rows)
    out = tok[:, None] * so_n + head * so_h + offs_d[None, :]
    o = acc / l_i[:, None]
    tl.store(OUT + out, o.to(OUT.dtype.element_ty), mask=rows[:, None])


@triton.jit
def delta_kernel(
    OUT, DO, DELTA, DQ_ACC, N, so_n, so_h, sd_n, sd_h,
    D: tl.constexpr, BM: tl.constexpr, ZERO: tl.constexpr,
):  # fmt: skip
    """DELTA[h, n] = sum_d O * dO in fp32; with ZERO also DQ_ACC[n, h, :] = 0 (fp32
    [N, H, D] contiguous); grid (token blocks, heads)."""
    pid, head = tl.program_id(0), tl.program_id(1)
    offs = pid.to(tl.int64) * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    rows = offs < N
    o = tl.load(
        OUT + offs[:, None] * so_n + head * so_h + offs_d[None, :],
        mask=rows[:, None],
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        DO + offs[:, None] * sd_n + head * sd_h + offs_d[None, :],
        mask=rows[:, None],
        other=0.0,
    ).to(tl.float32)
    tl.store(DELTA + head * N + offs, tl.sum(o * do, 1), mask=rows)
    if ZERO:
        acc = DQ_ACC + (offs[:, None] * tl.num_programs(1) + head) * D + offs_d[None, :]
        tl.store(acc, tl.zeros([BM, D], tl.float32), mask=rows[:, None])


@triton.jit
def _load_rows(qt_ptrs, do_ptrs, stat, rows, MASKED: tl.constexpr):
    """q^T [D, BM], dO [BM, D] and the row statistics of one query tile."""
    m_ptr, l_ptr, d_ptr = stat
    if MASKED:
        qt = tl.load(qt_ptrs, mask=rows[None, :], other=0.0)
        do = tl.load(do_ptrs, mask=rows[:, None], other=0.0)
        m = tl.load(m_ptr, mask=rows, other=0.0)
        lg = tl.load(l_ptr, mask=rows, other=0.0)
        delta = tl.load(d_ptr, mask=rows, other=0.0)
    else:
        qt = tl.load(qt_ptrs)
        do = tl.load(do_ptrs)
        m = tl.load(m_ptr)
        lg = tl.load(l_ptr)
        delta = tl.load(d_ptr)
    return qt, do, m, lg, delta


@triton.jit
def _dkdv_tile(dk, dv, k, v, qt_ptrs, do_ptrs, stat, rows, c, MASKED: tl.constexpr):
    qt, do, m, lg, delta = _load_rows(qt_ptrs, do_ptrs, stat, rows, MASKED)
    pt = tl.exp2((tl.dot(k, qt) - m[None, :]) * c - lg[None, :])
    if MASKED:
        pt = tl.where(rows[None, :], pt, 0.0)
    dv = _mma(pt, do, dv)
    dst = pt * (tl.dot(v, tl.trans(do)) - delta[None, :])
    dk = _mma(dst, tl.trans(qt), dk)
    return dk, dv


@triton.jit
def dkdv_kernel(
    Q, K, V, DO, M, L, DELTA, DK, DV, CU, scale, c, N,
    sq_n, sq_h, sk_n, sk_h, sv_n, sv_h, sd_n, sd_h, sg_n, sg_h,
    D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
):  # fmt: skip
    """Grid (key blocks of the longest segment, heads, segments); dK/dV in DK's strides."""
    pid, head, seg = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start = tl.load(CU + seg).to(tl.int64)
    seqlen = tl.load(CU + seg + 1) - start
    if pid * BN >= seqlen:
        return
    c = c.to(tl.float32)
    offs_n = pid * BN + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    cols = offs_n < seqlen
    k = tl.load(
        K + (start + offs_n)[:, None] * sk_n + head * sk_h + offs_d[None, :],
        mask=cols[:, None],
        other=0.0,
    )
    v = tl.load(
        V + (start + offs_n)[:, None] * sv_n + head * sv_h + offs_d[None, :],
        mask=cols[:, None],
        other=0.0,
    )
    dk = tl.zeros([BN, D], tl.float32)
    dv = tl.zeros([BN, D], tl.float32)
    q_base = Q + start * sq_n + head * sq_h
    do_base = DO + start * sd_n + head * sd_h
    st_base = head * N + start
    full = seqlen // BM * BM
    for m0 in range(0, full, BM):
        offs_m = m0 + tl.arange(0, BM)
        dk, dv = _dkdv_tile(
            dk, dv, k, v,
            q_base + offs_m[None, :] * sq_n + offs_d[:, None],
            do_base + offs_m[:, None] * sd_n + offs_d[None, :],
            (M + st_base + offs_m, L + st_base + offs_m, DELTA + st_base + offs_m),
            offs_m < seqlen, c, False,
        )  # fmt: skip
    if full < seqlen:
        offs_m = full + tl.arange(0, BM)
        dk, dv = _dkdv_tile(
            dk, dv, k, v,
            q_base + offs_m[None, :] * sq_n + offs_d[:, None],
            do_base + offs_m[:, None] * sd_n + offs_d[None, :],
            (M + st_base + offs_m, L + st_base + offs_m, DELTA + st_base + offs_m),
            offs_m < seqlen, c, True,
        )  # fmt: skip
    dk = dk * scale.to(tl.float32)
    out = (start + offs_n)[:, None] * sg_n + head * sg_h + offs_d[None, :]
    tl.store(DK + out, dk.to(DK.dtype.element_ty), mask=cols[:, None])
    tl.store(DV + out, dv.to(DV.dtype.element_ty), mask=cols[:, None])


@triton.jit
def _fused_tile(dk, dv, k, v, qt_ptrs, do_ptrs, dq_ptrs, stat, rows, cols, c,
                MASKED: tl.constexpr, BM: tl.constexpr, D: tl.constexpr):  # fmt: skip
    qt, do, m, lg, delta = _load_rows(qt_ptrs, do_ptrs, stat, rows, MASKED)
    pt = tl.exp2((tl.dot(k, qt) - m[None, :]) * c - lg[None, :])
    # padded keys of the tail key block carry no probability
    pt = tl.where(cols[:, None], pt, 0.0)
    if MASKED:
        pt = tl.where(rows[None, :], pt, 0.0)
    dv = _mma(pt, do, dv)
    dst = (pt * (tl.dot(v, tl.trans(do)) - delta[None, :])).to(k.dtype)
    dk = tl.dot(dst, tl.trans(qt), dk)
    part = tl.dot(tl.trans(dst), k, tl.zeros([BM, D], tl.float32))
    if MASKED:
        tl.atomic_add(dq_ptrs, part, mask=rows[:, None], sem="relaxed")
    else:
        tl.atomic_add(dq_ptrs, part, sem="relaxed")
    return dk, dv


@triton.jit
def fused_bwd_kernel(
    Q, K, V, DO, M, L, DELTA, DK, DV, DQ_ACC, CU, scale, c, N,
    sq_n, sq_h, sk_n, sk_h, sv_n, sv_h, sd_n, sd_h, sg_n, sg_h,
    D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
):  # fmt: skip
    """dK/dV per key block plus unscaled fp32 dQ partials added atomically to DQ_ACC
    ([N, H, D] contiguous); grid (key blocks of the longest segment, heads, segments).
    """
    pid, head, seg = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start = tl.load(CU + seg).to(tl.int64)
    seqlen = tl.load(CU + seg + 1) - start
    if pid * BN >= seqlen:
        return
    c = c.to(tl.float32)
    offs_n = pid * BN + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    cols = offs_n < seqlen
    k = tl.load(
        K + (start + offs_n)[:, None] * sk_n + head * sk_h + offs_d[None, :],
        mask=cols[:, None],
        other=0.0,
    )
    v = tl.load(
        V + (start + offs_n)[:, None] * sv_n + head * sv_h + offs_d[None, :],
        mask=cols[:, None],
        other=0.0,
    )
    dk = tl.zeros([BN, D], tl.float32)
    dv = tl.zeros([BN, D], tl.float32)
    q_base = Q + start * sq_n + head * sq_h
    do_base = DO + start * sd_n + head * sd_h
    dq_base = DQ_ACC + (start * tl.num_programs(1) + head) * D
    dq_n = tl.num_programs(1) * D
    st_base = head * N + start
    full = seqlen // BM * BM
    for m0 in range(0, full, BM):
        offs_m = m0 + tl.arange(0, BM)
        dk, dv = _fused_tile(
            dk, dv, k, v,
            q_base + offs_m[None, :] * sq_n + offs_d[:, None],
            do_base + offs_m[:, None] * sd_n + offs_d[None, :],
            dq_base + offs_m[:, None] * dq_n + offs_d[None, :],
            (M + st_base + offs_m, L + st_base + offs_m, DELTA + st_base + offs_m),
            offs_m < seqlen, cols, c, False, BM, D,
        )  # fmt: skip
    if full < seqlen:
        offs_m = full + tl.arange(0, BM)
        dk, dv = _fused_tile(
            dk, dv, k, v,
            q_base + offs_m[None, :] * sq_n + offs_d[:, None],
            do_base + offs_m[:, None] * sd_n + offs_d[None, :],
            dq_base + offs_m[:, None] * dq_n + offs_d[None, :],
            (M + st_base + offs_m, L + st_base + offs_m, DELTA + st_base + offs_m),
            offs_m < seqlen, cols, c, True, BM, D,
        )  # fmt: skip
    dk = dk * scale.to(tl.float32)
    out = (start + offs_n)[:, None] * sg_n + head * sg_h + offs_d[None, :]
    tl.store(DK + out, dk.to(DK.dtype.element_ty), mask=cols[:, None])
    tl.store(DV + out, dv.to(DV.dtype.element_ty), mask=cols[:, None])


@triton.jit
def dq_cast_kernel(DQ_ACC, DQ, scale, NUMEL, BLOCK: tl.constexpr):
    """DQ = (DQ_ACC * scale) cast to DQ's dtype (both contiguous)."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    keep = offs < NUMEL
    x = tl.load(DQ_ACC + offs, mask=keep, other=0.0)
    tl.store(DQ + offs, (x * scale.to(tl.float32)).to(DQ.dtype.element_ty), mask=keep)


@triton.jit
def _dq_tile(dq, q, do, m, lg, delta, k_ptrs, v_ptrs, cols, c, MASKED: tl.constexpr):
    if MASKED:
        k = tl.load(k_ptrs, mask=cols[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=cols[:, None], other=0.0)
    else:
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
    s = tl.dot(q, tl.trans(k))
    p = tl.exp2((s - m[:, None]) * c - lg[:, None])
    if MASKED:
        p = tl.where(cols[None, :], p, 0.0)
    ds = p * (tl.dot(do, tl.trans(v)) - delta[:, None])
    return _mma(ds, k, dq)


@triton.jit
def dq_kernel(
    Q, K, V, DO, M, L, DELTA, DQ, CU, scale, c, N,
    sq_n, sq_h, sk_n, sk_h, sv_n, sv_h, sd_n, sd_h, sg_n, sg_h,
    D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
):  # fmt: skip
    """Grid (query blocks of the longest segment, heads, segments)."""
    pid, head, seg = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start = tl.load(CU + seg).to(tl.int64)
    seqlen = tl.load(CU + seg + 1) - start
    if pid * BM >= seqlen:
        return
    c = c.to(tl.float32)
    offs_m = pid * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    rows = offs_m < seqlen
    tok = start + offs_m
    q = tl.load(
        Q + tok[:, None] * sq_n + head * sq_h + offs_d[None, :],
        mask=rows[:, None],
        other=0.0,
    )
    do = tl.load(
        DO + tok[:, None] * sd_n + head * sd_h + offs_d[None, :],
        mask=rows[:, None],
        other=0.0,
    )
    m = tl.load(M + head * N + tok, mask=rows, other=0.0)
    lg = tl.load(L + head * N + tok, mask=rows, other=0.0)
    delta = tl.load(DELTA + head * N + tok, mask=rows, other=0.0)
    dq = tl.zeros([BM, D], tl.float32)
    k_base = K + start * sk_n + head * sk_h
    v_base = V + start * sv_n + head * sv_h
    full = seqlen // BN * BN
    for n0 in range(0, full, BN):
        offs_n = n0 + tl.arange(0, BN)
        dq = _dq_tile(
            dq, q, do, m, lg, delta,
            k_base + offs_n[:, None] * sk_n + offs_d[None, :],
            v_base + offs_n[:, None] * sv_n + offs_d[None, :],
            offs_n < seqlen, c, False,
        )  # fmt: skip
    if full < seqlen:
        offs_n = full + tl.arange(0, BN)
        dq = _dq_tile(
            dq, q, do, m, lg, delta,
            k_base + offs_n[:, None] * sk_n + offs_d[None, :],
            v_base + offs_n[:, None] * sv_n + offs_d[None, :],
            offs_n < seqlen, c, True,
        )  # fmt: skip
    dq = dq * scale.to(tl.float32)
    tl.store(
        DQ + tok[:, None] * sg_n + head * sg_h + offs_d[None, :],
        dq.to(DQ.dtype.element_ty),
        mask=rows[:, None],
    )
