"""Gated delta rule (Qwen3.5 Gated DeltaNet) in Triton, forward only, varlen.

Chunked form, chunk 64 = four 16-row sub-blocks: `prep_kernel` (chunk x value head)
builds the UT transform T = (I + A)^-1; `state_kernel` (segment x value head x value
tile, serial over chunks) writes each chunk's start state H and V'; `out_kernel`
(chunk x value head x value tile) writes the output; `fused_kernel` makes each chunk's
output inside the serial pass; `recurrent_kernel` is token-serial vector math.

Every product is an MMA on input-dtype (`DT`) operands with fp32 accumulation; the
state and every scale / decay stay fp32. T, H, V' and the decayed QK go through
memory in `DT`: they are only ever MMA operands, so the store is that rounding. Raw
q / k rows enter exact and are rescaled by their fp32 l2 norms. Chunk
offsets of a segment come from `CCU`; segment s spans [SB[s], SE[s]). See
docs/internals/text-encoder.md.
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice

BT = 64


@triton.jit
def gates(A, B, ALOG, DTB, tok, ok, hv, s_ab):
    """(g, beta) fp32 of the rows `tok` of value head `hv`; zero outside `ok`."""
    a = tl.load(A + tok * s_ab + hv, mask=ok, other=0.0).to(tl.float32)
    b = tl.load(B + tok * s_ab + hv, mask=ok, other=0.0).to(tl.float32)
    x = a + tl.load(DTB + hv).to(tl.float32)
    sp = tl.where(x > 20.0, x, libdevice.log1p(libdevice.exp(x)))
    g = -libdevice.exp(tl.load(ALOG + hv).to(tl.float32)) * sp
    beta = 1.0 / (1.0 + tl.exp(-b))
    return tl.where(ok, g, 0.0), tl.where(ok, beta, 0.0)


@triton.jit
def load_rows(P, tok, ok, s_n, off, cols):
    return tl.load(
        P + tok[:, None] * s_n + off + cols[None, :], mask=ok[:, None], other=0.0
    )


@triton.jit
def inv_norm(x):
    xf = x.to(tl.float32)
    return 1.0 / tl.sqrt_rn(tl.sum(xf * xf, 1) + 1e-6)


@triton.jit
def _chunk_base(CCU, seg):
    """Global index of segment `seg`'s first chunk (`CCU`: chunk prefix sums)."""
    return tl.load(CCU + seg)


@triton.jit
def mma(a, b, DT: tl.constexpr):
    """a @ b on DT operands, fp32 accumulation."""
    return tl.dot(a.to(DT), b.to(DT))


@triton.jit
def pair_decay(g, N: tl.constexpr):
    """D[i, j] = sum_{j < t <= i} g_t (0 for i <= j), without prefix-sum differences."""
    r = tl.arange(0, N)
    return tl.cumsum(tl.where(r[:, None] > r[None, :], g[:, None], 0.0), 0)


@triton.jit
def suffix_sum(g, N: tl.constexpr):
    """sum_{t > j} g_t, indexed by j."""
    r = tl.arange(0, N)
    return tl.sum(tl.where(r[:, None] > r[None, :], g[:, None], 0.0), 0)


@triton.jit
def _a_scale(kk, rki, rkj, bi, d, DIAG: tl.constexpr):
    """A_ij = beta_i rk_i rk_j kk_ij e^d_ij from raw kk = k_i . k_j, strictly lower on
    the diagonal."""
    a = kk * (bi * rki)[:, None] * rkj[None, :] * tl.exp(d)
    if DIAG:
        r = tl.arange(0, 16)
        a = tl.where(r[:, None] > r[None, :], a, 0.0)
    return a


@triton.jit
def a_block(ki, kj, rki, rkj, bi, d, DIAG: tl.constexpr):
    """A_ij = beta_i rk_i rk_j (k_i . k_j) e^d_ij, strictly lower on the diagonal."""
    return _a_scale(tl.dot(ki, tl.trans(kj)), rki, rkj, bi, d, DIAG)


@triton.jit
def _inv_diag(a):
    """(I + a)^-1 of a strictly lower 16 x 16 block, row by row (fp32 vector math)."""
    r = tl.arange(0, 16)
    t = tl.where(r[:, None] == r[None, :], 1.0, 0.0)
    for i in tl.static_range(1, 16):
        m = -tl.sum(tl.where(r[:, None] == i, a, 0.0), 0)
        row = tl.sum(m[:, None] * t, 0)
        t = tl.where(r[:, None] == i, t + row[None, :], t)
    return t


@triton.jit
def invert(a00, a11, a22, a33, a10, a20, a21, a30, a31, a32, DT: tl.constexpr):
    """Blocks (i00, i11, i22, i33, i10, i20, i21, i30, i31, i32) of (I + A)^-1 for a
    strictly lower 64 x 64 A in 16 x 16 blocks: T_ij = -T_ii sum_{j<=m<i} A_im T_mj."""
    i00, i11, i22, i33 = _inv_diag(a00), _inv_diag(a11), _inv_diag(a22), _inv_diag(a33)
    i10 = -mma(i11, mma(a10, i00, DT), DT)
    i21 = -mma(i22, mma(a21, i11, DT), DT)
    i20 = -mma(i22, mma(a20, i00, DT) + mma(a21, i10, DT), DT)
    i32 = -mma(i33, mma(a32, i22, DT), DT)
    i31 = -mma(i33, mma(a31, i11, DT) + mma(a32, i21, DT), DT)
    i30 = -mma(i33, mma(a30, i00, DT) + mma(a31, i10, DT) + mma(a32, i20, DT), DT)
    return i00, i11, i22, i33, i10, i20, i21, i30, i31, i32


@triton.jit
def _put(T, base, i: tl.constexpr, j: tl.constexpr, blk):
    r = tl.arange(0, 16)
    ptr = T + base + (i * 16 + r[:, None]) * 64 + j * 16 + r[None, :]
    tl.store(ptr, blk.to(T.dtype.element_ty))


@triton.jit
def qk_block(qi, kj, rqi, rkj, d, DIAG: tl.constexpr):
    """(q_n,i . k_n,j) e^d_ij, lower triangle with the diagonal on diagonal blocks."""
    qk = tl.dot(qi, tl.trans(kj)) * rqi[:, None] * rkj[None, :] * tl.exp(d)
    if DIAG:
        r = tl.arange(0, 16)
        qk = tl.where(r[:, None] >= r[None, :], qk, 0.0)
    return qk


@triton.jit
def _factors(FAC, tok, ok, hv, HV, beta, rk, rq, gc, suf):
    """Per-row fp32 factors [beta, rk beta e^Gc, rq e^Gc, rk e^suffix]."""
    base = FAC + (tok.to(tl.int64) * HV + hv) * 4
    tl.store(base, beta, mask=ok)
    tl.store(base + 1, rk * beta * tl.exp(gc), mask=ok)
    tl.store(base + 2, rq * tl.exp(gc), mask=ok)
    tl.store(base + 3, rk * tl.exp(suf), mask=ok)


@triton.jit
def _qk_out(
    Q, QK, FAC, EGL, base, gi, HV, hv, sq_n, hk, scale,
    k0, k1, k2, k3, rk0, rk1, rk2, rk3, b0, b1, b2, b3, g0, g1, g2, g3,
    t0, t1, t2, t3, m0, m1, m2, m3, DK: tl.constexpr,
):  # fmt: skip
    """The fused path's extra prep outputs: decayed QK [64, 64], row factors, e^Gc_last."""
    ck = tl.arange(0, DK)
    q0 = load_rows(Q, t0, m0, sq_n, hk * DK, ck)
    q1 = load_rows(Q, t1, m1, sq_n, hk * DK, ck)
    q2 = load_rows(Q, t2, m2, sq_n, hk * DK, ck)
    q3 = load_rows(Q, t3, m3, sq_n, hk * DK, ck)
    rq0, rq1 = inv_norm(q0) * scale, inv_norm(q1) * scale
    rq2, rq3 = inv_norm(q2) * scale, inv_norm(q3) * scale
    s0, s1, s2, s3 = tl.sum(g0, 0), tl.sum(g1, 0), tl.sum(g2, 0), tl.sum(g3, 0)
    e0, e1, e2 = suffix_sum(g0, 16), suffix_sum(g1, 16), suffix_sum(g2, 16)
    p0, p1 = tl.cumsum(g0, 0), tl.cumsum(g1, 0)
    p2, p3 = tl.cumsum(g2, 0), tl.cumsum(g3, 0)
    zero = tl.zeros([16, 16], tl.float32)
    _put(QK, base, 0, 0, qk_block(q0, k0, rq0, rk0, pair_decay(g0, 16), True))
    _put(QK, base, 1, 1, qk_block(q1, k1, rq1, rk1, pair_decay(g1, 16), True))
    _put(QK, base, 2, 2, qk_block(q2, k2, rq2, rk2, pair_decay(g2, 16), True))
    _put(QK, base, 3, 3, qk_block(q3, k3, rq3, rk3, pair_decay(g3, 16), True))
    _put(QK, base, 1, 0, qk_block(q1, k0, rq1, rk0, p1[:, None] + e0[None, :], False))
    _put(QK, base, 2, 0, qk_block(q2, k0, rq2, rk0, p2[:, None] + s1 + e0[None, :], False))  # fmt: skip
    _put(QK, base, 2, 1, qk_block(q2, k1, rq2, rk1, p2[:, None] + e1[None, :], False))
    _put(QK, base, 3, 0, qk_block(q3, k0, rq3, rk0, p3[:, None] + (s1 + s2) + e0[None, :], False))  # fmt: skip
    _put(QK, base, 3, 1, qk_block(q3, k1, rq3, rk1, p3[:, None] + s2 + e1[None, :], False))  # fmt: skip
    _put(QK, base, 3, 2, qk_block(q3, k2, rq3, rk2, p3[:, None] + e2[None, :], False))
    _put(QK, base, 0, 1, zero)
    _put(QK, base, 0, 2, zero)
    _put(QK, base, 0, 3, zero)
    _put(QK, base, 1, 2, zero)
    _put(QK, base, 1, 3, zero)
    _put(QK, base, 2, 3, zero)
    _factors(FAC, t0, m0, hv, HV, b0, rk0, rq0, p0, e0 + (s1 + s2 + s3))
    _factors(FAC, t1, m1, hv, HV, b1, rk1, rq1, p1 + s0, e1 + (s2 + s3))
    _factors(FAC, t2, m2, hv, HV, b2, rk2, rq2, p2 + (s0 + s1), e2 + s3)
    _factors(FAC, t3, m3, hv, HV, b3, rk3, rq3, p3 + (s0 + s1 + s2), suffix_sum(g3, 16))
    tl.store(EGL + gi * HV + hv, tl.exp(s0 + s1 + s2 + s3))


@triton.jit
def prep_kernel(
    K, A, B, ALOG, DTB, SB, SE, CCU, T, Q, QK, FAC, EGL,
    sk_n, s_ab, HV, sq_n, scale,
    GROUPS: tl.constexpr, DK: tl.constexpr, DT: tl.constexpr, WITH_QK: tl.constexpr,
):  # fmt: skip
    """Grid (chunks of the longest segment, value heads, segments). With WITH_QK it
    also writes the fused path's decayed QK, row factors and e^Gc_last."""
    cid, hv, seg = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start = tl.load(SB + seg)
    seqlen = tl.load(SE + seg) - start
    c0 = cid * 64
    if c0 >= seqlen:
        return
    gi = (_chunk_base(CCU, seg) + cid).to(tl.int64)
    hk = hv // GROUPS
    r = tl.arange(0, 16)
    ck = tl.arange(0, DK)
    p0, p1, p2, p3 = c0 + r, c0 + 16 + r, c0 + 32 + r, c0 + 48 + r
    m0, m1, m2, m3 = p0 < seqlen, p1 < seqlen, p2 < seqlen, p3 < seqlen
    t0, t1, t2, t3 = start + p0, start + p1, start + p2, start + p3
    k0 = load_rows(K, t0, m0, sk_n, hk * DK, ck)
    k1 = load_rows(K, t1, m1, sk_n, hk * DK, ck)
    k2 = load_rows(K, t2, m2, sk_n, hk * DK, ck)
    k3 = load_rows(K, t3, m3, sk_n, hk * DK, ck)
    rk0, rk1, rk2, rk3 = inv_norm(k0), inv_norm(k1), inv_norm(k2), inv_norm(k3)
    kk00, kk11 = tl.dot(k0, tl.trans(k0)), tl.dot(k1, tl.trans(k1))
    kk22, kk33 = tl.dot(k2, tl.trans(k2)), tl.dot(k3, tl.trans(k3))
    kk10, kk20, kk21 = tl.dot(k1, tl.trans(k0)), tl.dot(k2, tl.trans(k0)), tl.dot(k2, tl.trans(k1))  # fmt: skip
    kk30, kk31, kk32 = tl.dot(k3, tl.trans(k0)), tl.dot(k3, tl.trans(k1)), tl.dot(k3, tl.trans(k2))  # fmt: skip
    _prep_head(
        A, B, ALOG, DTB, T, Q, QK, FAC, EGL, s_ab, HV, sq_n, scale, gi, hv, hk,
        kk00, kk11, kk22, kk33, kk10, kk20, kk21, kk30, kk31, kk32,
        k0, k1, k2, k3, rk0, rk1, rk2, rk3, t0, t1, t2, t3, m0, m1, m2, m3,
        DK, DT, WITH_QK,
    )  # fmt: skip


@triton.jit
def _prep_head(
    A, B, ALOG, DTB, T, Q, QK, FAC, EGL, s_ab, HV, sq_n, scale, gi, hv, hk,
    kk00, kk11, kk22, kk33, kk10, kk20, kk21, kk30, kk31, kk32,
    k0, k1, k2, k3, rk0, rk1, rk2, rk3, t0, t1, t2, t3, m0, m1, m2, m3,
    DK: tl.constexpr, DT: tl.constexpr, WITH_QK: tl.constexpr,
):  # fmt: skip
    """T (and the WITH_QK outputs) of value head `hv` from the raw kk blocks."""
    base = (gi * HV + hv) * 4096
    g0, b0 = gates(A, B, ALOG, DTB, t0, m0, hv, s_ab)
    g1, b1 = gates(A, B, ALOG, DTB, t1, m1, hv, s_ab)
    g2, b2 = gates(A, B, ALOG, DTB, t2, m2, hv, s_ab)
    g3, b3 = gates(A, B, ALOG, DTB, t3, m3, hv, s_ab)
    s1, s2 = tl.sum(g1, 0), tl.sum(g2, 0)
    e0, e1, e2 = suffix_sum(g0, 16)[None, :], suffix_sum(g1, 16)[None, :], suffix_sum(g2, 16)[None, :]  # fmt: skip
    c1 = tl.cumsum(g1, 0)[:, None]
    c2 = tl.cumsum(g2, 0)[:, None]
    c3 = tl.cumsum(g3, 0)[:, None]
    a00 = _a_scale(kk00, rk0, rk0, b0, pair_decay(g0, 16), True)
    a11 = _a_scale(kk11, rk1, rk1, b1, pair_decay(g1, 16), True)
    a22 = _a_scale(kk22, rk2, rk2, b2, pair_decay(g2, 16), True)
    a33 = _a_scale(kk33, rk3, rk3, b3, pair_decay(g3, 16), True)
    a10 = _a_scale(kk10, rk1, rk0, b1, c1 + e0, False)
    a20 = _a_scale(kk20, rk2, rk0, b2, c2 + s1 + e0, False)
    a21 = _a_scale(kk21, rk2, rk1, b2, c2 + e1, False)
    a30 = _a_scale(kk30, rk3, rk0, b3, c3 + (s1 + s2) + e0, False)
    a31 = _a_scale(kk31, rk3, rk1, b3, c3 + s2 + e1, False)
    a32 = _a_scale(kk32, rk3, rk2, b3, c3 + e2, False)
    i00, i11, i22, i33, i10, i20, i21, i30, i31, i32 = invert(
        a00, a11, a22, a33, a10, a20, a21, a30, a31, a32, DT
    )
    zero = tl.zeros([16, 16], tl.float32)
    _put(T, base, 0, 0, i00)
    _put(T, base, 0, 1, zero)
    _put(T, base, 0, 2, zero)
    _put(T, base, 0, 3, zero)
    _put(T, base, 1, 0, i10)
    _put(T, base, 1, 1, i11)
    _put(T, base, 1, 2, zero)
    _put(T, base, 1, 3, zero)
    _put(T, base, 2, 0, i20)
    _put(T, base, 2, 1, i21)
    _put(T, base, 2, 2, i22)
    _put(T, base, 2, 3, zero)
    _put(T, base, 3, 0, i30)
    _put(T, base, 3, 1, i31)
    _put(T, base, 3, 2, i32)
    _put(T, base, 3, 3, i33)
    if WITH_QK:
        _qk_out(
            Q, QK, FAC, EGL, base, gi, HV, hv, sq_n, hk, scale,
            k0, k1, k2, k3, rk0, rk1, rk2, rk3, b0, b1, b2, b3, g0, g1, g2, g3,
            t0, t1, t2, t3, m0, m1, m2, m3, DK,
        )  # fmt: skip


@triton.jit
def state_kernel(
    K, V, A, B, ALOG, DTB, T, H, VN, SB, SE, CCU,
    sk_n, sv_n, s_ab, svn_n, HV,
    GROUPS: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr,
    DT: tl.constexpr,
):  # fmt: skip
    """Grid (segments, value heads, value tiles); serial over the segment's chunks. The
    first chunk runs before the loop (zero start state: no H, no state product); each
    loop step first folds the previous chunk into the state, so the last chunk's end
    state is never formed."""
    seg, hv, vt = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start = tl.load(SB + seg)
    seqlen = tl.load(SE + seg) - start
    base = _chunk_base(CCU, seg)
    hk = hv // GROUPS
    r = tl.arange(0, 64)
    ck = tl.arange(0, DK)
    cv = vt * BV + tl.arange(0, BV)
    ok = r < seqlen
    tok = start + r
    k = load_rows(K, tok, ok, sk_n, hk * DK, ck)
    v = load_rows(V, tok, ok, sv_n, hv * DV, cv)
    g, beta = gates(A, B, ALOG, DTB, tok, ok, hv, s_ab)
    rk = inv_norm(k)
    t = tl.load(T + (base.to(tl.int64) * HV + hv) * 4096 + r[:, None] * 64 + r[None, :])
    vn = mma(t, v.to(tl.float32) * beta[:, None], DT)
    tl.store(VN + tok[:, None] * svn_n + hv * DV + cv[None, :], vn.to(VN.dtype.element_ty), mask=ok[:, None])  # fmt: skip
    kd = vn * (rk * tl.exp(suffix_sum(g, 64)))[:, None]
    egl = tl.exp(tl.sum(g, 0))
    state = tl.zeros([DK, BV], tl.float32)
    for c0 in range(64, seqlen, 64):
        state = state * egl + tl.dot(tl.trans(k), kd.to(DT))
        gi = (base + c0 // 64).to(tl.int64)
        h_ptr = H + ((gi * HV + hv) * DK + ck[:, None]) * DV + cv[None, :]
        tl.store(h_ptr, state.to(H.dtype.element_ty))
        ok = c0 + r < seqlen
        tok = start + c0 + r
        k = load_rows(K, tok, ok, sk_n, hk * DK, ck)
        v = load_rows(V, tok, ok, sv_n, hv * DV, cv)
        g, beta = gates(A, B, ALOG, DTB, tok, ok, hv, s_ab)
        rk = inv_norm(k)
        t = tl.load(T + (gi * HV + hv) * 4096 + r[:, None] * 64 + r[None, :])
        ks = tl.dot(k, state.to(DT))
        y = v.to(tl.float32) * beta[:, None]
        y = y - (rk * beta * tl.exp(tl.cumsum(g, 0)))[:, None] * ks
        vn = mma(t, y, DT)
        vn_ptr = VN + tok[:, None] * svn_n + hv * DV + cv[None, :]
        tl.store(vn_ptr, vn.to(VN.dtype.element_ty), mask=ok[:, None])
        kd = vn * (rk * tl.exp(suffix_sum(g, 64)))[:, None]
        egl = tl.exp(tl.sum(g, 0))


@triton.jit
def fused_kernel(
    Q, K, V, T, QK, FAC, EGL, SB, SE, CCU, OUT,
    sq_n, sk_n, sv_n, so_n, HV,
    GROUPS: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr,
    DT: tl.constexpr,
):  # fmt: skip
    """Grid (segments, value heads, value tiles); serial over chunks with T, decayed QK
    and the row factors from `prep_kernel` (WITH_QK); every chunk's output is made in
    the same pass, so no chunk state or V' goes through memory. A segment's first
    chunk skips the state products (the state is zero there)."""
    seg, hv, vt = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start = tl.load(SB + seg)
    seqlen = tl.load(SE + seg) - start
    base = _chunk_base(CCU, seg)
    hk = hv // GROUPS
    r = tl.arange(0, 64)
    ck = tl.arange(0, DK)
    cv = vt * BV + tl.arange(0, BV)
    state = tl.zeros([DK, BV], tl.float32)
    for c0 in range(0, seqlen, 64):
        gi = (base + c0 // 64).to(tl.int64)
        ok = c0 + r < seqlen
        tok = start + c0 + r
        fac = FAC + (tok.to(tl.int64) * HV + hv) * 4
        beta = tl.load(fac, mask=ok, other=0.0)
        fy = tl.load(fac + 1, mask=ok, other=0.0)
        fq = tl.load(fac + 2, mask=ok, other=0.0)
        fk = tl.load(fac + 3, mask=ok, other=0.0)
        k = load_rows(K, tok, ok, sk_n, hk * DK, ck)
        v = load_rows(V, tok, ok, sv_n, hv * DV, cv)
        mat = (gi * HV + hv) * 4096 + r[:, None] * 64 + r[None, :]
        y = v.to(tl.float32) * beta[:, None]
        if c0 > 0:
            y = y - fy[:, None] * tl.dot(k, state.to(DT))
        vn = mma(tl.load(T + mat), y, DT)
        o = mma(tl.load(QK + mat), vn, DT)
        if c0 > 0:
            q = load_rows(Q, tok, ok, sq_n, hk * DK, ck)
            o = o + fq[:, None] * tl.dot(q, state.to(DT))
        tl.store(
            OUT + tok[:, None] * so_n + hv * DV + cv[None, :],
            o.to(OUT.dtype.element_ty),
            mask=ok[:, None],
        )
        if c0 + 64 < seqlen:
            kd = vn * fk[:, None]
            egl = tl.load(EGL + gi * HV + hv)
            state = state * egl + tl.dot(tl.trans(k), kd.to(DT))


@triton.jit
def out_kernel(
    Q, K, A, B, ALOG, DTB, H, VN, SB, SE, CCU, OUT, scale,
    sq_n, sk_n, s_ab, svn_n, so_n, HV, NVT,
    GROUPS: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr,
    DT: tl.constexpr,
):  # fmt: skip
    """Grid (chunks of the longest segment, value heads x value tiles, segments); a
    segment's first chunk skips the start-state product (H is zero there)."""
    cid, hvt, seg = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start = tl.load(SB + seg)
    seqlen = tl.load(SE + seg) - start
    c0 = cid * 64
    if c0 >= seqlen:
        return
    hv, vt = hvt // NVT, hvt % NVT
    gi = (_chunk_base(CCU, seg) + cid).to(tl.int64)
    hk = hv // GROUPS
    r = tl.arange(0, 64)
    ck = tl.arange(0, DK)
    cv = vt * BV + tl.arange(0, BV)
    ok = c0 + r < seqlen
    tok = start + c0 + r
    q = load_rows(Q, tok, ok, sq_n, hk * DK, ck)
    k = load_rows(K, tok, ok, sk_n, hk * DK, ck)
    rq = inv_norm(q) * scale
    rk = inv_norm(k)
    g, _beta = gates(A, B, ALOG, DTB, tok, ok, hv, s_ab)
    vn = tl.load(VN + tok[:, None] * svn_n + hv * DV + cv[None, :], mask=ok[:, None], other=0.0)  # fmt: skip
    causal = r[:, None] >= r[None, :]
    qk = tl.dot(q, tl.trans(k)) * rq[:, None] * rk[None, :]
    qk = tl.where(causal, qk * tl.exp(pair_decay(g, 64)), 0.0)
    o = mma(qk, vn, DT)
    if cid > 0:
        h = tl.load(H + ((gi * HV + hv) * DK + ck[:, None]) * DV + cv[None, :])
        o = o + (rq * tl.exp(tl.cumsum(g, 0)))[:, None] * tl.dot(q, h.to(DT))
    tl.store(
        OUT + tok[:, None] * so_n + hv * DV + cv[None, :],
        o.to(OUT.dtype.element_ty),
        mask=ok[:, None],
    )


@triton.jit
def recurrent_kernel(
    Q, K, V, A, B, ALOG, DTB, OUT, SB, SE, scale,
    sq_n, sk_n, sv_n, s_ab, so_n,
    GROUPS: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr,
):  # fmt: skip
    """Grid (segments, value heads, value tiles); one token per iteration, fp32 state
    and vector math only. o_t = e^g S^T q + delta (k . q) with S the state before t."""
    seg, hv, vt = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start = tl.load(SB + seg)
    end = tl.load(SE + seg)
    hk = hv // GROUPS
    ck = tl.arange(0, DK)
    cv = vt * BV + tl.arange(0, BV)
    neg_a = -tl.exp(tl.load(ALOG + hv).to(tl.float32))
    dtb = tl.load(DTB + hv).to(tl.float32)
    state = tl.zeros([DK, BV], tl.float32)
    for t in range(start, end):
        k = tl.load(K + t * sk_n + hk * DK + ck).to(tl.float32)
        q = tl.load(Q + t * sq_n + hk * DK + ck).to(tl.float32)
        v = tl.load(V + t * sv_n + hv * DV + cv).to(tl.float32)
        x = tl.load(A + t * s_ab + hv).to(tl.float32) + dtb
        e = tl.exp(neg_a * tl.where(x > 20.0, x, libdevice.log1p(libdevice.exp(x))))
        beta = 1.0 / (1.0 + tl.exp(-tl.load(B + t * s_ab + hv).to(tl.float32)))
        k = k / tl.sqrt_rn(tl.sum(k * k, 0) + 1e-6)
        q = q * (scale / tl.sqrt_rn(tl.sum(q * q, 0) + 1e-6))
        sk = tl.sum(state * k[:, None], 0)
        sq = tl.sum(state * q[:, None], 0)
        delta = (v - e * sk) * beta
        o = e * sq + delta * tl.sum(k * q, 0)
        state = state * e + k[:, None] * delta[None, :]
        tl.store(OUT + t * so_n + hv * DV + cv, o.to(OUT.dtype.element_ty))
