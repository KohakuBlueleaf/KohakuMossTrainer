"""Gated delta rule for segments of at most 64 tokens, several per 64-row tile.

A tile `[ts, te)` holds whole consecutive segments; the state is zero at every segment
start, so a tile's output is in-tile only: o = (Q K^T . decay . same-segment) V' with
V' = T (beta v) and T = (I + A)^-1, A masked to same-segment pairs. One program per
(tile, value head) writes every value column; no state or V' leaves the program.
Products are DT-operand MMAs with fp32 accumulation, as in the chunk kernels. See
docs/internals/text-encoder.md.
"""

import triton
import triton.language as tl

from kohakumoss_kernels.backends.triton.textenc._delta_kernels import (
    a_block,
    gates,
    inv_norm,
    invert,
    load_rows,
    mma,
    pair_decay,
    qk_block,
    suffix_sum,
)


@triton.jit
def _seg(SEG, tok, ok):
    return tl.load(SEG + tok, mask=ok, other=-1)


@triton.jit
def _same(si, sj):
    return si[:, None] == sj[None, :]


@triton.jit
def _rows_out(
    V, OUT, sv_n, so_n, hv,
    i00, i11, i22, i33, i10, i20, i21, i30, i31, i32,
    q00, q11, q22, q33, q10, q20, q21, q30, q31, q32,
    t0, t1, t2, t3, m0, m1, m2, m3, b0, b1, b2, b3, c,
    DV: tl.constexpr, BC: tl.constexpr, DT: tl.constexpr,
):  # fmt: skip
    """Output columns [c, c + BC) of the tile: V' = T (beta v), o = QK V'."""
    cols = c + tl.arange(0, BC)
    off = hv * DV
    y0 = load_rows(V, t0, m0, sv_n, off, cols).to(tl.float32) * b0[:, None]
    y1 = load_rows(V, t1, m1, sv_n, off, cols).to(tl.float32) * b1[:, None]
    y2 = load_rows(V, t2, m2, sv_n, off, cols).to(tl.float32) * b2[:, None]
    y3 = load_rows(V, t3, m3, sv_n, off, cols).to(tl.float32) * b3[:, None]
    v0 = mma(i00, y0, DT)
    v1 = mma(i10, y0, DT) + mma(i11, y1, DT)
    v2 = mma(i20, y0, DT) + mma(i21, y1, DT) + mma(i22, y2, DT)
    v3 = mma(i30, y0, DT) + mma(i31, y1, DT) + mma(i32, y2, DT) + mma(i33, y3, DT)
    o0 = mma(q00, v0, DT)
    o1 = mma(q10, v0, DT) + mma(q11, v1, DT)
    o2 = mma(q20, v0, DT) + mma(q21, v1, DT) + mma(q22, v2, DT)
    o3 = mma(q30, v0, DT) + mma(q31, v1, DT) + mma(q32, v2, DT) + mma(q33, v3, DT)
    base = OUT + off + cols[None, :]
    ty = OUT.dtype.element_ty
    tl.store(base + t0[:, None] * so_n, o0.to(ty), mask=m0[:, None])
    tl.store(base + t1[:, None] * so_n, o1.to(ty), mask=m1[:, None])
    tl.store(base + t2[:, None] * so_n, o2.to(ty), mask=m2[:, None])
    tl.store(base + t3[:, None] * so_n, o3.to(ty), mask=m3[:, None])


@triton.jit
def short_kernel(
    Q, K, V, A, B, ALOG, DTB, TILES, SEG, OUT, scale,
    sq_n, sk_n, sv_n, s_ab, so_n,
    GROUPS: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, BC: tl.constexpr,
    DT: tl.constexpr,
):  # fmt: skip
    """Grid (tiles, value heads); `TILES` int32 [tiles, 2] token ranges, `SEG` int32
    [N] each token's segment."""
    pid, hv = tl.program_id(0), tl.program_id(1)
    ts = tl.load(TILES + 2 * pid)
    te = tl.load(TILES + 2 * pid + 1)
    hk = hv // GROUPS
    r = tl.arange(0, 16)
    ck = tl.arange(0, DK)
    t0, t1, t2, t3 = ts + r, ts + 16 + r, ts + 32 + r, ts + 48 + r
    m0, m1, m2, m3 = t0 < te, t1 < te, t2 < te, t3 < te
    s0, s1, s2, s3 = _seg(SEG, t0, m0), _seg(SEG, t1, m1), _seg(SEG, t2, m2), _seg(SEG, t3, m3)  # fmt: skip
    k0 = load_rows(K, t0, m0, sk_n, hk * DK, ck)
    k1 = load_rows(K, t1, m1, sk_n, hk * DK, ck)
    k2 = load_rows(K, t2, m2, sk_n, hk * DK, ck)
    k3 = load_rows(K, t3, m3, sk_n, hk * DK, ck)
    q0 = load_rows(Q, t0, m0, sq_n, hk * DK, ck)
    q1 = load_rows(Q, t1, m1, sq_n, hk * DK, ck)
    q2 = load_rows(Q, t2, m2, sq_n, hk * DK, ck)
    q3 = load_rows(Q, t3, m3, sq_n, hk * DK, ck)
    rk0, rk1, rk2, rk3 = inv_norm(k0), inv_norm(k1), inv_norm(k2), inv_norm(k3)
    rq0, rq1 = inv_norm(q0) * scale, inv_norm(q1) * scale
    rq2, rq3 = inv_norm(q2) * scale, inv_norm(q3) * scale
    g0, b0 = gates(A, B, ALOG, DTB, t0, m0, hv, s_ab)
    g1, b1 = gates(A, B, ALOG, DTB, t1, m1, hv, s_ab)
    g2, b2 = gates(A, B, ALOG, DTB, t2, m2, hv, s_ab)
    g3, b3 = gates(A, B, ALOG, DTB, t3, m3, hv, s_ab)
    u1, u2 = tl.sum(g1, 0), tl.sum(g2, 0)
    e0, e1, e2 = suffix_sum(g0, 16)[None, :], suffix_sum(g1, 16)[None, :], suffix_sum(g2, 16)[None, :]  # fmt: skip
    c1 = tl.cumsum(g1, 0)[:, None]
    c2 = tl.cumsum(g2, 0)[:, None]
    c3 = tl.cumsum(g3, 0)[:, None]
    d00, d11 = pair_decay(g0, 16), pair_decay(g1, 16)
    d22, d33 = pair_decay(g2, 16), pair_decay(g3, 16)
    d10, d20, d21 = c1 + e0, c2 + u1 + e0, c2 + e1
    d30, d31, d32 = c3 + (u1 + u2) + e0, c3 + u2 + e1, c3 + e2
    a00 = tl.where(_same(s0, s0), a_block(k0, k0, rk0, rk0, b0, d00, True), 0.0)
    a11 = tl.where(_same(s1, s1), a_block(k1, k1, rk1, rk1, b1, d11, True), 0.0)
    a22 = tl.where(_same(s2, s2), a_block(k2, k2, rk2, rk2, b2, d22, True), 0.0)
    a33 = tl.where(_same(s3, s3), a_block(k3, k3, rk3, rk3, b3, d33, True), 0.0)
    a10 = tl.where(_same(s1, s0), a_block(k1, k0, rk1, rk0, b1, d10, False), 0.0)
    a20 = tl.where(_same(s2, s0), a_block(k2, k0, rk2, rk0, b2, d20, False), 0.0)
    a21 = tl.where(_same(s2, s1), a_block(k2, k1, rk2, rk1, b2, d21, False), 0.0)
    a30 = tl.where(_same(s3, s0), a_block(k3, k0, rk3, rk0, b3, d30, False), 0.0)
    a31 = tl.where(_same(s3, s1), a_block(k3, k1, rk3, rk1, b3, d31, False), 0.0)
    a32 = tl.where(_same(s3, s2), a_block(k3, k2, rk3, rk2, b3, d32, False), 0.0)
    i00, i11, i22, i33, i10, i20, i21, i30, i31, i32 = invert(
        a00, a11, a22, a33, a10, a20, a21, a30, a31, a32, DT
    )
    q00 = tl.where(_same(s0, s0), qk_block(q0, k0, rq0, rk0, d00, True), 0.0)
    q11 = tl.where(_same(s1, s1), qk_block(q1, k1, rq1, rk1, d11, True), 0.0)
    q22 = tl.where(_same(s2, s2), qk_block(q2, k2, rq2, rk2, d22, True), 0.0)
    q33 = tl.where(_same(s3, s3), qk_block(q3, k3, rq3, rk3, d33, True), 0.0)
    q10 = tl.where(_same(s1, s0), qk_block(q1, k0, rq1, rk0, d10, False), 0.0)
    q20 = tl.where(_same(s2, s0), qk_block(q2, k0, rq2, rk0, d20, False), 0.0)
    q21 = tl.where(_same(s2, s1), qk_block(q2, k1, rq2, rk1, d21, False), 0.0)
    q30 = tl.where(_same(s3, s0), qk_block(q3, k0, rq3, rk0, d30, False), 0.0)
    q31 = tl.where(_same(s3, s1), qk_block(q3, k1, rq3, rk1, d31, False), 0.0)
    q32 = tl.where(_same(s3, s2), qk_block(q3, k2, rq3, rk2, d32, False), 0.0)
    for c in tl.static_range(0, DV, BC):
        _rows_out(
            V, OUT, sv_n, so_n, hv,
            i00, i11, i22, i33, i10, i20, i21, i30, i31, i32,
            q00, q11, q22, q33, q10, q20, q21, q30, q31, q32,
            t0, t1, t2, t3, m0, m1, m2, m3, b0, b1, b2, b3, c, DV, BC, DT,
        )  # fmt: skip
