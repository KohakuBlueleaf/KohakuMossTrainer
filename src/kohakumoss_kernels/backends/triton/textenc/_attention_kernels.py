"""Causal GQA varlen attention forward in Triton (the text encoder's full-attention
layers). Grid (query blocks of the longest segment, query heads, segments); query head
h reads kv head h // GROUPS; key blocks past the query block's last row are skipped,
the diagonal block is masked. fp32 accumulator and row statistics, probabilities
rounded to the input dtype for the PV product, one rounding at the store.
"""

import triton
import triton.language as tl


@triton.jit
def _tile(
    acc, m_i, l_i, q, k_ptrs, v_ptrs, rows, cols, seqlen, c,
    MASKED: tl.constexpr, PREC: tl.constexpr,
):  # fmt: skip
    if MASKED:
        ok = cols < seqlen
        k = tl.load(k_ptrs, mask=ok[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=ok[:, None], other=0.0)
    else:
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
    s = tl.dot(q, tl.trans(k), input_precision=PREC)
    if MASKED:
        s = tl.where((cols[None, :] <= rows[:, None]) & (cols[None, :] < seqlen), s, float("-inf"))  # fmt: skip
    m_new = tl.maximum(m_i, tl.max(s, 1))
    alpha = tl.exp2((m_i - m_new) * c)
    p = tl.exp2((s - m_new[:, None]) * c)
    acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None], input_precision=PREC)
    l_i = l_i * alpha + tl.sum(p, 1)
    return acc, m_new, l_i


@triton.jit
def causal_fwd_kernel(
    Q, K, V, OUT, CU, c,
    sq_n, sq_h, sk_n, sk_h, sv_n, sv_h, so_n, so_h,
    GROUPS: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
    PREC: tl.constexpr,
):  # fmt: skip
    pid, head, seg = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start = tl.load(CU + seg)
    seqlen = tl.load(CU + seg + 1) - start
    if pid * BM >= seqlen:
        return
    c = c.to(tl.float32)
    kvh = head // GROUPS
    rows = pid * BM + tl.arange(0, BM)
    cd = tl.arange(0, D)
    q = tl.load(
        Q + (start + rows)[:, None] * sq_n + head * sq_h + cd[None, :],
        mask=(rows < seqlen)[:, None],
        other=0.0,
    )
    acc = tl.zeros([BM, D], tl.float32)
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    k_base = K + start * sk_n + kvh * sk_h
    v_base = V + start * sv_n + kvh * sv_h
    lo = pid * BM
    full = tl.minimum(lo // BN * BN, seqlen // BN * BN)
    for n0 in range(0, full, BN):
        cols = n0 + tl.arange(0, BN)
        acc, m_i, l_i = _tile(
            acc, m_i, l_i, q,
            k_base + cols[:, None] * sk_n + cd[None, :],
            v_base + cols[:, None] * sv_n + cd[None, :],
            rows, cols, seqlen, c, False, PREC,
        )  # fmt: skip
    hi = tl.minimum(lo + BM, seqlen)
    for n0 in range(full, hi, BN):
        cols = n0 + tl.arange(0, BN)
        acc, m_i, l_i = _tile(
            acc, m_i, l_i, q,
            k_base + cols[:, None] * sk_n + cd[None, :],
            v_base + cols[:, None] * sv_n + cd[None, :],
            rows, cols, seqlen, c, True, PREC,
        )  # fmt: skip
    o = acc / l_i[:, None]
    tl.store(
        OUT + (start + rows)[:, None] * so_n + head * so_h + cd[None, :],
        o.to(OUT.dtype.element_ty),
        mask=(rows < seqlen)[:, None],
    )
