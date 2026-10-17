"""Double-float (fp32 pair) arithmetic for TileLang prims, on IEEE round-to-nearest
intrinsics so no product is contracted into an addition. A pair (hi, lo) stands for
hi + lo with |lo| <= ulp(hi) / 2.

The macros return expressions. Arguments that are buffer loads are bound once; pure
expression arguments are substituted, so callers pass loads and store each result in
a fragment before the next pair operation (a chain of substituted pair operations
grows the kernel source geometrically). See docs/internals/fp16.md.
"""

import math

import tilelang.language as T

F32 = "float32"


@T.macro
def two_sum(a, b):
    """(s, e) with s + e == a + b exactly."""
    s = T.ieee_add(a, b)
    bb = T.ieee_sub(s, a)
    return s, T.ieee_add(T.ieee_sub(a, T.ieee_sub(s, bb)), T.ieee_sub(b, bb))


@T.macro
def fast_two_sum(a, b):
    """two_sum for |a| >= |b|."""
    s = T.ieee_add(a, b)
    return s, T.ieee_sub(b, T.ieee_sub(s, a))


@T.macro
def two_prod(a, b):
    """(p, e) with p + e == a * b exactly (FMA)."""
    p = T.ieee_mul(a, b)
    return p, T.ieee_fmaf(a, b, -p)


@T.macro
def df_add(ah, al, bh, bl):
    """(ah + al) + (bh + bl)."""
    s, e = two_sum(ah, bh)
    return fast_two_sum(s, T.ieee_add(e, T.ieee_add(al, bl)))


@T.macro
def df_mul(ah, al, bh, bl):
    """(ah + al) * (bh + bl)."""
    p, e = two_prod(ah, bh)
    e2 = T.ieee_add(e, T.ieee_add(T.ieee_mul(ah, bl), T.ieee_mul(al, bh)))
    return fast_two_sum(p, e2)


@T.macro
def df_scale(a, bh, bl):
    """a * (bh + bl) for an fp32 a."""
    p, e = two_prod(a, bh)
    return fast_two_sum(p, T.ieee_add(e, T.ieee_mul(a, bl)))


@T.macro
def df_div(h, lo, n):
    """(h + lo) / n for an fp32 n."""
    q = T.ieee_fdiv(h, n)
    ql = T.ieee_fdiv(T.ieee_add(T.ieee_fmaf(-q, n, h), lo), n)
    return fast_two_sum(q, ql)


@T.macro
def rstd_into(oh, ol, s0, s1, s2, i, h, lo, eps):
    """(oh[i], ol[i]) = 1 / sqrt(h + lo + eps) for loads h, lo: fp32 seed and one Newton
    step in pairs; s0, s1, s2 are scratch buffers of oh's shape."""
    qh, ql = df_add(h, lo, eps, T.float32(0))
    s0[i] = qh
    s1[i] = ql
    s2[i] = T.ieee_fdiv(T.float32(1), T.ieee_fsqrt(s0[i]))
    ph, pl = two_prod(s2[i], s2[i])
    oh[i] = ph
    ol[i] = pl
    th, tlo = df_mul(s0[i], s1[i], oh[i], ol[i])
    s0[i] = th
    s1[i] = tlo
    e = T.ieee_sub(T.ieee_sub(T.float32(1), s0[i]), s1[i])
    rh, rl = two_sum(s2[i], T.ieee_mul(T.ieee_mul(T.float32(0.5), s2[i]), e))
    oh[i] = rh
    ol[i] = rl


def extract_bits(n: int) -> float:
    """Exponent margin K of `split_hi` for sums of n terms (2**K > 2 n)."""
    return float(math.ceil(math.log2(max(n, 1))) + 2)


@T.macro
def split_sigma(mx, k):
    """The power of two 2**(ceil(log2 mx) + k) that `split_hi` splits against."""
    return T.exp2(T.ceil(T.log2(mx)) + k)


@T.macro
def split_hi(v, sigma):
    """The high part of v on the grid of sigma: sums of up to n such parts (|v| <=
    sigma / 2**K) are exact in fp32; v - hi is exact and below ulp(sigma)."""
    return T.ieee_sub(T.ieee_add(v, sigma), sigma)
