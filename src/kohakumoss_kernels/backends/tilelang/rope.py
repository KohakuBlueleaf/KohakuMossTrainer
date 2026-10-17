"""TileLang qk_norm_rope: per-head RMSNorm of q and k plus split-half multi-axis RoPE
with in-kernel sin/cos from positions, forward and backward (fp16 I/O, fp32 math).

Rows are (token, head) pairs. Angles are the fp32 sin/cos of the rounded phase plus the
first-order term of its rounding error. The backward program walks CHUNK rows and writes
one fp32-pair partial of each weight gradient per program row lane (double-float in the
`df` reduce mode), pair-summed afterwards. See docs/internals/fp16.md.
"""

import tilelang.language as T
import torch
from torch import Tensor
from torch.library import custom_op

from kohakumoss_kernels.backends.compile.race import raced, shape_key
from kohakumoss_kernels.backends.eager import rope as eager_rope
from kohakumoss_kernels.backends.tilelang._blockops import (
    DTYPE,
    CompileError,
    cdiv,
    compiled,
    total,
    tuned_run,
)
from kohakumoss_kernels.backends.tilelang._df import (
    df_add,
    df_div,
    df_scale,
    extract_bits,
    rstd_into,
    split_hi,
    split_sigma,
    two_prod,
    two_sum,
)
from kohakumoss_kernels.constraints import requires
from kohakumoss_kernels.hardware.device import probe
from kohakumoss_kernels.reduce import DF, MODE
from kohakumoss_kernels.registry import register

__all__ = ["qk_norm_rope"]

F32 = "float32"
FWD_CFGS = [
    {"bm": bm, "threads": th}
    for bm, th in ((8, 128), (16, 128), (4, 64), (16, 256), (32, 256))
]
BWD_CFGS = [
    {"bm": bm, "threads": th, "per_sm": ps}
    for bm, th, ps in ((8, 128, 4), (16, 128, 4), (8, 128, 8), (16, 256, 2), (4, 64, 8))
]


@T.macro
def _angle(p, band):
    """(sin, cos) of the exact phase p * band."""
    ph, pl = two_prod(p, band)
    s = T.sin(ph)
    c = T.cos(ph)
    return T.ieee_fmaf(c, pl, s), T.ieee_fmaf(-s, pl, c)


def _fwd_prim(heads, half, nb, axes, BM, THREADS, dtype):
    r = T.dynamic("r")
    n = T.dynamic("n")

    @T.prim_func
    def main(
        q: T.Tensor((r, 2 * half), dtype),
        k: T.Tensor((r, 2 * half), dtype),
        qw: T.Tensor((2 * half,), F32),
        kw: T.Tensor((2 * half,), F32),
        pos: T.Tensor((n, axes), F32),
        bands: T.Tensor((nb,), F32),
        qo: T.Tensor((r, 2 * half), dtype),
        ko: T.Tensor((r, 2 * half), dtype),
        qr: T.Tensor((r,), F32),
        kr: T.Tensor((r,), F32),
        eps: T.float32,
    ):
        with T.Kernel(T.ceildiv(r, BM), threads=THREADS) as bx:
            sn = T.alloc_fragment((BM, half), F32)
            cs = T.alloc_fragment((BM, half), F32)
            qa = T.alloc_fragment((BM, half), F32)
            qb = T.alloc_fragment((BM, half), F32)
            ka = T.alloc_fragment((BM, half), F32)
            kb = T.alloc_fragment((BM, half), F32)
            sq = T.alloc_fragment((BM, half), F32)
            sk = T.alloc_fragment((BM, half), F32)
            ssq = T.alloc_fragment((BM,), F32)
            ssk = T.alloc_fragment((BM,), F32)
            for i, j in T.Parallel(BM, half):
                row = T.min(bx * BM + i, r - 1)
                s, c = _angle(pos[row // heads, j // nb], bands[j % nb])
                sn[i, j] = s
                cs[i, j] = c
                qa[i, j] = T.cast(q[row, j], F32)
                qb[i, j] = T.cast(q[row, j + half], F32)
                ka[i, j] = T.cast(k[row, j], F32)
                kb[i, j] = T.cast(k[row, j + half], F32)
                sq[i, j] = qa[i, j] * qa[i, j] + qb[i, j] * qb[i, j]
                sk[i, j] = ka[i, j] * ka[i, j] + kb[i, j] * kb[i, j]
            T.reduce_sum(sq, ssq, dim=1)
            T.reduce_sum(sk, ssk, dim=1)
            for i in T.Parallel(BM):
                ssq[i] = 1.0 / T.sqrt(ssq[i] / (2 * half) + eps)
                ssk[i] = 1.0 / T.sqrt(ssk[i] / (2 * half) + eps)
            for i in T.Parallel(BM):
                if bx * BM + i < r:
                    qr[bx * BM + i] = ssq[i]
                    kr[bx * BM + i] = ssk[i]
            for i, j in T.Parallel(BM, half):
                row = bx * BM + i
                an = qa[i, j] * ssq[i] * qw[j]
                bn = qb[i, j] * ssq[i] * qw[j + half]
                ak = ka[i, j] * ssk[i] * kw[j]
                bk = kb[i, j] * ssk[i] * kw[j + half]
                if row < r:
                    qo[row, j] = T.cast(an * cs[i, j] - bn * sn[i, j], dtype)
                    qo[row, j + half] = T.cast(bn * cs[i, j] + an * sn[i, j], dtype)
                    ko[row, j] = T.cast(ak * cs[i, j] - bk * sn[i, j], dtype)
                    ko[row, j + half] = T.cast(bk * cs[i, j] + ak * sn[i, j], dtype)

    return main


def _bwd_prim(heads, half, nb, axes, chunk, BM, THREADS, dtype, df):
    """Backward: dx into the flat `dx` (token rows `tok` apart) and per-lane fp32-pair
    weight-gradient partials, double-float with `df` (row statistics from an exact
    hi/lo split sum, exact angle products), Kahan otherwise."""
    r = T.dynamic("r")
    n = T.dynamic("n")
    p = T.dynamic("p")
    ldx = T.dynamic("ldx")
    kbits = extract_bits(2 * half)

    @T.prim_func
    def main(
        g: T.Tensor((r, 2 * half), dtype),
        x: T.Tensor((r, 2 * half), dtype),
        rstd: T.Tensor((r,), F32),
        w: T.Tensor((2 * half,), F32),
        pos: T.Tensor((n, axes), F32),
        bands: T.Tensor((nb,), F32),
        dx: T.Tensor((ldx,), dtype),
        part: T.Tensor((p, BM, 1, 2, 2 * half), F32),
        eps: T.float32,
        tok: T.int32,
    ):
        with T.Kernel(p, threads=THREADS) as bx:
            acc_a = T.alloc_fragment((BM, half), F32)
            acc_b = T.alloc_fragment((BM, half), F32)
            c_a = T.alloc_fragment((BM, half), F32)
            c_b = T.alloc_fragment((BM, half), F32)
            da = T.alloc_fragment((BM, half), F32)
            db = T.alloc_fragment((BM, half), F32)
            xa = T.alloc_fragment((BM, half), F32)
            xb = T.alloc_fragment((BM, half), F32)
            prod = T.alloc_fragment((BM, half), F32)
            sq = T.alloc_fragment((BM, half), F32)
            sl = T.alloc_fragment((BM, half), F32)
            kk = T.alloc_fragment((BM,), F32)
            rs = T.alloc_fragment((BM,), F32)
            mx = T.alloc_fragment((BM,), F32)
            shs = T.alloc_fragment((BM,), F32)
            sls = T.alloc_fragment((BM,), F32)
            rh = T.alloc_fragment((BM,), F32)
            rl = T.alloc_fragment((BM,), F32)
            snf = T.alloc_fragment((BM, half), F32)
            csf = T.alloc_fragment((BM, half), F32)
            cr_h = T.alloc_fragment((BM, half), F32)
            cr_l = T.alloc_fragment((BM, half), F32)
            sr_h = T.alloc_fragment((BM, half), F32)
            sr_l = T.alloc_fragment((BM, half), F32)
            pl_a = T.alloc_fragment((BM, half), F32)
            t0 = T.alloc_fragment((BM,), F32)
            t1 = T.alloc_fragment((BM,), F32)
            t2 = T.alloc_fragment((BM,), F32)
            t3 = T.alloc_fragment((BM,), F32)
            t4 = T.alloc_fragment((BM,), F32)
            T.clear(acc_a)
            T.clear(acc_b)
            T.clear(c_a)
            T.clear(c_b)
            for it in T.serial(chunk // BM):
                for i in T.Parallel(BM):
                    row = bx * chunk + it * BM + i
                    rs[i] = T.if_then_else(row < r, rstd[T.min(row, r - 1)], 0.0)
                for i, j in T.Parallel(BM, half):
                    row = bx * chunk + it * BM + i
                    ok = row < r
                    rr = T.min(row, r - 1)
                    sn, cs = _angle(pos[rr // heads, j // nb], bands[j % nb])
                    snf[i, j] = sn
                    csf[i, j] = cs
                    ga = T.if_then_else(ok, T.cast(g[rr, j], F32), 0.0)
                    gb = T.if_then_else(ok, T.cast(g[rr, j + half], F32), 0.0)
                    a32 = T.if_then_else(ok, T.cast(x[rr, j], F32), 0.0)
                    b32 = T.if_then_else(ok, T.cast(x[rr, j + half], F32), 0.0)
                    dan = ga * cs + gb * sn
                    dbn = gb * cs - ga * sn
                    xa[i, j] = a32 * rs[i]
                    xb[i, j] = b32 * rs[i]
                    if not df:
                        ya = dan * xa[i, j] - c_a[i, j]
                        ta = acc_a[i, j] + ya
                        c_a[i, j] = (ta - acc_a[i, j]) - ya
                        acc_a[i, j] = ta
                        yb = dbn * xb[i, j] - c_b[i, j]
                        tb = acc_b[i, j] + yb
                        c_b[i, j] = (tb - acc_b[i, j]) - yb
                        acc_b[i, j] = tb
                    da[i, j] = dan * w[j]
                    db[i, j] = dbn * w[j + half]
                    prod[i, j] = da[i, j] * xa[i, j] + db[i, j] * xb[i, j]
                    sq[i, j] = T.max(T.ieee_mul(a32, a32), T.ieee_mul(b32, b32))
                T.reduce_sum(prod, kk, dim=1)
                if df:
                    T.reduce_max(sq, mx, dim=1)
                    for i, j in T.Parallel(BM, half):
                        row = bx * chunk + it * BM + i
                        rr = T.min(row, r - 1)
                        ok = row < r
                        a32 = T.if_then_else(ok, T.cast(x[rr, j], F32), 0.0)
                        b32 = T.if_then_else(ok, T.cast(x[rr, j + half], F32), 0.0)
                        sig = split_sigma(mx[i], kbits)
                        qa = T.ieee_mul(a32, a32)
                        qb = T.ieee_mul(b32, b32)
                        ha = split_hi(qa, sig)
                        hb = split_hi(qb, sig)
                        sq[i, j] = T.ieee_add(ha, hb)
                        sl[i, j] = T.ieee_add(T.ieee_sub(qa, ha), T.ieee_sub(qb, hb))
                    T.reduce_sum(sq, shs, dim=1)
                    T.reduce_sum(sl, sls, dim=1)
                    for i in T.Parallel(BM):
                        sh, sl2 = two_sum(shs[i], sls[i])
                        t0[i] = sh
                        t1[i] = sl2
                    for i in T.Parallel(BM):
                        qh, ql = df_div(t0[i], t1[i], T.float32(2 * half))
                        t2[i] = qh
                        t3[i] = ql
                    for i in T.Parallel(BM):
                        rstd_into(rh, rl, t0, t1, t4, i, t2[i], t3[i], eps)
                    for i in T.Parallel(BM):
                        ok = bx * chunk + it * BM + i < r
                        rh[i] = T.if_then_else(ok, rh[i], 0.0)
                        rl[i] = T.if_then_else(ok, rl[i], 0.0)
                    for i, j in T.Parallel(BM, half):
                        crh, crl = df_scale(csf[i, j], rh[i], rl[i])
                        cr_h[i, j] = crh
                        cr_l[i, j] = crl
                        srh, srl = df_scale(snf[i, j], rh[i], rl[i])
                        sr_h[i, j] = srh
                        sr_l[i, j] = srl
                    for i, j in T.Parallel(BM, half):
                        row = bx * chunk + it * BM + i
                        rr = T.min(row, r - 1)
                        ok = row < r
                        ga = T.if_then_else(ok, T.cast(g[rr, j], F32), 0.0)
                        gb = T.if_then_else(ok, T.cast(g[rr, j + half], F32), 0.0)
                        a32 = T.if_then_else(ok, T.cast(x[rr, j], F32), 0.0)
                        b32 = T.if_then_else(ok, T.cast(x[rr, j + half], F32), 0.0)
                        uh, ul = df_scale(T.ieee_mul(ga, a32), cr_h[i, j], cr_l[i, j])
                        sq[i, j] = uh
                        sl[i, j] = ul
                        vh, vl = df_scale(T.ieee_mul(gb, a32), sr_h[i, j], sr_l[i, j])
                        prod[i, j] = vh
                        pl_a[i, j] = vl
                        wh, wl = df_scale(T.ieee_mul(gb, b32), cr_h[i, j], cr_l[i, j])
                        cr_h[i, j] = wh
                        cr_l[i, j] = wl
                        zh, zl = df_scale(-T.ieee_mul(ga, b32), sr_h[i, j], sr_l[i, j])
                        sr_h[i, j] = zh
                        sr_l[i, j] = zl
                    for i, j in T.Parallel(BM, half):
                        th_, tl_ = df_add(sq[i, j], sl[i, j], prod[i, j], pl_a[i, j])
                        sq[i, j] = th_
                        sl[i, j] = tl_
                        uh2, ul2 = df_add(cr_h[i, j], cr_l[i, j], sr_h[i, j], sr_l[i, j])
                        cr_h[i, j] = uh2
                        cr_l[i, j] = ul2
                    for i, j in T.Parallel(BM, half):
                        ah, al = df_add(acc_a[i, j], c_a[i, j], sq[i, j], sl[i, j])
                        acc_a[i, j] = ah
                        c_a[i, j] = al
                        bh, bl = df_add(acc_b[i, j], c_b[i, j], cr_h[i, j], cr_l[i, j])
                        acc_b[i, j] = bh
                        c_b[i, j] = bl

                for i, j in T.Parallel(BM, half):
                    row = bx * chunk + it * BM + i
                    c = kk[i] / (2 * half)
                    at = (row // heads) * tok + (row % heads) * (2 * half) + j
                    if row < r:
                        dx[at] = T.cast(rs[i] * (da[i, j] - xa[i, j] * c), dtype)
                        dx[at + half] = T.cast(rs[i] * (db[i, j] - xb[i, j] * c), dtype)
            for i, j in T.Parallel(BM, half):
                part[bx, i, 0, 0, j] = acc_a[i, j]
                part[bx, i, 0, 0, j + half] = acc_b[i, j]
                if df:
                    part[bx, i, 0, 1, j] = c_a[i, j]
                    part[bx, i, 0, 1, j + half] = c_b[i, j]
                else:
                    part[bx, i, 0, 1, j] = -c_a[i, j]
                    part[bx, i, 0, 1, j + half] = -c_b[i, j]

    return main


def _flat(t: Tensor) -> Tensor:
    return t.reshape(-1, t.shape[-1]).contiguous()


def _forward(q, k, qw, kw, pos, bands, eps):
    n, h, dh = q.shape
    dev = q.device
    qo, ko = torch.empty_like(q), torch.empty_like(k)
    qr = torch.empty(n * h, device=dev, dtype=torch.float32)
    kr = torch.empty_like(qr)
    meta = (h, dh // 2, bands.shape[0], pos.shape[1])
    args = (
        _flat(q),
        _flat(k),
        qw,
        kw,
        pos,
        bands,
        qo.view(n * h, dh),
        ko.view(n * h, dh),
    )

    def run(c):
        kernel = compiled(_fwd_prim, dev, *meta, c["bm"], c["threads"], DTYPE[q.dtype])
        kernel(*args, qr, kr, float(eps))

    tuned_run("tilelang_qk_rope_fwd", dev, f"h{h}-dh{dh}-{q.dtype}", FWD_CFGS, run)
    return qo, ko, qr, kr


def _backward_one(g, x, rstd, w, pos, bands, eps, need_w=True, dx=None):
    """(dx, d weight fp32 or empty without `need_w`) of one of q / k [N, H, Dh]; `dx`
    may be a given [N, >= H Dh] row-strided view (unit column stride) to write into."""
    n, h, dh = x.shape
    dev = x.device
    rows = n * h
    if dx is None:
        dx = torch.empty_like(x)
    tok = dx.stride(0)
    flat = dx.as_strided(((n - 1) * tok + h * dh,), (1,), dx.storage_offset())
    meta = (h, dh // 2, bands.shape[0], pos.shape[1])
    df = DF and need_w

    def run(c):
        bm = c["bm"]
        chunk = max(bm, cdiv(cdiv(rows, c["per_sm"] * max(1, probe(dev).sms)), bm) * bm)
        progs = cdiv(rows, chunk)
        part = torch.empty(progs, bm, 1, 2, dh, device=dev, dtype=torch.float32)
        kernel = compiled(
            _bwd_prim, dev, *meta, chunk, bm, c["threads"], DTYPE[x.dtype], df
        )
        kernel(_flat(g), _flat(x), rstd, w, pos, bands, flat, part, float(eps), tok)
        return part

    key = f"h{h}-dh{dh}-{x.dtype}-r{rows}-{df and MODE}"
    part = tuned_run("tilelang_qk_rope_bwd", dev, key, BWD_CFGS, run)
    return dx, total(part)[0] if need_w else x.new_empty(0)


@custom_op("kohakumoss::tilelang_qk_norm_rope_fwd", mutates_args=())
def _fwd_op(
    q: Tensor, k: Tensor, qw: Tensor, kw: Tensor, pos: Tensor, bands: Tensor, eps: float
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _forward(q, k, qw, kw, pos, bands, eps)


@_fwd_op.register_fake
def _(q, k, qw, kw, pos, bands, eps):
    rows = q.shape[0] * q.shape[1]
    r = q.new_empty(rows, dtype=torch.float32)
    return torch.empty_like(q), torch.empty_like(k), r, torch.empty_like(r)


@custom_op("kohakumoss::tilelang_qk_norm_rope_bwd", mutates_args=())
def _bwd_op(
    gq: Tensor,
    gk: Tensor,
    q: Tensor,
    k: Tensor,
    qr: Tensor,
    kr: Tensor,
    qw: Tensor,
    kw: Tensor,
    pos: Tensor,
    bands: Tensor,
    eps: float,
    need_w: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    dq, dqw = _backward_one(gq, q, qr, qw, pos, bands, eps, need_w)
    dk, dkw = _backward_one(gk, k, kr, kw, pos, bands, eps, need_w)
    return dq, dk, dqw, dkw


@_bwd_op.register_fake
def _(gq, gk, q, k, qr, kr, qw, kw, pos, bands, eps, need_w):
    w = q.new_empty(q.shape[2] if need_w else 0, dtype=torch.float32)
    return torch.empty_like(q), torch.empty_like(k), w, torch.empty_like(w)


def _setup(ctx, inputs, output):
    q, k, qw, kw, pos, bands, eps = inputs
    ctx.save_for_backward(q, k, output[2], output[3], qw, kw, pos, bands)
    ctx.eps = eps


def _backward(ctx, gqo, gko, _gqr, _gkr):
    q, k, qr, kr, qw, kw, pos, bands = ctx.saved_tensors
    gq = torch.zeros_like(q) if gqo is None else gqo.contiguous()
    gk = torch.zeros_like(k) if gko is None else gko.contiguous()
    need = ctx.needs_input_grad
    dq, dk, dqw, dkw = _bwd_op(
        gq, gk, q, k, qr, kr, qw, kw, pos, bands, ctx.eps, need[2] or need[3]
    )
    return dq, dk, dqw if need[2] else None, dkw if need[3] else None, None, None, None


_fwd_op.register_autograd(_backward, setup_context=_setup)


class _Cast(torch.autograd.Function):
    """fp32 view of a small parameter whose gradient returns in its own dtype."""

    @staticmethod
    def forward(ctx, t):
        ctx.dtype = t.dtype
        return t.to(torch.float32).contiguous()

    @staticmethod
    def backward(ctx, g):
        return g.to(ctx.dtype)


def qk_norm_rope(
    q: Tensor,
    k: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    pos: Tensor,
    bands: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """q, k [N, H, Dh]: RMSNorm over Dh with weight, then rotate pairs (j, j + Dh/2)."""
    half = q.shape[-1] // 2
    if q.shape[-1] % 2 or pos.shape[1] * bands.shape[0] != half:
        return eager_rope.qk_norm_rope(q, k, q_weight, k_weight, pos, bands, eps)
    try:
        qo, ko, _, _ = _fwd_op(
            q.contiguous(),
            k.contiguous(),
            _Cast.apply(q_weight),
            _Cast.apply(k_weight),
            pos.float().contiguous(),
            bands.float().contiguous(),
            eps,
        )
    except CompileError:
        return eager_rope.qk_norm_rope(q, k, q_weight, k_weight, pos, bands, eps)
    return qo, ko


register(
    "qk_norm_rope",
    "tilelang",
    raced("tilelang", "qk_norm_rope", qk_norm_rope, shape_key),
    requires(kinds=("cuda",), min_sm=70, dtypes=(torch.float16,)),
)
