"""TileLang qk_norm_rope: per-head RMSNorm of q and k plus split-half multi-axis RoPE
with in-kernel sin/cos from positions, forward and backward (fp16 I/O, fp32 math).

Rows are (token, head) pairs. Angles are the fp32 sin/cos of the fp32 phase. The
backward runs one program per BM rows (as the forward) and writes one fp32 partial of
each weight gradient per program, summed afterwards. See docs/internals/fp16.md.
"""

import tilelang.language as T
import torch
from torch import Tensor
from torch.library import custom_op

from kohakumoss_kernels.backends.compile.choice import kernel_or_compile, shape_key
from kohakumoss_kernels.backends.eager import rope as eager_rope
from kohakumoss_kernels.backends.tilelang.blocks._blockops import (
    DTYPE,
    F32,
    FP16_SUPPORTS,
    CompileError,
    compiled,
    total,
    tuned_run,
)
from kohakumoss_kernels.registry import register
from kohakumoss_kernels.tuning.buckets import cdiv, pow2

__all__ = ["qk_norm_rope"]

FWD_CFGS = [
    {"bm": bm, "threads": th}
    for bm, th in ((8, 128), (16, 128), (4, 64), (16, 256), (32, 256))
]
BWD_CFGS = [
    {"bm": bm, "threads": th}
    for bm, th in ((8, 128), (16, 128), (16, 256), (32, 256), (4, 64))
]


@T.macro
def _angle(p, band):
    """(sin, cos) of the fp32 phase p * band (as the eager reference computes it)."""
    ph = p * band
    return T.sin(ph), T.cos(ph)


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


def _bwd_prim(heads, half, nb, axes, BM, THREADS, dtype, dense):
    """Backward, one program per BM rows: dx (a dense [r, 2 half] tensor, or with
    `dense` False a flat buffer with token rows `tok` apart) and the per-program
    weight-gradient partials part [programs, 1, 1, 2 half] (summed over its rows)."""
    r = T.dynamic("r")
    n = T.dynamic("n")
    p = T.dynamic("p")
    ldx = T.dynamic("ldx")
    dx_shape = (r, 2 * half) if dense else (ldx,)

    @T.prim_func
    def main(
        g: T.Tensor((r, 2 * half), dtype),
        x: T.Tensor((r, 2 * half), dtype),
        rstd: T.Tensor((r,), F32),
        w: T.Tensor((2 * half,), F32),
        pos: T.Tensor((n, axes), F32),
        bands: T.Tensor((nb,), F32),
        dx: T.Tensor(dx_shape, dtype),
        part: T.Tensor((p, 1, 1, 2 * half), F32),
        tok: T.int32,
    ):
        with T.Kernel(T.ceildiv(r, BM), threads=THREADS) as bx:
            pa = T.alloc_fragment((BM, half), F32)
            pb = T.alloc_fragment((BM, half), F32)
            da = T.alloc_fragment((BM, half), F32)
            db = T.alloc_fragment((BM, half), F32)
            xa = T.alloc_fragment((BM, half), F32)
            xb = T.alloc_fragment((BM, half), F32)
            prod = T.alloc_fragment((BM, half), F32)
            kk = T.alloc_fragment((BM,), F32)
            rs = T.alloc_fragment((BM,), F32)
            sa = T.alloc_fragment((half,), F32)
            sb = T.alloc_fragment((half,), F32)
            for i in T.Parallel(BM):
                row = bx * BM + i
                rs[i] = T.if_then_else(row < r, rstd[T.min(row, r - 1)], 0.0)
            for i, j in T.Parallel(BM, half):
                row = bx * BM + i
                ok = row < r
                rr = T.min(row, r - 1)
                sn, cs = _angle(pos[rr // heads, j // nb], bands[j % nb])
                ga = T.if_then_else(ok, T.cast(g[rr, j], F32), 0.0)
                gb = T.if_then_else(ok, T.cast(g[rr, j + half], F32), 0.0)
                a32 = T.cast(x[rr, j], F32)
                b32 = T.cast(x[rr, j + half], F32)
                dan = ga * cs + gb * sn
                dbn = gb * cs - ga * sn
                xa[i, j] = a32 * rs[i]
                xb[i, j] = b32 * rs[i]
                pa[i, j] = dan * xa[i, j]
                pb[i, j] = dbn * xb[i, j]
                da[i, j] = dan * w[j]
                db[i, j] = dbn * w[j + half]
                prod[i, j] = da[i, j] * xa[i, j] + db[i, j] * xb[i, j]
            T.reduce_sum(prod, kk, dim=1)
            T.reduce_sum(pa, sa, dim=0)
            T.reduce_sum(pb, sb, dim=0)
            for i, j in T.Parallel(BM, half):
                row = bx * BM + i
                c = kk[i] / (2 * half)
                va = T.cast(rs[i] * (da[i, j] - xa[i, j] * c), dtype)
                vb = T.cast(rs[i] * (db[i, j] - xb[i, j] * c), dtype)
                if row < r:
                    if dense:
                        dx[row, j] = va
                        dx[row, j + half] = vb
                    else:
                        at = (row // heads) * tok + (row % heads) * (2 * half) + j
                        dx[at] = va
                        dx[at + half] = vb
            for j in T.Parallel(half):
                part[bx, 0, 0, j] = sa[j]
                part[bx, 0, 0, j + half] = sb[j]

    return main


def _flat(t: Tensor) -> Tensor:
    return t.reshape(-1, t.shape[-1]).contiguous()


def rope_forward(q, k, qw, kw, pos, bands, eps):
    """(q out, k out, q rstd [N H] fp32, k rstd) of q, k [N, H, Dh] with fp32 weights
    (no autograd); shared with fused_linear."""
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
    dense = tok == h * dh
    out = (
        dx.view(rows, dh)
        if dense
        else dx.as_strided(((n - 1) * tok + h * dh,), (1,), dx.storage_offset())
    )
    meta = (h, dh // 2, bands.shape[0], pos.shape[1])

    def run(c):
        bm = c["bm"]
        part = torch.empty(cdiv(rows, bm), 1, 1, dh, device=dev, dtype=torch.float32)
        kernel = compiled(
            _bwd_prim, dev, *meta, bm, c["threads"], DTYPE[x.dtype], dense
        )
        kernel(_flat(g), _flat(x), rstd, w, pos, bands, out, part, tok)
        return part

    key = f"h{h}-dh{dh}-{x.dtype}-{dense}-r{pow2(rows, 1024)}"
    part = tuned_run("tilelang_qk_rope_bwd", dev, key, BWD_CFGS, run)
    return dx, total(part)[0] if need_w else x.new_empty(0)


def rope_backward(gq, gk, q, k, qr, kr, qw, kw, pos, bands, eps, need_w, dqk=None):
    """(dq, dk, d q_weight, d k_weight; fp32, empty without `need_w`) of q, k [N, H, Dh]
    (no autograd); with `dqk` [N, 2 H Dh] dq and dk are written into its two halves."""
    d = q.shape[1] * q.shape[2]
    dq_out = None if dqk is None else dqk[:, :d]
    dk_out = None if dqk is None else dqk[:, d:]
    dq, dqw = _backward_one(gq, q, qr, qw, pos, bands, eps, need_w, dq_out)
    dk, dkw = _backward_one(gk, k, kr, kw, pos, bands, eps, need_w, dk_out)
    return dq, dk, dqw, dkw


@custom_op("kohakumoss::tilelang_qk_norm_rope_fwd", mutates_args=())
def _fwd_op(
    q: Tensor, k: Tensor, qw: Tensor, kw: Tensor, pos: Tensor, bands: Tensor, eps: float
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return rope_forward(q, k, qw, kw, pos, bands, eps)


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
    return rope_backward(gq, gk, q, k, qr, kr, qw, kw, pos, bands, eps, need_w)


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
    kernel_or_compile("tilelang", "qk_norm_rope", qk_norm_rope, shape_key),
    FP16_SUPPORTS,
)
