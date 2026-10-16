"""TileLang qk_norm_rope: per-head RMSNorm of q and k plus split-half multi-axis RoPE
with in-kernel sin/cos from positions, forward and backward (fp16 I/O, fp32 math).

Rows are (token, head) pairs. The backward program walks CHUNK rows and writes one
fp32 partial of each weight gradient per program row lane, summed in torch.
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
    tuned_run,
)
from kohakumoss_kernels.constraints import requires
from kohakumoss_kernels.hardware.device import probe
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
                ph = pos[row // heads, j // nb] * bands[j % nb]
                sn[i, j] = T.sin(ph)
                cs[i, j] = T.cos(ph)
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


def _bwd_prim(heads, half, nb, axes, chunk, BM, THREADS, dtype):
    r = T.dynamic("r")
    n = T.dynamic("n")
    p = T.dynamic("p")

    @T.prim_func
    def main(
        g: T.Tensor((r, 2 * half), dtype),
        x: T.Tensor((r, 2 * half), dtype),
        rstd: T.Tensor((r,), F32),
        w: T.Tensor((2 * half,), F32),
        pos: T.Tensor((n, axes), F32),
        bands: T.Tensor((nb,), F32),
        dx: T.Tensor((r, 2 * half), dtype),
        part: T.Tensor((p, BM, 2 * half), F32),
    ):
        with T.Kernel(p, threads=THREADS) as bx:
            acc_a = T.alloc_fragment((BM, half), F32)
            acc_b = T.alloc_fragment((BM, half), F32)
            da = T.alloc_fragment((BM, half), F32)
            db = T.alloc_fragment((BM, half), F32)
            xa = T.alloc_fragment((BM, half), F32)
            xb = T.alloc_fragment((BM, half), F32)
            prod = T.alloc_fragment((BM, half), F32)
            kk = T.alloc_fragment((BM,), F32)
            rs = T.alloc_fragment((BM,), F32)
            T.clear(acc_a)
            T.clear(acc_b)
            for it in T.serial(chunk // BM):
                for i in T.Parallel(BM):
                    row = bx * chunk + it * BM + i
                    rs[i] = T.if_then_else(row < r, rstd[T.min(row, r - 1)], 0.0)
                for i, j in T.Parallel(BM, half):
                    row = bx * chunk + it * BM + i
                    ok = row < r
                    rr = T.min(row, r - 1)
                    ph = pos[rr // heads, j // nb] * bands[j % nb]
                    sn = T.sin(ph)
                    cs = T.cos(ph)
                    ga = T.if_then_else(ok, T.cast(g[rr, j], F32), 0.0)
                    gb = T.if_then_else(ok, T.cast(g[rr, j + half], F32), 0.0)
                    dan = ga * cs + gb * sn
                    dbn = gb * cs - ga * sn
                    xa[i, j] = T.if_then_else(ok, T.cast(x[rr, j], F32), 0.0) * rs[i]
                    xb[i, j] = (
                        T.if_then_else(ok, T.cast(x[rr, j + half], F32), 0.0) * rs[i]
                    )
                    acc_a[i, j] += dan * xa[i, j]
                    acc_b[i, j] += dbn * xb[i, j]
                    da[i, j] = dan * w[j]
                    db[i, j] = dbn * w[j + half]
                    prod[i, j] = da[i, j] * xa[i, j] + db[i, j] * xb[i, j]
                T.reduce_sum(prod, kk, dim=1)
                for i, j in T.Parallel(BM, half):
                    row = bx * chunk + it * BM + i
                    c = kk[i] / (2 * half)
                    if row < r:
                        dx[row, j] = T.cast(rs[i] * (da[i, j] - xa[i, j] * c), dtype)
                        dx[row, j + half] = T.cast(
                            rs[i] * (db[i, j] - xb[i, j] * c), dtype
                        )
            for i, j in T.Parallel(BM, half):
                part[bx, i, j] = acc_a[i, j]
                part[bx, i, j + half] = acc_b[i, j]

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

    tuned_run("f6tl_qk_rope_fwd", dev, f"h{h}-dh{dh}-{q.dtype}", FWD_CFGS, run)
    return qo, ko, qr, kr


def _backward_one(g, x, rstd, w, pos, bands):
    n, h, dh = x.shape
    dev = x.device
    rows = n * h
    dx = torch.empty_like(x)
    meta = (h, dh // 2, bands.shape[0], pos.shape[1])

    def run(c):
        bm = c["bm"]
        chunk = max(bm, cdiv(cdiv(rows, c["per_sm"] * max(1, probe(dev).sms)), bm) * bm)
        progs = cdiv(rows, chunk)
        part = torch.empty(progs, bm, dh, device=dev, dtype=torch.float32)
        kernel = compiled(
            _bwd_prim, dev, *meta, chunk, bm, c["threads"], DTYPE[x.dtype]
        )
        kernel(_flat(g), _flat(x), rstd, w, pos, bands, dx.view(rows, dh), part)
        return part

    key = f"h{h}-dh{dh}-{x.dtype}-r{rows}"
    part = tuned_run("f6tl_qk_rope_bwd", dev, key, BWD_CFGS, run)
    return dx, part.sum((0, 1))


@custom_op("kohakumoss::f6tl_qk_norm_rope_fwd", mutates_args=())
def _fwd_op(
    q: Tensor, k: Tensor, qw: Tensor, kw: Tensor, pos: Tensor, bands: Tensor, eps: float
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _forward(q, k, qw, kw, pos, bands, eps)


@_fwd_op.register_fake
def _(q, k, qw, kw, pos, bands, eps):
    rows = q.shape[0] * q.shape[1]
    r = q.new_empty(rows, dtype=torch.float32)
    return torch.empty_like(q), torch.empty_like(k), r, torch.empty_like(r)


@custom_op("kohakumoss::f6tl_qk_norm_rope_bwd", mutates_args=())
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
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    dq, dqw = _backward_one(gq, q, qr, qw, pos, bands)
    dk, dkw = _backward_one(gk, k, kr, kw, pos, bands)
    return dq, dk, dqw, dkw


@_bwd_op.register_fake
def _(gq, gk, q, k, qr, kr, qw, kw, pos, bands):
    dh = q.shape[2]
    w = q.new_empty(dh, dtype=torch.float32)
    return torch.empty_like(q), torch.empty_like(k), w, torch.empty_like(w)


def _setup(ctx, inputs, output):
    q, k, qw, kw, pos, bands, _ = inputs
    ctx.save_for_backward(q, k, output[2], output[3], qw, kw, pos, bands)


def _backward(ctx, gqo, gko, _gqr, _gkr):
    q, k, qr, kr, qw, kw, pos, bands = ctx.saved_tensors
    gq = torch.zeros_like(q) if gqo is None else gqo.contiguous()
    gk = torch.zeros_like(k) if gko is None else gko.contiguous()
    dq, dk, dqw, dkw = _bwd_op(gq, gk, q, k, qr, kr, qw, kw, pos, bands)
    need = ctx.needs_input_grad
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
