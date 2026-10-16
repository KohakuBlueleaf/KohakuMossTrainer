"""TileLang optimizer step kernels (sm70+): AdamW over fp32 / 16-bit states and 8-bit
blockwise states, nearest / stochastic (bf16) / Kahan writeback. Spec:
kohakumoss_kernels/optim/reference.py.
"""

import tilelang.language as T
import torch

from kohakumoss_kernels.backends.tilelang._blockops import DTYPE, compiled
from kohakumoss_kernels.constraints import requires
from kohakumoss_kernels.optim import looped_multi
from kohakumoss_kernels.optim.reference import BLOCK as QBLOCK
from kohakumoss_kernels.optim.reference import MODES
from kohakumoss_kernels.registry import register

__all__ = ["adamw_step", "adamw8bit_step"]

F32 = "float32"
U64 = "uint64"
BLOCK, THREADS = 1024, 256


def _hash(idx, seed):
    h = (T.cast(idx, U64) * T.cast(0x9E3779B1, U64) + seed) & T.cast(0xFFFFFFFF, U64)
    h = h ^ (h >> 16)
    h = (h * T.cast(0x7FEB352D, U64)) & T.cast(0xFFFFFFFF, U64)
    h = h ^ (h >> 15)
    h = (h * T.cast(0x846CA68B, U64)) & T.cast(0xFFFFFFFF, U64)
    return h ^ (h >> 16)


def _adamw_prim(mode, pt, mt, ct):
    n = T.dynamic("n")

    @T.prim_func
    def main(
        p: T.Tensor((n,), pt),
        g: T.Tensor((n,), pt),
        m: T.Tensor((n,), mt),
        r: T.Tensor((n,), mt),
        c: T.Tensor((n,), ct),
        lr: T.float32,
        b1: T.float32,
        b2: T.float32,
        eps: T.float32,
        wd: T.float32,
        c1: T.float32,
        c2: T.float32,
        seed: T.uint64,
    ):
        with T.Kernel(T.ceildiv(n, BLOCK), threads=THREADS) as bx:
            for i in T.Parallel(BLOCK):
                idx = bx * BLOCK + i
                ii = T.min(idx, n - 1)
                pv = T.cast(p[ii], F32)
                if mode == 2:
                    pv = pv + T.cast(c[ii], F32)
                gv = T.cast(g[ii], F32)
                mv = b1 * T.cast(m[ii], F32) + (1.0 - b1) * gv
                rv0 = T.cast(r[ii], F32)
                rv = T.sqrt(b2 * rv0 * rv0 + (1.0 - b2) * gv * gv)
                new = pv - lr * ((mv * c1) / (rv * c2 + eps) + wd * pv)
                if idx < n:
                    m[idx] = T.cast(mv, mt)
                    r[idx] = T.cast(rv, mt)
                    if mode == 1:
                        bits = T.cast(T.reinterpret(new, "uint32"), U64)
                        rnd = _hash(idx, seed) & T.cast(0xFFFF, U64)
                        out = T.cast((bits + rnd) & T.cast(0xFFFF0000, U64), "uint32")
                        p[idx] = T.cast(T.reinterpret(out, F32), pt)
                    elif mode == 2:
                        pn = T.cast(new, pt)
                        p[idx] = pn
                        c[idx] = T.cast(new - T.cast(pn, F32), ct)
                    else:
                        p[idx] = T.cast(new, pt)

    return main


def _scalars(lr, beta1, beta2, eps, wd, step):
    c1 = 1.0 / (1.0 - beta1**step)
    c2 = 1.0 / (1.0 - beta2**step) ** 0.5
    return float(lr), float(beta1), float(beta2), float(eps), float(wd), c1, c2


def adamw_step(
    p, g, m, r, comp, lr, beta1, beta2, eps, wd, step, mode="nearest", seed=0
):
    """In-place AdamW; see kohakumoss_kernels.optim.reference.adamw_step."""
    if p.numel() == 0:
        return
    code = MODES.index(mode)
    c = comp if comp is not None else p
    kernel = compiled(
        _adamw_prim, p.device, code, DTYPE[p.dtype], DTYPE[m.dtype], DTYPE[c.dtype]
    )
    kernel(
        p.view(-1), g.view(-1), m.view(-1), r.view(-1), c.view(-1),
        *_scalars(lr, beta1, beta2, eps, wd, step), int(seed) & 0xFFFFFFFF,
    )  # fmt: skip


def _adamw8_prim(mode, pt, ct):
    n = T.dynamic("n")
    nb = T.dynamic("nb")

    @T.prim_func
    def main(
        p: T.Tensor((n,), pt),
        g: T.Tensor((n,), pt),
        mq: T.Tensor((n,), "int8"),
        ms: T.Tensor((nb,), F32),
        rq: T.Tensor((n,), "uint8"),
        rs: T.Tensor((nb,), F32),
        c: T.Tensor((n,), ct),
        lr: T.float32,
        b1: T.float32,
        b2: T.float32,
        eps: T.float32,
        wd: T.float32,
        c1: T.float32,
        c2: T.float32,
        seed: T.uint64,
    ):
        with T.Kernel(nb, threads=QBLOCK) as bx:
            mf = T.alloc_fragment((QBLOCK,), F32)
            rf = T.alloc_fragment((QBLOCK,), F32)
            am = T.alloc_fragment((QBLOCK,), F32)
            nv = T.alloc_fragment((QBLOCK,), F32)
            smax = T.alloc_fragment((1,), F32)
            rmax = T.alloc_fragment((1,), F32)
            for i in T.Parallel(QBLOCK):
                idx = bx * QBLOCK + i
                ok = idx < n
                ii = T.min(idx, n - 1)
                pv = T.cast(p[ii], F32)
                if mode == 2:
                    pv = pv + T.cast(c[ii], F32)
                gv = T.cast(g[ii], F32)
                qm = T.cast(mq[ii], F32)
                tm = T.abs(qm) / 127.0
                sg = T.if_then_else(qm < 0, -1.0, T.if_then_else(qm > 0, 1.0, 0.0))
                tr = T.cast(rq[ii], F32) / 255.0
                mv = b1 * (tm * tm * ms[bx] * sg) + (1.0 - b1) * gv
                r0 = tr * tr * rs[bx]
                rv = T.sqrt(b2 * r0 * r0 + (1.0 - b2) * gv * gv)
                nv[i] = pv - lr * ((mv * c1) / (rv * c2 + eps) + wd * pv)
                mf[i] = T.if_then_else(ok, mv, 0.0)
                rf[i] = T.if_then_else(ok, rv, 0.0)
                am[i] = T.abs(mf[i])
            T.reduce_max(am, smax, dim=0)
            T.reduce_max(rf, rmax, dim=0)
            for i in T.Parallel(QBLOCK):
                idx = bx * QBLOCK + i
                sm = T.max(smax[0], 1e-30)
                sr = T.max(rmax[0], 1e-30)
                sg = T.if_then_else(
                    mf[i] < 0, -1.0, T.if_then_else(mf[i] > 0, 1.0, 0.0)
                )
                cm = sg * T.nearbyint(T.sqrt(T.abs(mf[i]) / sm) * 127.0)
                cr = T.nearbyint(T.sqrt(rf[i] / sr) * 255.0)
                if idx < n:
                    mq[idx] = T.cast(cm, "int8")
                    rq[idx] = T.cast(cr, "uint8")
                    if mode == 1:
                        bits = T.cast(T.reinterpret(nv[i], "uint32"), U64)
                        rnd = _hash(idx, seed) & T.cast(0xFFFF, U64)
                        out = T.cast((bits + rnd) & T.cast(0xFFFF0000, U64), "uint32")
                        p[idx] = T.cast(T.reinterpret(out, F32), pt)
                    elif mode == 2:
                        pn = T.cast(nv[i], pt)
                        p[idx] = pn
                        c[idx] = T.cast(nv[i] - T.cast(pn, F32), ct)
                    else:
                        p[idx] = T.cast(nv[i], pt)
                if i == 0:
                    ms[bx] = sm
                    rs[bx] = sr

    return main


def adamw8bit_step(
    p, g, mq, ms, rq, rs, comp, lr, beta1, beta2, eps, wd, step, mode="nearest", seed=0
):
    """In-place 8-bit AdamW; see kohakumoss_kernels.optim.reference.adamw8bit_step."""
    if p.numel() == 0:
        return
    code = MODES.index(mode)
    c = comp if comp is not None else p
    kernel = compiled(_adamw8_prim, p.device, code, DTYPE[p.dtype], DTYPE[c.dtype])
    kernel(
        p.view(-1), g.view(-1), mq, ms, rq, rs, c.view(-1),
        *_scalars(lr, beta1, beta2, eps, wd, step), int(seed) & 0xFFFFFFFF,
    )  # fmt: skip


_SUPPORTS = requires(kinds=("cuda",), min_sm=70, dtypes=(torch.float16, torch.float32))
register("adamw_step", "tilelang", adamw_step, _SUPPORTS)
register("adamw_step_multi", "tilelang", looped_multi(adamw_step), _SUPPORTS)
register("adamw8bit_step", "tilelang", adamw8bit_step, _SUPPORTS)
