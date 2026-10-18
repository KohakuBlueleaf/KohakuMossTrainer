"""Triton optimizer step kernels: AdamW over fp32 / 16-bit states and 8-bit blockwise
states, with nearest / stochastic (bf16) / Kahan writeback, each timed once per problem
key against the reference step (kohakumoss_kernels/optim/choice.py). The eager
reference is kohakumoss_kernels/optim/reference.py.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from kohakumoss_kernels.backends.triton._common import Launcher
from kohakumoss_kernels.constraints import requires
from kohakumoss_kernels.optim import reference
from kohakumoss_kernels.optim.choice import kernel_or_reference
from kohakumoss_kernels.optim.reference import BLOCK as QBLOCK
from kohakumoss_kernels.optim.reference import MODES
from kohakumoss_kernels.registry import register
from kohakumoss_kernels.tuning.buckets import cdiv

__all__ = ["adamw_step", "adamw8bit_step"]

MASK32 = 0xFFFFFFFF


@triton.jit
def hash32(idx, seed):
    """Counter hash of (index, seed) in [0, 2^32) as uint64 (reference.hash32)."""
    h = (idx.to(tl.uint64) * 0x9E3779B1 + seed) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x7FEB352D) & 0xFFFFFFFF
    h ^= h >> 15
    h = (h * 0x846CA68B) & 0xFFFFFFFF
    h ^= h >> 16
    return h


@triton.jit
def write_param(p_ptr, c_ptr, offs, mask, new, seed, MODE: tl.constexpr):
    """Store fp32 `new` to the parameter: nearest (0), stochastic bf16 (1), Kahan (2)."""
    if MODE == 1:
        bits = new.to(tl.uint32, bitcast=True).to(tl.uint64)
        rnd = hash32(offs, seed) & 0xFFFF
        out = ((bits + rnd) & 0xFFFF0000).to(tl.uint32).to(tl.float32, bitcast=True)
        tl.store(p_ptr + offs, out.to(p_ptr.dtype.element_ty), mask=mask)
    elif MODE == 2:
        pn = new.to(p_ptr.dtype.element_ty)
        tl.store(p_ptr + offs, pn, mask=mask)
        tl.store(
            c_ptr + offs,
            (new - pn.to(tl.float32)).to(c_ptr.dtype.element_ty),
            mask=mask,
        )
    else:
        tl.store(p_ptr + offs, new.to(p_ptr.dtype.element_ty), mask=mask)


@triton.jit
def adam_update(p, g, m, r, lr, ob1, ob2, eps, wd, c1, c2):
    """fp32 (new p, m, r) of one AdamW step; `r` is the root of the second moment."""
    m = (1.0 - ob1) * m + ob1 * g
    r = tl.sqrt_rn((1.0 - ob2) * r * r + ob2 * g * g)
    upd = tl.div_rn(m * c1, r * c2 + eps)
    return p - lr * (upd + wd * p), m, r


@triton.jit
def _adamw_kernel(
    p_ptr, g_ptr, m_ptr, r_ptr, c_ptr, n, lr, ob1, ob2, eps, wd, c1, c2, seed,
    MODE: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    p = tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if MODE == 2:
        p += tl.load(c_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(m_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    new, m, r = adam_update(p, g, m, r, lr, ob1, ob2, eps, wd, c1, c2)
    tl.store(m_ptr + offs, m.to(m_ptr.dtype.element_ty), mask=mask)
    tl.store(r_ptr + offs, r.to(r_ptr.dtype.element_ty), mask=mask)
    write_param(p_ptr, c_ptr, offs, mask, new, seed, MODE)


@triton.jit
def _adamw8_kernel(
    p_ptr, g_ptr, mq_ptr, ms_ptr, rq_ptr, rs_ptr, c_ptr, n, lr, ob1, ob2, eps, wd, c1, c2,
    seed, MODE: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    blk = tl.program_id(0)
    offs = blk.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    p = tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if MODE == 2:
        p += tl.load(c_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    qm = tl.load(mq_ptr + offs, mask=mask, other=0).to(tl.float32)
    qr = tl.load(rq_ptr + offs, mask=mask, other=0).to(tl.float32)
    tm = tl.abs(qm) / 127.0
    tr = qr / 255.0
    sgn = tl.where(qm < 0, -1.0, tl.where(qm > 0, 1.0, 0.0))
    m = tm * tm * tl.load(ms_ptr + blk) * sgn
    r = tr * tr * tl.load(rs_ptr + blk)
    new, m, r = adam_update(p, g, m, r, lr, ob1, ob2, eps, wd, c1, c2)
    m = tl.where(mask, m, 0.0)
    r = tl.where(mask, r, 0.0)
    sm = tl.maximum(tl.max(tl.abs(m), axis=0), 1e-30)
    sr = tl.maximum(tl.max(r, axis=0), 1e-30)
    msg = tl.where(m < 0, -1.0, tl.where(m > 0, 1.0, 0.0))
    cm = msg * libdevice.rint(tl.sqrt_rn(tl.div_rn(tl.abs(m), sm)) * 127.0)
    cr = libdevice.rint(tl.sqrt_rn(tl.div_rn(r, sr)) * 255.0)
    tl.store(mq_ptr + offs, cm.to(tl.int8), mask=mask)
    tl.store(rq_ptr + offs, cr.to(tl.uint8), mask=mask)
    tl.store(ms_ptr + blk, sm)
    tl.store(rs_ptr + blk, sr)
    write_param(p_ptr, c_ptr, offs, mask, new, seed, MODE)


_ADAMW = Launcher(_adamw_kernel)
_ADAMW8 = Launcher(_adamw8_kernel)


def adam_scalars(lr, beta1, beta2, eps, wd, step):
    """Kernel scalars: lr, 1 - beta1, 1 - beta2 (complements formed on the host, as the
    torch reference forms them), eps, wd and the bias corrections."""
    c1 = 1.0 / (1.0 - beta1**step)
    c2 = 1.0 / (1.0 - beta2**step) ** 0.5
    return float(lr), 1.0 - beta1, 1.0 - beta2, float(eps), float(wd), c1, c2


def adamw_step(
    p, g, m, r, comp, lr, beta1, beta2, eps, wd, step, mode="nearest", seed=0
):
    """In-place AdamW; see kohakumoss_kernels.optim.reference.adamw_step."""
    n = p.numel()
    if n == 0:
        return
    code = MODES.index(mode)
    _ADAMW[(cdiv(n, 1024),)](
        p, g, m, r, comp if comp is not None else p, n,
        *adam_scalars(lr, beta1, beta2, eps, wd, step), int(seed) & MASK32,
        MODE=code, BLOCK=1024, num_warps=4,
    )  # fmt: skip


def adamw8bit_step(
    p, g, mq, ms, rq, rs, comp, lr, beta1, beta2, eps, wd, step, mode="nearest", seed=0
):
    """In-place 8-bit AdamW; see kohakumoss_kernels.optim.reference.adamw8bit_step."""
    n = p.numel()
    if n == 0:
        return
    code = MODES.index(mode)
    _ADAMW8[(cdiv(n, QBLOCK),)](
        p, g, mq, ms, rq, rs, comp if comp is not None else p, n,
        *adam_scalars(lr, beta1, beta2, eps, wd, step), int(seed) & MASK32,
        MODE=code, BLOCK=QBLOCK, num_warps=4,
    )  # fmt: skip


_SUPPORTS = requires(
    kinds=("cuda", "xpu"),
    min_sm=70,
    dtypes=(torch.float16, torch.bfloat16, torch.float32),
)
register(
    "adamw_step",
    "triton",
    kernel_or_reference("triton", "adamw_step", adamw_step, reference.adamw_step),
    _SUPPORTS,
)
register(
    "adamw8bit_step",
    "triton",
    kernel_or_reference(
        "triton", "adamw8bit_step", adamw8bit_step, reference.adamw8bit_step
    ),
    _SUPPORTS,
)
