"""Triton qk_norm_rope: per-head RMSNorm of q and k, then multi-axis split-half RoPE
with sin/cos computed in-kernel from token positions (no table), forward and backward.

Rows are (token, head) pairs; one program owns ROWS rows of both q and k, so the angles
are computed once for the pair. The backward rotates the gradient back, applies the
RMSNorm backward with the saved fp32 rstd, and writes per-program partials of the
q/k weight gradients that are summed afterwards (no atomics).
"""

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op
from triton.language.extra import libdevice

from kohakumoss_kernels.backends.compile.race import raced, shape_key
from kohakumoss_kernels.backends.eager import rope as eager_rope
from kohakumoss_kernels.backends.triton._blockops import num_sms
from kohakumoss_kernels.backends.triton._common import Launcher, cdiv
from kohakumoss_kernels.constraints import requires
from kohakumoss_kernels.registry import register
from kohakumoss_kernels.tuning.tuner import select

__all__ = ["qk_norm_rope"]

DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _angles(rows, rok, pos_ptr, bands_ptr, H, A, NB: tl.constexpr, HALF: tl.constexpr):
    j = tl.arange(0, HALF)
    tok = (rows // H).to(tl.int64)
    p = tl.load(
        pos_ptr + tok[:, None] * A + (j // NB)[None, :], mask=rok[:, None], other=0.0
    )
    phase = p.to(tl.float32) * tl.load(bands_ptr + (j % NB)).to(tl.float32)[None, :]
    return libdevice.sin(phase), libdevice.cos(phase)


@triton.jit
def _norm_rotate_fwd(
    src, dst, rstd_ptr, w_ptr, rows, rok, s, c, eps, HALF: tl.constexpr
):
    j = tl.arange(0, HALF)
    offs = rows.to(tl.int64)[:, None] * (2 * HALF) + j[None, :]
    m = rok[:, None]
    a = tl.load(src + offs, mask=m, other=0.0).to(tl.float32)
    b = tl.load(src + offs + HALF, mask=m, other=0.0).to(tl.float32)
    ms = tl.div_rn(tl.sum(a * a, axis=1) + tl.sum(b * b, axis=1), 2.0 * HALF)
    rstd = tl.div_rn(1.0, tl.sqrt_rn(ms + eps.to(tl.float32)))
    an = a * rstd[:, None] * tl.load(w_ptr + j).to(tl.float32)[None, :]
    bn = b * rstd[:, None] * tl.load(w_ptr + HALF + j).to(tl.float32)[None, :]
    tl.store(dst + offs, (an * c - bn * s).to(dst.dtype.element_ty), mask=m)
    tl.store(dst + offs + HALF, (bn * c + an * s).to(dst.dtype.element_ty), mask=m)
    tl.store(rstd_ptr + rows, rstd, mask=rok)


@triton.jit
def _fwd_kernel(
    q_ptr,
    k_ptr,
    qw_ptr,
    kw_ptr,
    pos_ptr,
    bands_ptr,
    qo_ptr,
    ko_ptr,
    qr_ptr,
    kr_ptr,
    R,
    H,
    A,
    eps,
    NB: tl.constexpr,
    HALF: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rok = rows < R
    s, c = _angles(rows, rok, pos_ptr, bands_ptr, H, A, NB, HALF)
    _norm_rotate_fwd(q_ptr, qo_ptr, qr_ptr, qw_ptr, rows, rok, s, c, eps, HALF)
    _norm_rotate_fwd(k_ptr, ko_ptr, kr_ptr, kw_ptr, rows, rok, s, c, eps, HALF)


@triton.jit
def _norm_rotate_bwd(
    g_ptr, x_ptr, rstd_ptr, w_ptr, dx_ptr, rows, rok, s, c, HALF: tl.constexpr
):
    j = tl.arange(0, HALF)
    offs = rows.to(tl.int64)[:, None] * (2 * HALF) + j[None, :]
    m = rok[:, None]
    ga = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
    gb = tl.load(g_ptr + offs + HALF, mask=m, other=0.0).to(tl.float32)
    dan = ga * c + gb * s
    dbn = gb * c - ga * s
    rs = tl.load(rstd_ptr + rows, mask=rok, other=0.0)
    xa = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32) * rs[:, None]
    xb = tl.load(x_ptr + offs + HALF, mask=m, other=0.0).to(tl.float32) * rs[:, None]
    da = dan * tl.load(w_ptr + j).to(tl.float32)[None, :]
    db = dbn * tl.load(w_ptr + HALF + j).to(tl.float32)[None, :]
    k = tl.div_rn(tl.sum(da * xa, axis=1) + tl.sum(db * xb, axis=1), 2.0 * HALF)
    dxa = rs[:, None] * (da - xa * k[:, None])
    dxb = rs[:, None] * (db - xb * k[:, None])
    tl.store(dx_ptr + offs, dxa.to(dx_ptr.dtype.element_ty), mask=m)
    tl.store(dx_ptr + offs + HALF, dxb.to(dx_ptr.dtype.element_ty), mask=m)
    return tl.sum(dan * xa, axis=0), tl.sum(dbn * xb, axis=0)


@triton.jit
def _bwd_kernel(
    gq_ptr,
    gk_ptr,
    q_ptr,
    k_ptr,
    qr_ptr,
    kr_ptr,
    qw_ptr,
    kw_ptr,
    pos_ptr,
    bands_ptr,
    dq_ptr,
    dk_ptr,
    part_ptr,
    R,
    H,
    A,
    ROWS_PER_PROG,
    NB: tl.constexpr,
    HALF: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    j = tl.arange(0, HALF)
    acc_qa = tl.zeros([HALF], tl.float32)
    acc_qb = tl.zeros([HALF], tl.float32)
    acc_ka = tl.zeros([HALF], tl.float32)
    acc_kb = tl.zeros([HALF], tl.float32)
    start = pid * ROWS_PER_PROG
    stop = tl.minimum(start + ROWS_PER_PROG, R)
    for r0 in range(start, stop, ROWS):
        rows = r0 + tl.arange(0, ROWS)
        rok = rows < stop
        s, c = _angles(rows, rok, pos_ptr, bands_ptr, H, A, NB, HALF)
        pa, pb = _norm_rotate_bwd(
            gq_ptr, q_ptr, qr_ptr, qw_ptr, dq_ptr, rows, rok, s, c, HALF
        )
        acc_qa += pa
        acc_qb += pb
        pa, pb = _norm_rotate_bwd(
            gk_ptr, k_ptr, kr_ptr, kw_ptr, dk_ptr, rows, rok, s, c, HALF
        )
        acc_ka += pa
        acc_kb += pb
    base = part_ptr + pid.to(tl.int64) * 4 * HALF
    tl.store(base + j, acc_qa)
    tl.store(base + HALF + j, acc_qb)
    tl.store(base + 2 * HALF + j, acc_ka)
    tl.store(base + 3 * HALF + j, acc_kb)


_fwd = Launcher(_fwd_kernel)
_bwd = Launcher(_bwd_kernel)


def _candidates() -> list[dict]:
    return [
        {"ROWS": r, "warps": w}
        for r, w in ((16, 4), (32, 4), (8, 2), (32, 8), (64, 8), (16, 2))
    ]


def _fwd_launch(cfg, q, k, qw, kw, pos, bands, qo, ko, qr, kr, eps):
    n, h, dh = q.shape
    r = n * h
    _fwd[(cdiv(r, cfg["ROWS"]),)](
        q, k, qw, kw, pos, bands, qo, ko, qr, kr, r, h, pos.shape[1], eps,
        NB=bands.shape[0], HALF=dh // 2, ROWS=cfg["ROWS"], num_warps=cfg["warps"],
    )  # fmt: skip


def rope_forward(q, k, q_weight, k_weight, pos, bands, eps):
    """(q_out, k_out, q_rstd, k_rstd) for q, k [N, H, Dh] (no autograd)."""
    n, h, _ = q.shape
    qo, ko = torch.empty_like(q), torch.empty_like(k)
    qr = torch.empty(n * h, device=q.device, dtype=torch.float32)
    kr = torch.empty_like(qr)
    key = f"dh{q.shape[2]}-h{h}-{q.dtype}"
    args = (q, k, q_weight, k_weight, pos, bands, qo, ko, qr, kr, eps)
    cfg = select(
        "f6_qk_rope_fwd",
        q.device,
        key,
        _candidates,
        lambda c: lambda: _fwd_launch(c, *args),
    )
    _fwd_launch(cfg, *args)
    return qo, ko, qr, kr


@triton_op("kohakumoss::f6_qk_norm_rope_fwd", mutates_args=())
def _fwd_op(
    q: Tensor,
    k: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    pos: Tensor,
    bands: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return rope_forward(q, k, q_weight, k_weight, pos, bands, eps)


def _bwd_launch(cfg, gq, gk, q, k, qr, kr, qw, kw, pos, bands, dq, dk, programs):
    n, h, dh = q.shape
    r = n * h
    per = cdiv(cdiv(r, programs), cfg["ROWS"]) * cfg["ROWS"]
    progs = cdiv(r, per)
    part = torch.empty(progs, 4, dh // 2, device=q.device, dtype=torch.float32)
    _bwd[(progs,)](
        gq, gk, q, k, qr, kr, qw, kw, pos, bands, dq, dk, part, r, h, pos.shape[1], per,
        NB=bands.shape[0], HALF=dh // 2, ROWS=cfg["ROWS"], num_warps=cfg["warps"],
    )  # fmt: skip
    return part


def rope_backward(gq, gk, q, k, qr, kr, q_weight, k_weight, pos, bands):
    """(dq, dk, d q_weight, d k_weight) from the saved pre-norm q, k and rstd."""
    dq, dk = torch.empty_like(q), torch.empty_like(k)
    args = (gq, gk, q, k, qr, kr, q_weight, k_weight, pos, bands, dq, dk)
    args = (*args, 4 * num_sms(q.device))
    key = f"dh{q.shape[2]}-h{q.shape[1]}-{q.dtype}"
    cfg = select(
        "f6_qk_rope_bwd",
        q.device,
        key,
        _candidates,
        lambda c: lambda: _bwd_launch(c, *args),
    )
    part = _bwd_launch(cfg, *args).sum(0)
    dqw = torch.cat((part[0], part[1])).to(q_weight.dtype)
    dkw = torch.cat((part[2], part[3])).to(k_weight.dtype)
    return dq, dk, dqw, dkw


@triton_op("kohakumoss::f6_qk_norm_rope_bwd", mutates_args=())
def _bwd_op(
    gq: Tensor,
    gk: Tensor,
    q: Tensor,
    k: Tensor,
    qr: Tensor,
    kr: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    pos: Tensor,
    bands: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return rope_backward(gq, gk, q, k, qr, kr, q_weight, k_weight, pos, bands)


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
    if half & (half - 1) or q.shape[-1] % 2 or pos.shape[1] * bands.shape[0] != half:
        return eager_rope.qk_norm_rope(q, k, q_weight, k_weight, pos, bands, eps)
    qo, ko, _, _ = _fwd_op(
        q.contiguous(),
        k.contiguous(),
        q_weight.contiguous(),
        k_weight.contiguous(),
        pos.float().contiguous(),
        bands.float().contiguous(),
        eps,
    )
    return qo, ko


register(
    "qk_norm_rope",
    "triton",
    raced("triton", "qk_norm_rope", qk_norm_rope, shape_key),
    requires(kinds=("cuda", "xpu"), min_sm=70, dtypes=DTYPES),
)
