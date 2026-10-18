"""Triton qk_norm_rope: per-head RMSNorm of q and k, then multi-axis split-half RoPE
with sin/cos computed in-kernel from token positions, forward and backward.

Rows are (token, head) pairs; one program owns ROWS rows of both q and k, so the angles
are computed once for the pair. The backward rotates the gradient back, applies the
RMSNorm backward with the saved fp32 rstd, and writes per-program fp32 partials of the
q/k weight gradients that are summed afterwards in a second pass.
See docs/internals/fp16.md.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op
from triton.language.extra import libdevice

from kohakumoss_kernels.backends.compile.choice import kernel_or_compile, shape_key
from kohakumoss_kernels.backends.eager import rope as eager_rope
from kohakumoss_kernels.backends.triton._common import Launcher, eager_function
from kohakumoss_kernels.backends.triton.blocks._blockops import num_sms, total
from kohakumoss_kernels.constraints import requires
from kohakumoss_kernels.registry import register
from kohakumoss_kernels.tuning.buckets import cdiv
from kohakumoss_kernels.tuning.tuner import select

__all__ = ["qk_norm_rope"]

DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _angles(rows, rok, pos_ptr, bands_ptr, H, A, NB: tl.constexpr, HALF: tl.constexpr):
    """sin, cos of the fp32 phase pos * band (as the eager reference computes it)."""
    j = tl.arange(0, HALF)
    tok = (rows // H).to(tl.int64)
    p = tl.load(
        pos_ptr + tok[:, None] * A + (j // NB)[None, :], mask=rok[:, None], other=0.0
    )
    band = tl.load(bands_ptr + (j % NB)).to(tl.float32)[None, :]
    ph = p.to(tl.float32) * band
    return libdevice.sin(ph), libdevice.cos(ph)


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
    g_ptr, x_ptr, rstd_ptr, w_ptr, dx_ptr, rows, rok, s, c, H, dx_tok,
    HALF: tl.constexpr,
):  # fmt: skip
    """Writes dx (token rows `dx_tok` apart); returns the fp32 weight-gradient column
    partials (pa, pb) of this row tile."""
    j = tl.arange(0, HALF)
    offs = rows.to(tl.int64)[:, None] * (2 * HALF) + j[None, :]
    tok = (rows // H).to(tl.int64)[:, None]
    doffs = tok * dx_tok + (rows % H)[:, None] * (2 * HALF) + j[None, :]
    m = rok[:, None]
    ga = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
    gb = tl.load(g_ptr + offs + HALF, mask=m, other=0.0).to(tl.float32)
    dan = ga * c + gb * s
    dbn = gb * c - ga * s
    rs = tl.load(rstd_ptr + rows, mask=rok, other=0.0)
    xa_raw = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    xb_raw = tl.load(x_ptr + offs + HALF, mask=m, other=0.0).to(tl.float32)
    xa = xa_raw * rs[:, None]
    xb = xb_raw * rs[:, None]
    da = dan * tl.load(w_ptr + j).to(tl.float32)[None, :]
    db = dbn * tl.load(w_ptr + HALF + j).to(tl.float32)[None, :]
    k = tl.div_rn(tl.sum(da * xa, axis=1) + tl.sum(db * xb, axis=1), 2.0 * HALF)
    dxa = rs[:, None] * (da - xa * k[:, None])
    dxb = rs[:, None] * (db - xb * k[:, None])
    tl.store(dx_ptr + doffs, dxa.to(dx_ptr.dtype.element_ty), mask=m)
    tl.store(dx_ptr + doffs + HALF, dxb.to(dx_ptr.dtype.element_ty), mask=m)
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
    dx_tok,
    NB: tl.constexpr,
    HALF: tl.constexpr,
    ROWS: tl.constexpr,
    NEED_W: tl.constexpr,
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
            gq_ptr, q_ptr, qr_ptr, qw_ptr, dq_ptr, rows, rok, s, c, H, dx_tok, HALF,
        )  # fmt: skip
        acc_qa += pa
        acc_qb += pb
        pa, pb = _norm_rotate_bwd(
            gk_ptr, k_ptr, kr_ptr, kw_ptr, dk_ptr, rows, rok, s, c, H, dx_tok, HALF,
        )  # fmt: skip
        acc_ka += pa
        acc_kb += pb
    if NEED_W:
        base = part_ptr + pid.to(tl.int64) * 4 * HALF + j
        tl.store(base, acc_qa)
        tl.store(base + HALF, acc_qb)
        tl.store(base + 2 * HALF, acc_ka)
        tl.store(base + 3 * HALF, acc_kb)


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
        "triton_qk_rope_fwd",
        q.device,
        key,
        _candidates,
        lambda c: lambda: _fwd_launch(c, *args),
    )
    _fwd_launch(cfg, *args)
    return qo, ko, qr, kr


@triton_op("kohakumoss::triton_qk_norm_rope_fwd", mutates_args=())
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


def _bwd_launch(
    cfg, gq, gk, q, k, qr, kr, qw, kw, pos, bands, dq, dk, dx_tok, programs, need_w
):
    n, h, dh = q.shape
    r = n * h
    per = cdiv(cdiv(r, programs), cfg["ROWS"]) * cfg["ROWS"]
    progs = cdiv(r, per)
    part = torch.empty(
        progs if need_w else 1, 4, dh // 2, device=q.device, dtype=torch.float32
    )
    _bwd[(progs,)](
        gq, gk, q, k, qr, kr, qw, kw, pos, bands, dq, dk, part, r, h, pos.shape[1], per,
        dx_tok, NB=bands.shape[0], HALF=dh // 2, ROWS=cfg["ROWS"], NEED_W=need_w,
        num_warps=cfg["warps"],
    )  # fmt: skip
    return part


def rope_backward(
    gq, gk, q, k, qr, kr, q_weight, k_weight, pos, bands, eps, need_w=True, dqk=None
):
    """(dq, dk, d q_weight, d k_weight) from the saved pre-norm q, k and rstd; with
    `need_w` False the weight gradients come back empty. With `dqk` [N, 2 H Dh], dq and
    dk are written into its two column halves (and returned as views of it)."""
    n, h, dh = q.shape
    if dqk is None:
        dq, dk = torch.empty_like(q), torch.empty_like(k)
        dx_tok = h * dh
    else:
        dq, dk = dqk[:, : h * dh], dqk[:, h * dh :]
        dx_tok = 2 * h * dh
    args = (gq, gk, q, k, qr, kr, q_weight, k_weight, pos, bands, dq, dk, dx_tok)
    args = (*args, 4 * num_sms(q.device), need_w)
    key = f"dh{q.shape[2]}-h{q.shape[1]}-{q.dtype}-{need_w}"
    cfg = select(
        "triton_qk_rope_bwd",
        q.device,
        key,
        _candidates,
        lambda c: lambda: _bwd_launch(c, *args),
        in_backward=True,
    )
    part = _bwd_launch(cfg, *args)
    if not need_w:
        return dq, dk, q.new_empty(0), k.new_empty(0)
    w = total(part)
    dqw = torch.cat((w[0], w[1])).to(q_weight.dtype)
    dkw = torch.cat((w[2], w[3])).to(k_weight.dtype)
    return dq, dk, dqw, dkw


@triton_op("kohakumoss::triton_qk_norm_rope_bwd", mutates_args=())
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
    eps: float,
    need_w: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return rope_backward(
        gq, gk, q, k, qr, kr, q_weight, k_weight, pos, bands, eps, need_w
    )


def _setup(ctx, inputs, output):
    q, k, qw, kw, pos, bands, eps = inputs
    ctx.save_for_backward(q, k, output[2], output[3], qw, kw, pos, bands)
    ctx.eps = eps


def _backward_with(bwd):
    """The autograd backward over `bwd` (the backward op, or its body in eager)."""

    def backward(ctx, gqo, gko, _gqr, _gkr):
        q, k, qr, kr, qw, kw, pos, bands = ctx.saved_tensors
        gq = torch.zeros_like(q) if gqo is None else gqo.contiguous()
        gk = torch.zeros_like(k) if gko is None else gko.contiguous()
        need = ctx.needs_input_grad
        dq, dk, dqw, dkw = bwd(
            gq, gk, q, k, qr, kr, qw, kw, pos, bands, ctx.eps, need[2] or need[3]
        )
        dqw = dqw if need[2] else None
        return dq, dk, dqw, dkw if need[3] else None, None, None, None

    return backward


_fwd_op.register_autograd(_backward_with(_bwd_op), setup_context=_setup)
_Eager = eager_function(
    "QkNormRope", rope_forward, _setup, _backward_with(rope_backward), nondiff=(2, 3)
)


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
    fwd = _fwd_op if torch.compiler.is_compiling() else _Eager.apply
    qo, ko, _, _ = fwd(
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
    kernel_or_compile("triton", "qk_norm_rope", qk_norm_rope, shape_key),
    requires(kinds=("cuda", "xpu"), min_sm=70, dtypes=DTYPES),
)
