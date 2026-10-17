"""Triton fused linear ops: GEMM + MOSS epilogue (SiTU-GLU, gated residual, qk-norm +
RoPE), registered as the optional `linear_situ`, `linear_gated_residual` and
`linear_qk_rope` ops.

Per shape the tuner races the fused kernel (`path=fused`) against cuBLAS + the
standalone epilogue kernel (`path=split`). Forward saves what the backward needs only
while autograd records (pre-activations / pre-norm projections / the projection
under a trained gate); the backward is the epilogue's backward kernel followed by
cuBLAS GEMMs for the input and weight gradients.
"""

import os
from collections.abc import Callable
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.library import triton_op

from kohakumoss_kernels.backends._master import activation, operand, weight_grad
from kohakumoss_kernels.backends.compile.race import raced, shape_key
from kohakumoss_kernels.backends.triton import _gemm_kernels as jit
from kohakumoss_kernels.backends.triton._blockops import strides2
from kohakumoss_kernels.backends.triton._common import Launcher, cdiv, pow2_bucket
from kohakumoss_kernels.backends.triton.ffn import situ_backward, situ_forward
from kohakumoss_kernels.backends.triton.residual import (
    residual_backward,
    residual_forward,
)
from kohakumoss_kernels.backends.triton.rope import rope_backward, rope_forward
from kohakumoss_kernels.constraints import requires
from kohakumoss_kernels.registry import register
from kohakumoss_kernels.tuning.tuner import decided, on_reset, select

__all__ = ["linear_situ", "linear_gated_residual", "linear_qk_rope"]

DTYPES = (torch.float16, torch.bfloat16)
SPLIT = {"path": "split"}
SAVE_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
    os.environ.get("KOHAKUMOSS_GATE_SAVE", "fp32")
]

_situ = Launcher(jit.situ_gemm_kernel)
_res = Launcher(jit.residual_gemm_kernel)
_qk = Launcher(jit.qk_rope_gemm_kernel)


def _tiles(*rows) -> list[dict]:
    return [
        {"path": "fused", "BM": bm, "BN": bn, "BK": bk, "warps": w, "stages": s}
        for bm, bn, bk, w, s in rows
    ]


SITU_CFGS = _tiles(
    (128, 64, 32, 8, 3), (128, 64, 64, 8, 3), (64, 64, 64, 4, 4), (128, 128, 32, 8, 2), (64, 128, 32, 4, 3)
) + [SPLIT]  # fmt: skip
RES_CFGS = _tiles(
    (128, 128, 32, 8, 3), (128, 64, 64, 4, 4), (64, 128, 64, 4, 4), (128, 256, 32, 8, 3), (64, 64, 64, 4, 4)
) + [SPLIT]  # fmt: skip
QK_CFGS = _tiles(
    (128, 128, 32, 8, 3), (64, 128, 64, 4, 4), (128, 128, 64, 8, 3), (64, 128, 32, 4, 3)
) + [SPLIT]  # fmt: skip


def _key(m: int, *rest) -> str:
    return "-".join(str(v) for v in (pow2_bucket(m, 256), *rest))


def _paths(split: Callable, fused: Callable) -> dict[str, Callable]:
    return {"split": split, "fused": fused}


FUSED_TEST_CFG = {
    "path": "fused",
    "BM": 64,
    "BN": 64,
    "BK": 32,
    "warps": 4,
    "stages": 3,
}
_FORCED: dict[str, dict] = {}
_BOUND: dict[tuple, Callable] = {}


def force_path(cfg: dict | None) -> None:
    """Pin every fused-linear op to one config (tests and benches); None = tuned."""
    _FORCED.clear()
    _BOUND.clear()
    if cfg is not None:
        _FORCED["cfg"] = cfg


def _kept(t: Tensor, needed: bool) -> Tensor:
    """`t` when the backward uses it, else an empty tensor of its dtype and device."""
    return t if needed else t.new_empty(0)


def _bound(name, device, key, cfgs, paths, args) -> Callable:
    """The runner (`paths[cfg["path"]]` with its tuned config) of this problem, bound
    once the choice is settled; called as runner(*args)."""
    ident = (name, str(device), key)
    fn = _BOUND.get(ident)
    if fn is not None:
        return fn
    cfg = _FORCED.get("cfg") or select(
        name,
        device,
        key,
        lambda: cfgs,
        lambda c: lambda: paths[c["path"]](c, *args),
        shortlist=len(cfgs),
    )
    fn = partial(paths[cfg["path"]], cfg)
    if _FORCED or decided(name, device, key):
        _BOUND[ident] = fn
    return fn


on_reset(_BOUND.clear)


# ---------------------------------------------------------------- SiTU-GLU


def _situ_split(cfg, y, w, bias, save):
    ug = F.linear(y, w, bias)
    return situ_forward(ug), (ug if save else y.new_empty(0))


def _situ_fused(cfg, y, w, bias, save):
    m, k = y.shape
    f = w.shape[0] // 2
    out = torch.empty(m, f, device=y.device, dtype=y.dtype)
    ug = (
        torch.empty(m, 2 * f, device=y.device, dtype=y.dtype)
        if save
        else y.new_empty(0)
    )
    grid = (cdiv(m, cfg["BM"]) * cdiv(f, cfg["BN"]),)
    _situ[grid](
        y, w, bias if bias is not None else w, out, ug if save else out, m, f, k,
        HAS_BIAS=bias is not None, SAVE=save, BM=cfg["BM"], BN=cfg["BN"], BK=cfg["BK"],
        GROUP_M=8, num_warps=cfg["warps"], num_stages=cfg["stages"],
    )  # fmt: skip
    return out, ug


SITU = _paths(_situ_split, _situ_fused)


@triton_op("kohakumoss::triton_linear_situ_fwd", mutates_args=())
def _situ_fwd(
    y: Tensor, w: Tensor, bias: Tensor | None, save: bool
) -> tuple[Tensor, Tensor]:
    key = _key(y.shape[0], y.shape[1], w.shape[0], y.dtype, bias is not None, save)
    args = (y, operand(w, y), operand(bias, y), save)
    return _bound("triton_linear_situ", y.device, key, SITU_CFGS, SITU, args)(*args)


@triton_op("kohakumoss::triton_linear_situ_bwd", mutates_args=())
def _situ_bwd(
    ga: Tensor,
    y: Tensor,
    w: Tensor,
    ug: Tensor,
    need_y: bool,
    need_w: bool,
    need_b: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    dug = situ_backward(ga, ug)
    dy = dug @ operand(w, dug) if need_y else y.new_empty(0)
    dw = weight_grad(dug, y, w.dtype) if need_w else y.new_empty(0)
    db = dug.float().sum(0) if need_b else y.new_empty(0)
    return dy, dw, db


def _situ_setup(ctx, inputs, output):
    y, w, bias, _ = inputs
    need = ctx.needs_input_grad
    ctx.save_for_backward(_kept(y, need[1]), _kept(w, need[0]), output[1])
    ctx.b_dtype = None if bias is None else bias.dtype


def _situ_backward(ctx, ga, _gug):
    y, w, ug = ctx.saved_tensors
    need = ctx.needs_input_grad
    dy, dw, db = _situ_bwd(ga.contiguous(), y, w, ug, need[0], need[1], need[2])
    return (
        dy if need[0] else None,
        dw if need[1] else None,
        db.to(ctx.b_dtype) if need[2] else None,
        None,
    )


_situ_fwd.register_autograd(_situ_backward, setup_context=_situ_setup)


def linear_situ(y: Tensor, w_up_gate: Tensor, bias: Tensor | None) -> Tensor:
    """situ_glu(y W^T + b) with W = [W_up; W_gate] [2F, K]."""
    shape = y.shape
    y2 = activation(y.reshape(-1, shape[-1])).contiguous()
    save = torch.is_grad_enabled() and (
        y.requires_grad
        or w_up_gate.requires_grad
        or (bias is not None and bias.requires_grad)
    )
    a, _ = _situ_fwd(y2, w_up_gate, bias, save)
    return a.view(*shape[:-1], a.shape[-1])


# ---------------------------------------------------------- gated residual


def _res_split(cfg, a, w, bias, x, gate, seg, r, save):
    o = F.linear(a, w, bias)
    return residual_forward(x, o, gate, seg, r), (o if save else a.new_empty(0))


def _res_fused(cfg, a, w, bias, x, gate, seg, r, save):
    m, k = a.shape
    n = w.shape[0]
    out = torch.empty_like(x)
    o = torch.empty(m, n, device=a.device, dtype=SAVE_DTYPE) if save else a.new_empty(0)
    grid = (cdiv(m, cfg["BM"]) * cdiv(n, cfg["BN"]),)
    _res[grid](
        a, w, bias if bias is not None else w, x, gate if gate is not None else x,
        seg if seg is not None else x, out, o if save else out, m, n, k, r,
        *strides2(gate), HAS_BIAS=bias is not None, HAS_GATE=gate is not None, SCALED=r != 1.0, SAVE=save,
        BM=cfg["BM"], BN=cfg["BN"], BK=cfg["BK"], GROUP_M=8, num_warps=cfg["warps"],
        num_stages=cfg["stages"],
    )  # fmt: skip
    return out, o


RES = _paths(_res_split, _res_fused)


@triton_op("kohakumoss::triton_linear_gres_fwd", mutates_args=())
def _res_fwd(
    a: Tensor,
    w: Tensor,
    bias: Tensor | None,
    x: Tensor,
    gate: Tensor | None,
    seg: Tensor | None,
    r: float,
    save: bool,
) -> tuple[Tensor, Tensor]:
    key = _key(a.shape[0], a.shape[1], w.shape[0], a.dtype, gate is not None, save)
    args = (a, operand(w, a), operand(bias, a), x, gate, seg, r, save)
    return _bound("triton_linear_gres", a.device, key, RES_CFGS, RES, args)(*args)


@triton_op("kohakumoss::triton_linear_gres_bwd", mutates_args=())
def _res_bwd(
    g: Tensor,
    a: Tensor,
    w: Tensor,
    o: Tensor,
    gate: Tensor | None,
    seg: Tensor | None,
    r: float,
    need_a: bool,
    need_w: bool,
    need_b: bool,
    need_gate: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """(d_x or empty when r == 1, d_a, d_w, d_b fp32, d_gate fp32)."""
    if gate is None:
        do = g
        dx = (g.float() * r).to(g.dtype) if r != 1.0 else g.new_empty(0)
        dgate = g.new_empty(0)
    else:
        dx, do, dgate = residual_backward(g, o, gate, seg, r, need_gate)
    do = do.to(a.dtype)
    da = do @ operand(w, do) if need_a else g.new_empty(0)
    dw = weight_grad(do, a, w.dtype) if need_w else g.new_empty(0)
    db = do.float().sum(0) if need_b else g.new_empty(0)
    return dx, da, dw, db, dgate


def _res_setup(ctx, inputs, output):
    a, w, bias, x, gate, seg, r, _ = inputs
    need = ctx.needs_input_grad
    ctx.save_for_backward(_kept(a, need[1]), _kept(w, need[0]), output[1], gate, seg)
    ctx.r = r
    ctx.dtypes = (None if bias is None else bias.dtype, x.dtype)


def _res_backward(ctx, gy, _go):
    a, w, o, gate, seg = ctx.saved_tensors
    need = ctx.needs_input_grad
    gy = gy.contiguous()
    dx, da, dw, db, dgate = _res_bwd(
        gy, a, w, o, gate, seg, ctx.r, need[0], need[1], need[2], need[4]
    )
    dx = gy if ctx.r == 1.0 else dx
    return (
        da if need[0] else None,
        dw if need[1] else None,
        db.to(ctx.dtypes[0]) if need[2] else None,
        dx if need[3] else None,
        dgate.to(gate.dtype) if need[4] else None,
        None,
        None,
        None,
    )


_res_fwd.register_autograd(_res_backward, setup_context=_res_setup)


def linear_gated_residual(
    a: Tensor,
    w: Tensor,
    bias: Tensor | None,
    x: Tensor,
    gate: Tensor | None,
    seg: Tensor | None,
    r: float,
) -> Tensor:
    """x * r + (a W^T + b) * gate[seg]; gate [B, D] per sample (any strides)."""
    shape = x.shape
    a2 = activation(a.reshape(-1, a.shape[-1])).contiguous()
    save = torch.is_grad_enabled() and gate is not None and gate.requires_grad
    out, _ = _res_fwd(
        a2,
        w,
        bias,
        x.reshape(-1, shape[-1]).contiguous(),
        gate,
        None if seg is None else seg.contiguous(),
        float(r),
        save,
    )
    return out.view(shape)


# --------------------------------------------------------- qk-norm + rope


def _qk_split(cfg, y, wq, wk, qw, kw, pos, bands, eps, heads, save):
    m = y.shape[0]
    d = wq.shape[0]
    dh = d // heads
    qp = F.linear(y, wq).view(m, heads, dh)
    kp = F.linear(y, wk).view(m, heads, dh)
    qo, ko, qr, kr = rope_forward(qp, kp, qw, kw, pos, bands, eps)
    if not save:
        qp, kp = y.new_empty(0), y.new_empty(0)
    return qo.view(m, d), ko.view(m, d), qp.reshape(-1), kp.reshape(-1), qr, kr


def _qk_fused(cfg, y, wq, wk, qw, kw, pos, bands, eps, heads, save):
    m, k = y.shape
    d = wq.shape[0]
    dh = d // heads
    qo = torch.empty(m, d, device=y.device, dtype=y.dtype)
    ko = torch.empty_like(qo)
    qp = torch.empty_like(qo) if save else y.new_empty(0)
    kp = torch.empty_like(qo) if save else y.new_empty(0)
    qr = torch.empty(m * heads, device=y.device, dtype=torch.float32)
    kr = torch.empty_like(qr)
    grid = (cdiv(m, cfg["BM"]) * 2 * heads,)
    _qk[grid](
        y, wq, wk, qw, kw, pos, bands, qo, ko, qp if save else qo, kp if save else ko,
        qr, kr, m, k, heads, pos.shape[1], eps, NB=bands.shape[0], HALF=dh // 2, SAVE=save,
        BM=cfg["BM"], BK=cfg["BK"], GROUP_M=8, num_warps=cfg["warps"],
        num_stages=cfg["stages"],
    )  # fmt: skip
    return qo, ko, qp.reshape(-1), kp.reshape(-1), qr, kr


QK = _paths(_qk_split, _qk_fused)


@triton_op("kohakumoss::triton_linear_qk_fwd", mutates_args=())
def _qk_fwd(
    y: Tensor,
    wq: Tensor,
    wk: Tensor,
    qw: Tensor,
    kw: Tensor,
    pos: Tensor,
    bands: Tensor,
    eps: float,
    heads: int,
    save: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    key = _key(y.shape[0], y.shape[1], wq.shape[0], heads, y.dtype, save)
    args = (y, operand(wq, y), operand(wk, y), qw, kw, pos, bands, eps, heads, save)
    return _bound("triton_linear_qk", y.device, key, QK_CFGS, QK, args)(*args)


@triton_op("kohakumoss::triton_linear_qk_bwd", mutates_args=())
def _qk_bwd(
    gq: Tensor,
    gk: Tensor,
    y: Tensor,
    wq: Tensor,
    wk: Tensor,
    qp: Tensor,
    kp: Tensor,
    qr: Tensor,
    kr: Tensor,
    qw: Tensor,
    kw: Tensor,
    pos: Tensor,
    bands: Tensor,
    heads: int,
    need_y: bool,
    need_wq: bool,
    need_wk: bool,
    eps: float,
    need_w: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    m, d = gq.shape
    shape = (m, heads, d // heads)
    dqk = torch.empty(m, 2 * d, device=gq.device, dtype=qp.dtype)
    dqp, dkp, dqw, dkw = rope_backward(
        gq.view(shape),
        gk.view(shape),
        qp.view(shape),
        kp.view(shape),
        qr,
        kr,
        qw,
        kw,
        pos,
        bands,
        eps,
        need_w,
        dqk,
    )
    dy = dqk @ torch.cat((wq, wk)).to(dqk.dtype) if need_y else y.new_empty(0)
    dwq = weight_grad(dqp, y, wq.dtype) if need_wq else y.new_empty(0)
    dwk = weight_grad(dkp, y, wk.dtype) if need_wk else y.new_empty(0)
    return dy, dwq, dwk, dqw, dkw


def _qk_setup(ctx, inputs, output):
    y, wq, wk, qw, kw, pos, bands, eps, heads, _ = inputs
    need = ctx.needs_input_grad
    ctx.save_for_backward(
        _kept(y, need[1] or need[2]), _kept(wq, need[0]), _kept(wk, need[0]),
        output[2], output[3], output[4], output[5], qw, kw, pos, bands,
    )  # fmt: skip
    ctx.heads = heads
    ctx.eps = eps
    ctx.out_shape = (y.shape[0], wq.shape[0])


def _qk_backward(ctx, gq, gk, *_rest):
    y, wq, wk, qp, kp, qr, kr, qw, kw, pos, bands = ctx.saved_tensors
    need = ctx.needs_input_grad
    out_shape = ctx.out_shape
    gq = y.new_zeros(out_shape) if gq is None else gq
    gk = y.new_zeros(out_shape) if gk is None else gk
    dy, dwq, dwk, dqw, dkw = _qk_bwd(
        gq.contiguous(), gk.contiguous(), y, wq, wk, qp, kp, qr, kr, qw, kw, pos, bands,
        ctx.heads, need[0], need[1], need[2], ctx.eps, need[3] or need[4],
    )  # fmt: skip
    return (
        dy if need[0] else None,
        dwq if need[1] else None,
        dwk if need[2] else None,
        dqw if need[3] else None,
        dkw if need[4] else None,
        None,
        None,
        None,
        None,
        None,
    )


_qk_fwd.register_autograd(_qk_backward, setup_context=_qk_setup)


def linear_qk_rope(
    y: Tensor,
    wq: Tensor,
    wk: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    pos: Tensor,
    bands: Tensor,
    eps: float,
    heads: int,
) -> tuple[Tensor, Tensor]:
    """(q, k) [N, H, Dh] = qk_norm_rope(y Wq^T, y Wk^T)."""
    m = y.reshape(-1, y.shape[-1]).shape[0]
    d = wq.shape[0]
    save = torch.is_grad_enabled() and any(
        t.requires_grad for t in (y, wq, wk, q_weight, k_weight)
    )
    qo, ko, _, _, _, _ = _qk_fwd(
        activation(y.reshape(m, -1)).contiguous(),
        wq,
        wk,
        q_weight.contiguous(),
        k_weight.contiguous(),
        pos.float().contiguous(),
        bands.float().contiguous(),
        eps,
        heads,
        save,
    )
    return qo.view(m, heads, d // heads), ko.view(m, heads, d // heads)


_SUPPORTS = requires(kinds=("cuda",), min_sm=80, dtypes=DTYPES)
register(
    "linear_situ",
    "triton",
    raced("triton", "linear_situ", linear_situ, shape_key),
    _SUPPORTS,
)
register(
    "linear_gated_residual",
    "triton",
    raced("triton", "linear_gated_residual", linear_gated_residual, shape_key),
    _SUPPORTS,
)
register(
    "linear_qk_rope",
    "triton",
    raced("triton", "linear_qk_rope", linear_qk_rope, shape_key),
    _SUPPORTS,
)
