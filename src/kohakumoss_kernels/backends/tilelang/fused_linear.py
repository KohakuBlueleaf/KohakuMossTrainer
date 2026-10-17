"""TileLang fused linear ops (fp16, sm70+): GEMM + MOSS epilogue, raced per shape
against cuBLAS + the standalone TileLang epilogue (`path=split`).

The backward is the epilogue backward kernel followed by cuBLAS GEMMs; the forward
saves the pre-activation / pre-norm / projection only while autograd records.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.library import custom_op

from kohakumoss_kernels.backends._master import activation, operand, weight_grad
from kohakumoss_kernels.backends.compile.ops import compiled_op
from kohakumoss_kernels.backends.compile.race import raced, shape_key
from kohakumoss_kernels.backends.tilelang import _gemm_prims as prims
from kohakumoss_kernels.backends.tilelang._blockops import (
    CompileError,
    compiled,
    per_sample,
    tuned,
)
from kohakumoss_kernels.backends.tilelang.ffn import situ_backward, situ_forward
from kohakumoss_kernels.backends.tilelang.residual import (
    residual_backward,
    residual_forward,
)
from kohakumoss_kernels.backends.tilelang.rope import _backward_one, _forward
from kohakumoss_kernels.constraints import requires
from kohakumoss_kernels.registry import register

__all__ = ["linear_situ", "linear_gated_residual", "linear_qk_rope"]

SPLIT = {"path": "split"}
FUSED_TEST_CFG = {
    "path": "fused",
    "bm": 64,
    "bn": 64,
    "bk": 32,
    "threads": 128,
    "stages": 1,
}
_FORCED: dict[str, dict] = {}


def _tiles(*rows) -> list[dict]:
    return [
        {"path": "fused", "bm": bm, "bn": bn, "bk": bk, "threads": th, "stages": st}
        for bm, bn, bk, th, st in rows
    ]


GEMM_CFGS = _tiles(
    (128, 128, 32, 256, 1), (128, 64, 32, 128, 1), (64, 128, 32, 128, 1), (128, 128, 64, 256, 1)
) + [SPLIT]  # fmt: skip
QK_CFGS = _tiles((64, 0, 32, 128, 1), (128, 0, 32, 256, 1), (64, 0, 64, 128, 1)) + [
    SPLIT
]


def force_path(cfg: dict | None) -> None:
    """Pin every fused-linear op to one config (tests and benches); None = tuned."""
    _FORCED.clear()
    if cfg is not None:
        _FORCED["cfg"] = cfg


def _choose(name, device, key, cfgs, run):
    if _FORCED:
        return _FORCED["cfg"]
    return tuned(name, device, key, cfgs, run)


def _bucket(m: int) -> int:
    b = 256
    while b < m:
        b *= 2
    return b


# ---------------------------------------------------------------- SiTU-GLU


def _situ_run(cfg, y, w, bias, save):
    m, k = y.shape
    f = w.shape[0] // 2
    if cfg["path"] == "split" or bias is not None:
        ug = F.linear(y, w, bias)
        return situ_forward(ug), (ug if save else y.new_empty(0))
    out = torch.empty(m, f, device=y.device, dtype=y.dtype)
    ug = torch.empty(m, 2 * f, device=y.device, dtype=y.dtype) if save else out
    kernel = compiled(
        prims.situ_gemm_prim, y.device, k, f, cfg["bm"], cfg["bn"], cfg["bk"],
        cfg["threads"], cfg["stages"], save,
    )  # fmt: skip
    kernel(y, w, out, ug)
    return out, (ug if save else y.new_empty(0))


@custom_op("kohakumoss::tilelang_linear_situ_fwd", mutates_args=())
def _situ_fwd(
    y: Tensor, w: Tensor, bias: Tensor | None, save: bool
) -> tuple[Tensor, Tensor]:
    key = f"{_bucket(y.shape[0])}-{y.shape[1]}-{w.shape[0]}-{bias is not None}-{save}"
    w, bias = operand(w, y), operand(bias, y)
    cfg = _choose(
        "tilelang_linear_situ",
        y.device,
        key,
        GEMM_CFGS,
        lambda c: _situ_run(c, y, w, bias, save),
    )
    return _situ_run(cfg, y, w, bias, save)


@_situ_fwd.register_fake
def _(y, w, bias, save):
    m, f2 = y.shape[0], w.shape[0]
    return y.new_empty(m, f2 // 2), (y.new_empty(m, f2) if save else y.new_empty(0))


@custom_op("kohakumoss::tilelang_linear_situ_bwd", mutates_args=())
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


@_situ_bwd.register_fake
def _(ga, y, w, ug, need_y, need_w, need_b):
    m, f2 = ga.shape[0], ug.shape[1]
    return (
        y.new_empty(m, w.shape[1]) if need_y else y.new_empty(0),
        y.new_empty(f2, y.shape[1], dtype=w.dtype) if need_w else y.new_empty(0),
        y.new_empty(f2, dtype=torch.float32) if need_b else y.new_empty(0),
    )


def _kept(t: Tensor, needed: bool) -> Tensor:
    """`t` when the backward uses it, else an empty tensor of its dtype and device."""
    return t if needed else t.new_empty(0)


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
    save = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (y, w_up_gate, bias)
    )
    try:
        y2 = activation(y.reshape(-1, shape[-1])).contiguous()
        a, _ = _situ_fwd(y2, w_up_gate, bias, save)
    except CompileError:
        return compiled_op("linear_situ")(y, w_up_gate, bias)
    return a.view(*shape[:-1], a.shape[-1])


# ---------------------------------------------------------- gated residual


def _res_run(cfg, a, w, bias, x, gate, seg, r, save):
    m, k = a.shape
    n = w.shape[0]
    if cfg["path"] == "split" or bias is not None or x.dtype != a.dtype:
        o = F.linear(a, w, bias)
        return residual_forward(x, o, gate, seg, r), (o if save else a.new_empty(0))
    out = torch.empty_like(x)
    o = torch.empty(m, n, device=a.device, dtype=a.dtype) if save else out
    gf, gr, gc = (
        per_sample(gate, *gate.shape, a.device)
        if gate is not None
        else per_sample(None, 1, n, a.device)
    )
    sg = (
        seg.to(torch.int32).contiguous()
        if seg is not None
        else torch.zeros(m, device=a.device, dtype=torch.int32)
    )
    kernel = compiled(
        prims.residual_gemm_prim, a.device, k, n, cfg["bm"], cfg["bn"], cfg["bk"],
        cfg["threads"], cfg["stages"], gate is not None, r != 1.0, save,
    )  # fmt: skip
    kernel(a, w, x, gf, sg, out, o, float(r), gr, gc)
    return out, (o if save else a.new_empty(0))


@custom_op("kohakumoss::tilelang_linear_gres_fwd", mutates_args=())
def _res_fwd(
    a: Tensor, w: Tensor, bias: Tensor | None, x: Tensor, gate: Tensor | None,
    seg: Tensor | None, r: float, save: bool,
) -> tuple[Tensor, Tensor]:  # fmt: skip
    key = f"{_bucket(a.shape[0])}-{a.shape[1]}-{w.shape[0]}-{gate is not None}-{r != 1.0}-{save}"
    w, bias = operand(w, a), operand(bias, a)
    run = lambda c: _res_run(c, a, w, bias, x, gate, seg, r, save)  # noqa: E731
    return run(_choose("tilelang_linear_gres", a.device, key, GEMM_CFGS, run))


@_res_fwd.register_fake
def _(a, w, bias, x, gate, seg, r, save):
    return torch.empty_like(x), (
        a.new_empty(a.shape[0], w.shape[0]) if save else a.new_empty(0)
    )


@custom_op("kohakumoss::tilelang_linear_gres_bwd", mutates_args=())
def _res_bwd(
    g: Tensor, a: Tensor, w: Tensor, o: Tensor, gate: Tensor | None, seg: Tensor | None,
    r: float, need_a: bool, need_w: bool, need_b: bool, need_gate: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:  # fmt: skip
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


@_res_bwd.register_fake
def _(g, a, w, o, gate, seg, r, need_a, need_w, need_b, need_gate):
    e = g.new_empty(0)
    return (
        torch.empty_like(g) if r != 1.0 else e,
        g.new_empty(g.shape[0], w.shape[1]) if need_a else e,
        g.new_empty(g.shape[1], a.shape[1], dtype=w.dtype) if need_w else e,
        g.new_empty(g.shape[1], dtype=torch.float32) if need_b else e,
        (
            g.new_empty(gate.shape, dtype=torch.float32)
            if (gate is not None and need_gate)
            else e
        ),
    )


def _res_setup(ctx, inputs, output):
    a, w, bias, x, gate, seg, r, _ = inputs
    need = ctx.needs_input_grad
    ctx.save_for_backward(_kept(a, need[1]), _kept(w, need[0]), output[1], gate, seg)
    ctx.r = r
    ctx.b_dtype = None if bias is None else bias.dtype


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
        db.to(ctx.b_dtype) if need[2] else None,
        dx if need[3] else None,
        dgate.to(gate.dtype) if need[4] else None,
        None,
        None,
        None,
    )


_res_fwd.register_autograd(_res_backward, setup_context=_res_setup)


def linear_gated_residual(a, w, bias, x, gate, seg, r) -> Tensor:
    """x * r + (a W^T + b) * gate[seg]."""
    shape = x.shape
    save = torch.is_grad_enabled() and gate is not None and gate.requires_grad
    try:
        out, _ = _res_fwd(
            activation(a.reshape(-1, a.shape[-1])).contiguous(), w, bias,
            x.reshape(-1, shape[-1]).contiguous(), gate, seg, float(r), save,
        )  # fmt: skip
    except CompileError:
        return compiled_op("linear_gated_residual")(a, w, bias, x, gate, seg, r)
    return out.view(shape)


# --------------------------------------------------------- qk-norm + rope


def _qk_run(cfg, y, wq, wk, qw, kw, pos, bands, eps, heads, save):
    m, k = y.shape
    d = wq.shape[0]
    dh = d // heads
    if cfg["path"] == "split":
        qp = F.linear(y, wq).view(m, heads, dh)
        kp = F.linear(y, wk).view(m, heads, dh)
        qo, ko, qr, kr = _forward(qp, kp, qw, kw, pos, bands, eps)
        if not save:
            qp, kp = y.new_empty(0), y.new_empty(0)
        return qo.view(m, d), ko.view(m, d), qp.reshape(-1), kp.reshape(-1), qr, kr
    qo = torch.empty(m, d, device=y.device, dtype=y.dtype)
    ko = torch.empty_like(qo)
    qp = torch.empty_like(qo) if save else qo
    kp = torch.empty_like(qo) if save else ko
    qr = torch.empty(m, heads, device=y.device, dtype=torch.float32)
    kr = torch.empty_like(qr)
    kernel = compiled(
        prims.qk_gemm_prim, y.device, k, heads, dh // 2, bands.shape[0], pos.shape[1],
        cfg["bm"], cfg["bk"], cfg["threads"], cfg["stages"], save,
    )  # fmt: skip
    kernel(y, wq, wk, qw, kw, pos, bands, qo, ko, qp, kp, qr, kr, float(eps))
    if not save:
        qp, kp = y.new_empty(0), y.new_empty(0)
    return qo, ko, qp.reshape(-1), kp.reshape(-1), qr.reshape(-1), kr.reshape(-1)


@custom_op("kohakumoss::tilelang_linear_qk_fwd", mutates_args=())
def _qk_fwd(
    y: Tensor, wq: Tensor, wk: Tensor, qw: Tensor, kw: Tensor, pos: Tensor, bands: Tensor,
    eps: float, heads: int, save: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:  # fmt: skip
    key = f"{_bucket(y.shape[0])}-{y.shape[1]}-{wq.shape[0]}-{heads}-{save}"
    wq, wk = operand(wq, y), operand(wk, y)

    def run(c):
        return _qk_run(c, y, wq, wk, qw, kw, pos, bands, eps, heads, save)  # noqa: E731

    return run(_choose("tilelang_linear_qk", y.device, key, QK_CFGS, run))


@_qk_fwd.register_fake
def _(y, wq, wk, qw, kw, pos, bands, eps, heads, save):
    m, d = y.shape[0], wq.shape[0]
    pre = y.new_empty(m * d) if save else y.new_empty(0)
    r = y.new_empty(m * heads, dtype=torch.float32)
    return (
        y.new_empty(m, d),
        y.new_empty(m, d),
        pre,
        torch.empty_like(pre),
        r,
        torch.empty_like(r),
    )


@custom_op("kohakumoss::tilelang_linear_qk_bwd", mutates_args=())
def _qk_bwd(
    gq: Tensor, gk: Tensor, y: Tensor, wq: Tensor, wk: Tensor, qp: Tensor, kp: Tensor,
    qr: Tensor, kr: Tensor, qw: Tensor, kw: Tensor, pos: Tensor, bands: Tensor, heads: int,
    need_y: bool, need_wq: bool, need_wk: bool, eps: float, need_w: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:  # fmt: skip
    m, d = gq.shape
    shape = (m, heads, d // heads)
    gqs, gks, qps, kps = gq.view(shape), gk.view(shape), qp.view(shape), kp.view(shape)
    dqk = torch.empty(m, 2 * d, device=gq.device, dtype=qp.dtype)
    dqp, dkp = dqk[:, :d], dqk[:, d:]
    _, dqw = _backward_one(gqs, qps, qr, qw, pos, bands, eps, need_w, dqp)
    _, dkw = _backward_one(gks, kps, kr, kw, pos, bands, eps, need_w, dkp)
    dy = dqk @ torch.cat((wq, wk)).to(dqk.dtype) if need_y else y.new_empty(0)
    dwq = weight_grad(dqp, y, wq.dtype) if need_wq else y.new_empty(0)
    dwk = weight_grad(dkp, y, wk.dtype) if need_wk else y.new_empty(0)
    return dy, dwq, dwk, dqw, dkw


@_qk_bwd.register_fake
def _(
    gq,
    gk,
    y,
    wq,
    wk,
    qp,
    kp,
    qr,
    kr,
    qw,
    kw,
    pos,
    bands,
    heads,
    need_y,
    need_wq,
    need_wk,
    eps,
    need_w,
):
    e = y.new_empty(0)
    m, d = gq.shape
    w = y.new_empty(d // heads if need_w else 0, dtype=torch.float32)
    return (
        y.new_empty(m, wq.shape[1]) if need_y else e,
        y.new_empty(d, y.shape[1], dtype=wq.dtype) if need_wq else e,
        y.new_empty(d, y.shape[1], dtype=wk.dtype) if need_wk else e,
        w,
        torch.empty_like(w),
    )


def _qk_setup(ctx, inputs, output):
    y, wq, wk, qw, kw, pos, bands, eps, heads, _ = inputs
    need = ctx.needs_input_grad
    ctx.save_for_backward(
        _kept(y, need[1] or need[2]), _kept(wq, need[0]), _kept(wk, need[0]),
        output[2], output[3], output[4], output[5], qw, kw, pos, bands,
    )  # fmt: skip
    ctx.heads = heads
    ctx.eps = eps
    ctx.w_dtypes = (qw.dtype, kw.dtype)
    ctx.out_shape = (y.shape[0], wq.shape[0])


def _qk_backward(ctx, gq, gk, *_rest):
    y, wq, wk, qp, kp, qr, kr, qw, kw, pos, bands = ctx.saved_tensors
    need = ctx.needs_input_grad
    out_shape = ctx.out_shape
    gq = y.new_zeros(out_shape) if gq is None else gq.contiguous()
    gk = y.new_zeros(out_shape) if gk is None else gk.contiguous()
    dy, dwq, dwk, dqw, dkw = _qk_bwd(
        gq, gk, y, wq, wk, qp, kp, qr, kr, qw, kw, pos, bands, ctx.heads,
        need[0], need[1], need[2], ctx.eps, need[3] or need[4],
    )  # fmt: skip
    return (
        dy if need[0] else None,
        dwq if need[1] else None,
        dwk if need[2] else None,
        dqw.to(ctx.w_dtypes[0]) if need[3] else None,
        dkw.to(ctx.w_dtypes[1]) if need[4] else None,
        None, None, None, None, None,
    )  # fmt: skip


_qk_fwd.register_autograd(_qk_backward, setup_context=_qk_setup)


def linear_qk_rope(y, wq, wk, q_weight, k_weight, pos, bands, eps, heads):
    """(q, k) [N, H, Dh] = qk_norm_rope(y Wq^T, y Wk^T)."""
    m = y.reshape(-1, y.shape[-1]).shape[0]
    d = wq.shape[0]
    save = torch.is_grad_enabled() and any(
        t.requires_grad for t in (y, wq, wk, q_weight, k_weight)
    )
    try:
        qo, ko, _, _, _, _ = _qk_fwd(
            activation(y.reshape(m, -1)).contiguous(), wq, wk, q_weight.float().contiguous(),
            k_weight.float().contiguous(), pos.float().contiguous(),
            bands.float().contiguous(), eps, heads, save,
        )  # fmt: skip
    except CompileError:
        return compiled_op("linear_qk_rope")(
            y, wq, wk, q_weight, k_weight, pos, bands, eps, heads
        )
    return qo.view(m, heads, d // heads), ko.view(m, heads, d // heads)


_SUPPORTS = requires(kinds=("cuda",), min_sm=70, dtypes=(torch.float16,))
for _name, _fn in (
    ("linear_situ", linear_situ),
    ("linear_gated_residual", linear_gated_residual),
    ("linear_qk_rope", linear_qk_rope),
):
    register(_name, "tilelang", raced("tilelang", _name, _fn, shape_key), _SUPPORTS)
