"""TileLang fused linear ops (fp16, sm70+): GEMM + MOSS epilogue, raced per shape
against cuBLAS + the standalone TileLang epilogue (`path=split`).

The backward is the epilogue backward kernel followed by cuBLAS GEMMs; the forward
saves the pre-activation / pre-norm / projection only while autograd records.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.library import custom_op

from kohakumoss_kernels.backends.compile.ops import compiled_op
from kohakumoss_kernels.backends.compile.race import raced, shape_key
from kohakumoss_kernels.backends.tilelang import _gemm_prims as prims
from kohakumoss_kernels.backends.tilelang._blockops import (
    CompileError,
    compiled,
    f32,
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


@custom_op("kohakumoss::f6tl_linear_situ_fwd", mutates_args=())
def _situ_fwd(
    y: Tensor, w: Tensor, bias: Tensor | None, save: bool
) -> tuple[Tensor, Tensor]:
    key = f"{_bucket(y.shape[0])}-{y.shape[1]}-{w.shape[0]}-{bias is not None}-{save}"
    cfg = _choose(
        "f6tl_linear_situ",
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


@custom_op("kohakumoss::f6tl_linear_situ_bwd", mutates_args=())
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
    dy = dug @ w if need_y else y.new_empty(0)
    dw = (dug.t() @ y).to(w.dtype) if need_w else y.new_empty(0)
    db = dug.float().sum(0) if need_b else y.new_empty(0)
    return dy, dw, db


@_situ_bwd.register_fake
def _(ga, y, w, ug, need_y, need_w, need_b):
    return (
        torch.empty_like(y) if need_y else y.new_empty(0),
        torch.empty_like(w) if need_w else y.new_empty(0),
        y.new_empty(w.shape[0], dtype=torch.float32) if need_b else y.new_empty(0),
    )


def _situ_setup(ctx, inputs, output):
    y, w, bias, _ = inputs
    ctx.save_for_backward(y, w, output[1])
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
        a, _ = _situ_fwd(
            y.reshape(-1, shape[-1]).contiguous(),
            w_up_gate.to(y.dtype).contiguous(),
            None if bias is None else bias.to(y.dtype),
            save,
        )
    except CompileError:
        return compiled_op("linear_situ")(y, w_up_gate, bias)
    return a.view(*shape[:-1], a.shape[-1])


# ---------------------------------------------------------- gated residual


def _res_run(cfg, a, w, bias, x, gate, seg, r, save):
    m, k = a.shape
    n = w.shape[0]
    if cfg["path"] == "split" or bias is not None:
        o = F.linear(a, w, bias)
        return residual_forward(x, o, gate, seg, r), (o if save else a.new_empty(0))
    out = torch.empty_like(x)
    o = torch.empty(m, n, device=a.device, dtype=a.dtype) if save else out
    gf = (
        f32(gate, (1, n), a.device)
        if gate is not None
        else torch.ones(1, n, device=a.device)
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
    kernel(a, w, x, gf, sg, out, o, float(r))
    return out, (o if save else a.new_empty(0))


@custom_op("kohakumoss::f6tl_linear_gres_fwd", mutates_args=())
def _res_fwd(
    a: Tensor, w: Tensor, bias: Tensor | None, x: Tensor, gate: Tensor | None,
    seg: Tensor | None, r: float, save: bool,
) -> tuple[Tensor, Tensor]:  # fmt: skip
    key = f"{_bucket(a.shape[0])}-{a.shape[1]}-{w.shape[0]}-{gate is not None}-{r != 1.0}-{save}"
    run = lambda c: _res_run(c, a, w, bias, x, gate, seg, r, save)  # noqa: E731
    return run(_choose("f6tl_linear_gres", a.device, key, GEMM_CFGS, run))


@_res_fwd.register_fake
def _(a, w, bias, x, gate, seg, r, save):
    return torch.empty_like(x), (
        a.new_empty(a.shape[0], w.shape[0]) if save else a.new_empty(0)
    )


@custom_op("kohakumoss::f6tl_linear_gres_bwd", mutates_args=())
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
    da = do @ w if need_a else g.new_empty(0)
    dw = (do.t() @ a).to(w.dtype) if need_w else g.new_empty(0)
    db = do.float().sum(0) if need_b else g.new_empty(0)
    return dx, da, dw, db, dgate


@_res_bwd.register_fake
def _(g, a, w, o, gate, seg, r, need_a, need_w, need_b, need_gate):
    e = g.new_empty(0)
    return (
        torch.empty_like(g) if r != 1.0 else e,
        torch.empty_like(a) if need_a else e,
        torch.empty_like(w) if need_w else e,
        g.new_empty(w.shape[0], dtype=torch.float32) if need_b else e,
        (
            g.new_empty(gate.shape, dtype=torch.float32)
            if (gate is not None and need_gate)
            else e
        ),
    )


def _res_setup(ctx, inputs, output):
    a, w, bias, x, gate, seg, r, _ = inputs
    ctx.save_for_backward(a, w, output[1], gate, seg)
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
            a.reshape(-1, a.shape[-1]).contiguous(), w.to(a.dtype).contiguous(),
            None if bias is None else bias.to(a.dtype), x.reshape(-1, shape[-1]).contiguous(),
            gate, seg, float(r), save,
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


@custom_op("kohakumoss::f6tl_linear_qk_fwd", mutates_args=())
def _qk_fwd(
    y: Tensor, wq: Tensor, wk: Tensor, qw: Tensor, kw: Tensor, pos: Tensor, bands: Tensor,
    eps: float, heads: int, save: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:  # fmt: skip
    key = f"{_bucket(y.shape[0])}-{y.shape[1]}-{wq.shape[0]}-{heads}-{save}"

    def run(c):
        return _qk_run(c, y, wq, wk, qw, kw, pos, bands, eps, heads, save)  # noqa: E731

    return run(_choose("f6tl_linear_qk", y.device, key, QK_CFGS, run))


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


@custom_op("kohakumoss::f6tl_linear_qk_bwd", mutates_args=())
def _qk_bwd(
    gq: Tensor, gk: Tensor, y: Tensor, wq: Tensor, wk: Tensor, qp: Tensor, kp: Tensor,
    qr: Tensor, kr: Tensor, qw: Tensor, kw: Tensor, pos: Tensor, bands: Tensor, heads: int,
    need_y: bool, need_wq: bool, need_wk: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:  # fmt: skip
    m, d = y.shape[0], wq.shape[0]
    shape = (m, heads, d // heads)
    dqp, dqw = _backward_one(gq.view(shape), qp.view(shape), qr, qw, pos, bands)
    dkp, dkw = _backward_one(gk.view(shape), kp.view(shape), kr, kw, pos, bands)
    dqp, dkp = dqp.view(m, d), dkp.view(m, d)
    dy = torch.addmm(dqp @ wq, dkp, wk) if need_y else y.new_empty(0)
    dwq = (dqp.t() @ y).to(wq.dtype) if need_wq else y.new_empty(0)
    dwk = (dkp.t() @ y).to(wk.dtype) if need_wk else y.new_empty(0)
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
):
    e = y.new_empty(0)
    dh = wq.shape[0] // heads
    w = y.new_empty(dh, dtype=torch.float32)
    return (
        torch.empty_like(y) if need_y else e,
        torch.empty_like(wq) if need_wq else e,
        torch.empty_like(wk) if need_wk else e,
        w,
        torch.empty_like(w),
    )


def _qk_setup(ctx, inputs, output):
    y, wq, wk, qw, kw, pos, bands, _, heads, _ = inputs
    ctx.save_for_backward(
        y, wq, wk, output[2], output[3], output[4], output[5], qw, kw, pos, bands
    )
    ctx.heads = heads
    ctx.w_dtypes = (qw.dtype, kw.dtype)


def _qk_backward(ctx, gq, gk, *_rest):
    y, wq, wk, qp, kp, qr, kr, qw, kw, pos, bands = ctx.saved_tensors
    need = ctx.needs_input_grad
    out_shape = (y.shape[0], wq.shape[0])
    gq = y.new_zeros(out_shape) if gq is None else gq.contiguous()
    gk = y.new_zeros(out_shape) if gk is None else gk.contiguous()
    dy, dwq, dwk, dqw, dkw = _qk_bwd(
        gq, gk, y, wq, wk, qp, kp, qr, kr, qw, kw, pos, bands, ctx.heads,
        need[0], need[1], need[2],
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
            y.reshape(m, -1).contiguous(), wq.to(y.dtype).contiguous(),
            wk.to(y.dtype).contiguous(), q_weight.float().contiguous(),
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
