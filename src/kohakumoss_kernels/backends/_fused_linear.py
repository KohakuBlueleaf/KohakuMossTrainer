"""Host code of the fused linear ops (`linear_situ`, `linear_gated_residual`,
`linear_qk_rope`), shared by the Triton and TileLang backends.

`FusedLinear(backend)` defines one backend's forward / backward custom ops, their
autograd, the per-problem path choice (`fused`: the GEMM + epilogue kernel, `split`:
cuBLAS + the standalone epilogue kernel) and the public functions; `FusedBackend`
holds what differs per backend: the op definer, the fused launchers and their tile
configs, the epilogue kernels, the tuner entry and the capability / failure handling.
The forward saves what the backward needs only while autograd records; the backward is
the epilogue's backward kernel followed by cuBLAS GEMMs. See
docs/internals/kernel-dev.md#fused-linear-ops.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor

from kohakumoss_common.calls import nothing
from kohakumoss_kernels.backends._master import (
    activation,
    operand,
    stacked,
    weight_grad,
)
from kohakumoss_kernels.backends.compile.ops import compiled_op
from kohakumoss_kernels.saved_ops import keep
from kohakumoss_kernels.tuning.buckets import TOKEN_FLOOR, pow2
from kohakumoss_kernels.tuning.tuner import decided, forget, on_reset

SPLIT = {"path": "split"}
OPS = ("situ", "gres", "qk")


def _key(m: int, *rest) -> str:
    """Tuning key: the token count's bucket, then the problem's fixed dims and flags."""
    return "-".join(str(v) for v in (pow2(m, TOKEN_FLOOR), *rest))


def _kept(t: Tensor, needed: bool) -> Tensor:
    """`t` when the backward uses it, else an empty tensor of its dtype and device."""
    return t if needed else t.new_empty(0)


def _always(args: tuple) -> bool:
    return True


@dataclass(frozen=True)
class FusedBackend:
    """One backend's part of the fused linear ops.

    `define(name, fn, fake)` registers `fn` as the custom op `name` (`fake`: its meta
    function, for definers that need one); `fused[op]` is the fused launcher
    `(cfg, *args) -> outputs` of op "situ" / "gres" / "qk" and `cfgs[op]` its tile
    configs; `situ`, `residual`, `rope` are the epilogue (forward, backward) kernels;
    `pick(name, device, key, cfgs, run)` is the tuned config of a problem;
    `fusable[op](args)` whether the fused kernel takes the problem (else `split` only);
    a `dead` error from a fused config drops it (`mark_dead`) and the problem picks
    again; a `fallback` error from the public function runs the compiled eager
    composition instead."""

    name: str
    define: Callable
    fused: dict[str, Callable]
    cfgs: dict[str, list[dict]]
    situ: tuple[Callable, Callable]
    residual: tuple[Callable, Callable]
    rope: tuple[Callable, Callable]
    pick: Callable
    fusable: dict[str, Callable] = field(default_factory=dict)
    dead: tuple[type[BaseException], ...] = ()
    mark_dead: Callable = nothing
    fallback: tuple[type[BaseException], ...] = ()


class FusedLinear:
    """The fused linear ops of `backend`: `linear_situ`, `linear_gated_residual`,
    `linear_qk_rope` (the registry implementations), `force_path`, and the per-problem
    runner binding (`bound`, `run`)."""

    def __init__(self, backend: FusedBackend):
        self.backend = backend
        self.forced: dict[str, dict] = {}
        self.bound: dict[tuple, Callable] = {}
        on_reset(self.bound.clear)
        self.paths = {
            "situ": {"split": self._situ_split, "fused": backend.fused["situ"]},
            "gres": {"split": self._res_split, "fused": backend.fused["gres"]},
            "qk": {"split": self._qk_split, "fused": backend.fused["qk"]},
        }
        self._define()

    def force_path(self, cfg: dict | None) -> None:
        """Pin every fused-linear op to one config (tests and benches; a problem that
        cannot fuse runs split); None = tuned."""
        self.forced.clear()
        self.bound.clear()
        if cfg is not None:
            self.forced["cfg"] = cfg

    def _candidates(self, op: str, args: tuple) -> list[dict]:
        if self.backend.fusable.get(op, _always)(args):
            return [*self.backend.cfgs[op], SPLIT]
        return [SPLIT]

    def _bound(self, name, device, key, cfgs, paths, args) -> Callable:
        """The runner (`paths[cfg["path"]]` with its tuned config) of this problem,
        bound once the choice is settled; called as runner(*args)."""
        ident = (name, str(device), key)
        fn = self.bound.get(ident)
        if fn is not None:
            return fn
        forced = self.forced.get("cfg")
        if forced is not None:
            cfg = forced if len(cfgs) > 1 else cfgs[0]
        else:
            cfg = self.backend.pick(
                name, device, key, cfgs, partial(_launch, paths, args)
            )
        fn = partial(paths[cfg["path"]], cfg)
        if self.forced or decided(name, device, key):
            self.bound[ident] = fn
        return fn

    def run(self, name, device, key, cfgs, paths, args):
        """The problem's bound runner on `args`; a fused config raising a `dead` error
        is dropped from the candidates and the problem picks again."""
        while True:
            fn = self._bound(name, device, key, cfgs, paths, args)
            try:
                return fn(*args)
            except self.backend.dead:
                cfg = fn.args[0]
                if self.forced or cfg["path"] != "fused":
                    raise
                self.backend.mark_dead(name, cfg)
                forget(name, device, key)
                self.bound.pop((name, str(device), key), None)

    def _op(self, op: str, device, key: str, args: tuple):
        name = f"{self.backend.name}_linear_{op}"
        cfgs = self._candidates(op, args)
        return self.run(name, device, key, cfgs, self.paths[op], args)

    # ------------------------------------------------------------ split paths

    def _situ_split(self, cfg, y, w, bias, save):
        ug = F.linear(y, w, bias)
        return self.backend.situ[0](ug), (ug if save else y.new_empty(0))

    def _res_split(self, cfg, a, w, bias, x, gate, seg, r, save):
        o = F.linear(a, w, bias)
        out = self.backend.residual[0](x, o, gate, seg, r)
        return out, (o if save else a.new_empty(0))

    def _qk_split(self, cfg, y, wq, wk, qw, kw, pos, bands, eps, heads, save):
        m, d = y.shape[0], wq.shape[0]
        dh = d // heads
        qp = F.linear(y, wq).view(m, heads, dh)
        kp = F.linear(y, wk).view(m, heads, dh)
        qo, ko, qr, kr = self.backend.rope[0](qp, kp, qw, kw, pos, bands, eps)
        if not save:
            qp, kp = y.new_empty(0), y.new_empty(0)
        return qo.view(m, d), ko.view(m, d), qp.reshape(-1), kp.reshape(-1), qr, kr

    # -------------------------------------------------------- op definitions

    def _define(self) -> None:
        name, define = self.backend.name, self.backend.define
        self.bodies = _op_bodies(self)
        for part, body in self.bodies.items():
            op = define(f"kohakumoss::{name}_linear_{part}", body, FAKES[part])
            setattr(self, f"_{part}", op)
        self._situ_fwd.register_autograd(self._situ_backward, setup_context=_situ_setup)
        self._gres_fwd.register_autograd(self._res_backward, setup_context=_res_setup)
        self._qk_fwd.register_autograd(self._qk_backward, setup_context=_qk_setup)
        space = torch.ops.kohakumoss
        keep(*(getattr(space, f"{name}_linear_{op}_fwd").default for op in OPS))

    def _situ_backward(self, ctx, ga, _gug):
        y, w, ug = ctx.saved_tensors
        need = ctx.needs_input_grad
        dy, dw, db = self._situ_bwd(
            ga.contiguous(), y, w, ug, need[0], need[1], need[2]
        )
        return (
            dy if need[0] else None,
            dw if need[1] else None,
            db.to(ctx.b_dtype) if need[2] else None,
            None,
        )

    def _res_backward(self, ctx, gy, _go):
        a, w, o, gate, seg = ctx.saved_tensors
        need = ctx.needs_input_grad
        gy = gy.contiguous()
        dx, da, dw, db, dgate = self._gres_bwd(
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

    def _qk_backward(self, ctx, gq, gk, *_rest):
        y, wq, wk, qp, kp, qr, kr, qw, kw, pos, bands = ctx.saved_tensors
        need = ctx.needs_input_grad
        out_shape = ctx.out_shape
        gq = y.new_zeros(out_shape) if gq is None else gq.contiguous()
        gk = y.new_zeros(out_shape) if gk is None else gk.contiguous()
        dy, dwq, dwk, dqw, dkw = self._qk_bwd(
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

    # ------------------------------------------------------- public functions

    def linear_situ(self, y: Tensor, w_up_gate: Tensor, bias: Tensor | None) -> Tensor:
        """situ_glu(y W^T + b) with W = [W_up; W_gate] [2F, K]."""
        shape = y.shape
        save = torch.is_grad_enabled() and any(
            t is not None and t.requires_grad for t in (y, w_up_gate, bias)
        )
        try:
            y2 = activation(y.reshape(-1, shape[-1])).contiguous()
            a, _ = self._situ_fwd(y2, w_up_gate, bias, save)
        except self.backend.fallback:
            return compiled_op("linear_situ")(y, w_up_gate, bias)
        return a.view(*shape[:-1], a.shape[-1])

    def linear_gated_residual(
        self,
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
        save = torch.is_grad_enabled() and gate is not None and gate.requires_grad
        try:
            out, _ = self._gres_fwd(
                activation(a.reshape(-1, a.shape[-1])).contiguous(), w, bias,
                x.reshape(-1, shape[-1]).contiguous(), gate,
                None if seg is None else seg.contiguous(), float(r), save,
            )  # fmt: skip
        except self.backend.fallback:
            return compiled_op("linear_gated_residual")(a, w, bias, x, gate, seg, r)
        return out.view(shape)

    def linear_qk_rope(
        self,
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
        """(q, k) [N, H, Dh] = qk_norm_rope(y Wq^T, y Wk^T); the norm weights enter as
        fp32 (their gradients come back in their own dtype)."""
        m = y.reshape(-1, y.shape[-1]).shape[0]
        d = wq.shape[0]
        save = torch.is_grad_enabled() and any(
            t.requires_grad for t in (y, wq, wk, q_weight, k_weight)
        )
        try:
            qo, ko, _, _, _, _ = self._qk_fwd(
                activation(y.reshape(m, -1)).contiguous(), wq, wk,
                q_weight.float().contiguous(), k_weight.float().contiguous(),
                pos.float().contiguous(), bands.float().contiguous(), eps, heads, save,
            )  # fmt: skip
        except self.backend.fallback:
            return compiled_op("linear_qk_rope")(
                y, wq, wk, q_weight, k_weight, pos, bands, eps, heads
            )
        return qo.view(m, heads, d // heads), ko.view(m, heads, d // heads)


def _launch(paths: dict, args: tuple, cfg: dict):
    return paths[cfg["path"]](cfg, *args)


# ------------------------------------------------------------------ op bodies


def _op_bodies(fl: FusedLinear) -> dict[str, Callable]:
    """The six custom-op bodies of `fl` (annotated for the op schemas)."""
    b = fl.backend

    def situ_fwd(
        y: Tensor, w: Tensor, bias: Tensor | None, save: bool
    ) -> tuple[Tensor, Tensor]:
        key = _key(y.shape[0], y.shape[1], w.shape[0], y.dtype, bias is not None, save)
        args = (y, operand(w, y), operand(bias, y), save)
        return fl._op("situ", y.device, key, args)

    def situ_bwd(
        ga: Tensor,
        y: Tensor,
        w: Tensor,
        ug: Tensor,
        need_y: bool,
        need_w: bool,
        need_b: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        dug = b.situ[1](ga, ug)
        dy = dug @ operand(w, dug) if need_y else y.new_empty(0)
        dw = weight_grad(dug, y, w.dtype) if need_w else y.new_empty(0)
        db = dug.float().sum(0) if need_b else y.new_empty(0)
        return dy, dw, db

    def gres_fwd(
        a: Tensor, w: Tensor, bias: Tensor | None, x: Tensor, gate: Tensor | None,
        seg: Tensor | None, r: float, save: bool,
    ) -> tuple[Tensor, Tensor]:  # fmt: skip
        flags = (gate is not None, r != 1.0, bias is not None, x.dtype == a.dtype, save)
        key = _key(a.shape[0], a.shape[1], w.shape[0], a.dtype, *flags)
        args = (a, operand(w, a), operand(bias, a), x, gate, seg, r, save)
        return fl._op("gres", a.device, key, args)

    def gres_bwd(
        g: Tensor, a: Tensor, w: Tensor, o: Tensor, gate: Tensor | None,
        seg: Tensor | None, r: float, need_a: bool, need_w: bool, need_b: bool,
        need_gate: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:  # fmt: skip
        """(d_x or empty when r == 1, d_a, d_w, d_b fp32, d_gate fp32)."""
        if gate is None:
            do = g
            dx = (g.float() * r).to(g.dtype) if r != 1.0 else g.new_empty(0)
            dgate = g.new_empty(0)
        else:
            dx, do, dgate = b.residual[1](g, o, gate, seg, r, need_gate)
        do = do.to(a.dtype)
        da = do @ operand(w, do) if need_a else g.new_empty(0)
        dw = weight_grad(do, a, w.dtype) if need_w else g.new_empty(0)
        db = do.float().sum(0) if need_b else g.new_empty(0)
        return dx, da, dw, db, dgate

    def qk_fwd(
        y: Tensor, wq: Tensor, wk: Tensor, qw: Tensor, kw: Tensor, pos: Tensor,
        bands: Tensor, eps: float, heads: int, save: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:  # fmt: skip
        key = _key(y.shape[0], y.shape[1], wq.shape[0], heads, y.dtype, save)
        args = (y, operand(wq, y), operand(wk, y), qw, kw, pos, bands, eps, heads, save)
        return fl._op("qk", y.device, key, args)

    def qk_bwd(
        gq: Tensor, gk: Tensor, y: Tensor, wq: Tensor, wk: Tensor, qp: Tensor,
        kp: Tensor, qr: Tensor, kr: Tensor, qw: Tensor, kw: Tensor, pos: Tensor,
        bands: Tensor, heads: int, need_y: bool, need_wq: bool, need_wk: bool,
        eps: float, need_w: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:  # fmt: skip
        m, d = gq.shape
        shape = (m, heads, d // heads)
        dqk = torch.empty(m, 2 * d, device=gq.device, dtype=qp.dtype)
        dqp, dkp, dqw, dkw = b.rope[1](
            gq.view(shape), gk.view(shape), qp.view(shape), kp.view(shape), qr, kr,
            qw, kw, pos, bands, eps, need_w, dqk,
        )  # fmt: skip
        dy = dqk @ stacked(wq, wk, dqk) if need_y else y.new_empty(0)
        dwq = weight_grad(dqp, y, wq.dtype) if need_wq else y.new_empty(0)
        dwk = weight_grad(dkp, y, wk.dtype) if need_wk else y.new_empty(0)
        return dy, dwq, dwk, dqw, dkw

    return {
        "situ_fwd": situ_fwd,
        "situ_bwd": situ_bwd,
        "gres_fwd": gres_fwd,
        "gres_bwd": gres_bwd,
        "qk_fwd": qk_fwd,
        "qk_bwd": qk_bwd,
    }


# --------------------------------------------------------- autograd setup


def _situ_setup(ctx, inputs, output):
    y, w, bias, _ = inputs
    need = ctx.needs_input_grad
    ctx.save_for_backward(_kept(y, need[1]), _kept(w, need[0]), output[1])
    ctx.b_dtype = None if bias is None else bias.dtype


def _res_setup(ctx, inputs, output):
    a, w, bias, x, gate, seg, r, _ = inputs
    need = ctx.needs_input_grad
    ctx.save_for_backward(_kept(a, need[1]), _kept(w, need[0]), output[1], gate, seg)
    ctx.r = r
    ctx.b_dtype = None if bias is None else bias.dtype


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


# ------------------------------------------------------------- meta functions


def _situ_fwd_fake(y, w, bias, save):
    m, f2 = y.shape[0], w.shape[0]
    return y.new_empty(m, f2 // 2), (y.new_empty(m, f2) if save else y.new_empty(0))


def _situ_bwd_fake(ga, y, w, ug, need_y, need_w, need_b):
    m, f2 = ga.shape[0], ug.shape[1]
    e = y.new_empty(0)
    return (
        y.new_empty(m, w.shape[1]) if need_y else e,
        y.new_empty(f2, y.shape[1], dtype=w.dtype) if need_w else e,
        y.new_empty(f2, dtype=torch.float32) if need_b else e,
    )


def _gres_fwd_fake(a, w, bias, x, gate, seg, r, save):
    o = a.new_empty(a.shape[0], w.shape[0]) if save else a.new_empty(0)
    return torch.empty_like(x), o


def _gres_bwd_fake(g, a, w, o, gate, seg, r, need_a, need_w, need_b, need_gate):
    e = g.new_empty(0)
    trained = gate is not None and need_gate
    return (
        torch.empty_like(g) if r != 1.0 else e,
        g.new_empty(g.shape[0], w.shape[1]) if need_a else e,
        g.new_empty(g.shape[1], a.shape[1], dtype=w.dtype) if need_w else e,
        g.new_empty(g.shape[1], dtype=torch.float32) if need_b else e,
        g.new_empty(gate.shape, dtype=torch.float32) if trained else e,
    )


def _qk_fwd_fake(y, wq, wk, qw, kw, pos, bands, eps, heads, save):
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


def _qk_bwd_fake(gq, gk, y, wq, wk, qp, kp, qr, kr, qw, kw, pos, bands, heads,
                 need_y, need_wq, need_wk, eps, need_w):  # fmt: skip
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


FAKES = {
    "situ_fwd": _situ_fwd_fake,
    "situ_bwd": _situ_bwd_fake,
    "gres_fwd": _gres_fwd_fake,
    "gres_bwd": _gres_bwd_fake,
    "qk_fwd": _qk_fwd_fake,
    "qk_bwd": _qk_bwd_fake,
}
