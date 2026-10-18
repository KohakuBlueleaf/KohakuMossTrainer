"""TileLang fused linear ops (fp16, sm70+): GEMM + MOSS epilogue. This module holds the
fused GEMM launches, their tile configs and what the TileLang kernels take (bias-free
GEMMs, the residual stream in the activation dtype); the host code (paths, custom ops,
autograd) is `backends/_fused_linear.py`. A fused config that does not compile is
dropped; a problem nothing compiles for runs the compiled eager composition.
"""

import torch
from torch.library import custom_op

from kohakumoss_kernels.backends._fused_linear import FusedBackend, FusedLinear
from kohakumoss_kernels.backends.compile.choice import kernel_or_compile, shape_key
from kohakumoss_kernels.backends.tilelang.blocks import _gemm_prims as prims
from kohakumoss_kernels.backends.tilelang.blocks._blockops import (
    FP16_SUPPORTS,
    CompileError,
    compiled,
    mark_dead,
    per_sample,
    segment_ids,
    tuned,
)
from kohakumoss_kernels.backends.tilelang.blocks.ffn import situ_backward, situ_forward
from kohakumoss_kernels.backends.tilelang.blocks.residual import (
    residual_backward,
    residual_forward,
)
from kohakumoss_kernels.backends.tilelang.blocks.rope import (
    rope_backward,
    rope_forward,
)
from kohakumoss_kernels.registry import register

__all__ = ["linear_situ", "linear_gated_residual", "linear_qk_rope"]

FUSED_TEST_CFG = {
    "path": "fused",
    "bm": 64,
    "bn": 64,
    "bk": 32,
    "threads": 128,
    "stages": 1,
}


def _tiles(*rows) -> list[dict]:
    return [
        {"path": "fused", "bm": bm, "bn": bn, "bk": bk, "threads": th, "stages": st}
        for bm, bn, bk, th, st in rows
    ]


GEMM_CFGS = _tiles(
    (128, 128, 32, 256, 1), (128, 64, 32, 128, 1), (64, 128, 32, 128, 1), (128, 128, 64, 256, 1)
)  # fmt: skip
QK_CFGS = _tiles((64, 0, 32, 128, 1), (128, 0, 32, 256, 1), (64, 0, 64, 128, 1))


def _situ_fused(cfg, y, w, bias, save):
    m, k = y.shape
    f = w.shape[0] // 2
    out = torch.empty(m, f, device=y.device, dtype=y.dtype)
    ug = torch.empty(m, 2 * f, device=y.device, dtype=y.dtype) if save else out
    kernel = compiled(
        prims.situ_gemm_prim, y.device, k, f, cfg["bm"], cfg["bn"], cfg["bk"],
        cfg["threads"], cfg["stages"], save,
    )  # fmt: skip
    kernel(y, w, out, ug)
    return out, (ug if save else y.new_empty(0))


def _res_fused(cfg, a, w, bias, x, gate, seg, r, save):
    m, k = a.shape
    n = w.shape[0]
    out = torch.empty_like(x)
    o = torch.empty(m, n, device=a.device, dtype=a.dtype) if save else out
    gf, gr, gc = (
        per_sample(gate, *gate.shape, a.device)
        if gate is not None
        else per_sample(None, 1, n, a.device)
    )
    sg = segment_ids(seg, m, a.device)
    kernel = compiled(
        prims.residual_gemm_prim, a.device, k, n, cfg["bm"], cfg["bn"], cfg["bk"],
        cfg["threads"], cfg["stages"], gate is not None, r != 1.0, save,
    )  # fmt: skip
    kernel(a, w, x, gf, sg, out, o, float(r), gr, gc)
    return out, (o if save else a.new_empty(0))


def _qk_fused(cfg, y, wq, wk, qw, kw, pos, bands, eps, heads, save):
    m, k = y.shape
    d = wq.shape[0]
    dh = d // heads
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


def _define(name, fn, fake):
    op = custom_op(name, mutates_args=())(fn)
    op.register_fake(fake)
    return op


def _situ_fusable(args) -> bool:
    """The SiTU GEMM kernel takes no bias."""
    return args[2] is None


def _gres_fusable(args) -> bool:
    """The residual GEMM kernel takes no bias and a residual stream in the activation
    dtype."""
    a, _, bias, x = args[:4]
    return bias is None and x.dtype == a.dtype


FUSED = FusedLinear(
    FusedBackend(
        name="tilelang",
        define=_define,
        fused={"situ": _situ_fused, "gres": _res_fused, "qk": _qk_fused},
        cfgs={"situ": GEMM_CFGS, "gres": GEMM_CFGS, "qk": QK_CFGS},
        situ=(situ_forward, situ_backward),
        residual=(residual_forward, residual_backward),
        rope=(rope_forward, rope_backward),
        pick=tuned,
        fusable={"situ": _situ_fusable, "gres": _gres_fusable},
        dead=(CompileError,),
        mark_dead=mark_dead,
        fallback=(CompileError,),
    )
)
force_path = FUSED.force_path
linear_situ = FUSED.linear_situ
linear_gated_residual = FUSED.linear_gated_residual
linear_qk_rope = FUSED.linear_qk_rope

for _name, _fn in (
    ("linear_situ", linear_situ),
    ("linear_gated_residual", linear_gated_residual),
    ("linear_qk_rope", linear_qk_rope),
):
    _op = kernel_or_compile("tilelang", _name, _fn, shape_key)
    register(_name, "tilelang", _op, FP16_SUPPORTS)
