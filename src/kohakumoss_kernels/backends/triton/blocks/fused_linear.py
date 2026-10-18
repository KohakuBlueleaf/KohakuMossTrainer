"""Triton fused linear ops: GEMM + MOSS epilogue (SiTU-GLU, gated residual, qk-norm +
RoPE), registered as the optional `linear_situ`, `linear_gated_residual` and
`linear_qk_rope` ops. This module holds the fused GEMM launches and their tile configs;
the host code (paths, custom ops, autograd) is `backends/_fused_linear.py`.
"""

import torch
from torch.library import triton_op

from kohakumoss_kernels.backends._fused_linear import FusedBackend, FusedLinear
from kohakumoss_kernels.backends.compile.choice import kernel_or_compile, shape_key
from kohakumoss_kernels.backends.triton._common import Launcher
from kohakumoss_kernels.backends.triton.blocks import _gemm_kernels as jit
from kohakumoss_kernels.backends.triton.blocks._blockops import strides2
from kohakumoss_kernels.backends.triton.blocks.ffn import situ_backward, situ_forward
from kohakumoss_kernels.backends.triton.blocks.residual import (
    residual_backward,
    residual_forward,
)
from kohakumoss_kernels.backends.triton.blocks.rope import rope_backward, rope_forward
from kohakumoss_kernels.constraints import requires
from kohakumoss_kernels.registry import register
from kohakumoss_kernels.tuning.buckets import cdiv
from kohakumoss_kernels.tuning.tuner import select

__all__ = ["linear_situ", "linear_gated_residual", "linear_qk_rope"]

DTYPES = (torch.float16, torch.bfloat16)

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
)  # fmt: skip
RES_CFGS = _tiles(
    (128, 128, 32, 8, 3), (128, 64, 64, 4, 4), (64, 128, 64, 4, 4), (128, 256, 32, 8, 3), (64, 64, 64, 4, 4)
)  # fmt: skip
QK_CFGS = _tiles(
    (128, 128, 32, 8, 3), (64, 128, 64, 4, 4), (128, 128, 64, 8, 3), (64, 128, 32, 4, 3)
)  # fmt: skip
FUSED_TEST_CFG = {
    "path": "fused",
    "BM": 64,
    "BN": 64,
    "BK": 32,
    "warps": 4,
    "stages": 3,
}


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


def _res_fused(cfg, a, w, bias, x, gate, seg, r, save):
    m, k = a.shape
    n = w.shape[0]
    out = torch.empty_like(x)
    o = torch.empty(m, n, device=a.device, dtype=a.dtype) if save else a.new_empty(0)
    grid = (cdiv(m, cfg["BM"]) * cdiv(n, cfg["BN"]),)
    _res[grid](
        a, w, bias if bias is not None else w, x, gate if gate is not None else x,
        seg if seg is not None else x, out, o if save else out, m, n, k, r,
        *strides2(gate), HAS_BIAS=bias is not None, HAS_GATE=gate is not None, SCALED=r != 1.0, SAVE=save,
        BM=cfg["BM"], BN=cfg["BN"], BK=cfg["BK"], GROUP_M=8, num_warps=cfg["warps"],
        num_stages=cfg["stages"],
    )  # fmt: skip
    return out, o


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


def _define(name, fn, fake):
    return triton_op(name, mutates_args=())(fn)


def _pick(name, device, key, cfgs, run) -> dict:
    return select(
        name, device, key, lambda: cfgs, lambda c: lambda: run(c), shortlist=len(cfgs)
    )


FUSED = FusedLinear(
    FusedBackend(
        name="triton",
        define=_define,
        fused={"situ": _situ_fused, "gres": _res_fused, "qk": _qk_fused},
        cfgs={"situ": SITU_CFGS, "gres": RES_CFGS, "qk": QK_CFGS},
        situ=(situ_forward, situ_backward),
        residual=(residual_forward, residual_backward),
        rope=(rope_forward, rope_backward),
        pick=_pick,
    )
)
force_path = FUSED.force_path
linear_situ = FUSED.linear_situ
linear_gated_residual = FUSED.linear_gated_residual
linear_qk_rope = FUSED.linear_qk_rope

_SUPPORTS = requires(kinds=("cuda", "xpu"), min_sm=80, dtypes=DTYPES)
for _name, _fn in (
    ("linear_situ", linear_situ),
    ("linear_gated_residual", linear_gated_residual),
    ("linear_qk_rope", linear_qk_rope),
):
    _op = kernel_or_compile("triton", _name, _fn, shape_key)
    register(_name, "triton", _op, _SUPPORTS)
