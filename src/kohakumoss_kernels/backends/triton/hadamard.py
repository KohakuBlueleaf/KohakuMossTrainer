"""Triton group Hadamard rotation: y = x . blockdiag(H_g) over the last dim.

One program holds a [BR, C] tile (C a multiple of g) in fp32 registers and applies
H_g as j = log4(g) 4-point butterfly stages (y_k = sum(x) - 2 x_(3-k)), then the 2^-j
scale, rounding once to the output dtype. `hadamard_tile` is also the prologue /
epilogue of the fused rotated int8 linear (`quant_rotated.py`).
"""

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op

from kohakumoss_kernels.backends.compile.choice import kernel_or_compile, shape_key
from kohakumoss_kernels.backends.triton._common import Launcher, eager_function
from kohakumoss_kernels.constraints import requires
from kohakumoss_kernels.hadamard import check_group
from kohakumoss_kernels.registry import register
from kohakumoss_kernels.tuning.buckets import cdiv, pow2
from kohakumoss_kernels.tuning.tuner import select

__all__ = ["group_hadamard", "hadamard_tile", "rotate_2d"]

DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _h4(x, OUT: tl.constexpr, IN: tl.constexpr, R: tl.constexpr, C: tl.constexpr):
    v = tl.reshape(x, [OUT, 4, IN])
    s = tl.sum(v, axis=1)
    v = s[:, None, :] - 2.0 * tl.flip(v, 1)
    return tl.reshape(v, [R, C])


@triton.jit
def hadamard_tile(x, R: tl.constexpr, C: tl.constexpr, G: tl.constexpr):
    """fp32 tile [R, C] (C a multiple of G) times blockdiag(H_G), unscaled (x 2^j)."""
    x = _h4(x, R * (C // G), G // 4, R, C)
    if G >= 16:
        x = _h4(x, R * (C // G) * 4, G // 16, R, C)
    if G >= 64:
        x = _h4(x, R * (C // G) * 16, G // 64, R, C)
    if G >= 256:
        x = _h4(x, R * (C // G) * 64, G // 256, R, C)
    if G >= 1024:
        x = _h4(x, R * (C // G) * 256, G // 1024, R, C)
    return x


@triton.jit
def _kernel(
    x_ptr,
    y_ptr,
    R,
    K,
    scale,
    G: tl.constexpr,
    BR: tl.constexpr,
    C: tl.constexpr,
):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cols = tl.program_id(1) * C + tl.arange(0, C)
    mask = (rows < R)[:, None]
    offs = rows.to(tl.int64)[:, None] * K + cols[None, :]
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = hadamard_tile(x, BR, C, G) * scale
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


_launcher = Launcher(_kernel)
_TILES = ((16, 1, 4), (8, 1, 4), (32, 1, 8), (4, 1, 4), (8, 2, 4), (16, 2, 8))


def candidates(k: int, group: int) -> list[dict]:
    """Tiles whose width (CG groups of `group`) divides `k`, at most 1024 wide."""
    return [
        {"BR": br, "CG": cg, "warps": w}
        for br, cg, w in _TILES
        if k % (group * cg) == 0 and group * cg <= 1024
    ]


def launch(cfg: dict, x: Tensor, y: Tensor, group: int, j: int) -> None:
    r, k = x.shape
    c = group * cfg["CG"]
    grid = (cdiv(r, cfg["BR"]), k // c)
    _launcher[grid](
        x, y, r, k, 2.0**-j, G=group, BR=cfg["BR"], C=c, num_warps=cfg["warps"]
    )


def rotate_2d(x: Tensor, group: int) -> Tensor:
    """[R, K] contiguous -> [R, K] rotated (no autograd); shared with the fused linear."""
    j = check_group(x, group)
    y = torch.empty_like(x)
    if x.numel() == 0:
        return y
    r, k = x.shape
    key = f"r{pow2(r, 64)}_k{k}_g{group}_{x.dtype}"
    cfg = select(
        "triton_hadamard",
        x.device,
        key,
        lambda: candidates(k, group),
        lambda c: lambda: launch(c, x, y, group, j),
        in_backward=True,
    )
    launch(cfg, x, y, group, j)
    return y


@triton_op("kohakumoss::triton_group_hadamard", mutates_args=())
def _op(x: Tensor, group: int) -> Tensor:
    return rotate_2d(x, group)


def _setup(ctx, inputs, output):
    ctx.group = inputs[1]


def _backward_with(fn):
    """The autograd backward: the rotation itself (H_g is symmetric and involutory)."""

    def backward(ctx, g):
        return fn(g.contiguous(), ctx.group), None

    return backward


_op.register_autograd(_backward_with(_op), setup_context=_setup)
_Eager = eager_function("GroupHadamard", rotate_2d, _setup, _backward_with(rotate_2d))


def _rotate(x: Tensor, group: int) -> Tensor:
    run = _op if torch.compiler.is_compiling() else _Eager.apply
    return run(x, group)


def group_hadamard(x: Tensor, group: int) -> Tensor:
    """x [..., K] -> x . blockdiag(H_group), as the eager reference computes."""
    shape = x.shape
    y = _rotate(x.reshape(-1, shape[-1]).contiguous(), group)
    return y.view(shape)


register(
    "group_hadamard",
    "triton",
    kernel_or_compile("triton", "group_hadamard", group_hadamard, shape_key),
    requires(kinds=("cuda", "xpu"), min_sm=70, dtypes=DTYPES),
)
