"""fp8 compute formats: the eager reference of every recipe's quantizer and dequantizer.

An operand is 2-D [rows, K] with K the contraction axis; values are e4m3 (max 448).

    tensor  one fp32 scale [1]
    row     one fp32 scale per row [rows]
    block   activations: fp32 per row and 128 columns [rows, K/128];
            weights: fp32 per 128x128 block [ceil(rows/128), K/128]
    mx      OCP MX: one e8m0 power-of-two scale per 32 columns [rows, K/32], uint8

fp32 scales are max(amax, 1e-12) / 448 and q = clamp(x / s). The mx exponent is the
smallest e with 2^e * 448 >= max(amax, 1e-30), floored at -127, taken from the bits of
amax, stored biased (+127); q = x * 2^-e. See docs/internals/quant-kernels.md.
"""

import torch
from torch import Tensor

from kohakumoss_kernels.tuning.buckets import cdiv

E4M3 = torch.float8_e4m3fn
E4M3_MAX = 448.0
MX_BLOCK = 32
BLOCK = 128
RECIPES = ("tensor", "row", "block", "mx")
ROLES = ("act", "weight")
AMAX_FLOOR = 1e-12
MX_AMAX_FLOOR = 1e-30
MX_BIAS = 127


def _check(x: Tensor, recipe: str, role: str) -> None:
    if recipe not in RECIPES:
        raise ValueError(f"fp8 recipe {recipe!r} not in {RECIPES}")
    if role not in ROLES:
        raise ValueError(f"fp8 role {role!r} not in {ROLES}")
    if x.dim() != 2:
        raise ValueError(f"fp8 operand must be 2-D, got {tuple(x.shape)}")
    sub = {"block": BLOCK, "mx": MX_BLOCK}.get(recipe, 1)
    if x.shape[1] % sub:
        raise ValueError(f"fp8 {recipe}: K={x.shape[1]} is not a multiple of {sub}")


def _cast(x: Tensor, s: Tensor) -> Tensor:
    return (x / s).clamp_(-E4M3_MAX, E4M3_MAX).to(E4M3)


def fp32_scale(amax: Tensor) -> Tensor:
    """max(amax, 1e-12) / 448 as a correctly rounded fp32 division (a tensor divisor;
    see docs/internals/quant-kernels.md)."""
    amax = amax.float().clamp_min(AMAX_FLOOR)
    return amax / torch.full_like(amax, E4M3_MAX)


def mx_biased_exponent(amax: Tensor) -> Tensor:
    """Biased exponent (int32, e + 127) of a block's scale 2^e, from the bits of fp32
    amax: amax = m 2^E needs e = E - 8, or E - 7 when m > 1.75 (448 = 1.75 2^8)."""
    bits = amax.float().clamp_min(MX_AMAX_FLOOR).view(torch.int32)
    biased = (bits >> 23) & 0xFF
    up = ((bits & 0x7FFFFF) > 0x600000).to(torch.int32)
    return (biased - 8 + up).clamp_min(0)


def pow2_from_biased(biased: Tensor, sign: int = 1) -> Tensor:
    """Exact fp32 2^(sign * (biased - 127)) of int32 biased exponents in [0, 254]."""
    e = biased.to(torch.int32) - MX_BIAS
    return ((MX_BIAS + sign * e) << 23).view(torch.float32)


def quantize(x: Tensor, recipe: str, role: str = "act") -> tuple[Tensor, Tensor]:
    """(q e4m3 [rows, K], scale) of a 2-D operand under `recipe`; `role` matters only
    for `block` (weights scale 128x128 blocks)."""
    _check(x, recipe, role)
    xf = x.detach().float().contiguous()
    rows, k = xf.shape
    match recipe, role:
        case "tensor", _:
            s = fp32_scale(xf.abs().amax().reshape(1))
            return _cast(xf, s), s
        case "row", _:
            s = fp32_scale(xf.abs().amax(1))
            return _cast(xf, s[:, None]), s
        case "block", "act":
            g = xf.view(rows, k // BLOCK, BLOCK)
            s = fp32_scale(g.abs().amax(-1))
            return _cast(g, s[..., None]).view(rows, k), s
        case "block", "weight":
            pad = -rows % BLOCK
            xp = torch.nn.functional.pad(xf, (0, 0, 0, pad))
            g = xp.view(-1, BLOCK, k // BLOCK, BLOCK)
            s = fp32_scale(g.abs().amax((1, 3)))
            q = _cast(g, s[:, None, :, None]).view(-1, k)[:rows]
            return q, s
        case "mx", _:
            g = xf.view(rows, k // MX_BLOCK, MX_BLOCK)
            biased = mx_biased_exponent(g.abs().amax(-1))
            inv = pow2_from_biased(biased, -1)
            q = (g * inv[..., None]).clamp_(-E4M3_MAX, E4M3_MAX).to(E4M3)
            return q.view(rows, k), biased.to(torch.uint8)
    raise ValueError(f"fp8 role {role!r} not in {ROLES}")


def dequantize(q: Tensor, scale: Tensor, recipe: str, dtype: torch.dtype) -> Tensor:
    """q * scale in fp32, rounded once to `dtype` (a block scale with fewer rows than
    `q` is a 128x128 weight-block scale)."""
    rows, k = q.shape
    qf = q.float()
    match recipe:
        case "tensor":
            out = qf * scale.float().reshape(())
        case "row":
            out = qf * scale.float()[:, None]
        case "mx":
            g = qf.view(rows, k // MX_BLOCK, MX_BLOCK)
            out = (g * pow2_from_biased(scale)[..., None]).view(rows, k)
        case "block" if scale.shape[0] == rows:
            g = qf.view(rows, k // BLOCK, BLOCK)
            out = (g * scale.float()[..., None]).view(rows, k)
        case "block":
            g = torch.nn.functional.pad(qf, (0, 0, 0, -rows % BLOCK))
            g = g.view(-1, BLOCK, k // BLOCK, BLOCK) * scale.float()[:, None, :, None]
            out = g.view(-1, k)[:rows]
        case _:
            raise ValueError(f"fp8 recipe {recipe!r} not in {RECIPES}")
    return out.to(dtype)


def swizzle_mx(scale: Tensor) -> Tensor:
    """[rows, K/32] uint8 -> the flat 128x4-tile SWIZZLE_32_4_4 layout the vendor
    block-scaled GEMM reads (zero-padded to whole tiles)."""
    rows, cols = scale.shape
    rt, ct = cdiv(rows, 128), cdiv(cols, 4)
    padded = scale.new_zeros(rt * 128, ct * 4)
    padded[:rows, :cols] = scale
    tiles = padded.view(rt, 128, ct, 4).permute(0, 2, 1, 3)
    return tiles.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1).contiguous()
