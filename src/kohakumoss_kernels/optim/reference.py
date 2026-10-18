"""Torch reference of the optimizer step kernels (what each computes; also the CPU
path): AdamW on `m` and `r = sqrt(v)`, writeback `MODES`, 8-bit blockwise states. See
docs/internals/kernel-dev.md#optimizer.
"""

import torch
from torch import Tensor

MODES = ("nearest", "stochastic", "kahan")
BLOCK = 256


def hash32(idx: Tensor, seed: int) -> Tensor:
    """Counter-based 32-bit hash of (index, seed) as int64 in [0, 2^32)."""
    m = 0xFFFFFFFF
    h = (idx.to(torch.int64) * 0x9E3779B1 + seed) & m
    h ^= h >> 16
    h = (h * 0x7FEB352D) & m
    h ^= h >> 15
    h = (h * 0x846CA68B) & m
    h ^= h >> 16
    return h


def sr_bf16(x: Tensor, seed: int) -> Tensor:
    """Stochastic rounding of fp32 `x` to bf16 (returned as fp32 holding a bf16 value)."""
    bits = x.float().contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    rnd = hash32(torch.arange(x.numel(), device=x.device).view(x.shape), seed) & 0xFFFF
    out = ((bits + rnd) & 0xFFFF0000).to(torch.int64)
    out = torch.where(out >= 2**31, out - 2**32, out).to(torch.int32)
    return out.view(torch.float32)


def _write(p: Tensor, value: Tensor, comp: Tensor | None, mode: str, seed: int) -> None:
    """Round fp32 `value` into `p` by writeback `mode`."""
    match mode:
        case "nearest":
            p.copy_(value)
        case "stochastic":
            p.copy_(sr_bf16(value, seed))
        case "kahan":
            p.copy_(value)
            comp.copy_(value - p.float())
        case _:
            raise ValueError(mode)


def adamw_step(
    p: Tensor,
    g: Tensor,
    m: Tensor,
    r: Tensor,
    comp: Tensor | None,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    wd: float,
    step: int,
    mode: str = "nearest",
    seed: int = 0,
) -> None:
    """In-place AdamW (decoupled decay) on p with states m and r = sqrt(v); in kahan
    mode the parameter value is p + comp."""
    gf, pf = g.float(), p.float()
    if mode == "kahan":
        pf = pf + comp.float()
    mf = m.float() * beta1 + gf * (1.0 - beta1)
    rf = (r.float().square() * beta2 + gf.square() * (1.0 - beta2)).sqrt()
    c1 = 1.0 / (1.0 - beta1**step)
    c2 = 1.0 / (1.0 - beta2**step) ** 0.5
    upd = (mf * c1) / (rf * c2 + eps)
    new = pf - lr * (upd + wd * pf)
    m.copy_(mf)
    r.copy_(rf)
    _write(p, new, comp, mode, seed)


def quant(x: Tensor, signed: bool) -> tuple[Tensor, Tensor]:
    """Blockwise sqrt-companded 8-bit codes and fp32 block scales of a flat tensor."""
    n = x.numel()
    pad = (-n) % BLOCK
    xf = torch.nn.functional.pad(x.float().reshape(-1), (0, pad)).view(-1, BLOCK)
    scale = xf.abs().amax(1).clamp_min(1e-30)
    t = (xf.abs() / scale[:, None]).sqrt()
    if signed:
        q = (torch.sign(xf) * torch.round(t * 127)).to(torch.int8)
    else:
        q = torch.round(t * 255).to(torch.uint8)
    return q.reshape(-1)[:n], scale


def dequant(q: Tensor, scale: Tensor, signed: bool) -> Tensor:
    """fp32 values of the blockwise codes `q` with block scales `scale` (flat)."""
    n = q.numel()
    pad = (-n) % BLOCK
    qf = torch.nn.functional.pad(q.float(), (0, pad)).view(-1, BLOCK)
    t = qf.abs() / (127.0 if signed else 255.0)
    v = t * t * scale[:, None]
    if signed:
        v = v * torch.sign(qf)
    return v.reshape(-1)[:n]


def adamw8bit_step(
    p: Tensor,
    g: Tensor,
    mq: Tensor,
    ms: Tensor,
    rq: Tensor,
    rs: Tensor,
    comp: Tensor | None,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    wd: float,
    step: int,
    mode: str = "nearest",
    seed: int = 0,
) -> None:
    """AdamW with m (int8) and sqrt(v) (uint8) stored blockwise-quantized; in place."""
    m = dequant(mq, ms, True).view(p.shape)
    r = dequant(rq, rs, False).view(p.shape)
    adamw_step(p, g, m, r, comp, lr, beta1, beta2, eps, wd, step, mode, seed)
    q, s = quant(m, True)
    mq.copy_(q)
    ms.copy_(s)
    q, s = quant(r, False)
    rq.copy_(q)
    rs.copy_(s)
