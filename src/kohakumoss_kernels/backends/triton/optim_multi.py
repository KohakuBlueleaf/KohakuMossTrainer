"""Triton multi-tensor AdamW: one launch for a whole parameter list (LoRA adapters are
hundreds of small tensors, where per-tensor launches dominate).

A device table holds the five base addresses (p, g, m, r, comp) of every tensor and a
block table maps each program to (tensor, offset). Tensor t uses seed
`(seed + t * 1000003) mod 2^32` for stochastic rounding, i.e. the same draws as
`adamw_step` on that tensor alone with that seed. Tables are rebuilt only when an
address changes.
"""

import torch
import triton
import triton.language as tl

from kohakumoss_kernels.backends.triton.optim import MASK32, _adam, _scalars, _write
from kohakumoss_kernels.constraints import requires
from kohakumoss_kernels.optim.reference import MODES
from kohakumoss_kernels.registry import register

__all__ = ["adamw_step_multi"]

BLOCK = 1024
SEED_STRIDE = 1000003
TL = {torch.float32: tl.float32, torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}


@triton.jit
def _multi_kernel(
    addr_ptr, btens_ptr, bstart_ptr, size_ptr, lr, b1, b2, eps, wd, c1, c2, seed,
    PT: tl.constexpr, GT: tl.constexpr, MT: tl.constexpr, CT: tl.constexpr,
    MODE: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    j = tl.program_id(0)
    t = tl.load(btens_ptr + j)
    start = tl.load(bstart_ptr + j)
    n = tl.load(size_ptr + t)
    base = addr_ptr + t * 5
    p_ptr = tl.load(base).to(tl.pointer_type(PT))
    g_ptr = tl.load(base + 1).to(tl.pointer_type(GT))
    m_ptr = tl.load(base + 2).to(tl.pointer_type(MT))
    r_ptr = tl.load(base + 3).to(tl.pointer_type(MT))
    c_ptr = tl.load(base + 4).to(tl.pointer_type(CT))
    offs = start + tl.arange(0, BLOCK)
    mask = offs < n
    p = tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if MODE == 2:
        p += tl.load(c_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(m_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    new, m, r = _adam(p, g, m, r, lr, b1, b2, eps, wd, c1, c2)
    tl.store(m_ptr + offs, m.to(MT), mask=mask)
    tl.store(r_ptr + offs, r.to(MT), mask=mask)
    tseed = (seed + t.to(tl.uint64) * 1000003) & 0xFFFFFFFF
    _write(p_ptr, c_ptr, offs, mask, new, tseed, MODE)


_TABLES: dict[tuple, tuple] = {}


def _table(tensors: list[tuple], device) -> tuple:
    key = tuple(tuple(t.data_ptr() for t in row) for row in tensors)
    hit = _TABLES.get(key)
    if hit is not None:
        return hit
    addrs = [t.data_ptr() for row in tensors for t in row]
    sizes = [row[0].numel() for row in tensors]
    btens, bstart = [], []
    for i, n in enumerate(sizes):
        for s in range(0, n, BLOCK):
            btens.append(i)
            bstart.append(s)
    out = (
        torch.tensor(addrs, dtype=torch.int64).to(device),
        torch.tensor(btens, dtype=torch.int32).to(device),
        torch.tensor(bstart, dtype=torch.int32).to(device),
        torch.tensor(sizes, dtype=torch.int32).to(device),
        len(btens),
    )
    if len(_TABLES) > 32:
        _TABLES.clear()
    _TABLES[key] = out
    return out


def adamw_step_multi(
    params,
    grads,
    ms,
    rs,
    comps,
    lr,
    beta1,
    beta2,
    eps,
    wd,
    step,
    mode="nearest",
    seed=0,
):
    """In-place AdamW over lists of same-dtype tensors (`comps` None or a list)."""
    if not params:
        return
    comps = comps if comps is not None else [None] * len(params)
    rows = [
        (p, g, m, r, c if c is not None else p)
        for p, g, m, r, c in zip(params, grads, ms, rs, comps)
    ]
    addrs, btens, bstart, sizes, blocks = _table(rows, params[0].device)
    p0, g0, m0, _, c0 = rows[0]
    _multi_kernel[(blocks,)](
        addrs, btens, bstart, sizes, *_scalars(lr, beta1, beta2, eps, wd, step),
        int(seed) & MASK32, PT=TL[p0.dtype], GT=TL[g0.dtype], MT=TL[m0.dtype],
        CT=TL[c0.dtype], MODE=MODES.index(mode), BLOCK=BLOCK, num_warps=4,
    )  # fmt: skip


register(
    "adamw_step_multi",
    "triton",
    adamw_step_multi,
    requires(kinds=("cuda", "xpu"), min_sm=70, dtypes=tuple(TL)),
)
