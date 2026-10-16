"""Prewarm of the block ops: run each op forward + backward on scratch MOSS-shaped
inputs (so kernel configs are tuned and kernel-vs-compile choices are measured and
stored) before training, outside any no_grad / checkpoint / graph capture."""

from collections.abc import Callable

import torch

from kohakumoss_kernels.backends.compile.race import retune_pending


def _segments(lengths: list[int], device) -> torch.Tensor:
    return torch.cat(
        [torch.full((n,), i, dtype=torch.int32) for i, n in enumerate(lengths)]
    ).to(device)


def _positions(n: int, device) -> torch.Tensor:
    pos = torch.zeros(n, 4, device=device)
    pos[:, 1] = torch.arange(n, device=device) % 512
    return pos


def prewarm_block_ops(
    ops: dict[str, Callable],
    device: torch.device | str,
    dtype: torch.dtype,
    lengths: list[list[int]],
    dim: int = 3072,
    heads: int = 24,
    ffn_dim: int = 8192,
    train_params: bool = False,
) -> int:
    """`ops` maps op names (registry names) to resolved callables; `lengths` lists
    batches of per-sample joint lengths. Returns how many problems were measured."""
    device = torch.device(device)
    hd = dim // heads
    bands = 1 / (10000.0 ** (torch.arange(hd // 8, device=device) / (hd // 8)))
    with torch.enable_grad():
        for lens in lengths:
            n, b = sum(lens), len(lens)
            seg = _segments(lens, device)
            x = torch.randn(n, dim, device=device).to(dtype).requires_grad_()
            mod = torch.randn(b, dim, device=device).requires_grad_(train_params)
            w = torch.ones(dim, device=device).requires_grad_(train_params)
            calls = {
                "rms_norm_mod": (x, w, mod, mod, seg, 1e-6),
                "rms_norm": (x, w, 1e-6),
                "gated_residual": (x, x, mod, seg, 1.0),
                "situ_glu": (
                    torch.randn(n, 2 * ffn_dim, device=device)
                    .to(dtype)
                    .requires_grad_(),
                ),
                "qk_norm_rope": (
                    x.view(n, heads, hd),
                    x.view(n, heads, hd),
                    torch.ones(hd, device=device),
                    torch.ones(hd, device=device),
                    _positions(n, device),
                    bands,
                    1e-6,
                ),
            }
            for name, args in calls.items():
                if name not in ops:
                    continue
                out = ops[name](*args)
                outs = out if isinstance(out, tuple) else (out,)
                torch.autograd.backward([o.float().sum() for o in outs])
    return retune_pending()
