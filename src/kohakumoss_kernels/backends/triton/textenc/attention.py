"""`attention_varlen_causal` (forward only) on sm80+ CUDA and XPU: per shape, the tuner
picks the fastest of the Triton causal GQA kernel tiles and the PyTorch path (torch's
flash-based `varlen_attn` where the build and card have it, else the per-segment SDPA
loop). See docs/internals/text-encoder.md."""

import math

import torch
from torch import Tensor

from kohakumoss_kernels import tuning
from kohakumoss_kernels.backends.eager.textenc import (
    attention_varlen_causal as sdpa_causal,
)
from kohakumoss_kernels.backends.triton._common import Launcher
from kohakumoss_kernels.backends.triton.textenc import _attention_kernels as K
from kohakumoss_kernels.backends.varlen_choice import flash_varlen_usable
from kohakumoss_kernels.constraints import requires
from kohakumoss_kernels.errors import BUILD_ERRORS, unavailable
from kohakumoss_kernels.registry import register
from kohakumoss_kernels.tuning.buckets import KEY_VERSION, cdiv, length_bucket
from kohakumoss_kernels.tuning.tuner import on_reset

try:
    from torch.nn.attention.varlen import varlen_attn as _varlen_attn
except ImportError:
    _varlen_attn = None

KERNEL = "te_attn_causal_triton"
LOG2E = 1.0 / math.log(2.0)
HEAD_DIMS = (64, 128, 256)
DTYPES = (torch.float16, torch.bfloat16)

_FWD = Launcher(K.causal_fwd_kernel)
_BOUND: dict[tuple, dict] = {}
on_reset(_BOUND.clear)


def _triton(cfg, q, k, v, cu, max_len, scale):
    n, h, d = q.shape
    out = torch.empty_like(q)
    if n == 0 or max_len == 0:
        return out
    _FWD[(cdiv(max_len, cfg["bm"]), h, cu.shape[0] - 1)](
        q, k, v, out, cu, scale * LOG2E,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
        out.stride(0), out.stride(1),
        GROUPS=h // k.shape[1], D=d, BM=cfg["bm"], BN=cfg["bn"],
        PREC="ieee", num_warps=cfg["w"], num_stages=cfg["st"],
    )  # fmt: skip
    return out


def _torch_varlen(cfg, q, k, v, cu, max_len, scale):
    return _varlen_attn(
        q, k, v, cu, cu, max_len, max_len,
        scale=scale, window_size=(-1, 0), enable_gqa=k.shape[1] != q.shape[1],
    )  # fmt: skip


def _sdpa(cfg, q, k, v, cu, max_len, scale):
    return sdpa_causal(q, k, v, cu, max_len, scale)


IMPLS = {"triton": _triton, "torch_varlen": _torch_varlen, "sdpa": _sdpa}


def candidates(device: torch.device, dtype: torch.dtype, d: int) -> list[dict]:
    """Triton tiles that fit the head dim, with the PyTorch path second: torch's flash
    varlen when usable, else the per-segment SDPA loop (the eager reference)."""
    out = []
    tiles = ((64, 32, 4), (64, 64, 8), (128, 32, 8), (32, 32, 4), (64, 16, 4))
    if d <= 128:
        tiles += ((128, 64, 8), (128, 32, 4))
    for bm, bn, w in tiles:
        for st in (1, 2):
            out.append({"impl": "triton", "bm": bm, "bn": bn, "w": w, "st": st})
    torch_impl = "torch_varlen" if flash_varlen_usable(device) else "sdpa"
    out.insert(1, {"impl": torch_impl})
    return out


def run(cfg: dict, q, k, v, cu, max_len, scale) -> Tensor:
    """One configured call (benches and precision tests)."""
    q, k, v = (t.contiguous() for t in (q, k, v))
    return IMPLS[cfg["impl"]](cfg, q, k, v, cu, int(max_len), float(scale))


def _key(q: Tensor, k: Tensor, max_len: int) -> str:
    return (
        f"{KEY_VERSION}-{q.dtype}-h{q.shape[1]}-kv{k.shape[1]}-d{q.shape[2]}"
        f"-L{length_bucket(max_len)}"
    )


def attention_varlen_causal(
    q: Tensor, k: Tensor, v: Tensor, cu_seqlens: Tensor, max_seqlen: int, scale: float
) -> Tensor:
    """Packed causal GQA attention through the fastest implementation for this shape."""
    max_len = int(max_seqlen)
    key = _key(q, k, max_len)
    ident = (q.device, key)
    try:
        cfg = _BOUND.get(ident)
        if cfg is None:
            args = (q, k, v, cu_seqlens, max_len, scale)
            cfg = tuning.select(
                KERNEL,
                q.device,
                key,
                lambda: candidates(q.device, q.dtype, q.shape[2]),
                lambda c: lambda: run(c, *args),
                shortlist=12,
                errors=BUILD_ERRORS,
            )
            if tuning.decided(KERNEL, q.device, key):
                _BOUND[ident] = cfg
        return run(cfg, q, k, v, cu_seqlens, max_len, scale)
    except Exception as err:
        raise unavailable("attention_varlen_causal", err, "triton") from err


def _head_dim_ok(query) -> str | None:
    d = query.get("head_dim")
    return None if d is None or d in HEAD_DIMS else f"head_dim {d} not in {HEAD_DIMS}"


SUPPORTS = requires(kinds=("cuda", "xpu"), min_sm=80, dtypes=DTYPES, extra=_head_dim_ok)
register("attention_varlen_causal", "triton", attention_varlen_causal, SUPPORTS)
