"""TileLang text-encoder ops (forward only): `gated_delta_rule_varlen` (token-serial)
and `attention_varlen_causal` (causal GQA), registered for sm70+. Kernels compile once
per (shape constants, dtype, tile config, arch); each problem key times the kernel tiles
against the PyTorch path (the eager reference) and binds the winner. See
docs/internals/text-encoder.md."""

import math

import torch
from torch import Tensor

from kohakumoss_kernels import tuning
from kohakumoss_kernels.backends.eager.textenc import (
    attention_varlen_causal as eager_causal,
)
from kohakumoss_kernels.backends.eager.textenc import (
    gated_delta_rule_varlen as eager_delta,
)
from kohakumoss_kernels.backends.tilelang.attention._kernels import compile_kernel
from kohakumoss_kernels.backends.tilelang.blocks._blockops import (
    DTYPE,
    CompileError,
    arch,
    compile_once,
    head_dims,
)
from kohakumoss_kernels.backends.tilelang.textenc import _kernels as K
from kohakumoss_kernels.constraints import requires
from kohakumoss_kernels.errors import BUILD_ERRORS, unavailable
from kohakumoss_kernels.registry import register
from kohakumoss_kernels.tuning.buckets import (
    KEY_VERSION,
    cdiv,
    delta_buckets,
    length_bucket,
)
from kohakumoss_kernels.tuning.tuner import on_reset

LOG2E = 1.0 / math.log(2.0)
ATTN_HEAD_DIMS = (64, 128, 256)
DELTA_HEAD_DIMS = (64, 128)
DELTA_KERNEL = "te_delta_rule_tilelang"
ATTN_KERNEL = "te_attn_causal_tilelang"

_BOUND: dict[tuple, dict] = {}
on_reset(_BOUND.clear)


def _compiled(device: torch.device, builder, *args):
    """builder(*args) compiled once per arch (a failure is kept, raised as
    `CompileError`)."""
    target = arch(device)
    return compile_once(
        (builder.__module__, builder.__qualname__, target, *args),
        lambda: compile_kernel(builder(*args), target),
        CompileError,
    )


def delta_candidates(dv: int = 128) -> list[dict]:
    """Value tiles that divide `dv` x thread counts, then the PyTorch path (the chunked
    eager reference)."""
    tiles = [
        {"bv": bv, "threads": th}
        for bv in (64, 32, 16)
        for th in (64, 32, 128)
        if dv % bv == 0
    ]
    return tiles[:1] + [{"impl": "torch"}] + tiles[1:]


def delta_run(cfg, q, k, v, a, b, a_log, dt_bias, cu, max_len) -> Tensor:
    """One configured launch of the token-serial delta rule (or the PyTorch path)."""
    if cfg.get("impl") == "torch":
        return eager_delta(q, k, v, a, b, a_log, dt_bias, cu, max_len)
    q, k, v, a, b = (t.contiguous() for t in (q, k, v, a, b))
    if v.shape[2] % cfg["bv"]:
        raise ValueError(f"value tile {cfg['bv']} does not divide dv {v.shape[2]}")
    out = torch.empty_like(v)
    if q.shape[0] == 0 or max_len == 0:
        return out
    hk, dk = q.shape[1], q.shape[2]
    hv, dv = v.shape[1], v.shape[2]
    kern = _compiled(
        q.device, K.delta_recurrent, hk, hv, dk, dv, cfg["bv"], DTYPE[q.dtype],
        cfg["threads"],
    )  # fmt: skip
    kern(q, k, v, a, b, a_log.float(), dt_bias.float(), cu, out, dk**-0.5)
    return out


def _delta_key(q: Tensor, v: Tensor, segments: int, max_len: int) -> str:
    nseg, length, mean = delta_buckets(q.shape[0], segments, max_len)
    return (
        f"{KEY_VERSION}-{q.dtype}-hk{q.shape[1]}-hv{v.shape[1]}-d{q.shape[2]}"
        f"-dv{v.shape[2]}-B{nseg}-L{length}-M{mean}"
    )


def gated_delta_rule_varlen(q, k, v, a, b, a_log, dt_bias, cu_seqlens, max_seqlen):
    """Packed gated delta rule through the tuned TileLang config for this shape; a
    build / launch failure raises `BackendUnavailable`."""
    max_len = int(max_seqlen)
    args = (q, k, v, a, b, a_log, dt_bias, cu_seqlens, max_len)
    key = _delta_key(q, v, cu_seqlens.shape[0] - 1, max_len)
    ident = ("delta", q.device, key)
    try:
        cfg = _BOUND.get(ident)
        if cfg is None:
            cfg = tuning.select(
                DELTA_KERNEL,
                q.device,
                key,
                lambda: delta_candidates(v.shape[2]),
                lambda c: lambda: delta_run(c, *args),
                shortlist=10,
                errors=BUILD_ERRORS,
            )
            if tuning.decided(DELTA_KERNEL, q.device, key):
                _BOUND[ident] = cfg
        return delta_run(cfg, *args)
    except Exception as err:
        raise unavailable("gated_delta_rule_varlen", err, "tilelang") from err


def attn_candidates(d: int) -> list[dict]:
    """TileLang tiles that fit the head dim, then the PyTorch path (the per-segment
    SDPA loop)."""
    tiles = ((64, 32, 128), (64, 64, 128), (32, 32, 128), (64, 32, 256), (128, 32, 256))
    if d <= 128:
        tiles += ((128, 64, 256), (128, 32, 128))
    out = [{"bm": bm, "bn": bn, "threads": th} for bm, bn, th in tiles]
    return out[:1] + [{"impl": "sdpa"}] + out[1:]


def attn_run(cfg, q, k, v, cu, max_len, scale) -> Tensor:
    """One configured launch of the causal GQA forward (or the PyTorch path)."""
    if cfg.get("impl") == "sdpa":
        return eager_causal(q, k, v, cu, max_len, scale)
    q, k, v = (t.contiguous() for t in (q, k, v))
    out = torch.empty_like(q)
    if q.shape[0] == 0 or max_len == 0:
        return out
    h, d = q.shape[1], q.shape[2]
    kern = _compiled(
        q.device, K.attn_causal, h, k.shape[1], d, DTYPE[q.dtype],
        cfg["bm"], cfg["bn"], cfg["threads"],
    )  # fmt: skip
    kern(q, k, v, cu, out, cdiv(max_len, cfg["bm"]), scale * LOG2E)
    return out


def attention_varlen_causal(q, k, v, cu_seqlens, max_seqlen, scale):
    """Packed causal GQA attention through the tuned TileLang config for this shape; a
    build / launch failure raises `BackendUnavailable`."""
    max_len, scale = int(max_seqlen), float(scale)
    args = (q, k, v, cu_seqlens, max_len, scale)
    key = (
        f"{KEY_VERSION}-{q.dtype}-h{q.shape[1]}-kv{k.shape[1]}-d{q.shape[2]}"
        f"-L{length_bucket(max_len)}"
    )
    ident = ("attn", q.device, key)
    try:
        cfg = _BOUND.get(ident)
        if cfg is None:
            cfg = tuning.select(
                ATTN_KERNEL,
                q.device,
                key,
                lambda: attn_candidates(q.shape[2]),
                lambda c: lambda: attn_run(c, *args),
                shortlist=8,
                errors=BUILD_ERRORS,
            )
            if tuning.decided(ATTN_KERNEL, q.device, key):
                _BOUND[ident] = cfg
        return attn_run(cfg, *args)
    except Exception as err:
        raise unavailable("attention_varlen_causal", err, "tilelang") from err


register(
    "gated_delta_rule_varlen",
    "tilelang",
    gated_delta_rule_varlen,
    requires(
        kinds=("cuda",),
        min_sm=70,
        dtypes=tuple(DTYPE),
        extra=head_dims(("head_dim_k", "head_dim_v"), DELTA_HEAD_DIMS),
    ),
)
register(
    "attention_varlen_causal",
    "tilelang",
    attention_varlen_causal,
    requires(
        kinds=("cuda",),
        min_sm=70,
        dtypes=(torch.float16, torch.bfloat16),
        extra=head_dims(("head_dim",), ATTN_HEAD_DIMS),
    ),
)
