"""Triton `gated_delta_rule_varlen` (forward only) for sm80+ CUDA and XPU.

Four designs timed per problem key: `short` (segments of <= 64 tokens packed several
to a 64-row tile, stateless; longer segments through `chunk` or `recurrent`), `chunk`
(prep + state + output kernels, chunk 64), `fused` (prep + one serial kernel that also
writes the output) and `recurrent` (token-serial fp32 state, vector math), against the
PyTorch path (`torch`, the chunked eager reference). Tile and warp choices are tuner
candidates; the winner is bound per key. See docs/internals/text-encoder.md.
"""

from dataclasses import dataclass

import torch
import triton.language as tl
from torch import Tensor

from kohakumoss_kernels import tuning
from kohakumoss_kernels.backends._memo import TensorMemo
from kohakumoss_kernels.backends.eager.textenc import (
    gated_delta_rule_varlen as eager_delta,
)
from kohakumoss_kernels.backends.triton._common import Launcher
from kohakumoss_kernels.backends.triton.textenc import _delta_kernels as K
from kohakumoss_kernels.backends.triton.textenc import _delta_short as S
from kohakumoss_kernels.constraints import requires
from kohakumoss_kernels.errors import BUILD_ERRORS, unavailable
from kohakumoss_kernels.registry import register
from kohakumoss_kernels.tuning.buckets import KEY_VERSION, cdiv, delta_buckets
from kohakumoss_kernels.tuning.tuner import on_reset

KERNEL = "te_delta_rule_triton"
SHORTLIST = 17
TORCH = {"impl": "torch"}
HEAD_DIMS = (64, 128)
DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_PREP = Launcher(K.prep_kernel)
_STATE = Launcher(K.state_kernel)
_OUT = Launcher(K.out_kernel)
_FUSED = Launcher(K.fused_kernel)
_RECUR = Launcher(K.recurrent_kernel)
_SHORT = Launcher(S.short_kernel)
_BOUND: dict[tuple, object] = {}
on_reset(_BOUND.clear)
_CACHE = TensorMemo()  # per `cu` tensor: shared by every layer of a pass


@dataclass
class Segs:
    """Segments a launch covers: segment s spans [sb[s], se[s]); `ccu` int32 chunk
    prefix sums; `n` segments, `tokens` their total, `max_len` the longest."""

    sb: Tensor
    se: Tensor
    ccu: Tensor
    n: int
    tokens: int
    max_len: int


@dataclass
class ShortPlan:
    """Tiles of whole short segments, per-token segment ids, and the long segments."""

    tiles: Tensor
    seg: Tensor
    long: Segs | None


def _ccu(sb: Tensor, se: Tensor) -> Tensor:
    lens = (se - sb).to(torch.int64)
    ccu = torch.zeros(sb.shape[0] + 1, device=sb.device, dtype=torch.int32)
    ccu[1:] = ((lens + K.BT - 1) // K.BT).cumsum(0).to(torch.int32)
    return ccu


def full_segs(cu: Tensor, n_tokens: int, max_len: int) -> Segs:
    """Every segment of `cu`."""
    sb, se = cu[:-1], cu[1:]
    ccu = _CACHE.get(cu, "ccu", lambda: _ccu(sb, se))
    return Segs(sb, se, ccu, cu.shape[0] - 1, n_tokens, max_len)


def _short_plan(cu: Tensor, n_tokens: int) -> ShortPlan:
    bounds = cu.tolist()
    tiles, longs = [], []
    for s, (b, e) in enumerate(zip(bounds[:-1], bounds[1:])):
        if e - b > K.BT:
            longs.append(s)
        elif e > b and tiles and tiles[-1][1] == b and e - tiles[-1][0] <= K.BT:
            tiles[-1][1] = e
        elif e > b:
            tiles.append([b, e])
    dev = cu.device
    idx = torch.arange(n_tokens, device=dev)
    seg = torch.searchsorted(cu[1:].to(torch.int64), idx, right=True).to(torch.int32)
    tiles_t = torch.tensor(tiles or [[0, 0]], dtype=torch.int32, device=dev)
    long = None
    if longs:
        sel = torch.tensor(longs, dtype=torch.int64, device=dev)
        sb, se = cu[:-1].index_select(0, sel), cu[1:].index_select(0, sel)
        lens = [bounds[s + 1] - bounds[s] for s in longs]
        long = Segs(sb, se, _ccu(sb, se), len(longs), sum(lens), max(lens))
    return ShortPlan(tiles_t if tiles else tiles_t[:0], seg, long)


def _tiles(cfg: dict) -> tuple[int, ...]:
    """The value-column tiles a config launches with."""
    return tuple(cfg[k] for k in ("bs", "bo", "bv", "bc") if k in cfg)


def candidates(device: torch.device, dtype: torch.dtype, dv: int = 128) -> list[dict]:
    """Configs whose value tiles divide `dv`, ranked: the tuner times the first
    SHORTLIST (each design's leaders, the overall leader first), the rest of the grid
    follows for benches. fp32 inputs run only `recurrent` (vector math: MMAs take
    16-bit operands only)."""
    fits = [c for c in _grid(dtype) if all(dv % t == 0 for t in _tiles(c))]
    lead = [c for c in _leaders(dtype) if c in fits]
    return lead[:1] + [TORCH] + lead[1:] + [c for c in fits if c not in lead]


def _short_cfg(w, bc, long):
    return {"impl": "short", "w": w, "bc": bc, "long": long}


def _chunk_cfg(wp, bs, ws, st, bo, wo):
    return {"impl": "chunk", "wp": wp, "bs": bs, "ws": ws, "st": st, "bo": bo, "wo": wo}


def _fused_cfg(wp, bv, w, st):
    return {"impl": "fused", "wp": wp, "bv": bv, "w": w, "st": st}


def _recur_cfg(bv, w):
    return {"impl": "recurrent", "bv": bv, "w": w}


def _leaders(dtype: torch.dtype) -> list[dict]:
    """Each design's leading configs (see docs/internals/text-encoder.md)."""
    if dtype == torch.float32:
        return [_recur_cfg(16, 1), _recur_cfg(64, 4), _recur_cfg(32, 1)]
    return [
        _chunk_cfg(2, 128, 8, 1, 128, 8),
        _chunk_cfg(2, 128, 8, 1, 64, 4),
        _chunk_cfg(2, 128, 8, 2, 128, 8),
        _chunk_cfg(2, 64, 8, 2, 128, 8),
        _chunk_cfg(2, 64, 4, 1, 128, 8),
        _chunk_cfg(2, 32, 4, 1, 128, 8),
        _chunk_cfg(2, 32, 8, 1, 64, 4),
        _short_cfg(4, 64, "chunk"),
        _short_cfg(2, 64, "chunk"),
        _short_cfg(4, 32, "recurrent"),
        _short_cfg(8, 32, "chunk"),
        _fused_cfg(4, 32, 4, 1),
        _fused_cfg(4, 32, 4, 2),
        _fused_cfg(2, 32, 4, 1),
        _fused_cfg(2, 16, 4, 2),
        _recur_cfg(16, 1),
    ]


def _grid(dtype: torch.dtype) -> list[dict]:
    """Every design's tile / warp / stage grid."""
    recur = [_recur_cfg(bv, w) for bv in (16, 32, 64) for w in (1, 2, 4)]
    if dtype == torch.float32:
        return recur
    short = [
        _short_cfg(w, bc, long)
        for w, bc in ((4, 32), (8, 64), (4, 64), (4, 16), (2, 32), (8, 32), (2, 64))
        for long in ("chunk", "recurrent")
    ]
    chunk = [
        _chunk_cfg(wp, bs, ws, st, bo, wo)
        for bs, ws in ((64, 4), (32, 4), (64, 8), (32, 8), (128, 8))
        for bo, wo in ((128, 8), (64, 4), (32, 4), (64, 8))
        for st in (1, 2)
        for wp in (2, 4, 1)
    ]
    fused = [
        _fused_cfg(wp, bv, w, st)
        for bv, w in ((32, 4), (16, 4), (64, 8), (128, 8))
        for st in (1, 2)
        for wp in (2, 4, 1)
    ]
    return short + chunk + fused + recur


def _dt(dtype: torch.dtype):
    """The MMA operand type of 16-bit inputs."""
    return tl.float16 if dtype == torch.float16 else tl.bfloat16


def _chunk(cfg, q, k, v, a, b, a_log, dt_bias, segs: Segs, out):
    _, hk, dk = q.shape
    hv, dv = v.shape[1], v.shape[2]
    dev = q.device
    chunks = segs.tokens // K.BT + segs.n
    t = torch.empty(chunks, hv, K.BT, K.BT, device=dev, dtype=q.dtype)
    h = torch.empty(chunks, hv, dk, dv, device=dev, dtype=q.dtype)
    vn = torch.empty(q.shape[0], hv, dv, device=dev, dtype=q.dtype)
    dt = _dt(q.dtype)
    groups = hv // hk
    nchunk = cdiv(segs.max_len, K.BT)
    sb, se, ccu = segs.sb, segs.se, segs.ccu
    _PREP[(nchunk, hv, segs.n)](
        k, a, b, a_log, dt_bias, sb, se, ccu, t, q, t, t, t,
        k.stride(0), a.stride(0), hv, q.stride(0), dk**-0.5,
        GROUPS=groups, DK=dk, DT=dt, WITH_QK=False, num_warps=cfg["wp"],
    )  # fmt: skip
    _STATE[(segs.n, hv, dv // cfg["bs"])](
        k, v, a, b, a_log, dt_bias, t, h, vn, sb, se, ccu,
        k.stride(0), v.stride(0), a.stride(0), vn.stride(0), hv,
        GROUPS=groups, DK=dk, DV=dv, BV=cfg["bs"], DT=dt,
        num_warps=cfg["ws"], num_stages=cfg["st"],
    )  # fmt: skip
    nvt = dv // cfg["bo"]
    _OUT[(nchunk, hv * nvt, segs.n)](
        q, k, a, b, a_log, dt_bias, h, vn, sb, se, ccu, out, dk**-0.5,
        q.stride(0), k.stride(0), a.stride(0), vn.stride(0), out.stride(0), hv, nvt,
        GROUPS=groups, DK=dk, DV=dv, BV=cfg["bo"], DT=dt, num_warps=cfg["wo"],
    )  # fmt: skip


def _fused(cfg, q, k, v, a, b, a_log, dt_bias, segs: Segs, out):
    _, hk, dk = q.shape
    hv, dv = v.shape[1], v.shape[2]
    dev = q.device
    chunks = segs.tokens // K.BT + segs.n
    t = torch.empty(chunks, hv, K.BT, K.BT, device=dev, dtype=q.dtype)
    qk = torch.empty(chunks, hv, K.BT, K.BT, device=dev, dtype=q.dtype)
    fac = torch.empty(q.shape[0], hv, 4, device=dev, dtype=torch.float32)
    egl = torch.empty(chunks, hv, device=dev, dtype=torch.float32)
    dt = _dt(q.dtype)
    groups = hv // hk
    sb, se, ccu = segs.sb, segs.se, segs.ccu
    _PREP[(cdiv(segs.max_len, K.BT), hv, segs.n)](
        k, a, b, a_log, dt_bias, sb, se, ccu, t, q, qk, fac, egl,
        k.stride(0), a.stride(0), hv, q.stride(0), dk**-0.5,
        GROUPS=groups, DK=dk, DT=dt, WITH_QK=True, num_warps=cfg["wp"],
    )  # fmt: skip
    _FUSED[(segs.n, hv, dv // cfg["bv"])](
        q, k, v, t, qk, fac, egl, sb, se, ccu, out,
        q.stride(0), k.stride(0), v.stride(0), out.stride(0), hv,
        GROUPS=groups, DK=dk, DV=dv, BV=cfg["bv"], DT=dt,
        num_warps=cfg["w"], num_stages=cfg["st"],
    )  # fmt: skip


def _recurrent(cfg, q, k, v, a, b, a_log, dt_bias, segs: Segs, out):
    _, hk, dk = q.shape
    hv, dv = v.shape[1], v.shape[2]
    _RECUR[(segs.n, hv, dv // cfg["bv"])](
        q, k, v, a, b, a_log, dt_bias, out, segs.sb, segs.se, dk**-0.5,
        q.stride(0), k.stride(0), v.stride(0), a.stride(0), out.stride(0),
        GROUPS=hv // hk, DK=dk, DV=dv, BV=cfg["bv"], num_warps=cfg["w"],
    )  # fmt: skip


LONG = {
    "chunk": {"wp": 2, "bs": 128, "ws": 8, "st": 1, "bo": 128, "wo": 8},
    "recurrent": {"bv": 16, "w": 1},
}


def _short(cfg, q, k, v, a, b, a_log, dt_bias, cu, n, out):
    _, hk, dk = q.shape
    hv, dv = v.shape[1], v.shape[2]
    plan = _CACHE.get(cu, "short", lambda: _short_plan(cu, n))
    if plan.tiles.shape[0]:
        _SHORT[(plan.tiles.shape[0], hv)](
            q, k, v, a, b, a_log, dt_bias, plan.tiles, plan.seg, out, dk**-0.5,
            q.stride(0), k.stride(0), v.stride(0), a.stride(0), out.stride(0),
            GROUPS=hv // hk, DK=dk, DV=dv, BC=cfg["bc"], DT=_dt(q.dtype),
            num_warps=cfg["w"],
        )  # fmt: skip
    if plan.long is not None:
        long = {"impl": cfg["long"], **LONG[cfg["long"]]}
        long.update({k: min(long[k], dv) for k in ("bs", "bo", "bv") if k in long})
        DESIGNS[cfg["long"]](long, q, k, v, a, b, a_log, dt_bias, plan.long, out)


DESIGNS = {"chunk": _chunk, "fused": _fused, "recurrent": _recurrent}


def _rows(t: Tensor) -> Tensor:
    """`t` when each token row is laid out [heads, dim] densely (any token stride)."""
    dense = t.stride(-1) == 1 and (t.dim() < 3 or t.stride(-2) == t.shape[-1])
    return t if dense else t.contiguous()


def run(cfg: dict, q, k, v, a, b, a_log, dt_bias, cu, max_len) -> Tensor:
    """One configured launch (benches and precision tests call this directly); the
    `torch` config is the PyTorch path (the chunked eager reference)."""
    if cfg["impl"] == "torch":
        return eager_delta(q, k, v, a, b, a_log, dt_bias, cu, max_len)
    q, k, v, a, b = (_rows(t) for t in (q, k, v, a, b))
    if any(v.shape[2] % t for t in _tiles(cfg)):
        raise ValueError(f"value tiles {_tiles(cfg)} do not divide dv {v.shape[2]}")
    n = q.shape[0]
    out = torch.empty(v.shape, device=v.device, dtype=v.dtype)
    if not n or not max_len:
        return out
    if cfg["impl"] == "short":
        _short(cfg, q, k, v, a, b, a_log, dt_bias, cu, n, out)
    else:
        segs = full_segs(cu, n, int(max_len))
        DESIGNS[cfg["impl"]](cfg, q, k, v, a, b, a_log, dt_bias, segs, out)
    return out


def _key(q: Tensor, v: Tensor, nseg: int, max_len: int) -> str:
    b, length, mean = delta_buckets(q.shape[0], nseg, max_len)
    return (
        f"{KEY_VERSION}-{q.dtype}-hk{q.shape[1]}-hv{v.shape[1]}-dk{q.shape[2]}"
        f"-dv{v.shape[2]}-B{b}-L{length}-M{mean}"
    )


def gated_delta_rule_varlen(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    b: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,
) -> Tensor:
    """Packed gated delta rule through the fastest Triton design for this shape."""
    max_len = int(max_seqlen)
    key = _key(q, v, cu_seqlens.shape[0] - 1, max_len)
    ident = (q.device, key)
    try:
        cfg = _BOUND.get(ident)
        if cfg is None:
            args = (q, k, v, a, b, a_log, dt_bias, cu_seqlens, max_len)
            cfg = tuning.select(
                KERNEL,
                q.device,
                key,
                lambda: candidates(q.device, q.dtype, v.shape[2]),
                lambda c: lambda: run(c, *args),
                shortlist=SHORTLIST,
                errors=BUILD_ERRORS,
            )
            if tuning.decided(KERNEL, q.device, key):
                _BOUND[ident] = cfg
        return run(cfg, q, k, v, a, b, a_log, dt_bias, cu_seqlens, max_len)
    except Exception as err:
        raise unavailable("gated_delta_rule_varlen", err, "triton") from err


def _dims_ok(query) -> str | None:
    for name in ("head_dim_k", "head_dim_v"):
        d = query.get(name)
        if d is not None and d not in HEAD_DIMS:
            return f"{name} {d} not in {HEAD_DIMS}"
    return None


SUPPORTS = requires(kinds=("cuda", "xpu"), min_sm=80, dtypes=DTYPES, extra=_dims_ok)
register("gated_delta_rule_varlen", "triton", gated_delta_rule_varlen, SUPPORTS)
