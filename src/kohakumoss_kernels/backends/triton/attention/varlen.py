"""Triton varlen attention ops `kohakumoss::attn_varlen_triton{,_bwd}` (sm80+ CUDA, XPU).

Tiles per problem key `dtype-h<heads>-d<head dim>-L<longest segment, pow2 bucket>-S<segment
count bucket>`; for the backward the tuner times split (dK/dV + dQ) against fused. The
registry entry is `entry.py`. See docs/internals/kernel-dev.md and tuning.md.
"""

import math
from functools import partial

import torch
from torch import Tensor
from torch.library import register_autograd, triton_op

from kohakumoss_kernels import tuning
from kohakumoss_kernels.backends.triton._common import Launcher
from kohakumoss_kernels.backends.triton.attention import _kernels as K
from kohakumoss_kernels.backends.varlen_choice import needs_backward
from kohakumoss_kernels.constraints import requires
from kohakumoss_kernels.errors import BUILD_ERRORS, unavailable
from kohakumoss_kernels.hardware.device import planning_device
from kohakumoss_kernels.saved_ops import keep_named
from kohakumoss_kernels.tuning.buckets import LENGTH_FLOOR, cdiv, pow2, segment_bucket
from kohakumoss_kernels.tuning.plan import Work, rank
from kohakumoss_kernels.tuning.tuner import on_reset

LOG2E = 1.0 / math.log(2.0)
HEAD_DIMS = (32, 64, 128, 256)
SHORTLIST = 10

_FWD = Launcher(K.fwd_kernel)
_DELTA = Launcher(K.delta_kernel)
_DKDV = Launcher(K.dkdv_kernel)
_DQ = Launcher(K.dq_kernel)
_FUSED = Launcher(K.fused_bwd_kernel)
_CAST = Launcher(K.dq_cast_kernel)

TILES = [
    {"BM": bm, "BN": bn, "warps": w, "stages": s}
    for bm in (16, 32, 64, 128)
    for bn in (16, 32, 64, 128)
    for w in (4, 8)
    for s in (1, 2, 3, 4)
]


def _smem(c: dict, d: int, esize: int, kind: str) -> int:
    """Shared bytes: the resident operand tiles plus `stages` copies of the streamed ones."""
    st = max(c["stages"], 1)
    match kind:
        case "fwd" | "dq":
            return (
                esize * d * (c["BM"] * (1 if kind == "fwd" else 2) + st * 2 * c["BN"])
            )
        case _:
            return esize * d * (2 * c["BN"] + st * 2 * c["BM"])


def _acc_regs(c: dict, d: int, kind: str) -> int:
    rows = c["BN"] if kind in ("dkdv", "fused") else c["BM"]
    accs = 2 if kind in ("dkdv", "fused") else 1
    tile = c["BM"] * c["BN"]
    return (rows * d * accs + 2 * tile) // (32 * c["warps"])


def _cfg(bm: int, bn: int, w: int, s: int) -> dict:
    return {"BM": bm, "BN": bn, "warps": w, "stages": s}


SEEDS = {
    "fwd": [_cfg(128, 32, 4, 2), _cfg(128, 64, 8, 2), _cfg(128, 128, 8, 2),
            _cfg(128, 64, 4, 2), _cfg(128, 32, 4, 3), _cfg(64, 64, 4, 3),
            _cfg(128, 32, 4, 1), _cfg(64, 128, 4, 2)],
    "dkdv": [_cfg(32, 32, 4, 2), _cfg(16, 128, 8, 2), _cfg(32, 128, 8, 3),
             _cfg(64, 64, 8, 2), _cfg(32, 64, 4, 3), _cfg(32, 128, 8, 2),
             _cfg(64, 64, 4, 2), _cfg(16, 128, 4, 2)],
    "dq": [_cfg(128, 32, 8, 3), _cfg(128, 64, 8, 2), _cfg(128, 32, 4, 2),
           _cfg(128, 32, 8, 2), _cfg(64, 64, 4, 2), _cfg(64, 32, 4, 3)],
    "fused": [_cfg(64, 128, 8, 1), _cfg(32, 128, 8, 2), _cfg(16, 128, 8, 2),
              _cfg(64, 64, 4, 2), _cfg(64, 64, 8, 2), _cfg(32, 64, 4, 3),
              _cfg(64, 128, 8, 2), _cfg(32, 128, 4, 2)],
}  # fmt: skip

ARCH_SEEDS = {
    80: {
        "fwd": [_cfg(128, 64, 4, 3), _cfg(128, 128, 8, 3), _cfg(128, 64, 8, 3)],
        "dkdv": [_cfg(64, 128, 8, 2), _cfg(32, 128, 4, 3), _cfg(64, 64, 4, 3)],
        "dq": [_cfg(128, 64, 8, 3), _cfg(128, 32, 4, 3), _cfg(128, 64, 8, 2)],
        "fused": [_cfg(64, 128, 8, 2), _cfg(32, 128, 4, 3), _cfg(16, 128, 8, 4)],
    },
    86: {
        "fwd": [_cfg(128, 32, 4, 2), _cfg(128, 128, 8, 2), _cfg(128, 64, 8, 2)],
        "dkdv": [_cfg(16, 128, 8, 2), _cfg(32, 128, 8, 3), _cfg(32, 32, 4, 2)],
        "dq": [_cfg(128, 64, 8, 2), _cfg(128, 32, 8, 3), _cfg(128, 32, 4, 2)],
        "fused": [_cfg(16, 128, 8, 2), _cfg(32, 32, 4, 2), _cfg(16, 128, 8, 3)],
    },
    120: {
        "fwd": [_cfg(128, 64, 8, 3), _cfg(128, 64, 8, 2), _cfg(128, 32, 4, 2)],
        "dkdv": [_cfg(32, 32, 4, 3), _cfg(64, 64, 8, 3), _cfg(32, 32, 4, 2)],
        "dq": [_cfg(128, 32, 8, 3), _cfg(128, 64, 8, 2), _cfg(128, 32, 8, 2)],
        "fused": [_cfg(16, 128, 8, 4), _cfg(16, 128, 8, 3), _cfg(64, 64, 8, 3)],
    },
}


ARCH_SEEDS[89] = ARCH_SEEDS[86]


def _seeds(dev, kind: str) -> list[dict]:
    """The arch's measured seeds (`ARCH_SEEDS[sm]`), then the shared ones."""
    own = ARCH_SEEDS.get(dev.sm, {}).get(kind, [])
    return own + [c for c in SEEDS[kind] if c not in own]


SAFE = {
    "fwd": _cfg(64, 32, 4, 1),
    "dkdv": _cfg(32, 32, 4, 1),
    "dq": _cfg(64, 32, 4, 1),
    "fused": _cfg(32, 32, 4, 1),
}


def _candidates(dev, kind, d, esize, heads, segs, max_len):
    """Seed tiles first (legal ones), then the planner's ranking. When the pick will be
    untimed (tracing, TUNE=off, a backward / recompute, a frozen or deferring tuner) a
    small tile that fits every sm80+ card leads."""
    seeds = [c for c in _seeds(dev, kind) if _fits(c, dev, kind, d, esize)]
    ranked = _ranked(dev, kind, d, esize, heads, segs, max_len)
    order = seeds + [c for c in ranked if c not in seeds]
    if not tuning.will_time(in_backward=kind != "fwd"):
        order = [SAFE[kind]] + [c for c in order if c != SAFE[kind]]
    return order


def _fits(c, dev, kind, d, esize) -> bool:
    return (
        _smem(c, d, esize, kind) <= dev.smem_per_block and _acc_regs(c, d, kind) <= 192
    )


def _ranked(dev, kind, d, esize, heads, segs, max_len):
    smem = dev.smem_per_block
    legal = [
        c
        for c in TILES
        if _smem(c, d, esize, kind) <= smem
        and _acc_regs(c, d, kind) <= 192
        and (kind == "fwd" and c["BM"] >= 32 or kind != "fwd")
    ]

    def work(c):
        outer = c["BN"] if kind in ("dkdv", "fused") else c["BM"]
        inner = c["BM"] if kind in ("dkdv", "fused") else c["BN"]
        progs = cdiv(max_len, outer) * heads * segs
        tiles = cdiv(max_len, inner)
        gemms = {"fwd": 2, "dq": 3, "dkdv": 4, "fused": 5}[kind]
        resident = max(1, min(dev.smem_per_sm // max(_smem(c, d, esize, kind), 1), 4))
        resident = min(resident, max(1, 64 // c["warps"] // 2))
        return Work(
            programs=progs,
            mma_flops=2.0 * gemms * progs * outer * inner * tiles * d,
            exp_ops=float(progs * outer * inner * tiles),
            dram_bytes=esize * d * heads * segs * max_len * (2 + 2 * tiles),
            resident=resident,
        )

    return rank(dev, legal, work)


def _key(q: Tensor, segs: int, max_len: int) -> str:
    """dtype, heads, head dim, longest-segment and segment-count buckets."""
    bucket = pow2(max_len, LENGTH_FLOOR)
    return f"{q.dtype}-h{q.shape[1]}-d{q.shape[2]}-L{bucket}-S{segment_bucket(segs)}"


FWD_KERNELS = ("attn_varlen_fwd_triton",)
BWD_KERNELS = (
    "attn_varlen_dkdv_triton",
    "attn_varlen_dq_triton",
    "attn_varlen_fused_triton",
    "attn_varlen_bwd_strategy_triton",
)


def _strides(*ts: Tensor) -> tuple[int, ...]:
    return tuple(s for t in ts for s in (t.stride(0), t.stride(1)))


def _run_fwd(cfg, q, k, v, cu, max_len, scale, o, m, lg):
    n, h, d = q.shape
    _FWD[(cdiv(max_len, cfg["BM"]), h, cu.numel() - 1)](
        q, k, v, o, m, lg, cu, scale * LOG2E, n, *_strides(q, k, v, o),
        D=d, BM=cfg["BM"], BN=cfg["BN"],
        num_warps=cfg["warps"], num_stages=cfg["stages"],
    )  # fmt: skip


def _run_delta(a, acc: Tensor | None) -> None:
    """DELTA = rowsum(O * dO); with `acc`, also zero the fp32 dQ accumulator."""
    q, do, delta, o = a[0], a[3], a[6], a[10]
    n, h, d = q.shape
    _DELTA[(cdiv(n, 64), h)](
        o, do, delta, delta if acc is None else acc, n, *_strides(o, do),
        D=d, BM=64, ZERO=acc is not None,
    )  # fmt: skip


def _run_dkdv(cfg, a, dk, dv):
    q, k, v, do, m, lg, delta, cu, max_len, scale, _ = a
    n, h, d = q.shape
    _DKDV[(cdiv(max_len, cfg["BN"]), h, cu.numel() - 1)](
        q, k, v, do, m, lg, delta, dk, dv, cu, scale, scale * LOG2E, n,
        *_strides(q, k, v, do, dk),
        D=d, BM=cfg["BM"], BN=cfg["BN"],
        num_warps=cfg["warps"], num_stages=cfg["stages"],
    )  # fmt: skip


def _run_dq(cfg, a, dq):
    q, k, v, do, m, lg, delta, cu, max_len, scale, _ = a
    n, h, d = q.shape
    _DQ[(cdiv(max_len, cfg["BM"]), h, cu.numel() - 1)](
        q, k, v, do, m, lg, delta, dq, cu, scale, scale * LOG2E, n,
        *_strides(q, k, v, do, dq),
        D=d, BM=cfg["BM"], BN=cfg["BN"],
        num_warps=cfg["warps"], num_stages=cfg["stages"],
    )  # fmt: skip


def _run_fused(cfg, a, dq, dk, dv):
    q, k, v, do, m, lg, delta, cu, max_len, scale, _ = a
    n, h, d = q.shape
    acc = torch.empty(n, h, d, device=q.device, dtype=torch.float32)
    _run_delta(a, acc)
    _FUSED[(cdiv(max_len, cfg["BN"]), h, cu.numel() - 1)](
        q, k, v, do, m, lg, delta, dk, dv, acc, cu, scale, scale * LOG2E, n,
        *_strides(q, k, v, do, dk),
        D=d, BM=cfg["BM"], BN=cfg["BN"],
        num_warps=cfg["warps"], num_stages=cfg["stages"],
    )  # fmt: skip
    _CAST[(cdiv(acc.numel(), 1024),)](acc, dq, scale, acc.numel(), BLOCK=1024)


def _run_split(dkdv, dqc, a, dq, dk, dv):
    _run_delta(a, None)
    _run_dkdv(dkdv, a, dk, dv)
    _run_dq(dqc, a, dq)


_BOUND: dict[tuple, object] = {}
on_reset(_BOUND.clear)


def _bound(ident: tuple, kernels: tuple[str, ...], build):
    """The launch callable bound for `ident` (direction, device, problem key), built
    from the tuner and memoised once every kernel's choice is settled (a provisional
    pick made while tracing or in a recompute is used once, not bound)."""
    fn = _BOUND.get(ident)
    if fn is None:
        fn = build()
        if all(tuning.decided(k, ident[1], ident[2]) for k in kernels):
            _BOUND[ident] = fn
    return fn


def _tile(kind, a, key, launch):
    q, cu, max_len = a[0], a[7], a[8]
    dev = planning_device(q.device)
    rest = (q.shape[2], q.element_size(), q.shape[1], cu.shape[0] - 1, max_len)
    return tuning.select(
        f"attn_varlen_{kind}_triton",
        q.device,
        key,
        lambda: _candidates(dev, kind, *rest),
        launch,
        shortlist=SHORTLIST,
        errors=BUILD_ERRORS,
        in_backward=True,
    )


def _build_backward(a, key, dq, dk, dv):
    """partial(_run_fused, cfg) or partial(_run_split, dkdv_cfg, dq_cfg); the dK/dV and
    dQ tiles are timed on this call's DELTA (computed first)."""
    _run_delta(a, None)
    dkdv = _tile("dkdv", a, key, lambda c: lambda: _run_dkdv(c, a, dk, dv))
    dqc = _tile("dq", a, key, lambda c: lambda: _run_dq(c, a, dq))
    split = partial(_run_split, dkdv, dqc)
    fused = partial(
        _run_fused,
        _tile("fused", a, key, lambda c: lambda: _run_fused(c, a, dq, dk, dv)),
    )
    options = {"fused": fused, "split": split}
    strategy = tuning.select(
        "attn_varlen_bwd_strategy_triton",
        a[0].device,
        key,
        lambda: [{"impl": "fused"}, {"impl": "split"}],
        lambda s: lambda: options[s["impl"]](a, dq, dk, dv),
        errors=BUILD_ERRORS,
        in_backward=True,
    )
    return options[strategy["impl"]]


def _build_forward(args, key):
    q, k, v, cu, max_len, scale, o, m, lg = args
    dev = planning_device(q.device)
    segs, d, h = cu.shape[0] - 1, q.shape[2], q.shape[1]
    cfg = tuning.select(
        "attn_varlen_fwd_triton",
        q.device,
        key,
        lambda: _candidates(dev, "fwd", d, q.element_size(), h, segs, max_len),
        lambda c: lambda: _run_fwd(c, *args),
        shortlist=SHORTLIST,
        errors=BUILD_ERRORS,
    )
    return partial(_run_fwd, cfg)


@triton_op("kohakumoss::attn_varlen_triton", mutates_args=())
def attn_varlen_triton(
    q: Tensor, k: Tensor, v: Tensor, cu: Tensor, max_seqlen: int, scale: float
) -> tuple[Tensor, Tensor, Tensor]:
    """(o [N,H,D], m [H,N] fp32, l [H,N] fp32) for packed q, k, v."""
    n, h, d = q.shape
    o = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    m = torch.empty(h, n, device=q.device, dtype=torch.float32)
    lg = torch.empty(h, n, device=q.device, dtype=torch.float32)
    if n == 0 or max_seqlen == 0:
        return o, m, lg
    args = (q, k, v, cu, max_seqlen, scale, o, m, lg)
    key = _key(q, cu.shape[0] - 1, max_seqlen)
    _bound(("fwd", q.device, key), FWD_KERNELS, lambda: _build_forward(args, key))(
        *args
    )
    return o, m, lg


@triton_op("kohakumoss::attn_varlen_triton_bwd", mutates_args=())
def attn_varlen_triton_bwd(
    do: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    m: Tensor,
    lg: Tensor,
    cu: Tensor,
    max_seqlen: int,
    scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """(dq, dk, dv) for `attn_varlen_triton`, contiguous [N, H, D]."""
    n, h, d = q.shape
    dq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    dk = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    dv = torch.empty(v.shape, device=v.device, dtype=v.dtype)
    if n == 0 or max_seqlen == 0:
        return dq, dk, dv
    delta = torch.empty(h, n, device=q.device, dtype=torch.float32)
    a = (q, k, v, do, m, lg, delta, cu, max_seqlen, scale, o)
    key = _key(q, cu.shape[0] - 1, max_seqlen)
    run = _bound(
        ("bwd", q.device, key), BWD_KERNELS, lambda: _build_backward(a, key, dq, dk, dv)
    )
    run(a, dq, dk, dv)
    return dq, dk, dv


def _setup(ctx, inputs, output):
    q, k, v, cu, max_seqlen, scale = inputs
    o, m, lg = output
    ctx.save_for_backward(q, k, v, o, m, lg, cu)
    ctx.max_seqlen = max_seqlen
    ctx.scale = scale


def _backward(ctx, do, _dm, _dl):
    q, k, v, o, m, lg, cu = ctx.saved_tensors
    dq, dk, dv = attn_varlen_triton_bwd(
        do.contiguous(), q, k, v, o, m, lg, cu, ctx.max_seqlen, ctx.scale
    )
    return dq, dk, dv, None, None, None


register_autograd("kohakumoss::attn_varlen_triton", _backward, setup_context=_setup)
keep_named("kohakumoss", "attn_varlen_triton")


def _bwd_bound(q: Tensor, cu: Tensor, max_len: int) -> bool:
    return ("bwd", q.device, _key(q, cu.shape[0] - 1, max_len)) in _BOUND


def attention_varlen(
    q: Tensor, k: Tensor, v: Tensor, cu_seqlens: Tensor, max_seqlen: int, scale: float
) -> Tensor:
    """Packed varlen attention through the Triton kernels only. A build or launch
    failure (forward, or the backward built on the first grad-enabled call of a
    problem key) raises `BackendUnavailable`."""
    max_seqlen, scale = int(max_seqlen), float(scale)
    q, k, v = (t if t.stride(-1) == 1 else t.contiguous() for t in (q, k, v))
    if torch.compiler.is_compiling():
        return attn_varlen_triton(q, k, v, cu_seqlens, max_seqlen, scale)[0]
    try:
        o, m, lg = attn_varlen_triton(q, k, v, cu_seqlens, max_seqlen, scale)
        if needs_backward(q, k, v) and not _bwd_bound(q, cu_seqlens, max_seqlen):
            with torch.no_grad():
                attn_varlen_triton_bwd(
                    torch.zeros_like(o),
                    q,
                    k,
                    v,
                    o,
                    m,
                    lg,
                    cu_seqlens,
                    max_seqlen,
                    scale,
                )
    except Exception as err:
        raise unavailable("attention_varlen", err, "triton") from err
    return o


def _head_dim_ok(query) -> str | None:
    d = query.get("head_dim")
    return None if d is None or d in HEAD_DIMS else f"head_dim {d} not in {HEAD_DIMS}"


SUPPORTS = requires(kinds=("cuda", "xpu"), min_sm=80, extra=_head_dim_ok)
