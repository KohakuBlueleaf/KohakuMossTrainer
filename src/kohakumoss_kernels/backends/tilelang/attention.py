"""TileLang varlen attention: `attention_varlen` on sm70+ (the sm70 / sm75 path), fwd + bwd.

Registered as `torch.ops.kohakumoss.attn_varlen_tilelang{,_bwd}` (opaque custom ops,
outputs allocated by torch and passed in) with autograd attached. Kernels compile once
per (heads, head dim, dtype, tile config, arch); token and segment counts are symbolic.
Per problem key the tuned forward, dK/dV and dQ kernels are bound once into launch
callables; the call path only looks the binding up.
"""

import math
import threading
from functools import partial

import torch
from torch import Tensor
from torch.library import custom_op, register_autograd, register_fake

from kohakumoss_kernels import tuning
from kohakumoss_kernels.backends.tilelang import _attention_kernels as K
from kohakumoss_kernels.constraints import requires
from kohakumoss_kernels.errors import BUILD_ERRORS, unavailable
from kohakumoss_kernels.hardware.current import on_device
from kohakumoss_kernels.hardware.device import planning_device
from kohakumoss_kernels.registry import register
from kohakumoss_kernels.tuning.plan import Work, rank
from kohakumoss_kernels.tuning.tuner import on_reset

LOG2E = 1.0 / math.log(2.0)
HEAD_DIMS = (64, 128)
DTYPES = {torch.float16: "float16", torch.bfloat16: "bfloat16"}
SHORTLIST = 6

_CACHE: dict[tuple, object] = {}
_LOCK = threading.Lock()
_BOUND: dict[tuple, object] = {}
on_reset(_BOUND.clear)

SPACES = {
    "fwd": [
        {"BM": bm, "BN": bn, "threads": th, "qreg": qr, "share": sh}
        for bm in (32, 64, 128)
        for bn in (32, 64)
        for th in (128, 256)
        for qr in (False, True)
        for sh in (False, True)
    ],
    "dkdv": [
        {"BM": bm, "BN": bn, "threads": th, "kvreg": kr, "pds": pd}
        for bm in (16, 32, 64)
        for bn in (32, 64, 128)
        for th in (128, 256)
        for kr in (False, True)
        for pd in (False, True)
    ],
    "dq": [
        {"BM": bm, "BN": bn, "threads": th, "qreg": qr, "share": sh}
        for bm in (32, 64, 128)
        for bn in (16, 32, 64)
        for th in (128, 256)
        for qr in (False, True)
        for sh in (False, True)
    ],
}
BUILDERS = {"fwd": K.fwd, "dkdv": K.dkdv, "dq": K.dq}
FLAGS = {"fwd": ("qreg", "share"), "dkdv": ("kvreg", "pds"), "dq": ("qreg", "share")}


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _arch(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}"


def _compiled(kind: str, device: torch.device, h: int, d: int, dt: str, c: dict):
    args = (h, d, dt, c["BM"], c["BN"], c["threads"], *(c[f] for f in FLAGS[kind]))
    key = (kind, _arch(device), *args)
    kernel = _CACHE.get(key)
    if kernel is None:
        with _LOCK:
            kernel = _CACHE.get(key)
            if kernel is None:
                kernel = K.compile_kernel(BUILDERS[kind](*args), _arch(device))
                _CACHE[key] = kernel
    return kernel


def _partitions(threads: int, *gemms: tuple[int, int]) -> bool:
    warps = threads // 32
    return all(
        any(
            warps % wm == 0 and m % (16 * wm) == 0 and n % (16 * (warps // wm)) == 0
            for wm in range(1, warps + 1)
        )
        for m, n in gemms
    )


def _usage(kind: str, c: dict, d: int) -> tuple[int, int]:
    """(shared bytes, 32-bit registers per thread) of one program."""
    bm, bn, th = c["BM"], c["BN"], c["threads"]
    match kind:
        case "fwd":
            smem = (0 if c["qreg"] else bm * d) + bn * d * (1 if c["share"] else 2)
            smem = 2 * (smem + bm * bn)
            regs = (bm * d + bm * bn + (bm * d // 2 if c["qreg"] else 0)) // th
        case "dkdv":
            smem = (0 if c["kvreg"] else 2 * bn * d) + 2 * bm * d
            smem = 2 * (smem + bn * bm * (1 if c["pds"] else 2)) + 12 * bm
            regs = (2 * bn * d + 2 * bn * bm + (bn * d if c["kvreg"] else 0)) // th
        case _:
            smem = (0 if c["qreg"] else 2 * bm * d) + bn * d * (1 if c["share"] else 2)
            smem = 2 * (smem + bm * bn)
            regs = (bm * d + 2 * bm * bn + (bm * d if c["qreg"] else 0)) // th
    return smem + 1024, regs


def _legal(kind: str, c: dict, d: int, smem: int) -> bool:
    used, regs = _usage(kind, c, d)
    bm, bn = c["BM"], c["BN"]
    gemms = {
        "fwd": ((bm, bn), (bm, d)),
        "dkdv": ((bn, bm), (bn, d)),
        "dq": ((bm, bn), (bm, d)),
    }[kind]
    return used <= smem and regs <= 168 and _partitions(c["threads"], *gemms)


def _fc(bm, bn, th, qreg, share):
    return {"BM": bm, "BN": bn, "threads": th, "qreg": qreg, "share": share}


def _kc(bm, bn, th, kvreg, pds):
    return {"BM": bm, "BN": bn, "threads": th, "kvreg": kvreg, "pds": pds}


SEEDS = {
    ("fwd", 70): [_fc(64, 64, 128, True, False), _fc(128, 64, 256, True, False),
                  _fc(128, 64, 256, False, False), _fc(128, 32, 128, False, True)],
    ("fwd", 75): [_fc(128, 32, 128, True, True), _fc(128, 64, 128, True, True),
                  _fc(128, 32, 128, True, False), _fc(64, 32, 128, True, True)],
    ("dkdv", 70): [_kc(16, 64, 128, False, False), _kc(16, 64, 128, False, True),
                   _kc(32, 128, 256, False, True), _kc(16, 128, 256, False, False)],
    ("dkdv", 75): [_kc(64, 64, 256, True, True), _kc(64, 64, 256, True, False),
                   _kc(32, 64, 256, True, True), _kc(64, 32, 256, True, True)],
    ("dq", 70): [_fc(64, 16, 128, True, False), _fc(64, 16, 128, True, True),
                 _fc(128, 16, 256, True, False), _fc(64, 32, 128, True, False)],
    ("dq", 75): [_fc(64, 16, 128, True, False), _fc(64, 16, 128, True, True),
                 _fc(128, 16, 256, True, True), _fc(64, 32, 256, True, True)],
}  # fmt: skip


SAFE = {
    "fwd": _fc(64, 32, 128, False, False),
    "dkdv": _kc(32, 32, 128, False, False),
    "dq": _fc(64, 32, 128, False, False),
}


def _candidates(kind, dev, d, esize, heads, segs, max_len):
    """Per-arch measured seeds first (the arch's own design), then the ranking; when
    nothing will be timed the per-arch first seed (or a small safe tile) leads."""
    seeds = [
        c
        for c in SEEDS.get((kind, dev.sm), [])
        if _legal(kind, c, d, dev.smem_per_block)
    ]
    order = seeds + [
        c for c in _ranked(kind, dev, d, esize, heads, segs, max_len) if c not in seeds
    ]
    if not seeds and not tuning.timing_allowed(in_backward=kind != "fwd"):
        order = [SAFE[kind]] + [c for c in order if c != SAFE[kind]]
    return order


def _ranked(kind, dev, d, esize, heads, segs, max_len):
    configs = [c for c in SPACES[kind] if _legal(kind, c, d, dev.smem_per_block)]

    def work(c):
        outer = c["BN"] if kind == "dkdv" else c["BM"]
        inner = c["BM"] if kind == "dkdv" else c["BN"]
        progs = _cdiv(max_len, outer) * heads * segs
        tiles = _cdiv(max_len, inner)
        gemms = {"fwd": 2, "dkdv": 4, "dq": 3}[kind]
        smem, regs = _usage(kind, c, d)
        resident = min(
            max(1, dev.smem_per_sm // smem),
            max(1, dev.regs_per_sm // max(regs * c["threads"], 1)),
        )
        return Work(
            programs=progs,
            mma_flops=2.0 * gemms * progs * outer * inner * tiles * d,
            exp_ops=float(progs * outer * inner * tiles),
            dram_bytes=esize * d * heads * segs * max_len * (2 + 2 * tiles),
            resident=resident,
        )

    return rank(dev, configs, work)


def _key(q: Tensor, segs: int, max_len: int) -> str:
    bucket = 64
    while bucket < max_len:
        bucket *= 2
    return f"{q.dtype}-h{q.shape[1]}-d{q.shape[2]}-L{bucket}"


def _launch_fwd(kernel, bm, q, k, v, cu, max_len, scale, o, m, lg):
    kernel(q, k, v, cu, o, m, lg, _cdiv(max_len, bm), scale * LOG2E)


def _launch_bwd(kdkdv, bn, kdq, bm, q, k, v, do, m, lg, delta, cu, max_len, scale, dq, dk, dv):  # fmt: skip
    c = scale * LOG2E
    kdkdv(q, k, v, do, m, lg, delta, cu, dk, dv, _cdiv(max_len, bn), scale, c)
    kdq(q, k, v, do, m, lg, delta, cu, dq, _cdiv(max_len, bm), scale, c)


def _select(kind, q, cu, max_len, key, launch):
    dev = planning_device(q.device)
    h, d = q.shape[1], q.shape[2]
    dt = DTYPES[q.dtype]
    cfg = tuning.select(
        f"attn_varlen_{kind}_tilelang",
        q.device,
        key,
        lambda: _candidates(
            kind, dev, d, q.element_size(), h, cu.shape[0] - 1, max_len
        ),
        lambda c: launch(_compiled(kind, q.device, h, d, dt, c), c),
        shortlist=SHORTLIST,
        errors=BUILD_ERRORS,
        in_backward=kind != "fwd",
    )
    return _compiled(kind, q.device, h, d, dt, cfg), cfg


FWD_KERNELS = ("attn_varlen_fwd_tilelang",)
BWD_KERNELS = ("attn_varlen_dkdv_tilelang", "attn_varlen_dq_tilelang")


def _bound(ident: tuple, kernels: tuple[str, ...], build):
    """The launch callable bound for `ident`, memoised once every kernel's choice is
    settled (a provisional pick while tracing or in a recompute is used once)."""
    fn = _BOUND.get(ident)
    if fn is None:
        fn = build()
        if all(tuning.decided(k, ident[1], ident[2]) for k in kernels):
            _BOUND[ident] = fn
    return fn


def _build_fwd(args, key):
    q, k, v, cu, max_len, scale, o, m, lg = args
    kern, cfg = _select(
        "fwd", q, cu, max_len, key,
        lambda kn, c: lambda: _launch_fwd(kn, c["BM"], *args),
    )  # fmt: skip
    return partial(_launch_fwd, kern, cfg["BM"])


def _build_bwd(args, key):
    q, k, v, do, m, lg, delta, cu, max_len, scale, dq, dk, dv = args
    c = scale * LOG2E
    kdkdv, cdkdv = _select(
        "dkdv", q, cu, max_len, key,
        lambda kn, cf: lambda: kn(
            q, k, v, do, m, lg, delta, cu, dk, dv, _cdiv(max_len, cf["BN"]), scale, c
        ),
    )  # fmt: skip
    kdq, cdq = _select(
        "dq", q, cu, max_len, key,
        lambda kn, cf: lambda: kn(
            q, k, v, do, m, lg, delta, cu, dq, _cdiv(max_len, cf["BM"]), scale, c
        ),
    )  # fmt: skip
    return partial(_launch_bwd, kdkdv, cdkdv["BN"], kdq, cdq["BM"])


@custom_op("kohakumoss::attn_varlen_tilelang", mutates_args=())
def attn_varlen_tilelang(
    q: Tensor, k: Tensor, v: Tensor, cu: Tensor, max_seqlen: int, scale: float
) -> tuple[Tensor, Tensor, Tensor]:
    """(o [N,H,D], m [H,N] fp32, l [H,N] fp32) for packed q, k, v."""
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    n, h, _ = q.shape
    o = torch.empty_like(q)
    m = torch.empty(h, n, device=q.device, dtype=torch.float32)
    lg = torch.empty(h, n, device=q.device, dtype=torch.float32)
    if n == 0 or max_seqlen == 0:
        return o, m, lg
    args = (q, k, v, cu, max_seqlen, scale, o, m, lg)
    key = _key(q, cu.shape[0] - 1, max_seqlen)
    with on_device(q.device):
        _bound(("fwd", q.device, key), FWD_KERNELS, lambda: _build_fwd(args, key))(
            *args
        )
    return o, m, lg


@register_fake("kohakumoss::attn_varlen_tilelang")
def _fwd_fake(q, k, v, cu, max_seqlen, scale):
    n, h, _ = q.shape
    stat = q.new_empty((h, n), dtype=torch.float32)
    return torch.empty_like(q), stat, torch.empty_like(stat)


@custom_op("kohakumoss::attn_varlen_tilelang_bwd", mutates_args=())
def attn_varlen_tilelang_bwd(
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
    """(dq, dk, dv) for `attn_varlen_tilelang`."""
    do, q, k, v, o = (t.contiguous() for t in (do, q, k, v, o))
    n, h, d = q.shape
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    if n == 0 or max_seqlen == 0:
        return dq, dk, dv
    delta = torch.empty(h, n, device=q.device, dtype=torch.float32)
    args = (q, k, v, do, m, lg, delta, cu, max_seqlen, scale, dq, dk, dv)
    key = _key(q, cu.shape[0] - 1, max_seqlen)
    with on_device(q.device):
        _delta_kernel(q.device, h, d, DTYPES[q.dtype])(o, do, delta)
        _bound(("bwd", q.device, key), BWD_KERNELS, lambda: _build_bwd(args, key))(
            *args
        )
    return dq, dk, dv


def _delta_kernel(device, h, d, dt):
    key = ("delta", _arch(device), h, d, dt)
    kernel = _CACHE.get(key)
    if kernel is None:
        kernel = K.compile_kernel(K.delta(h, d, dt), _arch(device))
        _CACHE[key] = kernel
    return kernel


@register_fake("kohakumoss::attn_varlen_tilelang_bwd")
def _bwd_fake(do, q, k, v, o, m, lg, cu, max_seqlen, scale):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _setup(ctx, inputs, output):
    q, k, v, cu, max_seqlen, scale = inputs
    o, m, lg = output
    ctx.save_for_backward(q, k, v, o, m, lg, cu)
    ctx.max_seqlen = max_seqlen
    ctx.scale = scale


def _backward(ctx, do, _dm, _dl):
    q, k, v, o, m, lg, cu = ctx.saved_tensors
    dq, dk, dv = attn_varlen_tilelang_bwd(
        do, q, k, v, o, m, lg, cu, ctx.max_seqlen, ctx.scale
    )
    return dq, dk, dv, None, None, None


register_autograd("kohakumoss::attn_varlen_tilelang", _backward, setup_context=_setup)


def attention_varlen(
    q: Tensor, k: Tensor, v: Tensor, cu_seqlens: Tensor, max_seqlen: int, scale: float
) -> Tensor:
    """Packed varlen attention (the registry entry). A build failure (forward, or the
    backward built on the first grad-enabled call of a problem key) raises
    `BackendUnavailable`."""
    max_seqlen, scale = int(max_seqlen), float(scale)
    if torch.compiler.is_compiling():
        return attn_varlen_tilelang(q, k, v, cu_seqlens, max_seqlen, scale)[0]
    try:
        o, m, lg = attn_varlen_tilelang(q, k, v, cu_seqlens, max_seqlen, scale)
        key = _key(q, cu_seqlens.shape[0] - 1, max_seqlen)
        grad = torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v))
        if grad and ("bwd", q.device, key) not in _BOUND:
            with torch.no_grad():
                attn_varlen_tilelang_bwd(
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
        raise unavailable("attention_varlen", err, "tilelang") from err
    return o


def _head_dim_ok(query) -> str | None:
    d = query.get("head_dim")
    return None if d is None or d in HEAD_DIMS else f"head_dim {d} not in {HEAD_DIMS}"


register(
    "attention_varlen",
    "tilelang",
    attention_varlen,
    requires(kinds=("cuda",), min_sm=70, extra=_head_dim_ok),
)
