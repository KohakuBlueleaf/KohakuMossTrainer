"""Every tile config the tuner may bind for a MOSS-sized key (the shortlist of each
attention kernel, Triton and TileLang) computes finite results within the precision
bar, on a packed case with a partial last block, an empty and a short segment."""

import importlib
import math

import pytest
import torch

from kohakumoss_kernels.hardware.device import planning_device
from tests._util.negative import must_fail
from tests.kernels._attention_case import check_bar, kernel_fn, make_case, mm16, ref64

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

LENS = [2337, 0, 77, 129]
KEYS = [(24, 2, 2420), (24, 6, 600)]
DTYPES = (torch.float16, torch.bfloat16)
BUILD = (RuntimeError, ImportError)


def _module(backend):
    return importlib.import_module(f"kohakumoss_kernels.backends.{backend}.attention")


def _case(device, dtype):
    case = make_case(LENS, device, heads=2, dtype=dtype)
    return case, mm16(*case), ref64(*case)


def _shortlist(mod, kind, device, dtype):
    """The union over `KEYS` (heads, segments, longest segment) of the configs the
    tuner times, plus the untimed safe tile."""
    dev = planning_device(device)
    out = [mod.SAFE[kind]]
    for heads, segs, max_len in KEYS:
        args = (128, torch.finfo(dtype).bits // 8, heads, segs, max_len)
        if mod.__name__.endswith("triton.attention"):
            cands = mod._candidates(dev, kind, *args)
        else:
            cands = mod._candidates(kind, dev, *args)
        out += [c for c in cands[: mod.SHORTLIST] if c not in out]
    return out


def _triton_fwd(tri, cfg, q, k, v, cu, max_len):
    n, h, _ = q.shape
    o = torch.empty_like(q)
    m, lg = (torch.empty(h, n, device=q.device) for _ in range(2))
    tri._run_fwd(cfg, q, k, v, cu, max_len, q.shape[-1] ** -0.5, o, m, lg)
    return o, m, lg


def _triton_bwd(tri, kind, cfg, case, stats):
    q, k, v, do, cu, max_len = case
    o, m, lg = stats
    n, h, d = q.shape
    delta = torch.empty(h, n, device=q.device)
    tri._DELTA[(tri.cdiv(n, 64), h)](o, do, delta, n, *tri._strides(o, do), D=d, BM=64)
    a = (q, k, v, do, m, lg, delta, cu, max_len, d**-0.5)
    dq, dk, dv = (torch.full_like(t, float("nan")) for t in (q, k, v))
    match kind:
        case "fused":
            tri._run_fused(cfg, a, dq, dk, dv)
        case "dkdv":
            tri._run_dkdv(cfg, a, dk, dv)
            tri._run_dq(tri.SAFE["dq"], a, dq)
        case _:
            tri._run_dq(cfg, a, dq)
            tri._run_dkdv(tri.SAFE["dkdv"], a, dk, dv)
    return [dq, dk, dv]


def check_triton_tiles(tri, device, dtype, run_bwd=None):
    """Each shortlisted Triton config of every kernel meets the bar (fwd: o; bwd kernels
    on the SAFE forward: dq, dk, dv); empty-segment and padded rows are written."""
    run_bwd = run_bwd or _triton_bwd
    case, (eo, eg), (ro, rg) = _case(device, dtype)
    q, k, v, do, cu, max_len = case
    tried = 0
    for cfg in _shortlist(tri, "fwd", device, dtype):
        try:
            o, _, _ = _triton_fwd(tri, cfg, q, k, v, cu, max_len)
        except BUILD:
            continue
        check_bar(f"triton fwd {cfg}", o, eg, eo, eg, ro, rg)
        tried += 1
    stats = _triton_fwd(tri, tri.SAFE["fwd"], q, k, v, cu, max_len)
    for kind in ("fused", "dkdv", "dq"):
        for cfg in _shortlist(tri, kind, device, dtype):
            try:
                grads = run_bwd(tri, kind, cfg, case, stats)
            except BUILD:
                continue
            check_bar(f"triton {kind} {cfg}", stats[0], grads, eo, eg, ro, rg)
            tried += 1
    assert tried > 0, "no config built"


@pytest.mark.parametrize("dtype", DTYPES, ids=("fp16", "bf16"))
def test_triton_shortlists(dtype, device):
    kernel_fn("triton", device, dtype)
    check_triton_tiles(_module("triton"), device, dtype)


def _stale_dq_acc(tri, kind, cfg, case, stats):
    """Broken variant: the fused dQ accumulator is not zeroed (stale memory)."""
    if kind != "fused":
        return _triton_bwd(tri, kind, cfg, case, stats)
    zeros = torch.zeros
    torch.zeros = lambda *s, **kw: torch.full(s, 1e3, **kw)
    try:
        return _triton_bwd(tri, kind, cfg, case, stats)
    finally:
        torch.zeros = zeros


@pytest.mark.parametrize("dtype", (torch.bfloat16,), ids=("bf16",))
def test_triton_shortlists_catch_stale_accumulator(dtype, device):
    kernel_fn("triton", device, dtype)
    tri = _module("triton")
    must_fail(
        lambda: check_triton_tiles(tri, device, dtype, _stale_dq_acc),
        "unzeroed dQ accumulator",
    )


def check_tilelang_tiles(tla, device, dtype):
    """Each shortlisted TileLang config of fwd / dK dV / dQ meets the bar."""
    case, (eo, eg), (ro, rg) = _case(device, dtype)
    q, k, v, do, cu, max_len = case
    n, h, d = q.shape
    dt, scale = tla.DTYPES[dtype], d**-0.5
    c = scale / math.log(2.0)
    safe_fwd = tla._compiled("fwd", device, h, d, dt, tla.SAFE["fwd"])
    o = torch.empty_like(q)
    m, lg = (torch.empty(h, n, device=device) for _ in range(2))
    tla._launch_fwd(safe_fwd, tla.SAFE["fwd"]["BM"], q, k, v, cu, max_len, scale, o, m, lg)
    delta = torch.empty_like(m)
    tla._delta_kernel(device, h, d, dt)(o, do, delta)
    tried = 0
    for kind in ("fwd", "dkdv", "dq"):
        for cfg in _shortlist(tla, kind, device, dtype):
            try:
                kern = tla._compiled(kind, device, h, d, dt, cfg)
            except Exception:  # noqa: BLE001 - TileLang raises its own compile errors
                continue
            dq, dk, dv = (torch.full_like(t, float("nan")) for t in (q, k, v))
            match kind:
                case "fwd":
                    out = torch.full_like(q, float("nan"))
                    tla._launch_fwd(kern, cfg["BM"], q, k, v, cu, max_len, scale, out, m.clone(), lg.clone())
                    check_bar(f"tilelang fwd {cfg}", out, eg, eo, eg, ro, rg)
                    tried += 1
                    continue
                case "dkdv":
                    kern(q, k, v, do, m, lg, delta, cu, dk, dv, -(-max_len // cfg["BN"]), scale, c)
                    dq = eg[0]
                case _:
                    kern(q, k, v, do, m, lg, delta, cu, dq, -(-max_len // cfg["BM"]), scale, c)
                    dk, dv = eg[1], eg[2]
            check_bar(f"tilelang {kind} {cfg}", o, [dq, dk, dv], eo, eg, ro, rg)
            tried += 1
    assert tried > 0, "no config built"


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", DTYPES, ids=("fp16", "bf16"))
def test_tilelang_shortlists(dtype, device):
    kernel_fn("tilelang", device, dtype)
    check_tilelang_tiles(_module("tilelang"), device, dtype)
