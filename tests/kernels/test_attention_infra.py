"""Attention infrastructure: host segment bounds of the eager spec, registry fallback
of the attention kernels, and the TileLang backward's row-stat guards."""

import importlib
import math
import warnings
from functools import partial

import pytest
import torch

from kohakumoss_kernels import health, registry, tuning
from kohakumoss_kernels.backends.eager import attention as eager_mod
from kohakumoss_kernels.hardware.arch import arch_of
from kohakumoss_kernels.tuning import tuner
from tests._util.negative import must_fail
from tests.kernels._attention_case import kernel_fn, make_case, run

COMPILER_ERROR = "Failed to find C++ compiler. Please specify via CXX"
OP = "test_attention_fallback"


def _optional(name: str):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


KERNEL_MODULES = {
    "triton": _optional("kohakumoss_kernels.backends.triton.attention"),
    "tilelang": _optional("kohakumoss_kernels.backends.tilelang.attention"),
}
RACE = _optional("kohakumoss_kernels.backends.triton.attention_race")


def _cpu_case(lens, heads=2, d=16):
    return make_case(lens, torch.device("cpu"), heads=heads, d=d, dtype=torch.float32)


def _per_segment(q, k, v, bounds, scale):
    """Independent reference: explicit softmax per segment, bounds given on the host."""
    out = torch.zeros_like(q)
    for s, e in zip(bounds[:-1], bounds[1:]):
        p = (torch.einsum("qhd,khd->hqk", q[s:e], k[s:e]) * scale).softmax(-1)
        out[s:e] = torch.einsum("hqk,khd->qhd", p, v[s:e])
    return out


def _attend(q, k, v, cu, bounds, scale):
    """Eager attention vs the per-segment reference; any error is a failed check."""
    try:
        out = eager_mod.attention_varlen(q, k, v, cu, max(bounds), scale)
    except RuntimeError as err:
        raise AssertionError(f"eager attention raised: {err}") from err
    assert torch.allclose(out, _per_segment(q, k, v, bounds, scale), atol=1e-5)


def check_follows_tensor():
    """Eager attention uses the current values of `cu_seqlens`: after an in-place
    rewrite, and for a new tensor that may reuse a freed one's id."""
    q, k, v, _, cu, _ = _cpu_case([3, 5])
    scale = q.shape[-1] ** -0.5
    _attend(q, k, v, cu, [0, 3, 8], scale)
    cu.copy_(torch.tensor([0, 6, 8], dtype=cu.dtype))
    _attend(q, k, v, cu, [0, 6, 8], scale)
    for split in (1, 2, 4, 7):
        fresh = torch.tensor([0, split, 8], dtype=torch.int32)
        _attend(q, k, v, fresh, [0, split, 8], scale)
        del fresh


def check_inference_mode():
    """Bounds registered and attention run under `inference_mode` (sampling)."""
    with torch.inference_mode():
        q, k, v, _, cu, _ = _cpu_case([4, 9])
        scale = q.shape[-1] ** -0.5
        try:
            eager_mod.remember_bounds(cu, [0, 4, 13])
        except RuntimeError as err:
            raise AssertionError(f"remember_bounds raised: {err}") from err
        _attend(q, k, v, cu, [0, 4, 13], scale)
        cu.copy_(torch.tensor([0, 10, 13], dtype=cu.dtype))
        _attend(q, k, v, cu, [0, 10, 13], scale)


def _stale_host_bounds():
    """Broken variant: bounds cached per tensor id, ignoring version and lifetime."""
    seen = {}
    return lambda cu: seen.setdefault(id(cu), cu.tolist())


def _versioned_remember(cu, bounds):
    """Broken variant: reads the version counter of every tensor."""
    eager_mod._BOUNDS[id(cu)] = (None, cu._version, [int(b) for b in bounds])


def _versioned_host_bounds(cu):
    """Broken variant: reads the version counter of every tensor."""
    hit = eager_mod._BOUNDS.get(id(cu))
    if hit is not None and hit[1] == cu._version:
        return hit[2]
    return cu.tolist()


def test_host_bounds_follow_the_tensor(monkeypatch):
    check_follows_tensor()
    monkeypatch.setattr(eager_mod, "host_bounds", _stale_host_bounds())
    must_fail(check_follows_tensor, "stale bounds")


def test_host_bounds_under_inference_mode(monkeypatch):
    check_inference_mode()
    with monkeypatch.context() as patch:
        patch.setattr(eager_mod, "host_bounds", _stale_host_bounds())
        must_fail(check_inference_mode, "inference tensors cached without a version")
    with monkeypatch.context() as patch:
        patch.setattr(eager_mod, "remember_bounds", _versioned_remember)
        patch.setattr(eager_mod, "host_bounds", _versioned_host_bounds)
        must_fail(check_inference_mode, "version counter read on an inference tensor")


def check_no_sync(cu, case):
    q, k, v = case[:3]
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        eager_mod.attention_varlen(q, k, v, cu, case[5], q.shape[-1] ** -0.5)
    except RuntimeError as err:
        raise AssertionError(f"host sync: {err}") from err
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.cuda
def test_remembered_bounds_launch_without_a_host_sync():
    """A `cu_seqlens` whose bounds the layout builder registered never syncs the host;
    an unregistered one syncs once, on first sight."""
    case = make_case([300, 0, 77], torch.device("cuda"), heads=2, d=64)
    host = [0, 300, 300, 377]
    cu = torch.tensor(host, dtype=torch.int32).cuda()
    eager_mod.remember_bounds(cu, host)
    check_no_sync(cu, case)
    again = torch.tensor(host, dtype=torch.int32).cuda()
    must_fail(lambda: check_no_sync(again, case), "bounds read back from the device")
    eager_mod.host_bounds(again)
    check_no_sync(again, case)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Fresh tuning tables and failure records under a temporary kernel cache; every
    attention binding cleared before and after."""
    mods = [m for m in (*KERNEL_MODULES.values(), RACE) if m is not None]

    def clear():
        health.clear_failures()
        tuner._TABLES.clear()
        tuner._CHOSEN.clear()
        for mod in mods:
            mod._BOUND.clear()

    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    mode = tuner.get_mode()
    clear()
    yield
    tuner.set_mode(mode)
    registry._STATE.impls.pop(OP, None)
    clear()


def _raise_compiler_error(*args, **kwargs):
    raise RuntimeError(COMPILER_ERROR)


def _broken_tilelang_compiled(original):
    """TileLang `_compiled` whose backward kernels cannot build."""

    def compiled(kind, *args):
        if kind != "fwd":
            raise RuntimeError(COMPILER_ERROR)
        return original(kind, *args)

    return compiled


def _break_backward(backend, mod, monkeypatch):
    match backend:
        case "triton":
            for name in ("_run_dkdv", "_run_dq", "_run_fused"):
                monkeypatch.setattr(mod, name, _raise_compiler_error)
        case _:
            monkeypatch.setattr(
                mod, "_compiled", _broken_tilelang_compiled(mod._compiled)
            )


def _break_forward(backend, mod, monkeypatch):
    name = "_run_fwd" if backend == "triton" else "_launch_fwd"
    monkeypatch.setattr(mod, name, _raise_compiler_error)


def _fallback_op(backend, device, monkeypatch):
    """`OP` with priority backend -> eager on this arch, resolved for fp16."""
    fn = kernel_fn(backend, device, torch.float16)
    registry.register(OP, backend, fn)
    registry.register(OP, "eager", eager_mod.attention_varlen)
    monkeypatch.setitem(
        registry._STATE.priority, arch_of(device).name, (backend, "eager")
    )
    return registry.resolve(OP, device, torch.float16)


def check_falls_back_in_forward(fn, backend, case):
    """The forward call already lands on eager (warning, recorded failure); the backward
    then runs eager's and matches it."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out, grads = run(fn, *case)
        except Exception as err:
            raise AssertionError(f"no fallback: {type(err).__name__}: {err}") from err
    eo, eg = run(eager_mod.attention_varlen, *case)
    assert getattr(fn, "backend", None) == "eager"
    assert any(f"backend {backend} failed" in str(w.message) for w in caught)
    assert any(COMPILER_ERROR in v for v in registry.failures().values())
    torch.testing.assert_close(out, eo)
    for a, b in zip(grads, eg):
        torch.testing.assert_close(a, b)


def _raw_forward(op, q, k, v, cu, max_len, scale):
    """Broken variant: the custom op alone (a backward failure surfaces in backward)."""
    return op(q, k, v, cu, max_len, scale)[0]


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ("auto", "off"))
@pytest.mark.parametrize("backend", ("triton", "tilelang"))
def test_backward_failure_falls_back_in_forward(
    backend, mode, device, isolated, monkeypatch
):
    """The backward kernels cannot build or launch (timed when tuning, the untimed first
    pick under TUNE=off): the first grad-enabled forward of the problem key already
    hands the op to the next backend."""
    fn = _fallback_op(backend, device, monkeypatch)
    mod = KERNEL_MODULES[backend]
    tuner.set_mode(mode)
    _break_backward(backend, mod, monkeypatch)
    case = make_case([300, 77], device)
    check_falls_back_in_forward(fn, backend, case)
    health.clear_failures()
    raw = partial(_raw_forward, getattr(mod, f"attn_varlen_{backend}"))
    must_fail(lambda: check_falls_back_in_forward(raw, backend, case), "raw custom op")


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ("auto", "off"))
@pytest.mark.parametrize("backend", ("triton", "tilelang"))
def test_forward_failure_falls_back(backend, mode, device, isolated, monkeypatch):
    """The forward kernel cannot launch (timed, or the untimed first pick)."""
    fn = _fallback_op(backend, device, monkeypatch)
    mod = KERNEL_MODULES[backend]
    tuner.set_mode(mode)
    _break_forward(backend, mod, monkeypatch)
    case = make_case([200, 61], device)
    check_falls_back_in_forward(fn, backend, case)
    health.clear_failures()
    raw = partial(_raw_forward, getattr(mod, f"attn_varlen_{backend}"))
    must_fail(lambda: check_falls_back_in_forward(raw, backend, case), "raw custom op")


def check_compiled_step(fn, case):
    """A torch.compile'd forward + backward through `fn` runs and matches eager's."""
    q, k, v, do, cu, max_len = case
    scale = q.shape[-1] ** -0.5
    torch._dynamo.reset()
    compiled = torch.compile(lambda q, k, v: fn(q, k, v, cu, max_len, scale))
    leaves = [t.clone().requires_grad_() for t in (q, k, v)]
    try:
        out = compiled(*leaves)
        out.backward(do)
    except Exception as err:
        raise AssertionError(f"compiled step: {type(err).__name__}: {err}") from err
    eo, eg = run(eager_mod.attention_varlen, *case)
    torch.testing.assert_close(out.detach(), eo)
    for a, b in zip(leaves, eg):
        torch.testing.assert_close(a.grad, b)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ("triton", "tilelang"))
def test_prewarm_falls_back_before_a_compiled_step(
    backend, device, isolated, monkeypatch
):
    """The backward kernels cannot build: prewarm (forward + backward per length bucket)
    hands the op to eager before the first compiled step, which then runs."""
    fn = _fallback_op(backend, device, monkeypatch)
    _break_backward(backend, KERNEL_MODULES[backend], monkeypatch)
    case = make_case([300, 77], device)
    must_fail(lambda: check_compiled_step(fn, case), "compiled without prewarm")
    health.clear_failures()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tuning.prewarm_attention(fn, device, torch.float16, 3, 128, max_len=300)
    assert fn.backend == "eager"
    check_compiled_step(fn, case)


def _tailed(t: torch.Tensor, fill: float) -> torch.Tensor:
    """`t` copied into the front of a larger buffer whose tail holds `fill`."""
    pad = 256 * math.prod(t.shape[1:])
    buf = torch.full((t.numel() + pad,), fill, dtype=t.dtype, device=t.device)
    buf[: t.numel()] = t.flatten()
    return buf[: t.numel()].view(t.shape)


def check_stat_guards(kdkdv, kdq, cfgs, case, stats):
    """dQ / dK / dV are finite and identical whether the memory after the row stats
    holds zeros or NaN."""
    q, k, v, do, cu, max_len = case
    scale = q.shape[-1] ** -0.5
    c = scale / math.log(2.0)

    def grads(fill):
        qt, kt, vt, dot = (_tailed(t, 0.0) for t in (q, k, v, do))
        mt, lt, dt = (_tailed(t, fill) for t in stats)
        dq, dk, dv = (_tailed(torch.zeros_like(q), 0.0) for _ in range(3))
        kdkdv(qt, kt, vt, dot, mt, lt, dt, cu, dk, dv, -(-max_len // cfgs[0]), scale, c)
        kdq(qt, kt, vt, dot, mt, lt, dt, cu, dq, -(-max_len // cfgs[1]), scale, c)
        return dq, dk, dv

    for name, a, b in zip(("dq", "dk", "dv"), grads(0.0), grads(float("nan"))):
        assert torch.isfinite(b).all(), f"{name}: NaN from past the last token"
        assert torch.equal(a, b), name


@pytest.mark.cuda
def test_tilelang_backward_reads_no_row_stats_past_the_last_token(device):
    """Built without TileLang's automatic bounds guards, the backward kernels read no
    row stat past the last token (the last segment ends inside a query block)."""
    kernel_fn("tilelang", device, torch.float16)
    tla = KERNEL_MODULES["tilelang"]
    h, d = 2, 64
    case = make_case([100, 37], device, heads=h, d=d)
    q, k, v, do, cu, max_len = case
    o, m, lg = tla.attn_varlen_tilelang(q, k, v, cu, max_len, d**-0.5)
    delta = torch.empty_like(m)
    tla._delta_kernel(device, h, d, "float16")(o, do, delta)
    dk_cfg, dq_cfg = tla.SAFE["dkdv"], tla.SAFE["dq"]
    arch = tla._arch(device)
    kdkdv, kdq = (
        tla.K.compile_kernel(
            build(h, d, "float16", cfg["BM"], cfg["BN"], cfg["threads"]),
            arch,
            safe=False,
        )
        for build, cfg in ((tla.K.dkdv, dk_cfg), (tla.K.dq, dq_cfg))
    )
    cfgs = (dk_cfg["BN"], dq_cfg["BM"])
    check_stat_guards(kdkdv, kdq, cfgs, case, (m, lg, delta))
