"""The registry's `attention_varlen` (the per-shape choice on sm80+ / XPU, TileLang on
sm70 / sm75): error statistics of whatever is chosen, and binding from a fresh cache
under no_grad, inference mode, checkpoint, prewarm and torch.compile."""

import importlib

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import kohakumoss_kernels as kk
from kohakumoss_kernels import tuning
from kohakumoss_kernels.tuning import tuner
from tests._util.error_stats import assert_no_regression
from tests._util.negative import must_fail
from tests.kernels.attention._attention_case import (
    check_stats,
    make_case,
    pytorch_sdpa,
    ref64,
    run,
)

pytestmark = pytest.mark.gpu

KERNELS = (
    "kohakumoss_kernels.backends.triton.attention.varlen",
    "kohakumoss_kernels.backends.tilelang.attention.varlen",
)
ENTRIES = (
    "kohakumoss_kernels.backends.triton.attention.entry",
    "kohakumoss_kernels.backends.tilelang.attention.entry",
)


def _optional(name: str):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


BINDINGS = [m._BOUND for m in map(_optional, KERNELS) if m is not None] + [
    m.CHOICE.bound for m in map(_optional, ENTRIES) if m is not None
]


def _clear_bindings() -> None:
    for bound in BINDINGS:
        bound.clear()


def _resolve(device, dtype=torch.float16):
    fn = kk.resolve("attention_varlen", device, dtype, head_dim=128)
    if fn.backend == "eager":
        pytest.skip("no attention kernel on this device")
    return fn


@pytest.fixture
def fresh_cache(tmp_path, monkeypatch):
    """No stored tuning choices and no bindings."""

    def clear():
        tuner._TABLES.clear()
        tuner._CHOSEN.clear()
        _clear_bindings()

    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    clear()
    yield
    clear()


@pytest.mark.budget(30)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
def test_chosen_implementation_has_no_error_regression(dtype, device):
    if dtype == torch.bfloat16 and device.type == "cuda":
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            pytest.skip("no bf16 on this card")
    fn = _resolve(device, dtype)
    case = make_case([300, 0, 77, 1029, 5], device, dtype=dtype)
    out, grads = run(fn, *case)
    eo, eg = pytorch_sdpa(*case)
    ro, rg = ref64(*case)
    check_stats(fn.backend, out, grads, eo, eg, ro, rg, dtype)


def _check(name, out, grads, case):
    """The error statistics of o (and dq, dk, dv when given) vs `pytorch_sdpa`, against
    fp64."""
    eo, eg = pytorch_sdpa(*case)
    ro, rg = ref64(*case)
    if grads is None:
        assert_no_regression(out, eo, ro, f"{name} o", test="attention")
        return
    check_stats(name, out, grads, eo, eg, ro, rg)


def _agrees(fn, case):
    out, grads = run(fn, *case)
    _check("choice", out, grads, case)


def test_no_grad_first_call(device, fresh_cache):
    """The first call of a key under no_grad, then a training call."""
    fn = _resolve(device)
    q, k, v, do, cu, max_len = case = make_case([300, 77], device)
    with torch.no_grad():
        out = fn(q, k, v, cu, max_len, q.shape[-1] ** -0.5)
    _check("no_grad", out, None, case)
    _agrees(fn, case)


def test_inference_mode_first_call(device, fresh_cache):
    fn = _resolve(device)
    with torch.inference_mode():
        q, k, v, do, cu, max_len = case = make_case([300, 77], device)
        out = fn(q, k, v, cu, max_len, q.shape[-1] ** -0.5)
    plain = [t.clone() for t in case[:5]] + [max_len]
    _check("inference_mode", out.clone(), None, plain)
    _agrees(fn, make_case([300, 77], device))


def test_checkpoint_first_call(device, fresh_cache):
    """The first call of a key inside a checkpointed forward and its recompute."""
    fn = _resolve(device)
    case = make_case([300, 77], device)
    q, k, v, do, cu, max_len = case
    scale = q.shape[-1] ** -0.5
    leaves = [t.clone().requires_grad_() for t in (q, k, v)]
    out = checkpoint(fn, *leaves, cu, max_len, scale, use_reentrant=False)
    out.backward(do)
    _check("checkpoint", out.detach(), [t.grad for t in leaves], case)


def check_live_calls_bound(fn, device):
    """Live calls of every length bucket up to 600 only look tuned choices up."""
    mode = tuner.get_mode()
    tuner.set_mode("off")
    try:
        chosen = dict(tuner._CHOSEN)
        for length in (64, 100, 300, 600):
            _agrees(fn, make_case([length, length // 2], device))
        assert tuner._CHOSEN == chosen, "a live call made a new choice"
    finally:
        tuner.set_mode(mode)


def test_prewarm_binds_every_bucket(device, fresh_cache):
    fn = _resolve(device)
    must_fail(lambda: check_live_calls_bound(fn, device), "no prewarm")
    tuner._CHOSEN.clear()
    _clear_bindings()
    tuning.prewarm_attention(
        fn, device, torch.float16, heads=3, head_dim=128, max_len=600, segments=2
    )
    check_live_calls_bound(fn, device)


def check_reset_unbinds(fn, device):
    """`tuner.reset` (run by `tuning.broadcast`) drops every attention binding."""
    _agrees(fn, make_case([300, 77], device))
    assert any(BINDINGS), "no attention binding was made"
    tuner.reset()
    assert not any(BINDINGS), "a binding survived reset"


def test_reset_unbinds(device, fresh_cache, monkeypatch):
    fn = _resolve(device)
    check_reset_unbinds(fn, device)
    monkeypatch.setattr(tuner, "_RESETS", [])
    must_fail(lambda: check_reset_unbinds(fn, device), "no reset hook")


def test_compile_fullgraph(device, fresh_cache):
    fn = _resolve(device)
    q, k, v, do, cu, max_len = make_case([300, 77], device)
    scale = q.shape[-1] ** -0.5
    compiled = torch.compile(
        lambda q, k, v: fn(q, k, v, cu, max_len, scale), fullgraph=True
    )
    leaves = [t.clone().requires_grad_() for t in (q, k, v)]
    out = compiled(*leaves)
    out.backward(do)
    case = (q, k, v, do, cu, max_len)
    _check("compiled", out.detach(), [t.grad for t in leaves], case)
