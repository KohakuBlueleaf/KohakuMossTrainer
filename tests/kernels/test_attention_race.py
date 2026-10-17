"""The registry's `attention_varlen` (the per-shape race on sm80+ / XPU, TileLang on
sm70 / sm75): precision of whatever wins, and binding from a fresh cache under no_grad,
inference mode, checkpoint, prewarm and torch.compile."""

import importlib

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import kohakumoss_kernels as kk
from kohakumoss_kernels import tuning
from kohakumoss_kernels.backends.eager.attention import attention_varlen as eager
from kohakumoss_kernels.tuning import tuner
from tests._util.negative import must_fail
from tests.kernels._attention_case import check_bar, make_case, mm16, ref64, run

pytestmark = pytest.mark.gpu

BINDINGS = (
    "kohakumoss_kernels.backends.triton.attention",
    "kohakumoss_kernels.backends.triton.attention_race",
    "kohakumoss_kernels.backends.tilelang.attention",
)


def _optional(name: str):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


MODULES = [m for m in map(_optional, BINDINGS) if m is not None]


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
        for mod in MODULES:
            mod._BOUND.clear()

    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    clear()
    yield
    clear()


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
def test_race_winner_meets_the_bar(dtype, device):
    if dtype == torch.bfloat16 and device.type == "cuda":
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            pytest.skip("no bf16 on this card")
    fn = _resolve(device, dtype)
    case = make_case([300, 0, 77, 1029, 5], device, dtype=dtype)
    out, grads = run(fn, *case)
    eo, eg = mm16(*case)
    ro, rg = ref64(*case)
    check_bar(fn.backend, out, grads, eo, eg, ro, rg, dtype)


def _agrees(fn, case):
    out, grads = run(fn, *case)
    eo, eg = run(eager, *case)
    torch.testing.assert_close(out, eo, atol=2e-3, rtol=0)
    for a, e in zip(grads, eg):
        torch.testing.assert_close(a, e, atol=5e-3, rtol=0)


def test_no_grad_first_call(device, fresh_cache):
    """The first call of a key under no_grad, then a training call."""
    fn = _resolve(device)
    q, k, v, do, cu, max_len = case = make_case([300, 77], device)
    with torch.no_grad():
        out = fn(q, k, v, cu, max_len, q.shape[-1] ** -0.5)
    ref = eager(q, k, v, cu, max_len, q.shape[-1] ** -0.5)
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=0)
    _agrees(fn, case)


def test_inference_mode_first_call(device, fresh_cache):
    fn = _resolve(device)
    with torch.inference_mode():
        q, k, v, do, cu, max_len = make_case([300, 77], device)
        out = fn(q, k, v, cu, max_len, q.shape[-1] ** -0.5)
        ref = eager(q, k, v, cu, max_len, q.shape[-1] ** -0.5)
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=0)
    _agrees(fn, make_case([300, 77], device))


def test_checkpoint_first_call(device, fresh_cache):
    """The first call of a key inside a checkpointed forward and its recompute."""
    fn = _resolve(device)
    q, k, v, do, cu, max_len = make_case([300, 77], device)
    scale = q.shape[-1] ** -0.5
    leaves = [t.clone().requires_grad_() for t in (q, k, v)]
    out = checkpoint(fn, *leaves, cu, max_len, scale, use_reentrant=False)
    out.backward(do)
    eo, eg = run(eager, q, k, v, do, cu, max_len)
    torch.testing.assert_close(out.detach(), eo, atol=2e-3, rtol=0)
    for a, e in zip(leaves, eg):
        torch.testing.assert_close(a.grad, e, atol=5e-3, rtol=0)


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
    for mod in MODULES:
        mod._BOUND.clear()
    tuning.prewarm_attention(
        fn, device, torch.float16, heads=3, head_dim=128, max_len=600
    )
    check_live_calls_bound(fn, device)


def check_reset_unbinds(fn, device):
    """`tuner.reset` (run by `tuning.broadcast`) drops every attention binding."""
    _agrees(fn, make_case([300, 77], device))
    assert any(mod._BOUND for mod in MODULES)
    tuner.reset()
    assert not any(mod._BOUND for mod in MODULES), "a binding survived reset"


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
    compiled(*leaves).backward(do)
    eo, eg = run(eager, q, k, v, do, cu, max_len)
    for a, e in zip(leaves, eg):
        torch.testing.assert_close(a.grad, e, atol=5e-3, rtol=0)
