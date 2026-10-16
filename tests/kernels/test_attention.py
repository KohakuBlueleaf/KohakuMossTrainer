"""attention_varlen backends vs the eager spec: precision (fp64 bar), huge logits,
boundaries, determinism and torch.compile."""

import importlib
import os

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import kohakumoss_kernels as kk
from kohakumoss_kernels import tuning
from kohakumoss_kernels.backends.eager.attention import attention_varlen as eager
from kohakumoss_kernels.tuning import tuner
from tests._util.negative import must_fail
from tests._util.precision import bar

pytestmark = pytest.mark.cuda

BACKENDS = ("triton", "tilelang")
TRITON_DETERMINISTIC = os.environ.get("KOHAKUMOSS_DETERMINISTIC", "0") == "1"
DTYPES = (torch.float16, torch.bfloat16)
LENS = [300, 0, 77, 1029, 5, 1, 128]


def _fn(backend, dtype):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    try:
        return kk.resolve("attention_varlen", "cuda", dtype, backend=backend)
    except kk.registry.NoImplementationError as err:
        pytest.skip(str(err))


def _case(lens, heads=3, d=128, dtype=torch.float16, qk_scale=1.0, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n = sum(lens)
    cu = torch.tensor(
        [0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
    )
    q, k, v, do = (
        torch.randn(n, heads, d, device="cuda", generator=g).to(dtype) for _ in range(4)
    )
    if qk_scale != 1.0:
        center = torch.randn(1, heads, d, device="cuda", generator=g)
        q = ((q.float() * 0.05 + center) * qk_scale).to(dtype)
        k = ((k.float() * 0.05 + center) * qk_scale).to(dtype)
    return q, k, v, do, cu, max(lens)


def _run(fn, q, k, v, do, cu, max_len):
    leaves = [t.clone().requires_grad_() for t in (q, k, v)]
    out = fn(*leaves, cu, max_len, q.shape[-1] ** -0.5)
    out.backward(do)
    return out.detach(), [t.grad for t in leaves]


def _ref64(q, k, v, do, cu, max_len):
    leaves = [t.double().requires_grad_() for t in (q, k, v)]
    out = eager(*leaves, cu, max_len, q.shape[-1] ** -0.5)
    out.backward(do.double())
    return out.detach(), [t.grad for t in leaves]


def _check(name, out, grads, eo, eg, ro, rg, dtype):
    """RMS error within 1.1x of the eager composition (PyTorch SDPA) in the same dtype,
    worst element within 2x (the max metric is dominated by near-zero outputs)."""
    for label, a, e, r in [("o", out, eo, ro), *zip(("dq", "dk", "dv"), grads, eg, rg)]:
        assert torch.isfinite(a).all(), f"{name} {label}: non-finite"
        res = bar(a, e, r, dtype)
        assert res.backend_rms <= max(
            1.1 * res.eager_rms, 2.0
        ), f"{name} {label}: {res}"
        assert res.backend_elem <= max(
            2.0 * res.eager_elem, 2.0
        ), f"{name} {label}: {res}"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", DTYPES, ids=("fp16", "bf16"))
def test_precision_bar(backend, dtype):
    fn = _fn(backend, dtype)
    case = _case(LENS, dtype=dtype)
    out, grads = _run(fn, *case)
    eo, eg = _run(eager, *case)
    ro, rg = _ref64(*case)
    _check(backend, out, grads, eo, eg, ro, rg, dtype)
    leak = _case([sum(LENS)], dtype=dtype)[4]
    broken, bgrads = _run(fn, *case[:4], leak, sum(LENS))
    must_fail(
        lambda: _check(backend, broken, bgrads, eo, eg, ro, rg, dtype),
        "attention across segments",
    )


def _cos(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return (a @ b / (a.norm() * b.norm()).clamp_min(1e-30)).item()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", DTYPES, ids=("fp16", "bf16"))
def test_huge_logits(backend, dtype):
    """Clustered q / k (logits ~1e5): cosine to fp64 at the KohakuFA floor
    (o, dV > 0.9999; dQ, dK > 0.995), or at least PyTorch SDPA's in the same dtype."""
    fn = _fn(backend, dtype)
    case = _case([700, 333], dtype=dtype, qk_scale=80.0)
    out, grads = _run(fn, *case)
    eo, eg = _run(eager, *case)
    ro, rg = _ref64(*case)
    assert torch.isfinite(out).all() and all(torch.isfinite(g).all() for g in grads)
    for a, e, r, floor in [
        (out, eo, ro, 0.9999),
        (grads[2], eg[2], rg[2], 0.9999),
        (grads[0], eg[0], rg[0], 0.995),
        (grads[1], eg[1], rg[1], 0.995),
    ]:
        assert _cos(a, r) > min(floor, _cos(e, r) - 1e-4), (_cos(a, r), _cos(e, r))


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_and_single_segments(backend):
    fn = _fn(backend, torch.float16)
    case = _case([0, 1, 0, 2, 0], heads=2)
    out, grads = _run(fn, *case)
    ro, rg = _ref64(*case)
    assert torch.allclose(out.double(), ro, atol=2e-3)
    for a, r in zip(grads, rg):
        assert torch.isfinite(a).all()
        assert torch.allclose(a.double(), r, atol=5e-3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_deterministic(backend):
    """o, dK, dV are bitwise repeatable; dQ too unless the fused (atomic) backward runs."""
    fn = _fn(backend, torch.float16)
    case = _case([513, 64, 1200])
    a = _run(fn, *case)
    b = _run(fn, *case)
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1][1], b[1][1]) and torch.equal(a[1][2], b[1][2])
    atomic = backend == "triton" and not TRITON_DETERMINISTIC
    if atomic:
        assert torch.allclose(a[1][0], b[1][0], rtol=0, atol=1e-3)
    else:
        assert torch.equal(a[1][0], b[1][0])


@pytest.mark.parametrize("backend", BACKENDS)
def test_strided_inputs(backend):
    fn = _fn(backend, torch.float16)
    q, k, v, do, cu, max_len = _case([200, 61])
    qkv = torch.stack((q, k, v), 1)
    out = fn(qkv[:, 0], qkv[:, 1], qkv[:, 2], cu, max_len, q.shape[-1] ** -0.5)
    ref = fn(q, k, v, cu, max_len, q.shape[-1] ** -0.5)
    assert torch.equal(out, ref)


@pytest.fixture
def fresh_cache(tmp_path, monkeypatch):
    """Empty on-disk tuning tables and no in-process choices or bindings."""
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    tuner._TABLES.clear()
    tuner._CHOSEN.clear()
    for module in _attention_modules():
        module._BOUND.clear()
    yield
    tuner._TABLES.clear()
    tuner._CHOSEN.clear()
    for module in _attention_modules():
        module._BOUND.clear()


def _attention_modules():
    mods = []
    for name in ("triton.attention", "triton.attention_race", "tilelang.attention"):
        try:
            mods.append(importlib.import_module(f"kohakumoss_kernels.backends.{name}"))
        except ImportError:
            continue
    return mods


@pytest.mark.parametrize("backend", BACKENDS)
def test_no_grad_fresh_cache(backend, fresh_cache):
    """A first call under no_grad / inference_mode tunes without autograd and matches."""
    fn = _fn(backend, torch.float16)
    q, k, v, do, cu, max_len = _case([333, 90])
    scale = q.shape[-1] ** -0.5
    with torch.no_grad():
        a = fn(q, k, v, cu, max_len, scale)
    with torch.inference_mode():
        b = fn(q, k, v, cu, max_len, scale)
    ref = eager(q.double(), k.double(), v.double(), cu, max_len, scale)
    assert torch.equal(a, b)
    assert ((a.double() - ref).norm() / ref.norm()) < 2e-3


@pytest.mark.parametrize("backend", BACKENDS)
def test_checkpoint_fresh_cache(backend, fresh_cache):
    """A first call inside a checkpointed block (recompute in backward) gives the same
    gradients as the plain call."""
    fn = _fn(backend, torch.float16)
    q, k, v, do, cu, max_len = _case([400, 129], seed=3)
    scale = q.shape[-1] ** -0.5

    def block(q, k, v):
        return fn(q, k, v, cu, max_len, scale) * 1.0

    leaves = [t.clone().requires_grad_() for t in (q, k, v)]
    checkpoint(block, *leaves, use_reentrant=False).backward(do)
    _, ref = _run(fn, q, k, v, do, cu, max_len)
    for a, r in zip(leaves, ref):
        assert torch.allclose(a.grad, r, atol=2e-3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_prewarm_binds(backend, fresh_cache):
    """After prewarm the live call of a covered length finds a bound launch."""
    fn = _fn(backend, torch.float16)
    tuning.prewarm_attention(fn, "cuda", torch.float16, 3, 128, 1029)
    module = importlib.import_module(f"kohakumoss_kernels.backends.{backend}.attention")
    assert any(ident[0] == "fwd" for ident in module._BOUND)
    assert any(ident[0] == "bwd" for ident in module._BOUND)


@pytest.mark.parametrize("backend", BACKENDS)
def test_compile_fullgraph(backend):
    fn = _fn(backend, torch.float16)
    q, k, v, do, cu, max_len = _case([300, 77])
    scale = q.shape[-1] ** -0.5

    def step(q, k, v):
        return fn(q, k, v, cu, max_len, scale)

    compiled = torch.compile(step, fullgraph=True)
    leaves = [t.clone().requires_grad_() for t in (q, k, v)]
    out = compiled(*leaves)
    out.backward(do)
    ref_out, ref_grads = _run(fn, q, k, v, do, cu, max_len)
    assert torch.allclose(out, ref_out, atol=1e-3)
    for a, r in zip(leaves, ref_grads):
        assert a.grad is not None and torch.allclose(a.grad, r, atol=1e-3)
