"""The attention kernels themselves (Triton, TileLang; not the race) vs PyTorch at the
same matmul precision (`mm16`): explosion bar, huge logits, empty segments,
determinism, strides, torch.compile."""

import os

import pytest
import torch

from tests._util.negative import must_fail
from tests.kernels._attention_case import (
    KERNELS,
    check_bar,
    cosine,
    kernel_fn,
    less_precise,
    make_case,
    mm16,
    ref64,
    run,
    unguarded,
    unshifted,
)

pytestmark = pytest.mark.gpu

TRITON_DETERMINISTIC = os.environ.get("KOHAKUMOSS_DETERMINISTIC", "0") == "1"
DTYPES = (torch.float16, torch.bfloat16)
LENS = [300, 0, 77, 1029, 5, 1, 128]
HUGE_SEEDS = range(16)


def _dtype_or_skip(device, dtype):
    if dtype == torch.bfloat16 and device.type == "cuda":
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            pytest.skip("no bf16 on this card")


@pytest.mark.parametrize("backend", KERNELS)
@pytest.mark.parametrize("dtype", DTYPES, ids=("fp16", "bf16"))
def test_precision_bar(backend, dtype, device):
    _dtype_or_skip(device, dtype)
    fn = kernel_fn(backend, device, dtype)
    case = make_case(LENS, device, dtype=dtype)
    out, grads = run(fn, *case)
    eo, eg = mm16(*case)
    ro, rg = ref64(*case)
    check_bar(backend, out, grads, eo, eg, ro, rg, dtype)
    leak = make_case([sum(LENS)], device, dtype=dtype)[4]
    broken, bgrads = run(fn, *case[:4], leak, sum(LENS))
    must_fail(
        lambda: check_bar(backend, broken, bgrads, eo, eg, ro, rg, dtype),
        "attention across segments",
    )
    worse = [less_precise(e, r, dtype) for e, r in zip(eg, rg)]
    must_fail(
        lambda: check_bar(backend, out, worse, eo, eg, ro, rg, dtype),
        "grads 3x less precise than PyTorch",
    )
    nan = [g.clone() for g in grads]
    nan[1].view(-1)[7] = float("nan")
    must_fail(
        lambda: check_bar(backend, out, nan, eo, eg, ro, rg, dtype), "one NaN in dK"
    )


def check_huge(name, out, grads, eo, eg, ro, rg, dtype):
    """The bar against `mm16` (cosines to fp64 in the message)."""
    cos = {
        label: (round(cosine(a, r), 6), round(cosine(e, r), 6))
        for label, a, e, r in [("o", out, eo, ro), *zip("qkv", grads, eg, rg)]
    }
    try:
        check_bar(name, out, grads, eo, eg, ro, rg, dtype)
    except AssertionError as err:
        raise AssertionError(f"{err}; cosine (ours, torch): {cos}") from err


def pooled(fn, cases):
    """(o, [dq, dk, dv]) of `fn` over every case, each flattened and concatenated."""
    runs = [fn(*case) for case in cases]
    out = torch.cat([o.flatten() for o, _ in runs])
    grads = [torch.cat([g[i].flatten() for _, g in runs]) for i in range(3)]
    return out, grads


@pytest.mark.parametrize("backend", KERNELS)
@pytest.mark.parametrize("dtype", DTYPES, ids=("fp16", "bf16"))
def test_huge_logits(backend, dtype, device):
    """Clustered q / k (logits ~1e5); errors pooled over `HUGE_SEEDS` draws (a few
    elements dominate each draw)."""
    _dtype_or_skip(device, dtype)
    fn = kernel_fn(backend, device, dtype)
    cases = [
        make_case([700, 333], device, dtype=dtype, qk_scale=80.0, seed=s)
        for s in HUGE_SEEDS
    ]
    out, grads = pooled(lambda *c: run(fn, *c), cases)
    eo, eg = pooled(mm16, cases)
    ro, rg = pooled(ref64, cases)
    check_huge(backend, out, grads, eo, eg, ro, rg, dtype)
    bo, bg = pooled(lambda *c: run(unshifted, *c), cases)
    must_fail(
        lambda: check_huge(backend, bo, bg, eo, eg, ro, rg, dtype),
        "exp without max shift",
    )


def check_empty(out, grads, ro, rg):
    assert torch.allclose(out.double(), ro, atol=2e-3)
    for a, r in zip(grads, rg):
        assert torch.isfinite(a).all()
        assert torch.allclose(a.double(), r, atol=5e-3)


@pytest.mark.parametrize("backend", KERNELS)
def test_empty_and_single_segments(backend, device):
    fn = kernel_fn(backend, device, torch.float16)
    case = make_case([0, 1, 0, 2, 0], device, heads=2)
    out, grads = run(fn, *case)
    ro, rg = ref64(*case)
    check_empty(out, grads, ro, rg)
    bo, bg = run(unguarded, *case)
    must_fail(lambda: check_empty(bo, bg, ro, rg), "no guard on all-masked rows")


@pytest.mark.parametrize("backend", KERNELS)
def test_deterministic(backend, device):
    """o, dK, dV are bitwise repeatable; dQ too unless the fused (atomic) backward runs."""
    fn = kernel_fn(backend, device, torch.float16)
    case = make_case([513, 64, 1200], device)
    a = run(fn, *case)
    b = run(fn, *case)
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1][1], b[1][1]) and torch.equal(a[1][2], b[1][2])
    if backend == "triton" and not TRITON_DETERMINISTIC:
        assert torch.allclose(a[1][0], b[1][0], rtol=0, atol=1e-3)
    else:
        assert torch.equal(a[1][0], b[1][0])


@pytest.mark.parametrize("backend", KERNELS)
def test_strided_inputs(backend, device):
    fn = kernel_fn(backend, device, torch.float16)
    q, k, v, do, cu, max_len = make_case([200, 61], device)
    qkv = torch.stack((q, k, v), 1)
    out = fn(qkv[:, 0], qkv[:, 1], qkv[:, 2], cu, max_len, q.shape[-1] ** -0.5)
    ref = fn(q, k, v, cu, max_len, q.shape[-1] ** -0.5)
    assert torch.equal(out, ref)


@pytest.mark.parametrize("backend", KERNELS)
def test_compile_fullgraph(backend, device):
    fn = kernel_fn(backend, device, torch.float16)
    q, k, v, do, cu, max_len = make_case([300, 77], device)
    scale = q.shape[-1] ** -0.5

    def step(q, k, v):
        return fn(q, k, v, cu, max_len, scale)

    compiled = torch.compile(step, fullgraph=True)
    leaves = [t.clone().requires_grad_() for t in (q, k, v)]
    out = compiled(*leaves)
    out.backward(do)
    ref_out, ref_grads = run(fn, q, k, v, do, cu, max_len)
    assert torch.allclose(out, ref_out, atol=1e-3)
    for a, r in zip(leaves, ref_grads):
        assert a.grad is not None and torch.allclose(a.grad, r, atol=1e-3)
