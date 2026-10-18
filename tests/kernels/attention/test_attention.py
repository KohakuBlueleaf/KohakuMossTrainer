"""The attention kernels themselves (Triton, TileLang; not the choice) vs plain PyTorch
(`pytorch_sdpa`, per-segment SDPA in the 16-bit dtype) with fp64 as the reference: error
statistics, huge logits, empty segments, determinism, strides, torch.compile."""

import pytest
import torch

from tests._util.negative import must_fail
from tests.kernels.attention._attention_case import (
    KERNELS,
    check_stats,
    dropped,
    kernel_fn,
    make_case,
    pytorch_sdpa,
    ref64,
    run,
    unguarded,
    unshifted,
)

pytestmark = pytest.mark.gpu

DTYPES = (torch.float16, torch.bfloat16)
LENS = [300, 0, 77, 1029, 5, 1, 128]
HUGE_SEEDS = range(16)


def _dtype_or_skip(device, dtype):
    if dtype == torch.bfloat16 and device.type == "cuda":
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            pytest.skip("no bf16 on this card")


@pytest.mark.budget(60)
@pytest.mark.parametrize("backend", KERNELS)
@pytest.mark.parametrize("dtype", DTYPES, ids=("fp16", "bf16"))
def test_error_stats(backend, dtype, device):
    _dtype_or_skip(device, dtype)
    fn = kernel_fn(backend, device, dtype)
    case = make_case(LENS, device, dtype=dtype)
    out, grads = run(fn, *case)
    eo, eg = pytorch_sdpa(*case)
    ro, rg = ref64(*case)
    check_stats(backend, out, grads, eo, eg, ro, rg, dtype)
    leak = make_case([sum(LENS)], device, dtype=dtype)[4]
    broken, bgrads = run(fn, *case[:4], leak, sum(LENS))
    must_fail(
        lambda: check_stats(backend, broken, bgrads, eo, eg, ro, rg, dtype),
        "attention across segments",
    )
    lost = [dropped(g, case[4]) for g in grads]
    must_fail(
        lambda: check_stats(backend, dropped(out, case[4]), lost, eo, eg, ro, rg),
        "a segment never written",
    )
    nan = [g.clone() for g in grads]
    nan[1].view(-1)[7] = float("nan")
    must_fail(
        lambda: check_stats(backend, out, nan, eo, eg, ro, rg, dtype), "one NaN in dK"
    )


HUGE_KEYS = ("cos_deficit", "abs_mean", "abs_median")


def check_huge(name, out, grads, eo, eg, ro, rg, dtype):
    """The error statistics against `pytorch_sdpa` on the cosine and absolute stats; the
    relative ones are recorded only. See
    docs/internals/kernel-dev.md#matmul-precision-and-error-statistics."""
    check_stats(name, out, grads, eo, eg, ro, rg, dtype, keys=HUGE_KEYS)


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
    eo, eg = pooled(pytorch_sdpa, cases)
    ro, rg = pooled(ref64, cases)
    check_huge(backend, out, grads, eo, eg, ro, rg, dtype)
    bo, bg = pooled(lambda *c: run(unshifted, *c), cases)
    must_fail(
        lambda: check_huge(backend, bo, bg, eo, eg, ro, rg, dtype),
        "exp without max shift",
    )


def check_empty(out, grads, eo, eg, ro, rg):
    for t in (out, *grads):
        assert torch.isfinite(t).all()
    check_stats("empty", out, grads, eo, eg, ro, rg)


@pytest.mark.budget(60)
@pytest.mark.parametrize("backend", KERNELS)
def test_empty_and_single_segments(backend, device):
    fn = kernel_fn(backend, device, torch.float16)
    case = make_case([0, 1, 0, 2, 0, 3], device, heads=2)
    out, grads = run(fn, *case)
    eo, eg = pytorch_sdpa(*case)
    ro, rg = ref64(*case)
    check_empty(out, grads, eo, eg, ro, rg)
    bo, bg = run(unguarded, *case)
    must_fail(
        lambda: check_empty(bo, bg, eo, eg, ro, rg), "no guard on all-masked rows"
    )


@pytest.mark.parametrize("backend", KERNELS)
def test_deterministic(backend, device):
    """o, dK, dV are bitwise repeatable; dQ too unless the fused (atomic) backward runs."""
    fn = kernel_fn(backend, device, torch.float16)
    case = make_case([513, 64, 1200], device)
    a = run(fn, *case)
    b = run(fn, *case)
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1][1], b[1][1]) and torch.equal(a[1][2], b[1][2])
    if backend == "triton":
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

    ref_out, ref_grads = run(fn, q, k, v, do, cu, max_len)
    compiled = torch.compile(step, fullgraph=True)
    leaves = [t.clone().requires_grad_() for t in (q, k, v)]
    out = compiled(*leaves)
    out.backward(do)
    ro, rg = ref64(q, k, v, do, cu, max_len)
    grads = [t.grad for t in leaves]
    check_stats("compiled", out.detach(), grads, ref_out, ref_grads, ro, rg)
