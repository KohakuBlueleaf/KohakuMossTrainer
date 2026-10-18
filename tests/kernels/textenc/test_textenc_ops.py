"""Text-encoder ops: the eager reference against independent fp64 references, and every
tuned Triton / TileLang candidate through the error statistics (abs and rel error vs
fp64 next to the eager reference in the same dtype), with a boundary-crossing variant
that must fail."""

import importlib

import pytest
import torch
import torch.nn.functional as F

import kohakumoss_kernels as kk
from kohakumoss_kernels import ops as kernel_ops
from kohakumoss_kernels.backends.eager import textenc as E
from tests._util.error_stats import assert_no_regression
from tests._util.negative import must_fail
from tests._util.textenc_ref import (
    attention_causal_dense,
    conv_dense,
    delta_rule_recurrent,
)

LENS = [37, 200, 0, 64, 130, 1, 65, 300]
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def _cu(lens, device):
    return torch.tensor(
        [0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device=device
    )


def _merged(lens, device):
    return _cu([sum(lens)], device)


def delta_case(lens, device, dtype, hk=4, hv=8, d=64, seed=0, decay=16.0):
    """Delta-rule inputs; `decay` bounds A = exp(a_log) (small: state carries far)."""
    g = torch.Generator(device=device).manual_seed(seed)
    n = sum(lens)

    def rnd(*shape, s=1.0):
        return torch.randn(*shape, device=device, generator=g) * s

    q = F.silu(rnd(n, hk, d)).to(dtype)
    k = F.silu(rnd(n, hk, d)).to(dtype)
    v = F.silu(rnd(n, hv, d)).to(dtype)
    a = rnd(n, hv, s=2.0).to(dtype)
    b = rnd(n, hv, s=2.0).to(dtype)
    a_log = torch.log(torch.rand(hv, device=device, generator=g) * decay + 0.01)
    dt_bias = rnd(hv)
    return q, k, v, a, b, a_log, dt_bias


def _f64(ts):
    return [t.double() for t in ts]


@pytest.mark.cpu
def test_eager_delta_matches_recurrence():
    case = delta_case(LENS, "cpu", torch.float64)
    cu = _cu(LENS, "cpu")
    ref = delta_rule_recurrent(*case, cu)

    def check(out):
        assert (out - ref).abs().max() <= 1e-10 * ref.abs().max()

    check(E.gated_delta_rule_varlen(*case, cu, max(LENS)))
    merged = E.gated_delta_rule_varlen(*case, _merged(LENS, "cpu"), sum(LENS))
    must_fail(lambda: check(merged), "delta rule state carried across segments")


@pytest.mark.cpu
def test_eager_conv_matches_per_segment_conv1d():
    g = torch.Generator().manual_seed(0)
    n = sum(LENS)
    x = torch.randn(n, 24, generator=g, dtype=torch.float64)
    w = torch.randn(24, 4, generator=g, dtype=torch.float64)
    cu = _cu(LENS, "cpu")
    ref = conv_dense(x, w, cu)

    def check(out):
        assert (out - ref).abs().max() <= 1e-12

    check(E.causal_conv_varlen(x, w, cu))
    must_fail(
        lambda: check(E.causal_conv_varlen(x, w, _merged(LENS, "cpu"))),
        "conv taps across segments",
    )


@pytest.mark.cpu
def test_eager_attention_matches_dense_mask():
    g = torch.Generator().manual_seed(0)
    n = sum(LENS)
    q = torch.randn(n, 8, 32, generator=g, dtype=torch.float64)
    k = torch.randn(n, 2, 32, generator=g, dtype=torch.float64)
    v = torch.randn(n, 2, 32, generator=g, dtype=torch.float64)
    cu = _cu(LENS, "cpu")
    ref = attention_causal_dense(q, k, v, cu, 0.2)

    def check(out):
        assert (out - ref).abs().max() <= 1e-12

    check(E.attention_varlen_causal(q, k, v, cu, max(LENS), 0.2))
    must_fail(
        lambda: check(E.attention_varlen_causal(q, k, v, _merged(LENS, "cpu"), n, 0.2)),
        "attention across segments",
    )


@pytest.mark.cpu
def test_eager_norm_gated_formula():
    g = torch.Generator().manual_seed(0)
    x, z = (torch.randn(50, 128, generator=g, dtype=torch.float64) for _ in range(2))
    w = torch.rand(128, generator=g, dtype=torch.float64)
    ref = x / x.square().mean(-1, keepdim=True).add(1e-6).sqrt() * w * z * z.sigmoid()

    def check(out):
        assert (out - ref).abs().max() <= 1e-12

    check(E.rms_norm_gated(x, z, w, 1e-6))
    must_fail(lambda: check(E.rms_norm_gated(x, x, w, 1e-6)), "gate input ignored")


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    return torch.device("cuda")


def _module(name):
    try:
        return importlib.import_module(f"kohakumoss_kernels.backends.{name}")
    except ImportError as err:
        pytest.skip(str(err))


def _delta_backend(backend):
    """(run, candidates(dev, dtype, dv)): every config the backend's tuner can bind."""
    match backend:
        case "triton":
            mod = _module("triton.textenc.delta")
            return (
                mod.run,
                lambda dev, dt, dv: mod.candidates(dev, dt, dv)[: mod.SHORTLIST],
            )
        case _:
            mod = _module("tilelang.textenc.ops")
            return mod.delta_run, lambda dev, dt, dv: mod.delta_candidates(dv)


def _supported(op, dev, dtype, backend):
    try:
        kk.resolve(op, dev, dtype, backend=backend)
    except kk.registry.NoImplementationError as err:
        pytest.skip(str(err))


@pytest.mark.cuda
def one_per_design(configs) -> list:
    """The first config of each design (`impl`, and its long-row path): one fixed tile
    per design keeps every code path's precision checked without a tile sweep."""
    seen, out = set(), []
    for cfg in configs:
        design = (cfg.get("impl"), cfg.get("long"))
        if design not in seen:
            seen.add(design)
            out.append(cfg)
    return out


@pytest.mark.budget(60)
@pytest.mark.parametrize("backend", ("triton", "tilelang"))
@pytest.mark.parametrize("dtype", tuple(DTYPES), ids=tuple(DTYPES))
@pytest.mark.parametrize("d", (64, 128))
def test_delta_candidates_precision(backend, dtype, d):
    """Every candidate the tuner can bind has no error regression against the eager
    reference in the same dtype (tests/_util/error_stats.py); a run with the segments
    merged fails."""
    dev, dt = _cuda(), DTYPES[dtype]
    _supported("gated_delta_rule_varlen", dev, dt, backend)
    run, candidates = _delta_backend(backend)
    lens = [37, 200, 0, 64, 130, 1, 700]
    case = delta_case(lens, dev, dt, hk=16, hv=32, d=d)
    cu = _cu(lens, dev)
    eager = E.gated_delta_rule_varlen(*case, cu, max(lens))
    ref = E.gated_delta_rule_varlen(*_f64(case), cu, max(lens))
    good = []
    for cfg in one_per_design(candidates(dev, dt, d)):
        try:
            out = run(cfg, *case, cu, max(lens))
            torch.cuda.synchronize()
        except Exception as err:  # noqa: BLE001
            if "out of resource" in str(err) or "shared memory" in str(err):
                continue
            raise
        assert_no_regression(out, eager, ref, str(cfg))
        good.append(cfg)
    assert good
    slow = delta_case(lens, dev, dt, hk=16, hv=32, d=d, decay=0.05)
    slow_eager = E.gated_delta_rule_varlen(*slow, cu, max(lens))
    slow_ref = E.gated_delta_rule_varlen(*_f64(slow), cu, max(lens))
    assert_no_regression(
        run(good[0], *slow, cu, max(lens)), slow_eager, slow_ref, "slow decay"
    )
    broken = run(good[0], *slow, _merged(lens, dev), sum(lens))
    must_fail(
        lambda: assert_no_regression(broken, slow_eager, slow_ref, "merged"),
        "delta rule state carried across segments",
    )


@pytest.mark.cuda
@pytest.mark.parametrize("impl", ("chunk", "short", "fused"))
def test_triton_delta_runs_under_inference_mode(impl):
    """The Triton designs that cache per-`cu_seqlens` plans run on inference tensors
    (a pipeline encode) and give the no-grad result."""
    dev, dt = _cuda(), torch.bfloat16
    _supported("gated_delta_rule_varlen", dev, dt, "triton")
    mod = _module("triton.textenc.delta")
    cfg = next(c for c in mod.candidates(dev, dt, 64) if c["impl"] == impl)
    lens = [37, 200, 1, 65]
    case = delta_case(lens, dev, dt, hk=2, hv=4, d=64)
    with torch.no_grad():
        want = mod.run(cfg, *case, _cu(lens, dev), max(lens))
    with torch.inference_mode():
        got = mod.run(cfg, *case, _cu(lens, dev), max(lens))
    assert torch.equal(got, want)


@pytest.mark.cuda
@pytest.mark.parametrize("backend", ("triton", "tilelang"))
def test_delta_rejects_value_tiles_wider_than_dv(backend):
    """A value tile that does not divide dv raises instead of leaving output columns
    unwritten; the tuner never offers one."""
    dev, dt = _cuda(), torch.bfloat16
    _supported("gated_delta_rule_varlen", dev, dt, backend)
    run, candidates = _delta_backend(backend)
    lens = [37, 70]
    case = delta_case(lens, dev, dt, hk=2, hv=4, d=32)
    wide = {"impl": "chunk", "wp": 2, "bs": 64, "ws": 4, "st": 1, "bo": 64, "wo": 4}
    if backend == "tilelang":
        wide = {"bv": 64, "threads": 64}
    with pytest.raises(ValueError):
        run(wide, *case, _cu(lens, dev), max(lens))
    assert all(wide != c for c in candidates(dev, dt, 32))


def _attn_backend(backend):
    match backend:
        case "triton":
            mod = _module("triton.textenc.attention")
            return mod.run, lambda dev, dt, d: mod.candidates(dev, dt, d)
        case _:
            mod = _module("tilelang.textenc.ops")
            return mod.attn_run, lambda dev, dt, d: mod.attn_candidates(d)


@pytest.mark.cuda
@pytest.mark.parametrize("backend", ("triton", "tilelang"))
@pytest.mark.parametrize("dtype", ("fp16", "bf16", "fp32"))
def test_attention_candidates_precision(backend, dtype):
    """Every tuned causal GQA candidate has no error regression against per-segment
    SDPA in the same dtype; merged segments must fail."""
    dev, dt = _cuda(), DTYPES[dtype]
    _supported("attention_varlen_causal", dev, dt, backend)
    run, candidates = _attn_backend(backend)
    lens = [37, 200, 0, 64, 130, 1, 300]
    g = torch.Generator(device=dev).manual_seed(0)
    n = sum(lens)
    q = (torch.randn(n, 16, 256, device=dev, generator=g) * 2).to(dt)
    k = (torch.randn(n, 4, 256, device=dev, generator=g) * 2).to(dt)
    v = torch.randn(n, 4, 256, device=dev, generator=g).to(dt)
    cu = _cu(lens, dev)
    scale = 256**-0.5
    eager = E.attention_varlen_causal(q, k, v, cu, max(lens), scale)
    ref = attention_causal_dense(q, k, v, cu, scale)
    good = []
    for cfg in candidates(dev, dt, 256):
        try:
            out = run(cfg, q, k, v, cu, max(lens), scale)
            torch.cuda.synchronize()
        except Exception as err:  # noqa: BLE001
            if "resource" in str(err) or "shared memory" in str(err):
                continue
            raise
        assert_no_regression(out, eager, ref, str(cfg))
        good.append(cfg)
    assert good
    broken = run(good[0], q, k, v, _merged(lens, dev), n, scale)
    must_fail(
        lambda: assert_no_regression(broken, eager, ref, "merged"),
        "attention across segments",
    )


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", ("fp16", "bf16"))
def test_compiled_conv_and_norm(dtype):
    """The resolved conv / gated-norm (torch.compile of the eager reference) against
    fp64."""
    dev, dt = _cuda(), DTYPES[dtype]
    if dt == torch.bfloat16 and not kk.arch_of(dev).has_bf16:
        pytest.skip("no bf16")
    g = torch.Generator(device=dev).manual_seed(0)
    n = sum(LENS)
    x = torch.randn(n, 512, device=dev, generator=g).to(dt)
    w = torch.randn(512, 4, device=dev, generator=g).to(dt)
    cu = _cu(LENS, dev)
    conv = kk.resolve("causal_conv_varlen", dev, dt)
    ref = E.causal_conv_varlen(x.double(), w.double(), cu)
    eager = E.causal_conv_varlen(x, w, cu)
    assert_no_regression(conv(x, w, cu), eager, ref, "conv")
    # merged segments only change each later segment's first width-1 rows: judge those
    heads = torch.cat(
        [torch.arange(s, s + w.shape[1] - 1) for s in cu[1:-1].tolist()]
    ).to(dev)
    merged = conv(x, w, _merged(LENS, dev))
    must_fail(
        lambda: assert_no_regression(merged[heads], eager[heads], ref[heads], "merged"),
        "conv across segments",
    )
    z = torch.randn(n, 128, device=dev, generator=g).to(dt)
    y = torch.randn(n, 128, device=dev, generator=g).to(dt)
    nw = torch.rand(128, device=dev, generator=g).to(dt)
    norm = kk.resolve("rms_norm_gated", dev, dt)
    ref = E.rms_norm_gated(y.double(), z.double(), nw.double(), 1e-6)
    assert_no_regression(
        norm(y, z, nw, 1e-6), E.rms_norm_gated(y, z, nw, 1e-6), ref, "norm"
    )


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", ("fp16", "bf16"))
def test_registry_serves_every_text_op(dtype):
    dev, dt = _cuda(), DTYPES[dtype]
    if dt == torch.bfloat16 and not kk.arch_of(dev).has_bf16:
        pytest.skip("no bf16")
    for op in kernel_ops.TEXT:
        assert callable(kk.resolve(op, dev, dt))
