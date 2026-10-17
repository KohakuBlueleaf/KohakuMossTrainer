"""fp16 fixes: every mode is the same function; overflow is fixed; adapter gradients match."""

import copy

import pytest
import torch
from lycoris.modules.locon import LoConModule

from kohakumoss.model.build import DEFAULT_KEEP_FP32, build_backbone, resolve_config
from kohakumoss.model.numerics import (
    SaturationCounters,
    apply_dtype_policy,
    apply_fp16_fix,
    choose_factors,
    clamp_counts,
    fold,
)
from tests._util.negative import must_fail
from tests.model.conftest import make_batch

pytestmark = pytest.mark.cpu
MODES = [
    ("weights", dict(c=2.0**-5)),
    ("weights", dict(c=2.0**-5, s_attn=2.0**-2, s_ffn=2.0**-3)),
    ("online", dict(c=2.0**-5)),
    ("online", dict(c=[2.0**-k for k in (3, 6, 4, 7)], s_attn=[0.5, 1, 0.25, 1])),
    ("fp32_residual", {}),
]


def _model(dtype=torch.float64, seed=3, keep_fp32=()):
    cfg = resolve_config("tiny")
    return build_backbone(cfg, dtype=dtype, keep_fp32=keep_fp32, init=True, seed=seed)


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def _inputs(cfg, dtype):
    img, text, sigma, seq, _, _ = make_batch(cfg)
    return img.to(dtype), text.to(dtype), sigma, seq


@pytest.mark.parametrize("mode, kw", MODES)
def test_modes_are_the_same_function(mode, kw):
    dtype = torch.float32 if mode == "fp32_residual" else torch.float64
    base = _model(dtype)
    x = _inputs(base.cfg, dtype)
    ref = base(*x)
    fixed = copy.deepcopy(base)
    apply_fp16_fix(fixed, mode, **kw)
    assert _rel(fixed(*x), ref) < (1e-6 if dtype == torch.float32 else 1e-12)


def test_online_needs_eps_rescale():
    base = _model()
    x = _inputs(base.cfg, torch.float64)
    ref = base(*x)
    fixed = copy.deepcopy(base)
    apply_fp16_fix(fixed, "online", c=2.0**-12)
    for name, eps in fixed._online_eps.items():
        fixed.get_submodule(name).eps = eps
    must_fail(lambda: _check(_rel(fixed(*x), ref) < 1e-8), "eps not rescaled")


@pytest.mark.parametrize("stream, bound", [(2.0**10, 1e-5), (1.0, 1e-1)])
def test_stock_eps_error_shrinks_with_stream_rms(stream, bound):
    base = _model()
    fold(base, stream, "gate", eps="stock")
    base.fold_scales = None
    x = _inputs(base.cfg, torch.float64)
    ref = base(*x)
    stock = copy.deepcopy(base)
    apply_fp16_fix(stock, "weights", c=2.0**-4, eps="stock")
    assert 1e-14 < _rel(stock(*x), ref) < bound


def _big_stream(dtype):
    """Gate biases 2^20 (added in fp32 after the 16-bit-input modulation GEMM): every
    weight and GEMM input fits fp16, the accumulated stream does not."""
    model = _model(torch.float32, keep_fp32=DEFAULT_KEEP_FP32)
    d = model.cfg.dim
    with torch.no_grad():
        bias = model.shared_block_modulation.bias
        bias[2 * d : 3 * d] = 2.0**20
        bias[5 * d : 6 * d] = 2.0**20
    apply_dtype_policy(model, dtype, DEFAULT_KEEP_FP32)
    model.bind_ops()
    return model


def test_fp16_overflow_and_fixes():
    ref_model = _big_stream(torch.float32)
    x32 = _inputs(ref_model.cfg, torch.float32)
    probe = SaturationCounters(ref_model)
    ref = ref_model(*x32)
    peak = max(v["maxabs"] for v in probe.read().values())
    c = choose_factors({"summary_max": {"stream": peak}})["c"]
    assert peak > 65504 and c < 1
    x16 = _inputs(ref_model.cfg, torch.float16)
    plain = _big_stream(torch.float16)
    assert not torch.isfinite(plain(*x16)).all()
    for mode, kw in (
        ("weights", dict(c=c)),
        ("online", dict(c=c)),
        ("fp32_residual", {}),
    ):
        model = _big_stream(torch.float16)
        model.bind_ops()
        apply_fp16_fix(model, mode, **kw)
        out = model(*x16)
        assert torch.isfinite(out).all(), mode
        assert _rel(out, ref) < 2e-2, (mode, _rel(out, ref))
    clamped = _big_stream(torch.float16)
    apply_fp16_fix(clamped, "none", clamp=True)
    out = clamped(*x16)
    assert torch.isfinite(out).all()
    assert sum(clamp_counts(clamped).values()) > 0


def _check(cond):
    assert cond


def _attach_lora(model, seed=0):
    torch.manual_seed(seed)
    loras = []
    for i, block in enumerate(model.layers):
        for name, lin in block.proj.items():
            lyco = LoConModule(f"l{i}_{name}", lin, 1.0, 4, 4.0, bypass_mode=True)
            torch.nn.init.normal_(lyco.lora_up.weight, std=0.05)
            lyco.to(lin.weight.dtype)
            lyco.apply_to()
            loras.append(lyco)
    model.bind_ops()
    return loras


def _lora_grads(mode, kw):
    model = _model()
    if mode != "none":
        apply_fp16_fix(model, mode, **kw)
    model.requires_grad_(False)
    loras = _attach_lora(model)
    x = _inputs(model.cfg, torch.float64)
    model(*x).square().sum().backward()
    return [p.grad.clone() for lyco in loras for p in lyco.parameters()]


def test_lora_gradients_identical_weights_vs_online():
    ref = _lora_grads("none", {})
    for mode, kw in (("weights", dict(c=2.0**-5)), ("online", dict(c=2.0**-5))):
        grads = _lora_grads(mode, kw)
        assert all(_rel(g, r) < 1e-10 for g, r in zip(grads, ref)), mode
    shifted = _lora_grads("weights", dict(c=2.0**-5, s_attn=0.5))
    must_fail(
        lambda: _check(all(_rel(g, r) < 1e-10 for g, r in zip(shifted, ref))),
        "pre-gate weight scale changes o-adapter gradients",
    )
