"""The kernel prewarm's weightless twin, model-free (the MOSS-Image base architecture
built on the meta device; no weight is created or read): one element of memory per
tensor, int8 bases on every quantized linear of an int8-base run, the built backbone's
dtypes and grad flags by name, an adapted linear's output requiring grad, the trainer's
prewarm recording from the twin; the recorded pass keeps no saved tensor."""

import contextlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from kohakumoss.model import build_backbone, resolve_config
from kohakumoss.model.linear import MossLinear
from kohakumoss.model.opset import OpSet
from kohakumoss.precision import build_precision
from kohakumoss.quant.convert import default_filter
from kohakumoss.quant.weights import QuantWeight
from kohakumoss.training.build import kernels
from kohakumoss.training.build.shapes import run_shapes
from kohakumoss.tuning.prewarm import keep_nothing
from kohakumoss.types.plans import MemoryPlan
from tests._util.negative import must_fail
from tests.training.knobs.test_knobs import runnable

pytestmark = pytest.mark.cpu
CPU = torch.device("cpu")
INT8 = MemoryPlan(base_precision="int8")


def _setup():
    k = runnable(TUNE="table")
    prec = build_precision("bf16", tuple(k["KEEP_FP32"]))
    cfg = resolve_config(k["VARIANT"])
    real = build_backbone(cfg, "meta", prec.compute, prec.keep_fp32)
    real.bind_ops(OpSet.resolve(CPU, prec.compute, {}))
    return k, prec, real


def _trained_norm(real: nn.Module) -> str:
    """The first norm weight of `real`, made an fp32 trainable parameter (TRAIN_NORMS)."""
    name, p = next((n, p) for n, p in real.named_parameters() if "norm" in n)
    p.data = p.data.to(torch.float32)
    p.requires_grad_(True)
    return name


def check_weightless(module: nn.Module, limit: int = 1 << 20) -> None:
    """Every parameter and buffer together hold under `limit` bytes of storage while
    the architecture has over a billion elements."""
    tensors = [*module.parameters(), *module.buffers()]
    held = sum(t.untyped_storage().nbytes() for t in tensors)
    assert held < limit, f"{held} bytes of storage"
    assert sum(t.numel() for t in tensors) > 10**9


def check_int8_bases(twin: nn.Module, keep_fp32) -> None:
    """Every linear the memory plan quantizes holds an int8 [out, in] base with fp32
    per-row scales."""
    accept = default_filter(keep_fp32, 256)
    count = 0
    for name, lin in twin.named_modules():
        if not isinstance(lin, MossLinear) or not accept(name, lin):
            continue
        base = lin.base
        assert isinstance(base, QuantWeight), name
        assert base.q.dtype == torch.int8, name
        assert tuple(base.q.shape) == (lin.out_features, lin.in_features), name
        assert base.scale.dtype == torch.float32, name
        assert tuple(base.scale.shape) == (lin.out_features,), name
        count += 1
    assert count > 0


def check_grad_out(lin: nn.Module) -> None:
    """An input without grad gives an output that requires grad."""
    x = torch.randn(2, lin.in_features).to(torch.bfloat16)
    assert lin(x).requires_grad


def test_twin_of_an_int8_run():
    k, prec, real = _setup()
    norm = _trained_norm(real)
    real.layers[0].proj["q"].add_module("lyco", nn.Identity())
    twin = run_shapes(k, prec, INT8, CPU, real)
    check_weightless(twin)
    check_int8_bases(twin, prec.keep_fp32)
    p = dict(twin.named_parameters())[norm]
    assert p.dtype == torch.float32 and p.requires_grad, (p.dtype, p.requires_grad)
    check_grad_out(twin.layers[0].proj["q"])
    must_fail(lambda: check_grad_out(twin.layers[1].proj["q"]), "a linear without one")
    must_fail(lambda: check_weightless(nn.Linear(1024, 1024)), "a weight-sized module")
    plain = run_shapes(k, prec, MemoryPlan(), CPU)
    must_fail(lambda: check_int8_bases(plain, prec.keep_fp32), "a 16-bit base")


def test_trainer_prewarm_records_from_the_twin(monkeypatch):
    k, prec, real = _setup()
    built = SimpleNamespace(backbone=real, plan=INT8)
    warm = kernels.build_prewarm(k, built, prec, CPU)
    assert warm.backbone is not real
    check_weightless(warm.backbone)
    monkeypatch.setattr(kernels, "run_shapes", lambda k, prec, plan, dev, real: real)
    broken = kernels.build_prewarm(k, built, prec, CPU)
    must_fail(lambda: check_weightless(broken.backbone), "the built backbone recorded")


def check_keeps_nothing(hooks) -> None:
    """The autograd graph built under `hooks` holds no saved tensor."""
    x = torch.randn(1 << 16, requires_grad=True)
    with hooks:
        y = x.sin()
    try:
        y.grad_fn._saved_self
    except RuntimeError:
        return
    raise AssertionError("the graph keeps a saved tensor")


def test_recording_pass_keeps_no_saved_tensor():
    check_keeps_nothing(keep_nothing())
    must_fail(lambda: check_keeps_nothing(contextlib.nullcontext()), "plain autograd")
