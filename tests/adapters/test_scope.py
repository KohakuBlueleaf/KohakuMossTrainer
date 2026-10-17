"""Tier prewarm on scratch factors, per-layer dense materialisation, and the kind
registry that Targets resolve from."""

from contextlib import nullcontext

import pytest
import torch
from _adcases import CASES, build, make_spec
from _adtoy import toy
from lycoris.kernels import tier
from test_stored_base import Int8Storage, quantized_toy

from kohakumoss.adapters import adapter_set, apply, network
from kohakumoss.adapters.bind.select import bind_target
from kohakumoss.adapters.kinds import REGISTRY, record_of
from kohakumoss.adapters.scope import slots_of
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
ROWS = [1, 2, 4]


def _raced(aset):
    """Re-bind every layer to raced slots (CPU operands resolve to the torch tier)."""
    for t in aset.targets:
        t.linear.forward = bind_target(t, "bypass", "triton", 1.0, False, 0.0, True)


class _Streamed(Int8Storage):
    """A base that is not readable outside its stream window."""

    def weight(self, dtype):
        raise RuntimeError("streamed base read outside its window")


def _streamed_toy():
    model = quantized_toy()
    for m in model.modules():
        if isinstance(m, Int8Storage):
            m.__class__ = _Streamed
    return model


def _prewarm_seen(monkeypatch, model_factory):
    torch.manual_seed(0)
    aset = apply(model_factory(), make_spec("lora"))
    _raced(aset)
    seen = []
    real = tier.best

    def spy(op, tensors, *rest):
        seen.append(float(tensors[2].detach().abs().max()))
        return real(op, tensors, *rest)

    monkeypatch.setattr(tier, "best", spy)
    calls = aset.prewarm(ROWS, torch.float32, [torch.device("cpu")])
    return aset, seen, calls


def test_prewarm_races_every_bucket_on_non_zero_factors(monkeypatch):
    aset, seen, calls = _prewarm_seen(monkeypatch, toy)
    slots = [s for t in aset.targets for s in slots_of(vars(t.linear)["forward"])]
    assert calls == len(slots) * len(ROWS) * 3
    assert seen and min(seen) > 0
    for s in slots:
        assert len(s.arms) == len(ROWS) * 3
    for t in aset.targets:
        assert not t.lyco.lora_up.weight.any()
        assert all(p.grad is None for p in t.lyco.parameters())


def test_prewarm_reaches_layers_whose_base_is_streamed(monkeypatch):
    _, _, calls = _prewarm_seen(monkeypatch, _streamed_toy)
    assert calls > 0


def test_prewarm_on_the_live_zero_factors_is_caught(monkeypatch):
    monkeypatch.setattr(adapter_set, "perturbed", lambda lyco: nullcontext())

    def zero():
        _, seen, _ = _prewarm_seen(monkeypatch, toy)
        assert min(seen) > 0

    must_fail(zero, "a race on zero-initialised factors")


class _Counting(Int8Storage):
    reads: list = []

    def materialize(self):
        _Counting.reads.append(id(self))
        return super().materialize()


def _materialised_once():
    model = quantized_toy()
    for m in model.modules():
        if isinstance(m, Int8Storage):
            m.__class__ = _Counting
    _Counting.reads = []
    aset = apply(model, make_spec("dora-out", preset="attn"))
    targeted = {id(t.linear.base) for t in aset.targets}
    assert set(_Counting.reads) == targeted
    assert len(_Counting.reads) == len(targeted)


class _EveryLayer:
    """The eager variant: every stored linear dense at once."""

    def __init__(self, linears):
        self.ctx = [network._DenseWeight(m) for m in linears]

    def __enter__(self):
        for c in self.ctx:
            c.__enter__()

    def __exit__(self, *exc):
        for c in self.ctx:
            c.__exit__(*exc)


def test_dense_init_materialises_only_targeted_layers_once(monkeypatch):
    _materialised_once()
    monkeypatch.setattr(network, "_LazyDense", _EveryLayer)
    must_fail(_materialised_once, "every stored linear materialised at once")


@pytest.mark.parametrize("name", sorted(CASES))
def test_targets_resolve_everything_from_their_record(name):
    _, aset = build(name)
    for t in aset.targets:
        assert t.record is record_of(t.lyco) is REGISTRY[t.kind]
        assert callable(t.unit_fn) and callable(t.base_fn)
