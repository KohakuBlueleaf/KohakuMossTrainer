"""OUTLIER_DECAY: each element's excess over cap x the start RMS shrinks by lr x rate x
excess after the optimiser step (elements under the cap untouched), in the plain bundle,
gated by the device flag, and per parameter in a fused step; the cap / rate schedules,
the targets, the start RMS, the coverage check and the knob checks (model-free:
explicit parameters and closed-form losses)."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from kohakumoss.precision import build_precision
from kohakumoss.training.optim import build_bundle, build_param_groups
from kohakumoss.training.optim.outlier import (
    DEFAULTS,
    OutlierDecay,
    attach_outlier_decay,
    cap_at,
    outlier_problems,
    rate_at,
    start_refs,
    targeted,
)
from tests._util.negative import must_fail

pytestmark = [pytest.mark.cpu, pytest.mark.budget(10)]

LR, RATE, CAP = 1e-2, 5.0, 2.0
SCHED = {"mode": "constant", "end": -1}


def _spec(**over) -> dict:
    return {**DEFAULTS, "rate": RATE, "cap": CAP, **over}


def _weight() -> torch.Tensor:
    w = torch.full((4, 4), 0.5)
    w[0, 0], w[1, 2] = 6.0, -9.0
    return w


def _rms(w: torch.Tensor) -> float:
    return float(w.square().mean().sqrt())


def _want(w: torch.Tensor, r0: float, k: float) -> torch.Tensor:
    ex = (w.abs() - CAP * r0).clamp_min(0)
    return w - k * ex * w.sign()


def check_decay(decay_of) -> None:
    """Over-cap elements move toward the cap by k x excess with their sign kept; every
    other element is unchanged; the metrics count the over-cap elements and give the
    largest |p| / (cap r0)."""
    w = _weight()
    r0 = _rms(w)
    p = w.clone()
    metrics = decay_of(p, r0)
    assert torch.allclose(p, _want(w, r0, LR * RATE), atol=1e-6)
    under = w.abs() <= CAP * r0
    assert torch.equal(p[under], w[under])
    assert metrics["outlier_over"].item() == 2
    assert metrics["outlier_max"].item() == pytest.approx(9.0 / (CAP * r0), rel=1e-5)


def _plain(p, r0):
    decay = OutlierDecay(_spec(), [(p, r0, {"lr": LR})], lambda: 0)
    decay.apply()
    return decay.current()


def test_decay_moves_only_the_excess():
    check_decay(_plain)


def test_decay_moves_only_the_excess_negative():
    def plain_wd(p, r0):
        out = _plain(p, r0)
        p.copy_(_weight() - LR * RATE * _weight())
        return out

    must_fail(lambda: check_decay(plain_wd), "a decay of every element")


def test_gated_and_fused_honour_the_flag():
    w = _weight()
    r0 = _rms(w)
    p = w.clone()
    decay = OutlierDecay(_spec(), [(p, r0, {"lr": LR})], lambda: 0)
    decay.apply_gated(torch.ones(()))
    assert torch.equal(p, w)
    decay.apply_gated(torch.zeros(()))
    assert torch.allclose(p, _want(w, r0, LR * RATE), atol=1e-6)
    q = w.clone()
    fused = OutlierDecay(_spec(), [(q, r0, None)], lambda: 0)
    fused.apply_param(q, LR, torch.ones(()))
    assert torch.equal(q, w)
    fused.apply_param(q, LR, torch.zeros(()))
    assert torch.allclose(q, _want(w, r0, LR * RATE), atol=1e-6)
    other = w.clone()
    fused.apply_param(other, LR, torch.zeros(()))
    assert torch.equal(other, w)


def test_schedules():
    s = _spec(cap=16.0, cap_start=64.0, cap_anneal=100, warmup=10)
    assert cap_at(s, 0) == pytest.approx(64.0)
    assert cap_at(s, 50) == pytest.approx(32.0)
    assert cap_at(s, 100) == cap_at(s, 10**6) == 16.0
    assert cap_at(_spec(cap=16.0), 0) == 16.0
    assert rate_at(s, 0) == pytest.approx(RATE / 10)
    assert rate_at(s, 9) == rate_at(s, 500) == RATE
    assert rate_at(_spec(), 0) == RATE


def test_targets():
    m = torch.zeros(2, 2)
    v = torch.zeros(2)
    assert targeted(_spec(), "layers.0.proj.o.weight", m)
    assert not targeted(_spec(), "layers.0.norm1.weight", v)
    assert targeted(_spec(include_1d=True), "layers.0.norm1.weight", v)
    assert not targeted(_spec(targets=r"ffn_"), "layers.0.proj.o.weight", m)
    assert not targeted(_spec(exclude=r"^layers\.0\."), "layers.0.proj.o.weight", m)


def _named() -> list[tuple[str, torch.nn.Parameter]]:
    return [
        ("mid.weight", torch.nn.Parameter(_weight())),
        ("mid.bias", torch.nn.Parameter(torch.zeros(4))),
    ]


def test_start_refs():
    named = _named()
    trainable = SimpleNamespace(named_trainable=lambda: named)
    refs = start_refs({"OUTLIER_DECAY": {"rate": RATE}}, trainable, resumed=False)
    assert set(refs) == {"mid.weight"}
    assert refs["mid.weight"] == pytest.approx(_rms(_weight()), rel=1e-6)
    assert start_refs({"OUTLIER_DECAY": {"rate": 0.0}}, trainable, False) is None
    assert start_refs({}, trainable, False) is None


def _bundle(named, k=None, device="cpu", **kind):
    groups = build_param_groups(named, LR, 0.0, "moss")
    bundle = build_bundle(
        groups,
        torch.device(device),
        "adamw",
        {"lr": LR, "betas": (0.9, 0.95), "eps": 1e-8},
        SCHED,
        10,
        build_precision("bf16", ()),
        0.0,
        **kind,
    )
    if k is not None:
        refs = {"mid.weight": _rms(_weight())}
        attach_outlier_decay(bundle, k, refs, [n for g in groups for n in g["names"]])
    return bundle


def _step(bundle, named, fused=False, device="cpu") -> list[torch.Tensor]:
    x = torch.randn(5, 4, generator=torch.Generator().manual_seed(1)).to(device)
    bundle.backward(F.linear(x, named[0][1], named[1][1]).square().mean())
    if fused:
        for _, p in named:
            bundle.step_param(p)
    bundle.step()
    if hasattr(bundle, "sync"):
        bundle.sync()
    return [p.detach().clone().cpu() for _, p in named]


def check_bundle_decays(**kind) -> None:
    """The decayed step equals the stock step with the excess decay applied to its
    result on the targeted weight; the untargeted bias is the stock step's."""
    device = kind.pop("device", "cpu")
    fused = kind.get("fused_step", False)
    stock_named = [(n, torch.nn.Parameter(p.detach().to(device))) for n, p in _named()]
    stock = _step(
        _bundle(stock_named, None, device, **kind), stock_named, fused, device
    )
    named = [(n, torch.nn.Parameter(p.detach().to(device))) for n, p in _named()]
    k = {"OUTLIER_DECAY": {"rate": RATE, "cap": CAP}}
    bundle = _bundle(named, k, device, **kind)
    got = _step(bundle, named, fused, device)
    assert torch.allclose(
        got[0], _want(stock[0], _rms(_weight()), LR * RATE), atol=1e-6
    )
    assert torch.equal(got[1], stock[1])
    assert set(bundle.extra_metrics()) == {"outlier_over", "outlier_max"}


def test_plain_bundle_decays_after_the_step():
    check_bundle_decays()
    assert _bundle(_named()).extra_metrics() == {}


def test_fused_bundle_decays_after_each_update():
    check_bundle_decays(fused_step=True)


@pytest.mark.cuda
def test_deferred_bundle_decays_after_the_step():
    if not torch.cuda.is_available():
        pytest.skip("deferred step needs torch's fused AdamW on CUDA")
    check_bundle_decays(device="cuda", sync_every=1)


def test_bundle_decay_negative():
    def undecayed(**kind):
        named = _named()
        got = _step(_bundle(named), named)
        stock_named = _named()
        stock = _step(_bundle(stock_named), stock_named)
        assert torch.allclose(
            got[0], _want(stock[0], _rms(_weight()), LR * RATE), atol=1e-6
        )

    must_fail(undecayed, "a bundle without the decay")


def test_coverage_check():
    bundle = SimpleNamespace(
        optimizer=SimpleNamespace(param_groups=[]), names=[], params=[], post_step=None
    )
    with pytest.raises(ValueError):
        attach_outlier_decay(
            bundle, {"OUTLIER_DECAY": {"rate": 1.0}}, {"a.weight": 1.0}, ["a.weight"]
        )


def test_outlier_problems():
    k = {"OUTLIER_DECAY": {"rate": 5.0, "cap": 16.0}, "OPTIMIZER": "adamw"}
    assert outlier_problems(k, full=True) == []
    assert outlier_problems({}, full=False) == []
    assert outlier_problems({"OUTLIER_DECAY": {"rate": 0.0}}, full=False) == []
    assert outlier_problems(k, full=False)
    bad = [
        {**k, "OPTIMIZER": "adamw16"},
        {**k, "PARALLEL": "fsdp"},
        {"OUTLIER_DECAY": {"rate": -1.0}},
        {"OUTLIER_DECAY": {"rate": 1.0, "cap": 0.0}},
        {"OUTLIER_DECAY": {"rate": 1.0, "cap": 16.0, "cap_start": 8.0}},
        {"OUTLIER_DECAY": {"rate": 1.0, "warmup": 1.5}},
        {"OUTLIER_DECAY": {"rate": 1.0, "targets": "("}},
        {"OUTLIER_DECAY": {"rate": 1.0, "include_1d": 1}},
        {"OUTLIER_DECAY": {"rate": 1.0, "cap_max": 3.0}},
        {"OUTLIER_DECAY": 5.0},
    ]
    for case in bad:
        assert outlier_problems(case, full=True), case
