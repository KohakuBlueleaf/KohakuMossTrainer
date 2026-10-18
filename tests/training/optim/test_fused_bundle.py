"""FusedStepBundle against the plain Bundle: same update, same decay-to-one, the real
gradient norm, and honest accounting of a step with a non-finite gradient (its
parameters are stepped inside backward, so it is applied, not skipped). Model-free:
an affine map then a layer norm over explicit parameters."""

import warnings

import pytest
import torch
import torch.nn.functional as F

from kohakumoss.precision import build_precision
from kohakumoss.training.optim import build_bundle, build_param_groups
from kohakumoss.training.optim.bundle.fused import FusedStepBundle
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
SCHED = {"mode": "constant", "end": -1}


class Params(list):
    """(name, parameter) pairs: `parameters()` in order, `[name]` by name."""

    def parameters(self) -> list[torch.nn.Parameter]:
        return [p for _, p in self]

    def __getitem__(self, key):
        if isinstance(key, str):
            return dict(list(self))[key]
        return super().__getitem__(key)


def _model(seed: int = 0) -> Params:
    """Affine 6 -> 5 (`0.*`), then a layer norm (`1.*`, gain 3.0)."""
    g = torch.Generator().manual_seed(seed)
    tensors = {
        "0.weight": torch.randn(5, 6, generator=g) / 6**0.5,
        "0.bias": torch.randn(5, generator=g) * 0.1,
        "1.weight": torch.full((5,), 3.0),
        "1.bias": torch.zeros(5),
    }
    return Params((n, torch.nn.Parameter(t)) for n, t in tensors.items())


def _forward(m: Params, x: torch.Tensor) -> torch.Tensor:
    h = F.linear(x, m["0.weight"], m["0.bias"])
    return F.layer_norm(h, (5,), m["1.weight"], m["1.bias"])


def _bundle(model: Params, fused: bool, precision="fp32", optimizer="adamw"):
    groups = build_param_groups(
        list(model), 1e-2, 0.5, "moss", decay_to_one=r"^1\.weight$"
    )
    return build_bundle(
        groups,
        torch.device("cpu"),
        optimizer,
        {"lr": 1e-2, "betas": (0.9, 0.95), "eps": 1e-8},
        SCHED,
        10,
        build_precision(precision, ()),
        0.0,
        fused_step=fused,
    )


def _train(fused: bool, steps: int = 3, x=None, optimizer="adamw"):
    model = _model()
    bundle = _bundle(model, fused, optimizer=optimizer)
    x = (
        torch.randn(4, 6, generator=torch.Generator().manual_seed(1))
        if x is None
        else x
    )
    for _ in range(steps):
        bundle.backward(_forward(model, x).square().mean())
        if fused:
            for p in model.parameters():
                bundle.step_param(p)
        bundle.step()
    return model, bundle


def _same(a, b) -> None:
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p, q, rtol=1e-6, atol=1e-7)


def test_fused_matches_plain_with_decay_to_one():
    plain, pb = _train(False)
    fused, fb = _train(True)
    assert float(plain["1.weight"].mean()) < 3.0 - 1e-3  # decayed towards one
    _same(plain, fused)
    no_decay = _model()
    must_fail(lambda: _same(no_decay, fused), "parameters not stepped")


def _in_train(bundle) -> None:
    opts = bundle._moded()
    assert opts and all(g["train_mode"] for o in opts for g in o.param_groups)


def test_schedule_free_fused_matches_plain(monkeypatch):
    pytest.importorskip("schedulefree")
    plain, _ = _train(False, optimizer="schedule_free")
    fused, fb = _train(True, optimizer="schedule_free")
    _same(plain, fused)
    _in_train(fb)
    fb.eval_mode()
    must_fail(lambda: _in_train(fb), "left in eval mode")
    monkeypatch.setattr(FusedStepBundle, "train_mode", lambda self: None)
    fresh = _bundle(_model(), True, optimizer="schedule_free")
    must_fail(lambda: _in_train(fresh), "train() never called at build")


def test_fused_reports_the_gradient_norm():
    plain, pb = _train(False, steps=1)
    fused, fb = _train(True, steps=1)
    assert fb.last_norm == pytest.approx(pb.last_norm, rel=1e-5)
    must_fail(lambda: _close(float("nan"), pb.last_norm), "norm reported as NaN")


def _close(a: float, b: float) -> None:
    assert a == pytest.approx(b, rel=1e-5)


def test_nonfinite_fused_step_is_applied_and_counted():
    model = _model()
    bundle = _bundle(model, True, "fp16")
    before = [p.detach().clone() for p in model.parameters()]
    scale0 = bundle.scaler.scale
    bundle.backward(_forward(model, torch.randn(4, 6)).square().mean())
    first = list(model.parameters())[0]
    first.grad.view(-1)[0] = float("inf")
    for p in model.parameters():
        bundle.step_param(p)
    assert bundle.step() is True
    assert bundle.skipped == 0 and bundle.nonfinite == 1 and not bundle.last_ok
    assert bundle.scaler.scale == scale0 / 2
    moved = [not torch.equal(p, q) for p, q in zip(model.parameters(), before)]
    assert all(moved[1:])


def _quiet(fn) -> None:
    """`fn` raises no LRScheduler "step() before optimizer.step()" warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    assert not [w for w in caught if "lr_scheduler.step()" in str(w.message)]


def _schedule_without_a_step() -> None:
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    torch.optim.lr_scheduler.StepLR(opt, 1).step()


def test_fused_step_counts_as_the_optimiser_step():
    _quiet(lambda: _train(True))
    must_fail(lambda: _quiet(_schedule_without_a_step), "schedule stepped first")


class _Blocks:
    param_groups: list = []

    def finish(self) -> None:
        return None

    def start(self, last: bool) -> None:
        return None


def _both_paths_count(bundle) -> None:
    """A non-finite block gradient and a finite device gradient, transformed through
    their own paths, both reach the step's flag (NaN norm, one non-finite parameter)."""
    blocks = _Blocks()
    bundle.attach_blocks(blocks)
    blocks.transform(torch.tensor([float("inf"), 0.0]))
    bundle.transform(torch.tensor([3.0, 4.0]))
    assert bundle._sq is not bundle._block_sq, "one accumulator shared by two streams"
    bundle.step()
    assert not bundle.last_ok and bundle.last_norm != bundle.last_norm
    assert bundle.last_nonfinite_params == 1


def test_block_and_device_paths_keep_their_own_accumulators(monkeypatch):
    _both_paths_count(_bundle(_model(), True))
    shared = _bundle(_model(), True)
    monkeypatch.setattr(type(shared), "_reset", _shared_reset)
    shared._reset()
    must_fail(lambda: _both_paths_count(shared), "shared accumulators")


def _shared_reset(self) -> None:
    self._bad = torch.zeros(1, dtype=torch.int32, device=self.device)
    self._sq = torch.zeros((), dtype=torch.float32, device=self.device)
    self._block_bad, self._block_sq = self._bad, self._sq
    self._loss = torch.zeros((), device=self.device)
    self._flags = []


def test_plain_bundle_counts_nonfinite_as_skipped():
    model = _model()
    bundle = _bundle(model, False, "fp16")
    bundle.backward(_forward(model, torch.full((2, 6), float("inf"))).sum())
    assert bundle.step() is False
    assert bundle.skipped == 1 and bundle.nonfinite == 1
