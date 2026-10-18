"""The fused step's per-gradient transform, model-free (explicit gradients): one norm
pass and one unscale + clip + finite-check pass give the unscaled gradient clipped by
the previous step's global norm (GRAD_CLIP is the global norm in every bundle), its
unscaled squared norm in the step's norm and the found_inf flag; per-tensor norms of
16-bit gradients are taken in fp32."""

import pytest
import torch

from kohakumoss.precision.scaler import DynamicScaler
from kohakumoss.training.optim.bundle import fused
from kohakumoss.training.optim.bundle.base import local_grad_norm
from kohakumoss.training.optim.bundle.fused import FusedStepBundle
from kohakumoss.training.optim.bundle.mirror import schedule_for
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
SCALE = 2.0**10
CLIP = 1.0


def _bundle() -> FusedStepBundle:
    p = torch.nn.Parameter(torch.zeros(64))
    groups = [{"params": [p], "lr": 1e-3, "names": ["p"]}]
    return FusedStepBundle(
        build_one=lambda one: torch.optim.SGD(one, lr=1e-3),
        schedule_factory=lambda opt: schedule_for(opt, {"mode": "constant"}, 10),
        scaler=DynamicScaler(SCALE),
        groups=groups,
        clip=CLIP,
        device=torch.device("cpu"),
    )


def check_transform(bundle: FusedStepBundle, norms: list) -> None:
    """Step 1 has no earlier global norm (no clip); step 2 clips every gradient by
    GRAD_CLIP / step 1's global norm."""
    raw = torch.linspace(-3, 3, 64)
    square = float((raw * raw).sum())
    g = raw * SCALE
    before = len(norms)
    found = bundle.transform(g)
    assert len(norms) - before == 1, f"{len(norms) - before} norm passes"
    torch.testing.assert_close(g, raw, rtol=1e-5, atol=1e-6)
    assert float(found) == 0.0
    assert float(bundle._sq) == pytest.approx(square, rel=1e-5)
    bundle.step()
    assert bundle.last_norm == pytest.approx(square**0.5, rel=1e-5)
    g2 = raw * SCALE
    bundle.transform(g2)
    want = raw * min(1.0, CLIP / (square**0.5 + 1e-6))
    torch.testing.assert_close(g2, want, rtol=1e-5, atol=1e-6)
    bad = torch.full((8,), float("nan"))
    assert float(bundle.transform(bad)) == 1.0
    assert int(bundle._bad.item()) == 1


def _counting(monkeypatch) -> list:
    seen = []
    real = torch.linalg.vector_norm

    def spy(*args, **kwargs):
        seen.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(fused.torch.linalg, "vector_norm", spy)
    return seen


def _old_accumulate(self, bad_acc, sq_acc, g):
    """The transform before: unscale, finite check, zeroing, norm, clip by its norm."""
    self.scaler.unscale_([g])
    finite = torch.isfinite(g).all()
    g.mul_(finite.to(g.dtype)).nan_to_num_(0.0, 0.0, 0.0)
    bad = (~finite).to(torch.int32)
    bad_acc += bad
    norm = torch.linalg.vector_norm(g, dtype=torch.float32)
    sq_acc += norm.square()
    clip = torch.linalg.vector_norm(g, dtype=torch.float32)
    g.mul_((CLIP / (clip + 1e-6)).clamp(max=1.0).to(g.dtype))
    return bad.float()


def test_two_passes_unscale_clip_flag(monkeypatch):
    check_transform(_bundle(), _counting(monkeypatch))
    monkeypatch.setattr(FusedStepBundle, "_accumulate", _old_accumulate)
    must_fail(lambda: check_transform(_bundle(), _counting(monkeypatch)), "old passes")


def test_per_tensor_norms_in_fp32(monkeypatch):
    grads = [torch.full((100, 100), 1000.0, dtype=torch.float16)]
    want = 1000.0 * 100
    assert float(local_grad_norm(grads)) == pytest.approx(want, rel=1e-3)

    def half_norms(grads):
        norms = torch._foreach_norm(grads, 2.0)
        return torch.linalg.vector_norm(torch.stack([n.float() for n in norms]), 2.0)

    must_fail(
        lambda: _approx(float(half_norms(grads)), want), "norm in the gradient dtype"
    )


def _approx(a: float, b: float) -> None:
    assert a == pytest.approx(b, rel=1e-3), (a, b)
