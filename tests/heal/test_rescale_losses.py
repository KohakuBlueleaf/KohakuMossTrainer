"""The uniform rescale's factor and exactness, the heal losses."""

import pytest
import torch

from kohakumoss.heal.losses import HealLoss, concentration, rel_sq
from kohakumoss.heal.rescale import power_of_two_scale
from kohakumoss.training.stream_loss import top_share
from kohakumoss_kernels.backends.eager.norm import rms_norm
from tests._util.negative import must_fail

pytestmark = [pytest.mark.cpu, pytest.mark.budget(10)]


def test_power_of_two_scale():
    assert power_of_two_scale([4.0, 16.0]) == 0.125
    assert power_of_two_scale([1.0]) == 1.0
    assert power_of_two_scale([0.25, 0.25]) == 4.0


def _rescale_check(eps_mult):
    """rms_norm(c x, eps * m) == rms_norm(x, eps) for c = 1/8 (m = c^2 is exact)."""
    c = 0.125
    x = torch.randn(6, 32, dtype=torch.float64) * 1e-3
    w = torch.rand(32, dtype=torch.float64)
    torch.testing.assert_close(
        rms_norm(c * x, w, 1e-6 * eps_mult(c)),
        rms_norm(x, w, 1e-6),
        rtol=1e-12,
        atol=1e-12,
    )


def test_rescale_exact_with_eps():
    _rescale_check(lambda c: c * c)


def test_rescale_exact_negative():
    must_fail(lambda: _rescale_check(lambda c: 1.0), "stock eps on a small stream")


def test_rel_sq():
    b = torch.tensor([3.0, 4.0])
    assert rel_sq(b, b).item() == 0.0
    assert rel_sq(torch.zeros(2), b).item() == pytest.approx(1.0)


def test_heal_loss_parts():
    v_t, target = torch.ones(4, 2), torch.zeros(4, 2) + 2.0
    v_s = v_t * 1.1
    blocks = [torch.randn(5, 16) for _ in range(3)]
    soft_only = HealLoss()
    loss, parts = soft_only(v_s, v_t, target, [])
    assert set(parts) == {"soft"} and loss.item() == pytest.approx(
        rel_sq(v_s, v_t).item()
    )
    full = HealLoss(hard_weight=0.02, conc_weight=0.05, conc_k=4)
    loss, parts = full(v_s, v_t, target, blocks)
    want = (
        rel_sq(v_s, v_t) + 0.02 * rel_sq(v_s, target) + 0.05 * concentration(blocks, 4)
    )
    assert loss.item() == pytest.approx(want.item())
    assert full.needs_blocks and not soft_only.needs_blocks
    mean = sum(top_share(h, 4) for h in blocks) / 3
    assert concentration(blocks, 4).item() == pytest.approx(mean.item())
