"""Native and Lightning drivers run the same TrainStep to the same parameters."""

import pytest
import torch

from kohakumoss.training.drivers import LightningDriver
from kohakumoss.training.drivers.lightning import L
from kohakumoss.training.state import TrainState
from tests.training.test_resume import make_run

pytestmark = pytest.mark.cpu


@pytest.mark.skipif(L is None, reason="lightning not installed")
def test_lightning_matches_native(tmp_path):
    model_a, drv, step, loader, bundle, ema, _, _ = make_run(tmp_path / "a", 4)
    drv.run(step, loader, bundle, ema, TrainState())
    model_b, native, step, loader, bundle, ema, ckpt, parts = make_run(
        tmp_path / "b", 4
    )
    lit = LightningDriver(
        4,
        2,
        native.batcher,
        native.normalizer,
        torch.device("cpu"),
        native.runner,
        ckpt,
        parts,
    )
    state = lit.run(step, loader, bundle, ema, TrainState())
    assert state.step == 4
    for p, q in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(p, q)
