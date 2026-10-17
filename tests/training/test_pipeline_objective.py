"""Pipeline engine micro-batches under token drop: the driver hands the kept-token
subset over, and the objective weights each sample's kept tokens only."""

from types import SimpleNamespace

import pytest
import torch

from kohakumoss.parallel.pipeline import MicroBatch
from kohakumoss.training.build.objective import build_loss
from kohakumoss.training.build.pipelines import pipeline_objective
from kohakumoss.training.drivers.pipeline import PipelineDriver, PipelineObjective
from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss_kernels.backends.eager.loss import token_mse
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
HW = torch.tensor([[2, 3], [3, 2]], dtype=torch.int32)
KEEP = [torch.tensor([0, 2, 5]), None]
WEIGHT = torch.tensor([1.0, 2.0])


def _micro() -> MicroBatch:
    g = torch.Generator().manual_seed(0)
    n = 3 + 6
    pred_like = torch.randn(n, 4, generator=g)
    return MicroBatch(
        img=pred_like,
        text=torch.zeros(5, 8),
        sigma=torch.rand(2, generator=g),
        hw=HW,
        text_len=torch.tensor([3, 2]),
        target=torch.randn(n, 4, generator=g),
        weight=WEIGHT,
        img_keep=KEEP,
    )


def _reference(pred, target) -> torch.Tensor:
    """Per-token channel-mean squared error; rows 0-2 are sample 0's kept tokens."""
    err = (pred - target).square().mean(-1)
    return err[:3].sum() * WEIGHT[0] + err[3:].sum() * WEIGHT[1]


def test_objective_uses_kept_token_counts():
    mb = _micro()
    pred = torch.randn_like(mb.target)
    objective = PipelineObjective(token_mse, SimpleNamespace(scale=1.0))
    want = _reference(pred, mb.target)
    torch.testing.assert_close(objective(pred, mb), want)

    def full_grid_ranges():
        sizes = (HW[:, 0] * HW[:, 1]).long()
        seg = torch.repeat_interleave(torch.arange(2), sizes)[: pred.shape[0]]
        got = token_mse(pred, mb.target, seg.to(torch.int32), WEIGHT, None)
        torch.testing.assert_close(got, want)

    must_fail(full_grid_ranges, "per-sample ranges from the full grid")


@pytest.mark.parametrize("loss", ["huber", "l1"])
def test_objective_follows_the_configured_loss(loss):
    built = SimpleNamespace(
        backbone=SimpleNamespace(ops=SimpleNamespace(token_mse=token_mse))
    )
    k = {"LOSS": loss, "LOSS_KWARGS": {}}
    objective = pipeline_objective(k, built)
    objective.scaler = SimpleNamespace(scale=1.0)
    mb = _micro()
    pred = mb.target + 3.0 * torch.randn_like(mb.target)
    seg = torch.repeat_interleave(torch.arange(2), torch.tensor([3, 6])).to(torch.int32)
    want = build_loss(k, built)(pred, mb.target, seg, WEIGHT, None)
    torch.testing.assert_close(objective(pred, mb), want)
    must_fail(
        lambda: torch.testing.assert_close(
            objective(pred, mb), token_mse(pred, mb.target, seg, WEIGHT, None)
        ),
        "pipeline loss fixed to MSE",
    )


def test_driver_micro_carries_img_keep():
    seq = build_seqinfo(HW, torch.tensor([3, 2]), img_keep=KEEP)
    hb = SimpleNamespace(hw=HW, img_keep=KEEP)
    p = SimpleNamespace(
        z=torch.zeros(9, 4),
        text=torch.zeros(5, 8),
        sigma=torch.zeros(2),
        seq=seq,
        target=torch.zeros(9, 4),
        weight=WEIGHT,
    )
    mb = PipelineDriver._micro(SimpleNamespace(micro=MicroBatch), hb, p)
    assert mb.img_keep is KEEP and mb.text_len.tolist() == [3, 2]
    assert mb.img_counts().tolist() == [3, 6] == [mb.img.shape[0] - 6, 6]
