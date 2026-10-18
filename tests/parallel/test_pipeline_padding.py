"""Pipeline static shapes (pure functions): padding samples of at most `pad_segment`
tokens filling every micro-batch to the shape, the schedule's micro-batch constraints,
and the last stage's loss over the real (kept) image tokens of a padded layout."""

from types import SimpleNamespace

import pytest
import torch

from kohakumoss.parallel.pipeline.distributed import (
    MicroBatch,
    PipeShape,
    TokenLoss,
    padded_seqinfo,
    padding,
    schedule_problems,
    stage_owner,
)
from kohakumoss_kernels.backends.eager.loss import token_mse
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
SHAPE = PipeShape(tokens=1200, max_samples=3, dim=8, microbatches=4, pad_segment=256)
HW = torch.tensor([[2, 3], [3, 2]])
KEEP = [torch.tensor([0, 2, 5]), None]
WEIGHT = torch.tensor([1.0, 2.0])


def check_padding(sizes: list[int], used: int, n: int, shape: PipeShape) -> None:
    assert sum(sizes) == shape.tokens - used, sizes
    assert all(0 <= s <= shape.pad_segment for s in sizes), sizes
    assert len(sizes) == shape.slots - n, (len(sizes), shape.slots)


@pytest.mark.parametrize("used", [0, 1, 700, 1199, 1200])
def test_padding_fills_the_shape(used):
    check_padding(padding(used, 2, SHAPE), used, 2, SHAPE)
    one_segment = [SHAPE.tokens - used] + [0] * (SHAPE.slots - 3)
    if SHAPE.tokens - used > SHAPE.pad_segment:
        must_fail(lambda: check_padding(one_segment, used, 2, SHAPE), "one pad sample")


def test_padding_rejects_overflow():
    with pytest.raises(ValueError, match="exceeds pipe shape"):
        padding(1201, 2, SHAPE)
    with pytest.raises(ValueError, match="exceeds pipe shape"):
        padding(10, 4, SHAPE)


def test_schedule_problems():
    assert schedule_problems("1f1b", 2, 8) == []
    assert schedule_problems("interleaved-1f1b", 4, 10) == []
    assert schedule_problems("interleaved-1f1b", 4, 9) != []
    assert schedule_problems("dualpipe-v", 2, 3) != []
    assert schedule_problems("1f1b", 4, 2) != []
    assert schedule_problems("nope", 2, 8) != []


def test_last_stage_owner():
    assert stage_owner(3, 4, "1f1b") == 3
    assert stage_owner(7, 4, "interleaved-1f1b") == 3
    assert stage_owner(7, 4, "zbv") == 0 and stage_owner(4, 4, "zbv") == 3


def _micro() -> MicroBatch:
    g = torch.Generator().manual_seed(0)
    n = 3 + 6
    return MicroBatch(
        img=torch.randn(n, 4, generator=g),
        text=torch.zeros(5, 8),
        sigma=torch.rand(2, generator=g),
        hw=HW,
        text_len=torch.tensor([3, 2]),
        target=torch.randn(n, 4, generator=g),
        weight=WEIGHT,
        img_keep=KEEP,
    )


def test_token_loss_uses_the_kept_tokens_of_the_padded_layout():
    mb = _micro()
    pred = torch.randn_like(mb.target)
    seq = padded_seqinfo(mb, SHAPE)
    n = mb.target.shape[0]
    loss = TokenLoss(token_mse, SimpleNamespace(scale=1.0))
    err = (pred - mb.target).square().mean(-1)
    want = err[:3].sum() * WEIGHT[0] + err[3:].sum() * WEIGHT[1]
    torch.testing.assert_close(loss(pred, mb, seq.seg_img[:n]), want)
    full = torch.repeat_interleave(torch.arange(2), (HW[:, 0] * HW[:, 1]))[:n]
    must_fail(
        lambda: torch.testing.assert_close(loss(pred, mb, full.to(torch.int32)), want),
        "sample ranges from the full grid",
    )
