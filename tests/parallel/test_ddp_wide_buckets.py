"""The DDP reducer's fp32 bucket copies (model-free, a one-rank gloo group, explicit fp16
parameters): at most WIDE_AHEAD + 1 copies are alive while backward launches the
buckets, and the summed gradients land."""

import pytest
import torch
import torch.distributed as dist

from kohakumoss.parallel import ddp
from kohakumoss.parallel.ddp import WIDE_AHEAD, GradReducer
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
SIZES = (5, 7, 3, 9, 4, 6)


@pytest.fixture
def group(tmp_path):
    store = dist.FileStore(str(tmp_path / "store"), 1)
    dist.init_process_group("gloo", store=store, rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


class _AllHeld(GradReducer):
    def _launch(self, bucket) -> None:
        bucket.gather()
        bucket.work = dist.all_reduce(bucket.reduced_buffer(), async_op=True)


def check_bounded(cls, monkeypatch) -> None:
    """Every launch sees at most WIDE_AHEAD + 1 fp32 copies alive; gradients are right."""
    params = [torch.nn.Parameter(torch.zeros(n, dtype=torch.float16)) for n in SIZES]
    reducer = cls(params, bucket_bytes=1)
    assert len(reducer.buckets) == len(SIZES)
    alive = []
    real = ddp._Bucket.reduced_buffer

    def counted(bucket):
        out = real(bucket)
        alive.append(sum(b.wide is not None for b in reducer.buckets))
        return out

    monkeypatch.setattr(ddp._Bucket, "reduced_buffer", counted)
    coef = [torch.arange(n, dtype=torch.float32) for n in SIZES]
    reducer.start(True)
    sum((p.float() * c).sum() for p, c in zip(params, coef)).backward()
    reducer.finish()
    monkeypatch.setattr(ddp._Bucket, "reduced_buffer", real)
    for p, c in zip(params, coef):
        assert torch.equal(p.grad.float(), c), (p.grad, c)
    assert all(b.wide is None for b in reducer.buckets)
    assert max(alive) <= WIDE_AHEAD + 1, alive


def test_fp32_copies_are_released_as_buckets_land(group, monkeypatch):
    check_bounded(GradReducer, monkeypatch)
    must_fail(lambda: check_bounded(_AllHeld, monkeypatch), "every copy held to finish")
