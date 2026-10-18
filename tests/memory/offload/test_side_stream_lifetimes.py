"""Device tensors handed to a side-stream copy (activation offload's D2H, the gradient
accumulator's D2H) stay alive until that stream has read them. The side stream is made
to lag, the caller drops the tensor, and fresh allocations refill freed blocks with 7.0:
the copy must still carry the original values."""

import pytest
import torch

from kohakumoss.offload.activation import ActivationOffload
from kohakumoss.offload.optim.host_acc import HostAccumulator
from tests._util.negative import must_fail

pytestmark = pytest.mark.cuda
LAG = 200_000_000
NUMEL = 1 << 20


def _lag(stream) -> None:
    with torch.cuda.stream(stream):
        torch.cuda._sleep(LAG)


def _refill() -> list:
    return [torch.full((NUMEL,), 7.0, device="cuda") for _ in range(8)]


def _activation_copy_survives() -> None:
    engine = ActivationOffload("cuda", min_bytes=0)
    t = torch.randn(NUMEL, device="cuda")
    ref = t.cpu()
    _lag(engine.d2h_stream)
    pk = engine._store_device(t)
    del t
    _refill()
    torch.cuda.synchronize()
    assert torch.equal(pk.host, ref), "D2H read a reused block"


def _accumulated_grad_survives() -> None:
    acc = HostAccumulator(pinned=True)
    stream = torch.cuda.Stream()
    g = torch.randn(NUMEL, device="cuda")
    ref = g.cpu()
    dst = torch.zeros(NUMEL)
    _lag(stream)
    acc.add(0, g, dst, stream)
    del g
    _refill()
    acc.drain()
    acc.close()
    assert torch.equal(dst, ref), "accumulated a reused block"


@pytest.mark.parametrize(
    "check", [_activation_copy_survives, _accumulated_grad_survives]
)
def test_side_stream_copy_reads_the_original_tensor(check, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    check()
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None)
    must_fail(check, "tensor not recorded on the side stream")
