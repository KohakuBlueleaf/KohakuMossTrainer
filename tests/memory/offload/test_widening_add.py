"""The host accumulator's widening add (a 16-bit gradient into an fp32 sum, chunked
through an fp32 scratch) gives bitwise the sum of `dst.add_(src)`, on sizes below,
at and across the chunk."""

import pytest
import torch

from kohakumoss.offload.optim import host_acc
from kohakumoss.offload.optim.host_acc import HostAccumulator, widening_add
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
CHUNK = 1000


def check_sums(add) -> None:
    gen = torch.Generator().manual_seed(0)
    scratch = torch.empty(CHUNK)
    for n in (7, CHUNK, 3 * CHUNK + 17):
        for dtype in (torch.bfloat16, torch.float16):
            src = torch.randn(n, 3, generator=gen).to(dtype)
            dst = torch.randn(n, 3, generator=gen)
            want = dst.clone().add_(src)
            add(dst, src, scratch)
            assert torch.equal(dst, want), (n, dtype)


def drop_tail(dst, src, scratch) -> None:
    flat_dst, flat_src = dst.view(-1), src.reshape(-1)
    full = flat_dst.numel() // CHUNK * CHUNK
    for start in range(0, full, CHUNK):
        part = scratch[:CHUNK]
        part.copy_(flat_src[start : start + CHUNK])
        flat_dst[start : start + CHUNK].add_(part)


def test_widening_add_is_the_plain_sum(monkeypatch):
    monkeypatch.setattr(host_acc, "WIDEN_CHUNK", CHUNK)
    check_sums(widening_add)
    must_fail(lambda: check_sums(drop_tail), "a chunked add that drops the last chunk")


def test_adder_widens_only_into_fp32():
    acc = HostAccumulator(pinned=False)
    try:
        assert acc._adder(torch.float32, torch.float32) is torch.Tensor.add_
        assert acc._adder(torch.bfloat16, torch.bfloat16) is torch.Tensor.add_
        assert acc._adder(torch.bfloat16, torch.float32).func is widening_add
    finally:
        acc.close()
