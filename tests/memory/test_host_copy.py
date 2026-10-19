"""The shared host-copy helpers of the offload placements, model-free: `host_copy`
gives a contiguous host tensor in the asked dtype with the source values,
`tensor_bytes` counts storage bytes, `HostCopies` round-trips a pass and counts its
bytes, `tensor_ids` names every tensor of its modules."""

import pytest
import torch

from kohakumoss.offload.encoders.placement import HostCopies
from kohakumoss.offload.weights.layers import host_copy, tensor_bytes, tensor_ids
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _source() -> torch.Tensor:
    return torch.arange(48, dtype=torch.float32).view(6, 8).t()


def assert_host_copy(copy_fn, dtype: torch.dtype | None) -> None:
    src = _source()
    out = copy_fn(src, False, dtype)
    want = dtype or src.dtype
    assert out.device.type == "cpu"
    assert out.dtype == want
    assert out.is_contiguous()
    assert out.shape == src.shape
    assert torch.equal(out, src.to(want))


def assert_bytes(count_fn) -> None:
    ts = [torch.empty(3, 5, dtype=torch.float16), torch.empty(7, dtype=torch.float32)]
    assert count_fn(ts) == sum(t.untyped_storage().nbytes() for t in ts)


@pytest.mark.parametrize("dtype", [None, torch.bfloat16, torch.float16])
def test_host_copy(dtype):
    assert_host_copy(host_copy, dtype)


def test_tensor_bytes():
    assert_bytes(tensor_bytes)


def assert_copies(make) -> None:
    lin = torch.nn.Linear(4, 3)
    want = [t.detach().clone() for t in (lin.weight, lin.bias)]
    copies = make([lin.weight, lin.bias])
    assert copies.nbytes == sum(t.untyped_storage().nbytes() for t in want)
    for _ in range(2):
        copies.begin()
        assert all(torch.equal(t, w) for t, w in zip((lin.weight, lin.bias), want))
        copies.end()
    assert copies.h2d_bytes == 2 * copies.nbytes


def assert_ids(ids_fn) -> None:
    a, b = torch.nn.Linear(2, 2), torch.nn.LayerNorm(2)
    want = {id(a.weight), id(a.bias), id(b.weight), id(b.bias)}
    assert ids_fn([a, b]) == want


def test_host_copies():
    assert_copies(lambda ts: HostCopies(ts, torch.device("cpu"), False))


def test_tensor_ids():
    assert_ids(tensor_ids)


class _Uncounted(HostCopies):
    def begin(self) -> None:
        for t, h, _ in self.pairs:
            t.data = h


def test_checks_fail_on_broken_helpers():
    must_fail(
        lambda: assert_copies(lambda ts: _Uncounted(ts, torch.device("cpu"), False)),
        "H2D bytes not counted",
    )
    must_fail(lambda: assert_ids(lambda ms: tensor_ids(ms[:1])), "module dropped")
    must_fail(
        lambda: assert_host_copy(lambda t, p, d: t.detach().to("cpu"), torch.float16),
        "dtype ignored",
    )
    must_fail(
        lambda: assert_host_copy(lambda t, p, d: t.detach().to("cpu", d), None),
        "not contiguous",
    )
    must_fail(lambda: assert_bytes(lambda ts: sum(t.numel() for t in ts)), "elements")
