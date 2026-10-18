"""What activation offload keeps, model-free (explicit tensors, the host engine): the
kept set is rebuilt from the live tensors (a replaced storage drops out), a saved dtype
cast of a parameter is recast in backward instead of copied (and raises when the
parameter changed in place), and one tensor saved twice in a block is one packet."""

import pytest
import torch
from torch import nn

from kohakumoss.memory.apply import MemoryEngines
from kohakumoss.offload import activation
from kohakumoss.offload.activation import ActivationOffload
from kohakumoss.types.plans import MemoryPlan
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


class _Holder(nn.Module):
    """Owns one parameter: the live backbone tensor the kept set is rebuilt from."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4, 4))


def _ptr(t: torch.Tensor) -> int:
    return t.untyped_storage().data_ptr()


def check_rebuilt(engines: MemoryEngines, holder: nn.Module) -> None:
    p = holder.weight
    old = _ptr(p)
    engines.offload.exclude([p])
    p.data = p.data.to(torch.float64)
    engines.exclude_params([p])
    assert _ptr(p) in engines.offload.skip
    assert old not in engines.offload.skip, "a freed storage stays kept"


def test_kept_set_is_rebuilt(monkeypatch):
    holder = _Holder()
    engines = MemoryEngines(MemoryPlan(), offload=ActivationOffload("cpu"))
    engines.backbone = holder
    check_rebuilt(engines, holder)
    monkeypatch.setattr(ActivationOffload, "rebuild", ActivationOffload.exclude)
    adding = MemoryEngines(MemoryPlan(), offload=ActivationOffload("cpu"))
    adding.backbone = _Holder()
    must_fail(lambda: check_rebuilt(adding, adding.backbone), "added, not rebuilt")


def _cast_matmul(off: ActivationOffload, p: nn.Parameter, x: torch.Tensor):
    with off.block(0):
        y = x @ p.to(torch.bfloat16).t()
    return y.float().square().sum()


def check_recast(off: ActivationOffload) -> None:
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(16, 32))
    x = torch.randn(8, 32, dtype=torch.bfloat16, requires_grad=True)
    _cast_matmul(off, p, x).backward()
    want_p, want_x = p.grad.clone(), x.grad.clone()
    p.grad = x.grad = None
    (x @ p.to(torch.bfloat16).t()).float().square().sum().backward()
    torch.testing.assert_close(p.grad, want_p)
    torch.testing.assert_close(x.grad, want_x)
    assert off.stats["recast"] == 1, off.stats


def test_parameter_cast_is_recast(monkeypatch):
    check_recast(ActivationOffload("cpu", min_bytes=0))
    monkeypatch.setattr(activation, "cast_source", lambda t: None)
    must_fail(lambda: check_recast(ActivationOffload("cpu", min_bytes=0)), "copied")


def test_recast_of_a_changed_parameter_raises():
    off = ActivationOffload("cpu", min_bytes=0)
    p = nn.Parameter(torch.randn(16, 32))
    x = torch.randn(8, 32, dtype=torch.bfloat16, requires_grad=True)
    loss = _cast_matmul(off, p, x)
    assert off.stats["recast"] == 1
    with torch.no_grad():
        p.add_(1.0)
    with pytest.raises(RuntimeError, match="modified in place"):
        loss.backward()


def check_shared(off: ActivationOffload) -> None:
    x = torch.randn(64, requires_grad=True)
    with off.block(0):
        y = (x * x).sum()
    y.backward()
    torch.testing.assert_close(x.grad, 2 * x.detach())
    assert off.stats["shared"] == 1 and off.stats["packed"] == 1, off.stats


def test_tensor_saved_twice_is_one_packet(monkeypatch):
    check_shared(ActivationOffload("cpu", min_bytes=0))
    keys = iter(range(1000))
    monkeypatch.setattr(activation, "_key", lambda t: next(keys))
    must_fail(lambda: check_shared(ActivationOffload("cpu", min_bytes=0)), "twice")
