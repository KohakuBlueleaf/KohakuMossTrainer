"""Side-stream optimisers read the caller's found_inf on their own stream. Each records
it there, so the caching allocator cannot give its memory to a later allocation on the
caller's stream before the side stream has read it. The side stream is made to lag, and
the freed block is refilled with 1.0 ("overflow"): a lost found_inf skips the update."""

import pytest
import torch
from torch import nn

from kohakumoss.offload.optim.cpu_optim import CPUAdamW
from kohakumoss.offload.optim.gpu_adamw import HostStateAdamW
from tests._util.negative import must_fail

pytestmark = pytest.mark.cuda
LAG = 200_000_000


def _step_then_reuse(step) -> list:
    """`step(found)` with a zero found_inf, then free it and fill new blocks with 1.0."""
    found = torch.zeros((), device="cuda")
    step(found)
    del found
    return [torch.ones((), device="cuda") for _ in range(64)]


def _param() -> nn.Parameter:
    p = nn.Parameter(torch.randn(64, 64, device="cuda"))
    p.grad = torch.randn_like(p)
    return p


def _host_state_updates() -> None:
    p = _param()
    opt = HostStateAdamW([p], lr=1e-2, weight_decay=0.0)
    before = p.detach().clone()
    with torch.cuda.stream(opt.opt):
        torch.cuda._sleep(LAG)
    _step_then_reuse(lambda found: opt.step_param(p, found))
    opt.finish()
    torch.cuda.synchronize()
    assert not torch.equal(p.detach(), before), "update skipped: found_inf reused"
    assert float(opt.steps[id(p)]) == 1.0


def _cpu_adamw_updates() -> None:
    p = _param()
    opt = CPUAdamW([p], lr=1e-2, weight_decay=0.0)
    before = p.detach().clone()
    d2h, _ = opt._streams[p.device]
    with torch.cuda.stream(d2h):
        torch.cuda._sleep(LAG)
    _step_then_reuse(lambda found: opt.step_param(p, found))
    opt.finish()
    torch.cuda.synchronize()
    opt.close()
    assert not torch.equal(p.detach(), before), "update skipped: found_inf reused"


def _no_record_stream(monkeypatch) -> None:
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None)


@pytest.mark.parametrize("check", [_host_state_updates, _cpu_adamw_updates])
def test_found_inf_survives_until_the_side_stream_reads_it(check, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    check()
    _no_record_stream(monkeypatch)
    must_fail(check, "found_inf not recorded on the side stream")
