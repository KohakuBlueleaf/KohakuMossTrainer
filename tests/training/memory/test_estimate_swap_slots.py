"""The training estimate covers a measured full-finetune device peak: `minimal` with 4
resident `swap` blocks at 1 MP, bf16, on an RTX 3090 (scripts/bench/train/memory.py)
measured 11.93 GiB allocated, 13.26 GiB reserved and 14.14 GiB device use. The
estimate's allocated parts, their reserved total (the side-stream fragmentation) and
its device total (plus the memory outside the allocator) must each reach the measured
number; without the offload optimisers' device buffers (StreamedAdamW's slot ring,
HostStateAdamW's state ring, gradient offload's staging ring), or with the
compute-stream fragmentation and a 0.45 GiB context, they do not."""

import importlib

import pytest
import torch

from kohakumoss.memory.profiles import resolve
from kohakumoss.model import resolve_config
from tests._util.negative import must_fail

E = importlib.import_module("kohakumoss.memory.estimate")
pytestmark = pytest.mark.cpu
GIB = 2**30
ALLOCATED, RESERVED, DEVICE = 11.93 * GIB, 13.26 * GIB, 14.14 * GIB
OVERRIDES = {"stream": "swap", "stream_resident": 4, "act_offload": True}


def _estimate():
    plan = resolve("minimal", OVERRIDES, "cpu", trainable_base=True)
    work = E.Workload(
        resolve_config("base"),
        img_tokens=4096,
        txt_tokens=300,
        full_ft=True,
        compute_bytes=torch.empty((), dtype=torch.bfloat16).element_size(),
    )
    return E.estimate(work, plan)


def check_covers() -> None:
    est = _estimate()
    torch_parts = [k for k in est.parts if k != "context"]
    allocated = sum(est.parts[k] for k in torch_parts)
    assert allocated >= ALLOCATED, allocated / GIB
    assert est.device(torch_parts) >= RESERVED, est.device(torch_parts) / GIB
    assert est.total >= DEVICE, est.gib()


def _without_rings(estimate):
    """The estimate without the offload optimisers' device buffers."""

    def run(work, plan):
        est = estimate(work, plan)
        for part in ("optim_slots", "state_ring", "grad_ring"):
            est.parts.pop(part, None)
        return est

    return run


def test_the_estimate_covers_the_measured_full_ft_peak(monkeypatch):
    check_covers()
    with monkeypatch.context() as m:
        m.setattr(E, "estimate", _without_rings(E.estimate))
        must_fail(check_covers, "optimiser device buffers left out")
    with monkeypatch.context() as m:
        m.setattr(E, "FRAGMENTATION_OFFLOAD", E.FRAGMENTATION)
        m.setattr(E, "CONTEXT_BYTES", int(0.45 * GIB))
        must_fail(check_covers, "compute-stream fragmentation and a 0.45 GiB context")
