"""Per-step collectives on an in-process gloo group of one rank (model-free, explicit
parameters): the step's loss and non-finite flag are one all-reduce, the block stats
one, a gradient no rank sent is not delivered by the DDP reducer, and a pipeline rank
hands only host tensors to `gather_object`."""

import pytest
import torch
import torch.distributed as dist

from kohakumoss.parallel import ddp
from kohakumoss.parallel.ddp import AsyncReducer, sum_block_stats
from kohakumoss.parallel.launch import bound_device
from kohakumoss.precision import build_precision
from kohakumoss.training.build.parallel import reduce_mean_loss, sum_loss_and_flag
from kohakumoss.training.drivers.pipeline import to_host
from kohakumoss.training.optim.bundle.base import all_reduce_max
from kohakumoss.training.optim.bundle.build import build_bundle
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
SCHED = {"mode": "constant", "end": -1}


@pytest.fixture
def group(tmp_path):
    store = dist.FileStore(str(tmp_path / "store"), 1)
    dist.init_process_group("gloo", store=store, rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


@pytest.fixture
def calls(monkeypatch):
    seen = []
    real = dist.all_reduce

    def spy(*args, **kwargs):
        seen.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(dist, "all_reduce", spy)
    return seen


def _bundle(fused: bool, folded: bool):
    p = torch.nn.Parameter(torch.ones(5))
    groups = [{"params": [p], "lr": 1e-2, "names": ["p"], "weight_decay": 0.0}]
    hooks = {"reduce_loss": reduce_mean_loss}
    if folded:
        hooks["reduce_step"] = sum_loss_and_flag
    bundle = build_bundle(
        groups,
        torch.device("cpu"),
        "adamw",
        {"lr": 1e-2, "betas": (0.9, 0.99), "eps": 1e-8},
        SCHED,
        10,
        build_precision("fp32", ()),
        0.0,
        fused_step=fused,
        agree=all_reduce_max,
        **hooks,
    )
    return p, bundle


def check_one_collective(fused: bool, folded: bool, calls: list) -> None:
    p, bundle = _bundle(fused, folded)
    bundle.backward((p * 2.0).sum())
    if fused:
        bundle.step_param(p)
    before = len(calls)
    assert bundle.step()
    assert len(calls) - before == 1, f"{len(calls) - before} all-reduces"
    assert bundle.last_loss == pytest.approx(10.0) and bundle.last_ok


@pytest.mark.parametrize("fused", [False, True])
def test_loss_and_flag_in_one_all_reduce(group, calls, fused):
    check_one_collective(fused, True, calls)
    must_fail(lambda: check_one_collective(fused, False, calls), "two collectives")


def test_block_stats_in_one_all_reduce(group, calls):
    bad = torch.tensor([2], dtype=torch.int32)
    sq = torch.tensor(9.0)
    sum_block_stats(bad, sq)
    assert len(calls) == 1 and int(bad) == 2 and float(sq) == 9.0
    assert bad.dtype == torch.int32


def check_absent(reducer: AsyncReducer) -> None:
    a, b = torch.nn.Parameter(torch.zeros(3)), torch.nn.Parameter(torch.zeros(4))
    got = {}

    def then(p):
        got[id(p)] = p.grad.clone()

    hook = reducer.reduced([a, b], then)
    reducer.start(True)
    a.grad = torch.ones(3)
    hook(a)
    reducer.finish()
    assert torch.equal(got[id(a)], torch.ones(3))
    assert id(b) not in got, "a gradient no rank sent was delivered (as zero)"


def _deliver_all(self, slab) -> None:
    with self._on_side():
        if slab.flat is None:
            slab.flat = torch.zeros(slab.size, dtype=slab.dtype, device=self.device)
        for key in slab.keys:
            if key not in slab.arrived:
                slab.view(key).zero_()
        self.reduce_flat(slab.flat)
        for key in slab.keys:
            slab.deliver[key](slab.view(key))
    slab.flat = None
    slab.done = True


def test_absent_everywhere_is_not_delivered(group, monkeypatch):
    check_absent(AsyncReducer("cpu"))
    monkeypatch.setattr(ddp.AsyncReducer, "_launch", _deliver_all)
    must_fail(lambda: check_absent(AsyncReducer("cpu")), "zero gradient delivered")


def _all_host(obj) -> None:
    match obj:
        case torch.Tensor():
            assert obj.device.type == "cpu" and not obj.requires_grad, obj
        case dict():
            for v in obj.values():
                _all_host(v)
        case list() | tuple():
            for v in obj:
                _all_host(v)


def test_gathered_state_is_on_the_host():
    state = {
        "optimizers": [{"state": {0: {"m": torch.ones(2, requires_grad=True)}}}],
        "pair": (torch.zeros(1, requires_grad=True), 3),
        "step": 7,
    }
    out = to_host(state)
    _all_host(out)
    assert out["step"] == 7 and out["pair"][1] == 3
    must_fail(lambda: _all_host(state), "the raw optimiser state")


def test_device_binding_follows_the_knob():
    cuda = torch.device("cuda", 1)
    assert bound_device("nccl", cuda, "ddp") == {}, "lazy is the default"
    assert bound_device("nccl", cuda, "ddp", "device") == {"device_id": cuda}
    assert bound_device("xccl", cuda, "fsdp", "device") == {"device_id": cuda}
    assert bound_device("nccl", cuda, "pipeline", "device") == {}
    assert bound_device("gloo", cuda, "ddp", "device") == {}
    assert bound_device("nccl", torch.device("cpu"), "ddp", "device") == {}
    with pytest.raises(ValueError):
        bound_device("nccl", cuda, "ddp", "eager")
    bound = bound_device("nccl", cuda, "pipeline", "device")
    must_fail(lambda: _equal(bound, {"device_id": cuda}), "pipeline bound")


def _equal(a, b) -> None:
    assert a == b, (a, b)
