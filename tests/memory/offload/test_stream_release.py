"""A trainable weight stream and the optimisers `build_memory_optim` wires over it free
by reference counting alone, model-free (blocks of one linear each, on the host): after
a real step through StreamedAdamW (GPU optim_state: stepped per block in backward) or
the host AdamW (cpu optim_state: host-stepped blocks), the bundle, the block optimiser,
the stream, its parts and its host masters are dead once dropped with gc off; each
check fails with one strong back-reference put back. See
docs/internals/reference-cycles.md."""

from functools import partial

import pytest
import torch
from torch import nn

from kohakumoss.memory.apply import MemoryEngines
from kohakumoss.model.linear import MossLinear
from kohakumoss.offload.weights.stream import WeightStream
from kohakumoss.training.memory.optimizers import build_memory_optim
from kohakumoss.types.plans import MemoryPlan, PrecisionConfig
from tests._util.negative import must_fail
from tests._util.release import check_released

pytestmark = pytest.mark.cpu
CPU = torch.device("cpu")
OUT, IN, BLOCKS = 8, 16, 3
HYPER = {"lr": 1e-2, "betas": (0.9, 0.999), "eps": 1e-8}
CONSTANT = {"mode": "constant"}
PREC = PrecisionConfig(compute=torch.float32, scaler=False)


def _blocks() -> list[nn.Module]:
    out = []
    for _ in range(BLOCKS):
        block = nn.Module()
        block.lin = MossLinear(IN, OUT, bias=False)
        nn.init.normal_(block.lin.weight)
        out.append(block)
    return out


def _wired(plan: MemoryPlan) -> tuple:
    stream = WeightStream(_blocks(), "cpu", slots=2, pinned=False, trainable=True)
    extra = nn.Parameter(torch.zeros(4))
    engines = MemoryEngines(plan, weight_stream=stream)
    params = [*stream.params.values(), extra]
    groups = [{"params": params, "lr": 1e-2, "weight_decay": 0.0}]
    bundle, accumulate = build_memory_optim(
        groups, plan, engines, CPU, "adamw", HYPER, CONSTANT, 10, PREC, 0.0
    )
    return stream, extra, bundle, accumulate


def _masters(stream: WeightStream) -> dict:
    return {k: p.detach().clone() for k, p in stream.params.items()}


def _all_moved(stream: WeightStream, before: dict) -> None:
    for key, p in stream.params.items():
        assert not torch.equal(p.detach(), before[key]), f"block {key} not stepped"


def _in_backward(strong: str = "") -> tuple:
    plan = MemoryPlan(
        stream="swap", stream_resident=2, optim_state="gpu", fused_step=True
    )
    stream, extra, bundle, accumulate = _wired(plan)
    blocks = bundle.blocks
    before = _masters(stream)
    with accumulate(True):
        for index in reversed(range(stream.n)):
            stream.on_block_backward(index)
            for key in blocks.keys[index]:
                stream.on_grad(key, torch.ones(stream.params[key].shape))
    bundle.step()
    _all_moved(stream, before)
    match strong:
        case "on_grad":
            stream.on_grad = blocks._on_grad
        case "transform":
            blocks.transform = bundle._block_transform
        case "copier":
            stream.copier._copy = stream.copier.copy_host
    master = next(iter(stream.params.values()))
    return bundle, blocks, stream, stream.copier, stream.grads, master, extra


def _host_stepped(strong: str = "") -> tuple:
    plan = MemoryPlan(stream="swap", stream_resident=2, optim_state="cpu")
    stream, extra, bundle, accumulate = _wired(plan)
    before = _masters(stream)
    with accumulate(True):
        for key, master in stream.params.items():
            stream.on_grad(key, torch.ones(master.shape))
    bundle.step()
    _all_moved(stream, before)
    match strong:
        case "mine":
            stream.reducer.mine = stream.mine
        case "sink":
            stream.grads._accumulate = stream.grads._accumulate_host
    master = next(iter(stream.params.values()))
    return bundle, stream, stream.reducer, stream.grads, master, extra


@pytest.mark.parametrize(
    ("build", "edges"),
    [
        (_in_backward, ("on_grad", "transform", "copier")),
        (_host_stepped, ("mine", "sink")),
    ],
    ids=["in_backward", "host_stepped"],
)
def test_stream_and_its_optimisers_are_released(build, edges):
    check_released(build)
    for edge in edges:
        must_fail(partial(check_released, partial(build, edge)), f"strong {edge}")
