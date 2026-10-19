"""The optimiser-side offload engines and their wiring free by reference counting alone,
model-free (small real parameters on the host): GradOffload, FusedStep, CPUAdamW and
HostStateAdamW after a real step, and the bundles `build_memory_optim` wires over them
(in-backward step, gradient offload, host AdamW, the DDP AsyncReducer) are dead once
dropped with gc off; each check fails with one strong back-reference put back. See
docs/internals/reference-cycles.md."""

from functools import partial
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from kohakumoss.memory.apply import MemoryEngines
from kohakumoss.offload.optim.cpu_optim import CPUAdamW
from kohakumoss.offload.optim.fused_step import FusedStep
from kohakumoss.offload.optim.gpu_adamw import HostStateAdamW
from kohakumoss.offload.optim.grad import GradOffload
from kohakumoss.parallel.ddp import AsyncReducer
from kohakumoss.training.memory.optimizers import build_memory_optim
from kohakumoss.types.plans import MemoryPlan, PrecisionConfig
from tests._util.negative import must_fail
from tests._util.release import check_released

pytestmark = pytest.mark.cpu
CPU = torch.device("cpu")
HYPER = {"lr": 1e-2, "betas": (0.9, 0.999), "eps": 1e-8}
CONSTANT = {"mode": "constant"}
PREC = PrecisionConfig(compute=torch.float32, scaler=False)


def _params() -> list[nn.Parameter]:
    return [nn.Parameter(torch.zeros(3, 4)) for _ in range(2)]


def _backward(params) -> None:
    sum((p * 2.0).sum() for p in params).backward()


def _grad_offload(strong: bool = False) -> tuple:
    params = _params()
    off = GradOffload(params, pinned=False)
    off.start(False)
    _backward(params)
    off.drain()
    assert off.lanes[id(params[0])].filled, "the weak hook did not reach its owner"
    if strong:
        params[0].register_post_accumulate_grad_hook(off.zero)
    return off, params[0], off.host[params[0]]


def _count(log: list, p: nn.Parameter) -> None:
    log.append(tuple(p.shape))


def _fused_step(strong: bool = False) -> tuple:
    params = _params()
    stepped = []
    fused = FusedStep(params, partial(_count, stepped), clip=1.0, scaled=True)
    fused.start(True, torch.ones(()))
    _backward(params)
    assert len(stepped) == 2, "the weak hook did not reach its owner"
    if strong:
        fused._transforms = (fused._clip,)
    return fused, params[0], fused.found_inf


def _stepped(opt, params) -> None:
    for p in params:
        p.grad = torch.ones_like(p)
    opt.step_param(params[0])
    opt.step()
    assert not torch.equal(params[0].detach(), torch.zeros(3, 4))


def _cpu_adamw(strong: bool = False) -> tuple:
    params = _params()
    opt = CPUAdamW(params, lr=1e-2, weight_decay=0.0, pinned=False)
    _stepped(opt, params)
    if strong:
        opt._lanes[id(params[0])].step = partial(opt._step_now)
    return opt, params[0], opt.state[params[0]]["master"]


def _host_state_adamw(strong: bool = False) -> tuple:
    params = _params()
    opt = HostStateAdamW(params, lr=1e-2, weight_decay=0.0, pinned=False)
    _stepped(opt, params)
    if strong:
        opt._update = MethodType(HostStateAdamW._update_host, opt)
    return opt, params[0], opt.host[id(params[0])][0]


@pytest.mark.parametrize(
    "build", [_grad_offload, _fused_step, _cpu_adamw, _host_state_adamw]
)
def test_engine_is_released(build):
    check_released(build)
    must_fail(lambda: check_released(partial(build, True)), "a strong self-reference")


def _wired(plan: MemoryPlan, hooks=None) -> tuple:
    lin = nn.Linear(4, 4)
    groups = [{"params": list(lin.parameters()), "lr": 1e-2, "weight_decay": 0.0}]
    engines = MemoryEngines(plan)
    bundle, accumulate = build_memory_optim(
        groups, plan, engines, CPU, "adamw", HYPER, CONSTANT, 10, PREC, 0.0, hooks
    )
    return lin, engines, bundle, accumulate


def _train_step(lin, bundle, accumulate) -> None:
    before = lin.weight.detach().clone()
    with accumulate(True):
        bundle.backward(lin(torch.ones(2, 4)).sum())
    bundle.step()
    assert not torch.equal(lin.weight.detach(), before), "no parameter was stepped"


def _fused(strong: bool = False) -> tuple:
    lin, engines, bundle, accumulate = _wired(MemoryPlan(fused_step=True))
    _train_step(lin, bundle, accumulate)
    if strong:
        engines.fused.step_param = bundle.step_param
    return bundle, engines.fused, lin.weight


def _fused_offload(strong: bool = False) -> tuple:
    plan = MemoryPlan(fused_step=True, grad_offload=True)
    lin, engines, bundle, accumulate = _wired(plan)
    _train_step(lin, bundle, accumulate)
    if strong:
        engines.grads.final = bundle.step_param
    return bundle, engines.grads, lin.weight


def _host_adamw_offload(strong: bool = False) -> tuple:
    plan = MemoryPlan(optim_state="cpu", grad_offload=True)
    lin, engines, bundle, accumulate = _wired(plan)
    _train_step(lin, bundle, accumulate)
    if strong:
        engines.grads.final = bundle.step
    return bundle, bundle.optimizer, engines.grads, lin.weight


def _ddp_fused(strong: bool = False) -> tuple:
    hooks = SimpleNamespace(ddp=AsyncReducer("cpu"))
    lin, engines, bundle, _ = _wired(MemoryPlan(fused_step=True), hooks)
    if strong:
        hooks.ddp.reduced([lin.weight], bundle.step_param)
    return bundle, hooks.ddp, engines.fused, lin.weight


def _ddp_offload(strong: bool = False) -> tuple:
    hooks = SimpleNamespace(ddp=AsyncReducer("cpu"))
    lin, engines, bundle, accumulate = _wired(MemoryPlan(grad_offload=True), hooks)
    accumulate(True)
    assert engines.grads.final is not None, "no DDP callback on the last micro-batch"
    if strong:
        engines.grads.final = hooks.ddp.add_param
    return bundle, hooks.ddp, engines.grads, lin.weight


@pytest.mark.parametrize(
    "build",
    [_fused, _fused_offload, _host_adamw_offload, _ddp_fused, _ddp_offload],
)
def test_wired_bundle_is_released(build):
    check_released(build)
    must_fail(lambda: check_released(partial(build, True)), "a strong back-reference")
