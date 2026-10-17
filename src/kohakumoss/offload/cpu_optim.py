"""Host-side AdamW (`torch.optim.Optimizer`): fp32 masters and moments in host memory,
update on the CPU, weights copied back to the device. See docs/internals/offload.md."""

import math
from dataclasses import dataclass
from functools import partial

import torch
from torch import Tensor
from torch.optim import Optimizer

from kohakumoss.offload.accel import module, streamed


def _fused_available() -> bool:
    p = [torch.zeros(4)]
    try:
        torch._fused_adamw_(
            p,
            [torch.zeros(4)],
            [torch.zeros(4)],
            [torch.zeros(4)],
            [],
            [torch.zeros(())],
            lr=1e-3,
            beta1=0.9,
            beta2=0.999,
            weight_decay=0.0,
            eps=1e-8,
            amsgrad=False,
            maximize=False,
        )
    except (RuntimeError, AttributeError, TypeError):
        return False
    return True


def _fused_adamw(master, grad, m, v, step, lr, beta1, beta2, eps, wd) -> None:
    step += 1
    torch._fused_adamw_(
        [master],
        [grad],
        [m],
        [v],
        [],
        [step],
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=wd,
        eps=eps,
        amsgrad=False,
        maximize=False,
    )


def _manual_adamw(master, grad, m, v, step, lr, beta1, beta2, eps, wd) -> None:
    step += 1
    t = step.item()
    master.mul_(1 - lr * wd)
    m.lerp_(grad, 1 - beta1)
    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    bc1, bc2 = 1 - beta1**t, 1 - beta2**t
    denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
    master.addcdiv_(m, denom, value=-lr / bc1)


FUSED_CPU = _fused_available()
adamw_host = _fused_adamw if FUSED_CPU else _manual_adamw
"""One AdamW update of host fp32 tensors (decoupled weight decay); bumps `step`."""


class _Done:
    def synchronize(self) -> None:
        return None


DONE = _Done()


@dataclass
class _Lane:
    """Per-parameter host buffers and the bound copy functions for its device."""

    grad_host: Tensor
    shadow: Tensor
    d2h: object
    h2d: object
    group: dict
    ready: object = None


class CPUAdamW(Optimizer):
    """AdamW whose state and fp32 master copy live on the host."""

    handles_decay_to_one = True

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        pinned: bool = True,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.pinned = pinned
        self._streams: dict[torch.device, tuple] = {}
        self._lanes: dict[int, _Lane] = {}
        for group in self.param_groups:
            for p in group["params"]:
                self._init(p, group)

    def _init(self, p: Tensor, group: dict) -> None:
        cuda = streamed(p.device)
        pin = self.pinned and cuda
        master = p.detach().to("cpu", torch.float32).clone()
        self.state[p].update(
            master=master,
            exp_avg=torch.zeros_like(master),
            exp_avg_sq=torch.zeros_like(master),
            step=torch.zeros((), dtype=torch.float32),
        )
        grad_host = torch.empty(p.shape, dtype=p.dtype, pin_memory=pin)
        shadow = torch.empty(p.shape, dtype=p.dtype, pin_memory=pin)
        if cuda:
            if p.device not in self._streams:
                self._streams[p.device] = (
                    module(p.device).Stream(p.device),
                    module(p.device).Stream(p.device),
                )
            d2h, h2d = self._streams[p.device]
            lane = _Lane(grad_host, shadow, None, None, group)
            lane.d2h = partial(_d2h_device, lane=lane, stream=d2h)
            lane.h2d = partial(_h2d_device, lane=lane, stream=h2d)
        else:
            lane = _Lane(grad_host, shadow, None, None, group)
            lane.d2h = partial(_d2h_host, lane=lane)
            lane.h2d = partial(_h2d_host, lane=lane)
        lane.ready = _has_grad
        self._lanes[id(p)] = lane

    def use_host_grads(self, offload) -> None:
        """Read the gradients of the offloaded parameters from a GradOffload's host fp32
        accumulators (no D2H here); other parameters keep their own lanes."""
        for group in self.param_groups:
            for p in group["params"]:
                if p not in offload.host:
                    continue
                lane = self._lanes[id(p)]
                lane.grad_host = offload.host[p]
                lane.d2h = _already_on_host
                lane.ready = _always

    def _update(self, p: Tensor, lane: _Lane, ev) -> None:
        st, group = self.state[p], lane.group
        ev.synchronize()
        b1, b2 = group["betas"]
        adamw_host(
            st["master"],
            lane.grad_host.float(),
            st["exp_avg"],
            st["exp_avg_sq"],
            st["step"],
            group["lr"],
            b1,
            b2,
            group["eps"],
            group["weight_decay"],
        )
        if group.get("decay_to_one"):
            k = group["lr"] * group["decay_to_one_coef"]
            st["master"].sub_(st["master"] - 1.0, alpha=k)
        lane.shadow.copy_(st["master"])
        lane.h2d(p)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        work = []
        for group in self.param_groups:
            for p in group["params"]:
                lane = self._lanes[id(p)]
                if lane.ready(p):
                    work.append((p, lane, lane.d2h(p)))
        for p, lane, ev in work:
            self._update(p, lane, ev)
        return loss

    @torch.no_grad()
    def step_param(self, p: Tensor, found_inf: Tensor | None = None) -> None:
        """Update one parameter now (its grad must be set); skipped when `found_inf`
        (a 0-d tensor) is nonzero."""
        lane = self._lanes[id(p)]
        ev = lane.d2h(p)
        if found_inf is not None and bool(found_inf.item()):
            return
        self._update(p, lane, ev)

    def finish(self) -> None:
        return None


def _d2h_host(p: Tensor, lane: _Lane):
    lane.grad_host.copy_(p.grad)
    return DONE


def _has_grad(p: Tensor) -> bool:
    return p.grad is not None


def _always(p: Tensor) -> bool:
    return True


def _already_on_host(p: Tensor):
    return DONE


def _h2d_host(p: Tensor, lane: _Lane) -> None:
    p.data.copy_(lane.shadow)


def _d2h_device(p: Tensor, lane: _Lane, stream):
    dm = module(p.device)
    stream.wait_stream(dm.current_stream(p.device))
    with dm.stream(stream):
        lane.grad_host.copy_(p.grad, non_blocking=True)
    ev = dm.Event()
    ev.record(stream)
    return ev


def _h2d_device(p: Tensor, lane: _Lane, stream) -> None:
    dm = module(p.device)
    with dm.stream(stream):
        p.data.copy_(lane.shadow, non_blocking=True)
    dm.current_stream(p.device).wait_stream(stream)
