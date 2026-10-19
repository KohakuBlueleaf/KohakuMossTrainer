"""Host-side AdamW (`torch.optim.Optimizer`): fp32 masters and moments in host memory,
update on the CPU, weights copied back to the device. See docs/internals/offload.md."""

import math
from dataclasses import dataclass
from functools import partial

import torch
from torch import Tensor
from torch.optim import Optimizer

from kohakumoss.offload.accel import DONE, module, streamed
from kohakumoss.offload.worker import HostWorker

HOST_STATE = ("master", "exp_avg", "exp_avg_sq")


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


@dataclass
class _Lane:
    """Per-parameter host buffers and the copy / step functions selected for its device
    (each called with the lane); the gradient's host copy is allocated on its first
    D2H (`pin`: pinned)."""

    grad_host: Tensor | None
    shadow: Tensor
    d2h: object
    h2d: object
    group: dict
    ready: object = None
    step: object = None
    found: Tensor | None = None
    pin: bool = False


class _Scratch:
    """One reusable fp32 host buffer: a 16-bit host gradient converted for the host
    AdamW without a new allocation per step."""

    def __init__(self):
        self.flat = torch.empty(0, dtype=torch.float32)

    def fp32(self, t: Tensor) -> Tensor:
        if t.dtype == torch.float32:
            return t
        if self.flat.numel() < t.numel():
            self.flat = torch.empty(t.numel(), dtype=torch.float32)
        return self.flat[: t.numel()].view(t.shape).copy_(t)


def _grad_host(p: Tensor, lane: _Lane) -> Tensor:
    """The lane's host gradient buffer (allocated on first use)."""
    if lane.grad_host is None:
        lane.grad_host = torch.empty(p.shape, dtype=p.dtype, pin_memory=lane.pin)
    return lane.grad_host


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
        self.stats = {"bytes_h2d": 0, "bytes_d2h": 0}
        self._streams: dict[torch.device, tuple] = {}
        self._lanes: dict[int, _Lane] = {}
        self._worker: HostWorker | None = None
        self._scratch = {"main": _Scratch(), "worker": _Scratch()}
        for group in self.param_groups:
            for p in group["params"]:
                self._init(p, group)

    def _init(self, p: Tensor, group: dict) -> None:
        side = streamed(p.device)
        pin = self.pinned and side
        master = p.detach().to("cpu", torch.float32).clone()
        self.state[p].update(
            master=master,
            exp_avg=torch.zeros_like(master),
            exp_avg_sq=torch.zeros_like(master),
            step=torch.zeros((), dtype=torch.float32),
        )
        shadow = torch.empty(p.shape, dtype=p.dtype, pin_memory=pin)
        lane = _Lane(None, shadow, None, None, group, pin=pin)
        if side:
            if p.device not in self._streams:
                self._streams[p.device] = (
                    module(p.device).Stream(p.device),
                    module(p.device).Stream(p.device),
                )
            d2h, h2d = self._streams[p.device]
            lane.d2h = partial(_d2h_device, stream=d2h)
            lane.h2d = partial(_h2d_device, stream=h2d)
            lane.found = torch.zeros((), dtype=torch.float32, pin_memory=pin)
            lane.step = partial(CPUAdamW._step_async, d2h=d2h, h2d=h2d)
            if self._worker is None:
                self._worker = HostWorker("kohakumoss-cpu-adamw")
        else:
            lane.d2h, lane.h2d, lane.step = _d2h_host, _h2d_host, CPUAdamW._step_now
        lane.ready = _has_grad
        self._lanes[id(p)] = lane

    def use_host_grads(self, offload) -> None:
        """Read the gradients of the offloaded parameters from a GradOffload's host fp32
        accumulators (no D2H here; stepped when their sum holds this step's
        gradients); other parameters keep their own lanes."""
        for group in self.param_groups:
            for p in group["params"]:
                if p not in offload.host:
                    continue
                lane = self._lanes[id(p)]
                lane.grad_host = offload.host[p]
                lane.d2h = _already_on_host
                lane.ready = partial(_filled, offload.lanes[id(p)])

    def _update(self, p: Tensor, lane: _Lane, ev) -> None:
        self._compute(p, lane, ev, self._scratch["main"])
        lane.h2d(p, lane)
        self.stats["bytes_h2d"] += p.nbytes

    def _fetch(self, p: Tensor, lane: _Lane):
        """`lane.d2h(p, lane)`, counting the copied gradient bytes."""
        if lane.d2h is not _already_on_host:
            self.stats["bytes_d2h"] += p.nbytes
        return lane.d2h(p, lane)

    def _compute(self, p: Tensor, lane: _Lane, ev, scratch: _Scratch) -> None:
        """The host AdamW update of p (after its gradient copy `ev`) into the shadow;
        a 16-bit gradient is converted into `scratch`."""
        st, group = self.state[p], lane.group
        ev.synchronize()
        b1, b2 = group["betas"]
        adamw_host(
            st["master"],
            scratch.fp32(lane.grad_host),
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

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        work = []
        for group in self.param_groups:
            for p in group["params"]:
                lane = self._lanes[id(p)]
                if lane.ready(p):
                    work.append((p, lane, self._fetch(p, lane)))
        for p, lane, ev in work:
            self._update(p, lane, ev)
        return loss

    @torch.no_grad()
    def step_param(self, p: Tensor, found_inf: Tensor | None = None) -> None:
        """Update one parameter (its grad must be set; it may be dropped on return);
        skipped when `found_inf` (a 0-d tensor) is nonzero. Device parameters update on
        a host worker: `finish()` before the parameter is read again."""
        lane = self._lanes[id(p)]
        lane.step(self, p, found_inf, lane)

    def _step_now(self, p: Tensor, found_inf: Tensor | None, lane: _Lane) -> None:
        ev = self._fetch(p, lane)
        if found_inf is not None and bool(found_inf.item()):
            return
        self._update(p, lane, ev)

    def _step_async(
        self, p: Tensor, found_inf: Tensor | None, lane: _Lane, d2h, h2d
    ) -> None:
        dm = module(p.device)
        host = _grad_host(p, lane)
        d2h.wait_stream(dm.current_stream(p.device))
        with dm.stream(d2h):
            host.copy_(p.grad, non_blocking=True)
            if found_inf is None:
                lane.found.zero_()
            else:
                lane.found.copy_(found_inf, non_blocking=True)
                found_inf.record_stream(d2h)
        p.grad.record_stream(d2h)
        self.stats["bytes_d2h"] += p.grad.nbytes
        ev = dm.Event()
        ev.record(d2h)
        self._worker.put(partial(self._update_async, p, lane, ev, h2d))

    def _update_async(self, p: Tensor, lane: _Lane, ev, h2d) -> None:
        ev.synchronize()
        if lane.found.item():
            return
        self._compute(p, lane, DONE, self._scratch["worker"])
        dm = module(p.device)
        h2d.wait_event(ev)
        with dm.stream(h2d):
            p.data.copy_(lane.shadow, non_blocking=True)
        self.stats["bytes_h2d"] += p.nbytes

    def finish(self) -> None:
        """Every queued update has run and the current stream waits for its copy back."""
        if self._worker is None:
            return
        self._worker.join()
        for device, (_, h2d) in self._streams.items():
            module(device).current_stream(device).wait_stream(h2d)

    def state_dict(self) -> dict:
        if self._worker is not None:
            self._worker.join()
        return super().state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """torch's load with the masters and moments kept as host fp32 tensors (torch
        casts state to each parameter's device and dtype) and every lane bound to its
        new group dict."""
        if self._worker is not None:
            self._worker.join()
        super().load_state_dict(state_dict)
        ids = [i for g in state_dict["param_groups"] for i in g["params"]]
        params = [p for g in self.param_groups for p in g["params"]]
        for i, p in zip(ids, params, strict=True):
            saved = state_dict["state"].get(i, {})
            for key in HOST_STATE:
                if key in saved:
                    host = saved[key].detach().to("cpu", torch.float32)
                    self.state[p][key] = host.clone()
        for group in self.param_groups:
            for p in group["params"]:
                self._lanes[id(p)].group = group

    def close(self) -> None:
        """Stop the host worker thread (after its queued updates)."""
        if self._worker is not None:
            self._worker.close()


def _d2h_host(p: Tensor, lane: _Lane):
    _grad_host(p, lane).copy_(p.grad)
    return DONE


def _has_grad(p: Tensor) -> bool:
    return p.grad is not None


def _filled(sum_lane, p: Tensor) -> bool:
    return sum_lane.filled


def _already_on_host(p: Tensor, lane: _Lane):
    return DONE


def _h2d_host(p: Tensor, lane: _Lane) -> None:
    p.data.copy_(lane.shadow)


def _d2h_device(p: Tensor, lane: _Lane, stream):
    dm = module(p.device)
    host = _grad_host(p, lane)
    stream.wait_stream(dm.current_stream(p.device))
    with dm.stream(stream):
        host.copy_(p.grad, non_blocking=True)
    ev = dm.Event()
    ev.record(stream)
    return ev


def _h2d_device(p: Tensor, lane: _Lane, stream) -> None:
    dm = module(p.device)
    with dm.stream(stream):
        p.data.copy_(lane.shadow, non_blocking=True)
    dm.current_stream(p.device).wait_stream(stream)
