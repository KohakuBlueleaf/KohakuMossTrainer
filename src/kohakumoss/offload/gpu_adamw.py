"""GPU AdamW with fp32 masters: resident states, or states streamed from host memory.

`MasterAdamW` keeps fp32 master / exp_avg / exp_avg_sq on the device. `HostStateAdamW`
keeps them in pinned host memory and streams each parameter's state through a 3-slot
device ring (H2D, update and D2H each on a side stream), so the update still runs on
the GPU. A slot is reused only after its previous write-back, and a parameter's state
is loaded only after its own last write-back; the next parameter's state is prefetched
in the order seen in the previous step. Both expose `step()` and `step_param(p)` (fused
per-parameter step) and take torch-style parameter groups: each group's lr / betas /
eps / weight_decay apply to its parameters, and `param_groups` holds the live dicts a
schedule edits.
"""

from collections.abc import Iterable

import torch
from torch import Tensor, nn


def fused_adamw(master, grad, m, v, step, group) -> None:
    """One AdamW update (`torch._fused_adamw_`) of fp32 tensors; bumps `step`."""
    step += 1
    b1, b2 = group["betas"]
    torch._fused_adamw_(
        [master],
        [grad],
        [m],
        [v],
        [],
        [step],
        lr=group["lr"],
        beta1=b1,
        beta2=b2,
        weight_decay=group["weight_decay"],
        eps=group["eps"],
        amsgrad=False,
        maximize=False,
    )


def make_groups(params, defaults: dict) -> list[dict]:
    """torch-style groups from an iterable of parameters or of dicts with `params`;
    missing hyperparameters come from `defaults`, extra keys are kept, frozen
    parameters are dropped."""
    items = list(params)
    if not items or not isinstance(items[0], dict):
        items = [{"params": items}]
    groups = []
    for item in items:
        group = {**defaults, **item}
        group["params"] = [p for p in item["params"] if p.requires_grad]
        groups.append(group)
    return groups


class _GroupsMixin:
    def _groups(self, params, lr, betas, eps, weight_decay):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.param_groups = make_groups(params, defaults)
        self.params = [p for g in self.param_groups for p in g["params"]]
        self.group_of = {id(p): g for g in self.param_groups for p in g["params"]}
        if len(self.group_of) != len(self.params):
            raise ValueError("a parameter appears in more than one group")

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            p.grad = None


class MasterAdamW(_GroupsMixin):
    """AdamW on device fp32 masters; the parameter gets the master cast to its dtype."""

    def __init__(
        self,
        params: Iterable[nn.Parameter | dict],
        lr: float = 1e-4,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self._groups(params, lr, betas, eps, weight_decay)
        self.state: dict[int, tuple] = {}
        for p in self.params:
            master = p.detach().float().clone()
            self.state[id(p)] = (
                master,
                torch.zeros_like(master),
                torch.zeros_like(master),
                torch.zeros((), dtype=torch.float32, device=p.device),
            )

    @torch.no_grad()
    def step_param(self, p: Tensor) -> None:
        master, m, v, step = self.state[id(p)]
        fused_adamw(master, p.grad.float(), m, v, step, self.group_of[id(p)])
        p.copy_(master)

    @torch.no_grad()
    def step(self) -> None:
        for p in self.params:
            if p.grad is not None:
                self.step_param(p)

    def state_dict(self) -> dict:
        return {i: self.state[id(p)] for i, p in enumerate(self.params)}

    def load_state_dict(self, state: dict) -> None:
        for i, p in enumerate(self.params):
            for dst, src in zip(self.state[id(p)], state[i]):
                dst.copy_(src)


class HostStateAdamW(_GroupsMixin):
    """AdamW computed on the device with master/moments held in pinned host memory."""

    def __init__(
        self,
        params: Iterable[nn.Parameter | dict],
        lr: float = 1e-4,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self._groups(params, lr, betas, eps, weight_decay)
        self.host: dict[int, tuple] = {}
        self.steps: dict[int, int] = {}
        largest = max((p.numel() for p in self.params), default=0)
        self.device = self.params[0].device if self.params else torch.device("cpu")
        cuda = self.device.type == "cuda"
        for p in self.params:
            master = p.detach().float().cpu()
            pin = master.pin_memory() if cuda else master
            zeros = [torch.zeros(pin.shape, pin_memory=cuda) for _ in range(2)]
            self.host[id(p)] = (pin, *zeros)
            self.steps[id(p)] = 0
        self.n_slots = 3
        self.slots = [
            torch.empty(3, largest, dtype=torch.float32, device=self.device)
            for _ in range(self.n_slots)
        ]
        self.next_slot = 0
        self.loaded: dict[int, int] = {}
        order = list(reversed(self.params))
        self.after = {id(a): b for a, b in zip(order, order[1:])}
        self.last: Tensor | None = None
        self.stats = {"bytes_h2d": 0, "bytes_d2h": 0}
        if cuda:
            self.h2d = torch.cuda.Stream(self.device)
            self.d2h = torch.cuda.Stream(self.device)
            self.opt = torch.cuda.Stream(self.device)
            cur = torch.cuda.current_stream(self.device)
            self.free = [torch.cuda.Event() for _ in range(self.n_slots)]
            self.ready = [torch.cuda.Event() for _ in range(self.n_slots)]
            for ev in self.free:
                ev.record(cur)
            self.written: dict[int, torch.cuda.Event] = {}
            for p in self.params:
                self.written[id(p)] = torch.cuda.Event()
                self.written[id(p)].record(cur)
            self._update = self._update_device
        else:
            self._update = self._update_host

    def _update_host(self, p: Tensor, ahead: Tensor | None) -> None:
        master, m, v = self.host[id(p)]
        step = torch.tensor(float(self.steps[id(p)]))
        fused_adamw(
            master,
            p.grad.float().reshape(master.shape),
            m,
            v,
            step,
            self.group_of[id(p)],
        )
        self.steps[id(p)] = int(step.item())
        p.copy_(master.reshape(p.shape))

    def _bufs(self, slot: int, p: Tensor) -> list[Tensor]:
        n = p.numel()
        return [self.slots[slot][k, :n].view(p.shape) for k in range(3)]

    def _load(self, p: Tensor) -> int:
        """Start the H2D of p's state into the next ring slot (side stream)."""
        slot = self.next_slot
        self.next_slot = (slot + 1) % self.n_slots
        for pid, s in list(self.loaded.items()):
            if s == slot:
                del self.loaded[pid]
        self.loaded[id(p)] = slot
        self.h2d.wait_event(self.free[slot])
        self.h2d.wait_event(self.written[id(p)])
        with torch.cuda.stream(self.h2d):
            for dst, src in zip(self._bufs(slot, p), self.host[id(p)]):
                dst.copy_(src.view(p.shape), non_blocking=True)
        self.ready[slot].record(self.h2d)
        self.stats["bytes_h2d"] += 12 * p.numel()
        return slot

    def prefetch(self, p: Tensor | None) -> None:
        if p is not None and id(p) not in self.loaded:
            self._load(p)

    def _update_device(self, p: Tensor, ahead: Tensor | None) -> None:
        slot = self.loaded.pop(id(p), None)
        if slot is None:
            slot = self._load(p)
            self.loaded.pop(id(p))
        bufs = self._bufs(slot, p)
        g = p.grad
        self.opt.wait_stream(torch.cuda.current_stream(self.device))
        self.opt.wait_event(self.ready[slot])
        with torch.cuda.stream(self.opt):
            step = torch.full((), float(self.steps[id(p)]), device=self.device)
            fused_adamw(
                bufs[0], g.float(), bufs[1], bufs[2], step, self.group_of[id(p)]
            )
            p.copy_(bufs[0])
        g.record_stream(self.opt)
        self.steps[id(p)] += 1
        self.d2h.wait_stream(self.opt)
        with torch.cuda.stream(self.d2h):
            for dst, src in zip(self.host[id(p)], bufs):
                dst.view(p.shape).copy_(src, non_blocking=True)
        self.free[slot].record(self.d2h)
        self.written[id(p)].record(self.d2h)
        self.stats["bytes_d2h"] += 12 * p.numel()
        self.prefetch(ahead)

    def finish(self) -> None:
        """The compute stream waits for every in-flight update (before the next forward)."""
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).wait_stream(self.opt)
        self.loaded.clear()
        self.last = None

    @torch.no_grad()
    def step_param(self, p: Tensor) -> None:
        """In-backward update; prefetches the parameter expected next: the one that
        followed p in the previous step (first step: reverse registration order)."""
        if self.last is not None:
            self.after[id(self.last)] = p
        self.last = p
        self._update(p, self.after.get(id(p)))

    @torch.no_grad()
    def step(self) -> None:
        live = [p for p in self.params if p.grad is not None]
        for p, ahead in zip(live, live[1:] + [None]):
            self._update(p, ahead)
        self.finish()

    def synchronize(self) -> None:
        """Wait until every state write-back has landed on the host."""
        if self.device.type == "cuda":
            self.d2h.synchronize()

    def state_dict(self) -> dict:
        self.synchronize()
        return {
            i: (*self.host[id(p)], self.steps[id(p)]) for i, p in enumerate(self.params)
        }

    def load_state_dict(self, state: dict) -> None:
        for i, p in enumerate(self.params):
            *tensors, step = state[i]
            for dst, src in zip(self.host[id(p)], tensors):
                dst.copy_(src)
            self.steps[id(p)] = step
