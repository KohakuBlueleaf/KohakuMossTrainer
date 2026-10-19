"""GPU AdamW on fp32 masters: states resident (`MasterAdamW`) or streamed from pinned
host memory per parameter (`HostStateAdamW`). See docs/internals/offload.md."""

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from kohakumoss.offload.accel import module, streamed

RING_SLOTS = 3


def state_ring_bytes(largest: int) -> int:
    """Device bytes of HostStateAdamW's state ring: RING_SLOTS slots of 4 fp32 rows
    (master, m, v, gradient) of the largest parameter."""
    return RING_SLOTS * 4 * 4 * largest


def fused_adamw(master, grad, m, v, step, group, found_inf=None) -> None:
    """One AdamW update (`torch._fused_adamw_`) of fp32 tensors; bumps `step`. A
    nonzero `found_inf` (0-d fp32 tensor) leaves every tensor unchanged. Groups with
    `decay_to_one` decay the master towards 1 by `lr * decay_to_one_coef`."""
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
        grad_scale=None,
        found_inf=found_inf,
    )
    if found_inf is not None:
        step -= found_inf
    if group.get("decay_to_one"):
        k = group["lr"] * group["decay_to_one_coef"]
        keep = 1.0 if found_inf is None else 1.0 - found_inf
        master.sub_((master - 1.0) * (k * keep))


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
    handles_decay_to_one = True

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
    def step_param(self, p: Tensor, found_inf: Tensor | None = None) -> None:
        master, m, v, step = self.state[id(p)]
        fused_adamw(master, p.grad.float(), m, v, step, self.group_of[id(p)], found_inf)
        p.copy_(master)

    @torch.no_grad()
    def step(self) -> None:
        for p in self.params:
            if p.grad is not None:
                self.step_param(p)

    def finish(self) -> None:
        return None

    def state_dict(self) -> dict:
        return {i: self.state[id(p)] for i, p in enumerate(self.params)}

    def load_state_dict(self, state: dict) -> None:
        for i, p in enumerate(self.params):
            for dst, src in zip(self.state[id(p)], state[i]):
                dst.copy_(src)
            p.data.copy_(self.state[id(p)][0])


class HostStateAdamW(_GroupsMixin):
    """AdamW computed on the device with master / moments in (pinned) host memory,
    streamed per parameter through a 3-slot device ring on side streams."""

    def __init__(
        self,
        params: Iterable[nn.Parameter | dict],
        lr: float = 1e-4,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        pinned: bool = True,
    ):
        self._groups(params, lr, betas, eps, weight_decay)
        self.host: dict[int, tuple] = {}
        largest = max((p.numel() for p in self.params), default=0)
        self.device = self.params[0].device if self.params else torch.device("cpu")
        side = streamed(self.device)
        self.streamed = side
        self.dm = module(self.device)
        pin = pinned and side
        step_device = self.device if side else torch.device("cpu")
        self.steps = {
            id(p): torch.zeros((), dtype=torch.float32, device=step_device)
            for p in self.params
        }
        for p in self.params:
            master = p.detach().float().cpu()
            master = master.pin_memory() if pin else master.clone()
            zeros = [torch.zeros(master.shape, pin_memory=pin) for _ in range(2)]
            self.host[id(p)] = (master, *zeros)
        self.n_slots = RING_SLOTS
        # rows: master, m, v, the fp32 gradient
        self.slots = [
            torch.empty(4, largest, dtype=torch.float32, device=self.device)
            for _ in range(self.n_slots)
        ]
        self.next_slot = 0
        self.loaded: dict[int, int] = {}
        order = list(reversed(self.params))
        self.after = {id(a): b for a, b in zip(order, order[1:])}
        self.last: Tensor | None = None
        self.stats = {"bytes_h2d": 0, "bytes_d2h": 0}
        if side:
            self.h2d = self.dm.Stream(self.device)
            self.d2h = self.dm.Stream(self.device)
            self.opt = self.dm.Stream(self.device)
            cur = self.dm.current_stream(self.device)
            self.free = [self.dm.Event() for _ in range(self.n_slots)]
            self.ready = [self.dm.Event() for _ in range(self.n_slots)]
            for ev in self.free:
                ev.record(cur)
            self.written: dict[int, object] = {}
            for p in self.params:
                self.written[id(p)] = self.dm.Event()
                self.written[id(p)].record(cur)
            self.no_inf = torch.zeros((), dtype=torch.float32, device=self.device)
            self._update = HostStateAdamW._update_device
        else:
            self._update = HostStateAdamW._update_host

    def _update_host(self, p: Tensor, ahead: Tensor | None, found_inf) -> None:
        master, m, v = self.host[id(p)]
        grad = p.grad.float().reshape(master.shape).to(master.device)
        found = None if found_inf is None else found_inf.to(master.device)
        step = self.steps[id(p)]
        fused_adamw(master, grad, m, v, step, self.group_of[id(p)], found)
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
        with self.dm.stream(self.h2d):
            for dst, src in zip(self._bufs(slot, p), self.host[id(p)]):
                dst.copy_(src.view(p.shape), non_blocking=True)
        self.ready[slot].record(self.h2d)
        self.stats["bytes_h2d"] += 12 * p.numel()
        return slot

    def prefetch(self, p: Tensor | None) -> None:
        if p is not None and id(p) not in self.loaded:
            self._load(p)

    def _update_device(self, p: Tensor, ahead: Tensor | None, found_inf) -> None:
        slot = self.loaded.pop(id(p), None)
        if slot is None:
            slot = self._load(p)
            self.loaded.pop(id(p))
        bufs = self._bufs(slot, p)
        g = p.grad
        g32 = self.slots[slot][3, : p.numel()].view(p.shape)
        found = self.no_inf if found_inf is None else found_inf
        self.opt.wait_stream(self.dm.current_stream(self.device))
        self.opt.wait_event(self.ready[slot])
        with self.dm.stream(self.opt):
            step = self.steps[id(p)]
            g32.copy_(g)
            fused_adamw(
                bufs[0], g32, bufs[1], bufs[2], step, self.group_of[id(p)], found
            )
            p.copy_(bufs[0])
        # mark the caller's tensors as used on the side stream
        g.record_stream(self.opt)
        found.record_stream(self.opt)
        self.d2h.wait_stream(self.opt)
        with self.dm.stream(self.d2h):
            for dst, src in zip(self.host[id(p)], bufs):
                dst.view(p.shape).copy_(src, non_blocking=True)
        self.free[slot].record(self.d2h)
        self.written[id(p)].record(self.d2h)
        self.stats["bytes_d2h"] += 12 * p.numel()
        self.prefetch(ahead)

    def finish(self) -> None:
        """The compute stream waits for every in-flight update (before the next forward)."""
        if self.streamed:
            self.dm.current_stream(self.device).wait_stream(self.opt)
        self.loaded.clear()
        self.last = None

    @torch.no_grad()
    def step_param(self, p: Tensor, found_inf: Tensor | None = None) -> None:
        """In-backward update; prefetches the parameter that followed p last step."""
        if self.last is not None:
            self.after[id(self.last)] = p
        self.last = p
        self._update(self, p, self.after.get(id(p)), found_inf)

    @torch.no_grad()
    def step(self) -> None:
        live = [p for p in self.params if p.grad is not None]
        for p, ahead in zip(live, live[1:] + [None]):
            self._update(self, p, ahead, None)
        self.finish()

    def synchronize(self) -> None:
        """Wait until every state write-back has landed on the host."""
        if self.streamed:
            self.d2h.synchronize()

    def state_dict(self) -> dict:
        self.synchronize()
        return {
            i: (*self.host[id(p)], self.steps[id(p)].clone())
            for i, p in enumerate(self.params)
        }

    def load_state_dict(self, state: dict) -> None:
        for i, p in enumerate(self.params):
            *tensors, step = state[i]
            for dst, src in zip(self.host[id(p)], tensors):
                dst.copy_(src)
            self.steps[id(p)].fill_(float(step))
            p.data.copy_(self.host[id(p)][0].view(p.shape))
