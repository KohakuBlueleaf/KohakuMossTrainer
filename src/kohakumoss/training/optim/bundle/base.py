"""OptimBundle: optimiser + schedule + scaler + clipping as one object.

`step()` makes the only host sync of a training step: the fp32 grad norm, the loss and
the agreed non-finite flag come back in one `.tolist()`.
"""

from collections.abc import Callable
from functools import partial

import torch
from torch import Tensor
from torch.optim import Optimizer

from kohakumoss.training.optim.bundle.nonfinite import (
    grad_report,
    nonfinite_count,
    param_names,
)
from kohakumoss_common.calls import identity, nothing


def local_grad_norm(grads: list[Tensor]) -> Tensor:
    """fp32 L2 norm of all grads (replicated grads: identical on every rank)."""
    if not grads:
        return torch.zeros((), dtype=torch.float32)
    norms = torch._foreach_norm(grads, 2.0, dtype=torch.float32)
    dev = grads[0].device
    return torch.linalg.vector_norm(torch.stack([n.to(dev) for n in norms]), 2.0)


def all_reduce_norm(grads: list[Tensor]) -> Tensor:
    """Global L2 norm when every rank holds a disjoint parameter subset (pipeline)."""
    sq = local_grad_norm(grads).square().reshape(1)
    torch.distributed.all_reduce(sq, op=torch.distributed.ReduceOp.SUM)
    return sq.sqrt()[0]


def all_reduce_max(flag: Tensor) -> Tensor:
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    return flag


def separate_reduce(
    reduce_loss: Callable, agree: Callable, loss: Tensor, bad: Tensor
) -> tuple[Tensor, Tensor]:
    """(reduced loss, agreed int32 flag) through the two hooks, one call each."""
    return reduce_loss(loss), agree(bad)


def step_reducer(reduce_loss: Callable, agree: Callable, reduce_step=None) -> Callable:
    """`(loss, bad) -> (loss, bad)` of a step: `reduce_step` when the parallel layout
    folds both into one collective, else `reduce_loss` and `agree`."""
    return reduce_step or partial(separate_reduce, reduce_loss, agree)


def _by_device(grads: list[Tensor]) -> dict:
    parts: dict = {}
    for g in grads:
        parts.setdefault((g.device, g.dtype), []).append(g)
    return parts


def make_clipper(
    max_norm: float, multi_device: bool = False
) -> Callable[[list[Tensor], Tensor], None]:
    """Clip by global norm without a host sync (coefficient stays on device).
    `multi_device`: grads span devices (threaded pipeline), one coefficient copy each.
    """
    if max_norm <= 0:
        return nothing

    def clip(grads: list[Tensor], norm: Tensor) -> None:
        coef = (max_norm / (norm + 1e-6)).clamp(max=1.0)
        torch._foreach_mul_(grads, coef.to(grads[0].device, grads[0].dtype))

    def clip_spread(grads: list[Tensor], norm: Tensor) -> None:
        coef = (max_norm / (norm + 1e-6)).clamp(max=1.0)
        for (dev, dtype), part in _by_device(grads).items():
            torch._foreach_mul_(part, coef.to(dev, dtype))

    return clip_spread if multi_device else clip


def decay_to_one(groups: list[dict], optimizer: Optimizer) -> Callable[[], None]:
    """p -= lr * coef * (p - 1) on decay-to-one groups, lr read from the optimiser's
    matching group (same order as `groups`); a no-op when there are none."""
    targets = [
        (optimizer.param_groups[i], g["decay_to_one_coef"], g["params"])
        for i, g in enumerate(groups)
        if g.get("decay_to_one")
    ]
    if not targets:
        return nothing

    @torch.no_grad()
    def apply() -> None:
        for live, coef, params in targets:
            k = live["lr"] * coef
            for p in params:
                p.sub_(p - 1.0, alpha=k)

    return apply


def _given(grads: Callable[[], list[Tensor]], bundle) -> list[Tensor]:
    return grads()


def close_optimizers(*opts) -> None:
    """Stop the host worker threads of the optimisers among `opts` that run them."""
    for opt in opts:
        close = getattr(opt, "close", None)
        if close is not None:
            close()


class Bundle:
    """The OptimBundle used by both drivers."""

    def __init__(
        self,
        optimizer: Optimizer,
        schedule,
        scaler,
        groups: list[dict],
        grad_clip: float,
        grad_norm: Callable[[list[Tensor]], Tensor] = local_grad_norm,
        agree: Callable[[Tensor], Tensor] = identity,
        train_mode: Callable[[], None] = nothing,
        eval_mode: Callable[[], None] = nothing,
        before: Callable[[], None] = nothing,
        after: Callable[[], None] = nothing,
        reduce_loss: Callable[[Tensor], Tensor] = identity,
        grads: Callable[[], list[Tensor]] | None = None,
        device: torch.device | None = None,
        reduce_step: Callable | None = None,
        grad_names: Callable[[], list[str]] | None = None,
    ):
        self.optimizer = optimizer
        self.shared = None
        self.blocks = None
        self.schedule = schedule
        self.scaler = scaler
        self.params = [p for g in groups for p in g["params"]]
        self.names = param_names(groups)
        self.nonfinite_grads: list[dict] = []
        if grads is None:
            self._report, self._grads = Bundle._device_report, Bundle._device_grads
        else:
            self._report, self._grads = Bundle._host_report, partial(_given, grads)
        self._grad_names = grad_names
        self.clip = make_clipper(
            grad_clip, len({p.device for p in self.params}) > 1 or grads is not None
        )
        self.grad_norm = grad_norm
        self.reduce = step_reducer(reduce_loss, agree, reduce_step)
        self.before = before
        self.after = after
        self.post_step = (
            nothing
            if getattr(optimizer, "handles_decay_to_one", False)
            else decay_to_one(groups, optimizer)
        )
        self.train_mode = train_mode
        self.eval_mode = eval_mode
        self.device = device if device is not None else self.params[0].device
        self._loss = torch.zeros((), device=self.device)
        self.last_loss = float("nan")
        self.last_norm = float("nan")
        self.last_ok = True
        self.last_nonfinite_params = 0
        self.skipped = 0
        self.nonfinite = 0

    def backward(self, loss: Tensor) -> None:
        """Accumulate one micro-batch: scaled backward, detached loss kept for logging."""
        self.scaler.scale_loss(loss).backward()
        self._loss = self._loss + loss.detach().float()

    def observe(self, loss: Tensor) -> None:
        """Record an (unscaled) loss whose backward an engine already ran."""
        self._loss = self._loss + loss.detach().float().to(self.device)

    def _device_grads(self) -> list[Tensor]:
        return [p.grad for p in self.params if p.grad is not None]

    def _device_report(self, grads: list[Tensor]) -> list[dict]:
        return grad_report([(n, p.grad) for n, p in zip(self.names, self.params)])

    def _host_report(self, grads: list[Tensor]) -> list[dict]:
        """Host gradients reported by their parameter names (`grad_names`)."""
        names = self._grad_names() if self._grad_names is not None else []
        names += [f"host_grad[{i}]" for i in range(len(names), len(grads))]
        return grad_report(list(zip(names, grads)))

    def step(self) -> bool:
        self.before()
        grads = self._grads(self)
        self.scaler.unscale_(grads)
        norm = self.grad_norm(grads).to(self.device)
        local = (~torch.isfinite(norm)).to(torch.int32).reshape(1)
        loss, bad = self.reduce(self._loss.reshape(1).clone(), local)
        bad = (bad > 0) | ~torch.isfinite(loss.reshape(1))
        loss_v, norm_v, bad_v = torch.cat(
            [loss.reshape(1).float(), norm.reshape(1).float(), bad.float()]
        ).tolist()
        ok = bad_v == 0
        self.nonfinite_grads = []
        self.last_nonfinite_params = 0
        if ok:
            self.clip(grads, norm)
            self.optimizer.step()
            self.post_step()
        else:
            self.skipped += 1
            self.nonfinite += 1
            self.nonfinite_grads = self._report(self, grads)
            self.last_nonfinite_params = nonfinite_count(grads)
        self.after()
        self.schedule.step()
        self.scaler.update(not ok)
        self.optimizer.zero_grad(set_to_none=True)
        self._loss = torch.zeros((), device=self.device)
        self.last_loss, self.last_norm, self.last_ok = loss_v, norm_v, ok
        return ok

    def step_param(self, param: torch.nn.Parameter) -> None:
        raise RuntimeError("fused per-parameter steps need FusedStepBundle")

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def close(self) -> None:
        close_optimizers(self.optimizer)

    @property
    def lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def state_dict(self) -> dict:
        return {
            "optimizer": self.optimizer.state_dict(),
            "schedule": self.schedule.state_dict(),
            "scaler": self.scaler.state_dict(),
            "skipped": self.skipped,
            "nonfinite": self.nonfinite,
        }

    def load_state_dict(self, state: dict) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        self.schedule.load_state_dict(state["schedule"])
        self.scaler.load_state_dict(state["scaler"])
        self.skipped = state.get("skipped", 0)
        self.nonfinite = state.get("nonfinite", self.skipped)
