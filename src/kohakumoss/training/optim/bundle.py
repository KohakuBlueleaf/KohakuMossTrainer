"""OptimBundle: optimiser + schedule + scaler + clipping as one object.

`step()` makes the only host sync of a training step: the fp32 grad norm, the loss and
the agreed non-finite flag come back in one `.tolist()`.
"""

from collections.abc import Callable

import torch
from torch import Tensor
from torch.optim import Optimizer


def local_grad_norm(grads: list[Tensor]) -> Tensor:
    """fp32 L2 norm of all grads (replicated grads: identical on every rank)."""
    if not grads:
        return torch.zeros((), dtype=torch.float32)
    norms = torch._foreach_norm(grads, 2.0)
    dev = grads[0].device
    return torch.linalg.vector_norm(
        torch.stack([n.float().to(dev) for n in norms]), 2.0
    )


def all_reduce_norm(grads: list[Tensor]) -> Tensor:
    """Global L2 norm when every rank holds a disjoint parameter subset (pipeline)."""
    sq = local_grad_norm(grads).square().reshape(1)
    torch.distributed.all_reduce(sq, op=torch.distributed.ReduceOp.SUM)
    return sq.sqrt()[0]


def no_agree(flag: Tensor) -> Tensor:
    return flag


def all_reduce_max(flag: Tensor) -> Tensor:
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    return flag


def _clip_off(grads: list[Tensor], norm: Tensor) -> None:
    return None


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
        return _clip_off

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
        return _noop

    @torch.no_grad()
    def apply() -> None:
        for live, coef, params in targets:
            k = live["lr"] * coef
            for p in params:
                p.sub_(p - 1.0, alpha=k)

    return apply


def _noop() -> None:
    return None


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
        agree: Callable[[Tensor], Tensor] = no_agree,
        train_mode: Callable[[], None] = _noop,
        eval_mode: Callable[[], None] = _noop,
        before: Callable[[], None] = _noop,
        after: Callable[[], None] = _noop,
        reduce_loss: Callable[[Tensor], Tensor] = no_agree,
        grads: Callable[[], list[Tensor]] | None = None,
        device: torch.device | None = None,
    ):
        self.optimizer = optimizer
        self.schedule = schedule
        self.scaler = scaler
        self.params = [p for g in groups for p in g["params"]]
        self._grads = grads or self._device_grads
        self.clip = make_clipper(
            grad_clip, len({p.device for p in self.params}) > 1 or grads is not None
        )
        self.grad_norm = grad_norm
        self.agree = agree
        self.before = before
        self.after = after
        self.reduce_loss = reduce_loss
        self.post_step = (
            _noop
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

    def step(self) -> bool:
        self.before()
        grads = self._grads()
        self.scaler.unscale_(grads)
        norm = self.grad_norm(grads).to(self.device)
        loss = self.reduce_loss(self._loss.reshape(1).clone())
        bad = (~torch.isfinite(norm) | ~torch.isfinite(loss)).to(torch.int32).reshape(1)
        bad = self.agree(bad)
        loss_v, norm_v, bad_v = torch.cat(
            [loss.reshape(1).float(), norm.reshape(1).float(), bad.float()]
        ).tolist()
        ok = bad_v == 0
        if ok:
            self.clip(grads, norm)
            self.optimizer.step()
            self.post_step()
        else:
            self.skipped += 1
            self.nonfinite += 1
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
