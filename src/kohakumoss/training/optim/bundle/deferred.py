"""STEP_SYNC='deferred': the optimiser step without a per-step host sync. The step's
non-finite flag stays on the device and gates the optimiser update (torch's fused AdamW
`found_inf`, or `step_masked`), the decay-to-one, the loss-scale update and the skip
counters; loss, norm, skip counts and the scale come to the host in one transfer at
`sync()`, every `every` steps (and at the first / last step and on `state_dict`). See
docs/concepts/training.md#deferred-step."""

from math import gcd

import torch
from torch import Tensor
from torch.optim import Optimizer

from kohakumoss.precision.scaler import DeviceScaler, DynamicScaler
from kohakumoss.training.optim.bundle.base import Bundle, anchor_pull

KNOB = "STEP_SYNC"
MODES = ("step", "deferred")


def gates_found_inf(optimizer: Optimizer) -> bool:
    """The optimiser skips its update on a device flag: torch fused (`found_inf`) or a
    KohakuMoss optimiser with `step_masked`."""
    return bool(getattr(optimizer, "defaults", {}).get("fused")) or hasattr(
        optimizer, "step_masked"
    )


def gated_decay_to_one(groups: list[dict], optimizer: Optimizer):
    """p -= lr * coef * (1 - bad) * (p - 1) on decay-to-one groups (`apply(bad)`), lr
    read from the optimiser's matching group; None when there are none."""
    targets = [
        (optimizer.param_groups[i], g["decay_to_one_coef"], g["params"])
        for i, g in enumerate(groups)
        if g.get("decay_to_one")
    ]
    if not targets or getattr(optimizer, "handles_decay_to_one", False):
        return None

    @torch.no_grad()
    def apply(bad: Tensor) -> None:
        keep = 1.0 - bad
        for live, coef, params in targets:
            k = keep * (live["lr"] * coef)
            for p in params:
                p.sub_((p - 1.0) * k.to(p.dtype))

    return apply


def sync_cadence(log_every: int, rows_every: int) -> int:
    """Steps between host syncs: every step the console line or a metric row reads."""
    return gcd(int(log_every), int(rows_every))


def step_sync_problems(k: dict) -> list[str]:
    """STEP_SYNC 'deferred': EMA off (its update is gated on the host flag), PARALLEL
    'none' or 'ddp'."""
    if k.get(KNOB, "step") != "deferred":
        return []
    p = []
    if float(k.get("EMA") or 0.0) > 0:
        p.append(f"{KNOB}='deferred' runs with EMA = 0 (the EMA update reads the skip)")
    if k.get("PARALLEL", "none") not in ("none", "ddp"):
        p.append(f"{KNOB}='deferred' runs with PARALLEL 'none' or 'ddp'")
    return p


class DeferredBundle(Bundle):
    """Bundle whose `step()` makes no host sync (see the module docstring); `sync_due(step)`
    reads the host values back when `step` is a multiple of `every`, the run's first
    step or `last`. Between syncs `last_*` / `skipped` hold the previous sync's values and
    `synced` is False."""

    def __init__(self, *args, groups: list[dict], every: int, last: int, **kw):
        super().__init__(*args, groups=groups, **kw)
        if not gates_found_inf(self.optimizer):
            raise ValueError(
                f"{KNOB}='deferred' needs an optimiser that skips on a device flag "
                f"(torch fused AdamW or step_masked), got {type(self.optimizer).__name__}"
            )
        if isinstance(self.scaler, DynamicScaler):
            self.scaler = DeviceScaler(self.scaler, self.device)
        self.every = max(int(every), 1)
        self.last = int(last)
        self.decay = gated_decay_to_one(groups, self.optimizer)
        self.pull = anchor_pull(groups, self.optimizer, gated=True)
        dev = self.device
        self._skipped = torch.zeros((), device=dev)
        self._bad_run = torch.zeros((), device=dev)
        self._last = torch.zeros(3, device=dev)
        self.bad_run = 0
        self.synced = True
        self.first = True

    def _step_masked(self, bad: Tensor) -> None:
        if self.optimizer.defaults.get("fused"):
            self.optimizer.found_inf = bad
            self.optimizer.step()
        else:
            self.optimizer.step_masked(bad)

    def step(self) -> bool:
        self.before()
        grads = self._grads(self)
        self.scaler.unscale_(grads)
        norm = self.grad_norm(grads).to(self.device).reshape(())
        local = (~torch.isfinite(norm)).to(torch.int32).reshape(1)
        loss, flag = self.reduce(self._loss.reshape(1).clone(), local)
        bad = ((flag > 0) | ~torch.isfinite(loss.reshape(1))).float().reshape(())
        self.clip(grads, norm)
        self._step_masked(bad)
        if self.decay is not None:
            self.decay(bad)
        if self.pull is not None:
            self.pull(bad)
        self.after()
        self.schedule.step()
        self.scaler.update_device(bad)
        self.optimizer.zero_grad(set_to_none=True)
        self._skipped += bad
        self._bad_run = (self._bad_run + 1.0) * bad
        self._last = torch.stack([loss.reshape(()).float(), norm.float(), bad])
        self._loss = torch.zeros((), device=self.device)
        self.synced = False
        return True

    def sync(self) -> None:
        """Host values of the last step: loss, norm, ok, skip counts and the scale, in
        one transfer."""
        scale = getattr(self.scaler, "scale", 1.0)
        scale = scale if isinstance(scale, Tensor) else torch.tensor(float(scale))
        loss, norm, bad, skipped, run, scale_v = torch.cat(
            [
                self._last,
                torch.stack([self._skipped, self._bad_run, scale.to(self.device)]),
            ]
        ).tolist()
        self.last_loss, self.last_norm, self.last_ok = loss, norm, bad == 0
        self.skipped = self.nonfinite = int(skipped)
        self.bad_run = int(run)
        self.host_scale = scale_v
        self.synced = True

    def sync_due(self, step: int) -> None:
        """`sync()` when `step` is a sync step (a multiple of `every`, the first step of
        this process, or the last)."""
        if self.first or step % self.every == 0 or step >= self.last:
            self.first = False
            self.sync()

    def state_dict(self) -> dict:
        self.sync()
        return super().state_dict()

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self._skipped.fill_(float(self.skipped))
