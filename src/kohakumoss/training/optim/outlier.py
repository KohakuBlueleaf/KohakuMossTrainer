"""OUTLIER_DECAY: a decoupled decay of each weight element's excess over a cap of its
tensor's start RMS, `p -= lr * rate * sign(p) * relu(|p| - cap * r0)`, after the
optimiser step, on the trainable weights it targets. See
docs/concepts/training.md#outlier-decay."""

import math
import os
import re
from collections.abc import Callable
from functools import partial

import torch
from torch import Tensor

from kohakumoss.parallel.fsdp import shard_like
from kohakumoss.training.checkpoint.io import load_trainable

KNOB = "OUTLIER_DECAY"
DEFAULTS = {
    "rate": 0.0,
    "cap": 16.0,
    "cap_start": 0.0,
    "cap_anneal": 0,
    "warmup": 0,
    "targets": "",
    "exclude": "",
    "include_1d": False,
}
OPTIMIZERS = ("adamw", "adamw8bit", "adamw8bit_kernel", "muon")


def spec_of(k: dict) -> dict | None:
    """The knob with its defaults filled, or None when off (unset or rate 0)."""
    given = k.get(KNOB) or {}
    if not isinstance(given, dict) or not given:
        return None
    spec = {**DEFAULTS, **given}
    return spec if float(spec["rate"]) > 0 else None


def _num(spec: dict, name: str, low: float, strict: bool) -> list[str]:
    v = spec.get(name)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return [f"{KNOB}[{name!r}] must be a number, got {v!r}"]
    if v < low or (strict and v == low):
        return [f"{KNOB}[{name!r}] must be {'>' if strict else '>='} {low}, got {v!r}"]
    return []


def outlier_problems(k: dict, full: bool) -> list[str]:
    """A dict of known fields: rate >= 0, cap > 0, cap_start 0 (= cap) or >= cap, ints
    cap_anneal / warmup >= 0, regex targets / exclude, a bool include_1d; when on, a full
    finetune with an optimiser in OPTIMIZERS (fp32 device weights) and no FSDP."""
    given = k.get(KNOB) or {}
    if not isinstance(given, dict):
        return [f"{KNOB} must be a dict, got {given!r}"]
    if not given:
        return []
    p = []
    unknown = sorted(set(given) - set(DEFAULTS))
    if unknown:
        p.append(f"{KNOB} has unknown fields {unknown}; valid: {sorted(DEFAULTS)}")
    spec = {**DEFAULTS, **given}
    p += _num(spec, "rate", 0.0, False) + _num(spec, "cap", 0.0, True)
    p += _num(spec, "cap_start", 0.0, False)
    if not p and 0 < spec["cap_start"] < spec["cap"]:
        p.append(
            f"{KNOB}['cap_start'] must be 0 (= cap) or >= cap, got {spec['cap_start']!r}"
        )
    for name in ("cap_anneal", "warmup"):
        v = spec[name]
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            p.append(f"{KNOB}[{name!r}] must be an int >= 0 (steps), got {v!r}")
    for name in ("targets", "exclude"):
        try:
            re.compile(spec[name])
        except (re.error, TypeError) as err:
            p.append(f"{KNOB}[{name!r}] must be a regex, got {spec[name]!r}: {err}")
    if not isinstance(spec["include_1d"], bool):
        p.append(f"{KNOB}['include_1d'] must be a bool, got {spec['include_1d']!r}")
    if p or spec_of(k) is None:
        return p
    if not full:
        p.append(f"{KNOB} decays the base weights (ALGO 'full')")
    if k.get("OPTIMIZER", "adamw") not in OPTIMIZERS:
        p.append(f"{KNOB} runs with OPTIMIZER in {OPTIMIZERS} (fp32 device weights)")
    if k.get("PARALLEL", "none") == "fsdp":
        p.append(f"{KNOB} runs without PARALLEL 'fsdp' (whole-tensor start RMS)")
    return p


def cap_at(spec: dict, t: int) -> float:
    """The cap at optimiser step `t`: `cap_start` to `cap` geometrically over
    `cap_anneal` steps (`cap` when cap_start is 0 or cap_anneal 0)."""
    cap, start, span = (
        float(spec["cap"]),
        float(spec["cap_start"]),
        int(spec["cap_anneal"]),
    )
    if start <= 0 or span <= 0 or t >= span:
        return cap
    return start * (cap / start) ** (t / span)


def rate_at(spec: dict, t: int) -> float:
    """The rate at optimiser step `t`, ramped linearly from 0 over `warmup` steps."""
    rate, warm = float(spec["rate"]), int(spec["warmup"])
    return rate if warm <= 0 else rate * min((t + 1) / warm, 1.0)


def targeted(spec: dict, name: str, p: Tensor) -> bool:
    """Whether the decay acts on parameter `name`: 2-D (any rank with include_1d),
    matching `targets` (all when empty) and not `exclude`."""
    if p.ndim < 2 and not spec["include_1d"]:
        return False
    if spec["targets"] and not re.search(spec["targets"], name):
        return False
    return not (spec["exclude"] and re.search(spec["exclude"], name))


@torch.no_grad()
def start_refs(k: dict, trainable, resumed: bool) -> dict[str, float] | None:
    """Name -> start RMS of every targeted trainable parameter (None when off): the
    live weights before a resume restores them, or on a resumed run with INIT_FROM that
    checkpoint's trainable state. One host read."""
    spec = spec_of(k)
    if spec is None:
        return None
    live = dict(trainable.named_trainable())
    names = [n for n, p in live.items() if targeted(spec, n, p)]
    if resumed and k.get("INIT_FROM"):
        init = k["INIT_FROM"]
        if not os.path.isfile(os.path.join(init, "trainable.safetensors")):
            raise SystemExit(
                f"{KNOB} on a resumed run reads its start RMS from INIT_FROM={init!r}, "
                "which is not a checkpoint dir (no trainable.safetensors)"
            )
        start = shard_like(load_trainable(init), {n: live[n].detach() for n in names})
        source = {n: start[n].to(live[n].device) for n in names}
    else:
        source = {n: live[n].detach() for n in names}
    if not names:
        return {}
    rms = [
        torch.linalg.vector_norm(source[n], dtype=torch.float32).to(
            live[names[0]].device
        )
        / math.sqrt(source[n].numel())
        for n in names
    ]
    return dict(zip(names, torch.stack(rms).tolist()))


class OutlierDecay:
    """The decay over `entries` (parameter, start RMS, live optimiser group or None) at
    the step `clock()` returns; `metrics` holds the step's device values (elements over
    the cap, the largest |p| / (cap * r0) among them, at least 1)."""

    def __init__(self, spec: dict, entries: list[tuple], clock: Callable[[], int]):
        self.spec = spec
        self.entries = entries
        self.by_id = {id(p): r0 for p, r0, _ in entries}
        self.clock = clock
        self.seen = -1
        self.metrics: dict[str, Tensor] = {}

    def current(self) -> dict[str, Tensor]:
        """The last decayed step's metrics (empty before the first)."""
        return self.metrics

    def _begin(self, device) -> tuple[float, float]:
        t = int(self.clock())
        if t != self.seen:
            self.seen = t
            if not self.metrics:
                self.metrics = {
                    "outlier_over": torch.zeros((), device=device),
                    "outlier_max": torch.ones((), device=device),
                }
            self.metrics["outlier_over"].zero_()
            self.metrics["outlier_max"].fill_(1.0)
        return cap_at(self.spec, t), rate_at(self.spec, t)

    def _shrink(self, p: Tensor, r0: float, cap: float, k) -> None:
        lim = cap * r0
        ex = (p.abs() - lim).clamp_min_(0.0)
        m = self.metrics
        m["outlier_over"] += torch.count_nonzero(ex).to(m["outlier_over"])
        m["outlier_max"].copy_(
            torch.maximum(
                m["outlier_max"], (1.0 + ex.amax() / lim).float().to(m["outlier_max"])
            )
        )
        p.sub_(ex.copysign_(p).mul_(k))

    @torch.no_grad()
    def apply(self) -> None:
        """The step's decay, every entry at its group's lr (after an applied step)."""
        cap, rate = self._begin(self.entries[0][0].device)
        for p, r0, group in self.entries:
            self._shrink(p, r0, cap, group["lr"] * rate)

    @torch.no_grad()
    def apply_gated(self, bad: Tensor) -> None:
        """`apply` with every update multiplied by 1 - `bad` (the device flag)."""
        cap, rate = self._begin(self.entries[0][0].device)
        keep = 1.0 - bad.float()
        for p, r0, group in self.entries:
            self._shrink(
                p, r0, cap, (keep * (group["lr"] * rate)).to(p.device, p.dtype)
            )

    @torch.no_grad()
    def apply_param(self, p: Tensor, lr: float, found: Tensor) -> None:
        """One parameter's decay inside a fused step (skipped where `found` is set)."""
        r0 = self.by_id.get(id(p))
        if r0 is None:
            return
        cap, rate = self._begin(p.device)
        self._shrink(p, r0, cap, ((1.0 - found) * (lr * rate)).to(p.device, p.dtype))


def _live_groups(optimizer) -> dict[int, dict]:
    return {id(p): g for g in optimizer.param_groups for p in g["params"]}


def attach_outlier_decay(
    bundle, k: dict, refs: dict[str, float] | None, names: list[str]
) -> None:
    """Install the decay on `bundle` for the targeted parameters among `names` (the
    bundle's groups): the per-parameter post of a fused bundle, the gated post of the
    deferred bundle, or the post-step chain; its metrics join the step's
    (`bundle.extra_metrics`). A no-op when `refs` is None; raises when a targeted
    parameter is stepped elsewhere (a per-block or host optimiser)."""
    if refs is None:
        return
    spec = spec_of(k)
    clock = partial(_last_epoch, bundle)
    if hasattr(bundle, "opts"):
        named = [
            (bundle.name_of[id(p)], p) for p in bundle.params if id(p) in bundle.opts
        ]
        entries = [(n, p, refs[n], None) for n, p in named if n in refs]
    else:
        live = _live_groups(bundle.optimizer)
        named = zip(bundle.names, bundle.params)
        entries = [
            (n, p, refs[n], live[id(p)])
            for n, p in named
            if n in refs and id(p) in live
        ]
    missing = sorted({n for n in names if n in refs} - {e[0] for e in entries})
    if missing:
        raise ValueError(
            f"{KNOB} needs every targeted weight stepped by the device optimiser; "
            f"{len(missing)} are not (first: {missing[0]!r}): use a plan with "
            "optim_state 'gpu' and stream 'off'"
        )
    if not entries:
        return
    decay = OutlierDecay(spec, [e[1:] for e in entries], clock)
    if hasattr(bundle, "opts"):
        for pid, (opt, gi, run, post) in list(bundle.opts.items()):
            if pid in decay.by_id:
                bundle.opts[pid] = (
                    opt,
                    gi,
                    run,
                    partial(_then, post, decay.apply_param),
                )
    elif hasattr(bundle, "gated_posts"):
        bundle.gated_posts.append(decay.apply_gated)
    else:
        bundle.post_step = partial(_then_plain, bundle.post_step, decay.apply)
    bundle.extra_metrics = decay.current


def _last_epoch(bundle) -> int:
    return bundle.schedule.last_epoch


def _then(first: Callable, second: Callable, p, lr, found) -> None:
    first(p, lr, found)
    second(p, lr, found)


def _then_plain(first: Callable, second: Callable) -> None:
    first()
    second()
