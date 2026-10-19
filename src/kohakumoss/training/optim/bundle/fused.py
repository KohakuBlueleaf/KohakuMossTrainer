"""Fused per-parameter optimiser steps, driven from backward hooks by the memory engine.
See docs/internals/offload.md (fused step, non-finite gradients)."""

import contextlib
from collections.abc import Callable
from functools import partial

import torch
from torch import Tensor
from torch.optim import Optimizer

from kohakumoss.training.optim.bundle.base import close_optimizers, step_reducer
from kohakumoss.training.optim.bundle.mirror import LRHolder
from kohakumoss.training.optim.bundle.nonfinite import flag_report, param_names
from kohakumoss.training.optim.optimizers.skip import (
    host_state,
    on_device,
    put_back,
    restore_host,
)
from kohakumoss_common.calls import identity, nothing
from kohakumoss_common.weak import weak_method


def _unclipped(bundle, inv: Tensor) -> Tensor:
    return inv


def _shared_update(step_param: Callable, bundle, param: Tensor, found: Tensor) -> None:
    step_param(param, found)


def next_clip(max_norm: float, norm: Tensor, prev: Tensor) -> Tensor:
    """min(1, max_norm / the step's global norm), or `prev` after a non-finite step."""
    coef = (max_norm / (norm + 1e-6)).clamp(max=1.0)
    return torch.where(torch.isfinite(norm), coef, prev)


def _decay_to_one(coef: float, p: Tensor, lr: float, found: Tensor) -> None:
    p.sub_((p - 1.0) * ((1.0 - found) * (lr * coef)).to(p.dtype))


def _step_plain(opt: Optimizer) -> None:
    opt.step()


def _step_blockwise(opt: Optimizer) -> None:
    """bitsandbytes' step body for its one parameter, without its per-parameter device
    synchronize."""
    if not opt.initialized:
        opt.check_overrides()
        opt.to_gpu()
        opt.initialized = True
    group = opt.param_groups[0]
    p = group["params"][0]
    if len(opt.state[p]) == 0:
        opt.init_state(group, p, 0, 0)
    opt.prefetch_state(p)
    opt.update_step(group, p, 0, 0)


def _step_found_inf(opt: Optimizer, found: Tensor) -> Callable[[], None]:
    """torch's fused step, which skips the update (step count included) itself."""
    opt.found_inf = found
    opt.step()
    return nothing


def _step_own(opt: Optimizer, found: Tensor) -> Callable[[], None]:
    """A KohakuMoss optimiser's `step_masked`; returns the host-state restore."""
    p = opt.param_groups[0]["params"][0]
    saved = host_state(opt, p)
    opt.step_masked(found)
    return partial(restore_host, opt, p, saved)


def _step_kept(body, opt: Optimizer, found: Tensor) -> Callable[[], None]:
    """`body(opt)`, then the parameter and its device state tensors put back where
    `found` is set; returns the host-state restore."""
    p = opt.param_groups[0]["params"][0]
    before = {k: v.clone() for k, v in opt.state[p].items() if on_device(v, p.device)}
    saved = host_state(opt, p)
    old = p.detach().clone()
    body(opt)
    skip = found.bool()
    put_back(skip, p, old)
    state = opt.state[p]
    for k, v in before.items():
        put_back(skip, state[k], v)
    return partial(restore_host, opt, p, saved)


def _step_runner(opt: Optimizer):
    """The per-parameter step of `opt`, selected once at build: `run(opt, found)`
    leaves the parameter and its state unchanged on device where `found` is set and
    returns the restore of its host-side state."""
    if opt.defaults.get("fused"):
        return _step_found_inf
    if hasattr(opt, "step_masked"):
        return _step_own
    if all(hasattr(opt, a) for a in ("update_step", "prefetch_state", "init_state")):
        return partial(_step_kept, _step_blockwise)
    return partial(_step_kept, _step_plain)


class FusedStepBundle:
    """OptimBundle whose optimiser step happens per parameter during backward.

    `shared`: an optimiser over every device parameter exposing `step_param(p,
    found_inf)`; otherwise `build_one(groups)` builds one optimiser per parameter.
    `attach_blocks(opt)` adds a per-block optimiser (StreamedAdamW) that steps in
    backward through the same gradient transform. `reduce_block_stats(bad, sq)` runs
    on the block steps' non-finite count and squared norm before they are read (DDP
    with block owners: their sum over the ranks).
    """

    def __init__(
        self,
        build_one: Callable[[list[dict]], Optimizer],
        schedule_factory: Callable[[Optimizer], object],
        scaler,
        groups: list[dict],
        agree: Callable[[Tensor], Tensor] = identity,
        before: Callable[[], None] = nothing,
        after: Callable[[], None] = nothing,
        reduce_loss: Callable[[Tensor], Tensor] = identity,
        clip: float = 0.0,
        shared=None,
        device: torch.device | None = None,
        reduce_step: Callable | None = None,
    ):
        self.scaler = scaler
        self.reduce = step_reducer(reduce_loss, agree, reduce_step)
        self.max_norm = clip
        self.factor = FusedStepBundle._clipped if clip > 0 else _unclipped
        self.hooks = None
        self.blocks = None
        self._start = _no_start
        self.before = before
        self.after = after
        self.params = [p for g in groups for p in g["params"]]
        self.name_of = {id(p): n for p, n in zip(self.params, param_names(groups))}
        self.nonfinite_grads: list[dict] = []
        self.shared = shared
        self.optimizer = None
        self.finishers: list[Callable[[], None]] = []
        self.reduce_block_stats: Callable[[Tensor, Tensor], None] = nothing
        self._lr0 = float(groups[0]["lr"]) if groups else 1.0
        self._block_lr: list[tuple[dict, float]] = []
        if shared is not None:
            self.lr_source = shared
            self.schedule = schedule_factory(shared)
            self._update = partial(_shared_update, shared.step_param)
            self.opts = {}
            if hasattr(shared, "finish"):
                self.finishers.append(shared.finish)
        else:
            self.lr_source = LRHolder(groups)
            self.schedule = schedule_factory(self.lr_source)
            self.opts: dict[int, tuple] = {}
            for gi, g in enumerate(groups):
                post = (
                    partial(_decay_to_one, g["decay_to_one_coef"])
                    if g.get("decay_to_one")
                    else nothing
                )
                for p in g["params"]:
                    one = {k: v for k, v in g.items() if k != "params"}
                    one["params"] = [p]
                    opt = build_one([one])
                    self.opts[id(p)] = (opt, gi, _step_runner(opt), post)
            self._update = FusedStepBundle._own_update
        self.device = device if device is not None else self.params[0].device
        self._clip = torch.ones(1, dtype=torch.float32, device=self.device)
        self._reset()
        self.last_loss = float("nan")
        self.last_norm = float("nan")
        self.last_ok = True
        self.last_nonfinite_params = 0
        self.skipped = 0
        self.nonfinite = 0

    def attach(self, hooks, finishes: bool = False) -> None:
        """The hook object this bundle starts per micro-batch. `finishes`: a
        GradOffload whose final callback on the last micro-batch is `step_param`; else
        a FusedStep."""
        self.hooks = hooks
        self._start = (
            partial(_start_offload, hooks, weak_method(self.step_param))
            if finishes
            else hooks.start
        )
        if hasattr(hooks, "finish"):
            self.finishers.insert(0, hooks.finish)

    def attach_blocks(self, blocks) -> None:
        """A per-block in-backward optimiser (StreamedAdamW) sharing the transform; its
        groups follow the schedule (lr scaled like group 0)."""
        self.blocks = blocks
        blocks.transform = weak_method(self._block_transform)
        self.finishers.append(blocks.finish)
        self._start = partial(_start_blocks, blocks, self._start)
        self._block_lr = [(g, float(g["lr"])) for g in blocks.param_groups]
        self._mirror_blocks()

    def _mirror_blocks(self) -> None:
        factor = self.lr / self._lr0
        for group, lr in self._block_lr:
            group["lr"] = lr * factor

    def start(self, last: bool):
        """Accumulate context: step parameters in this backward only on the last one."""
        self._start(last)
        return contextlib.nullcontext()

    def _reset(self) -> None:
        # One accumulator pair per stream that runs the transform (device / blocks).
        self._bad = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._sq = torch.zeros((), dtype=torch.float32, device=self.device)
        self._block_bad = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._block_sq = torch.zeros((), dtype=torch.float32, device=self.device)
        self._loss = torch.zeros((), device=self.device)
        self._inv = torch.full(
            (1,), 1.0 / self.scaler.scale, dtype=torch.float32, device=self.device
        )
        self._flags: list[tuple[str, Tensor]] = []
        self._undo: list[tuple[Tensor, Callable[[], None]]] = []

    def backward(self, loss: Tensor) -> None:
        self.scaler.scale_loss(loss).backward()
        self._loss = self._loss + loss.detach().float()

    def observe(self, loss: Tensor) -> None:
        self._loss = self._loss + loss.detach().float().to(self.device)

    def _own_update(self, param: torch.nn.Parameter, found: Tensor) -> None:
        opt, gi, run, post = self.opts[id(param)]
        lr = self.lr_source.param_groups[gi]["lr"]
        opt.param_groups[0]["lr"] = lr
        self._undo.append((found, run(opt, found)))
        post(param, lr, found)

    def _restore(self) -> None:
        """Host-side optimiser state of every skipped parameter back to its value before
        the step (one host sync, on non-finite steps only)."""
        if not self._undo:
            return
        flags = torch.stack([f.reshape(()).to(self.device) for f, _ in self._undo])
        for flag, (_, undo) in zip(flags.tolist(), self._undo):
            if flag:
                undo()

    @torch.no_grad()
    def _accumulate(self, bad_acc: Tensor, sq_acc: Tensor, g: Tensor) -> Tensor:
        norm = torch.linalg.vector_norm(g, dtype=torch.float32).reshape(1)
        inv = self._inv.to(g.device)
        sq_acc += (norm * inv).square().reshape(()).to(sq_acc.device)
        found = torch.zeros(1, dtype=torch.float32, device=g.device)
        factor = self.factor(self, inv)
        torch._amp_foreach_non_finite_check_and_unscale_([g], found, factor)
        bad_acc += found.to(torch.int32).to(bad_acc.device)
        return found.reshape(())

    def _clipped(self, inv: Tensor) -> Tensor:
        return inv * self._clip.to(inv.device)

    def transform(self, g: Tensor) -> Tensor:
        """Add the gradient's unscaled squared norm to the step's norm, then unscale and
        clip it in one pass that also flags a non-finite value (two passes over `g`).
        The clip factor is min(1, GRAD_CLIP / the previous step's global norm). Returns
        the 0-d fp32 found_inf the update must honour (a flagged gradient keeps its
        values: every update path skips it). Device-parameter path (its own
        accumulators). See docs/internals/offload.md (fused step)."""
        return self._accumulate(self._bad, self._sq, g)

    def _block_transform(self, g: Tensor) -> Tensor:
        """`transform` for the per-block optimiser's stream (its own accumulators)."""
        return self._accumulate(self._block_bad, self._block_sq, g)

    @torch.no_grad()
    def step_param(self, param: torch.nn.Parameter) -> None:
        """Called by the engine's post-accumulate-grad hook (grad is present)."""
        found = self.transform(param.grad)
        self._update(self, param, found)
        self._flags.append((self.name_of[id(param)], found))
        param.grad = None

    def step(self) -> bool:
        """Finish the step. Parameters were updated inside backward (one with a
        non-finite gradient is not), so the step counts as applied: returns True;
        `nonfinite` counts steps with a non-finite gradient or loss, and `last_ok` is
        False on those. `last_norm` is the unclipped gradient norm of the step, NaN when
        any parameter's gradient was non-finite (as in the non-fused bundle);
        `last_nonfinite_params` counts those parameters."""
        for finish in self.finishers:
            finish()
        self.reduce_block_stats(self._block_bad, self._block_sq)
        self.before()
        count = self._bad + self._block_bad.to(self._bad.device)
        loss, bad = self.reduce(self._loss.reshape(1).clone(), (count > 0).int())
        bad = (bad > 0) | ~torch.isfinite(loss.reshape(1))
        norm = (self._sq + self._block_sq.to(self._sq.device)).sqrt().reshape(1)
        norm = torch.where(count > 0, torch.full_like(norm, float("nan")), norm)
        if self.max_norm > 0:
            self._clip = next_clip(self.max_norm, norm, self._clip)
        loss_v, norm_v, bad_v, count_v = torch.cat(
            [loss.reshape(1).float(), norm, bad.float(), count.float()]
        ).tolist()
        ok = bad_v == 0
        self.last_nonfinite_params = int(count_v)
        self.nonfinite += int(not ok)
        self.nonfinite_grads = [] if ok else flag_report(self._flags)
        if not ok:
            self._restore()
        self.after()
        # mark the schedule's optimiser as stepped; see docs/internals/offload.md
        self.lr_source._opt_called = True
        self.schedule.step()
        if self._block_lr:
            self._mirror_blocks()
        self.scaler.update(not ok)
        self._reset()
        self.last_loss, self.last_norm, self.last_ok = loss_v, norm_v, ok
        return True

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def close(self) -> None:
        close_optimizers(self.shared, self.blocks)

    @property
    def lr(self) -> float:
        return float(self.lr_source.param_groups[0]["lr"])

    def train_mode(self) -> None:
        """train() on every schedule-free optimiser (the others have no modes)."""
        for opt in self._moded():
            opt.train()

    def eval_mode(self) -> None:
        """eval() on every schedule-free optimiser."""
        for opt in self._moded():
            opt.eval()

    def _moded(self) -> list:
        return [
            o for o in self._optimizers() if hasattr(o, "train") and hasattr(o, "eval")
        ]

    def _optimizers(self) -> list:
        own = (
            [self.shared]
            if self.shared is not None
            else [o for o, *_ in self.opts.values()]
        )
        return own + ([self.blocks] if self.blocks is not None else [])

    def state_dict(self) -> dict:
        return {
            "optimizers": [opt.state_dict() for opt in self._optimizers()],
            "schedule": self.schedule.state_dict(),
            "scaler": self.scaler.state_dict(),
            "skipped": self.skipped,
            "nonfinite": self.nonfinite,
            "clip": float(self._clip),
        }

    def load_state_dict(self, state: dict) -> None:
        for opt, sd in zip(self._optimizers(), state["optimizers"], strict=True):
            opt.load_state_dict(sd)
        self._clip.fill_(float(state.get("clip", 1.0)))
        self.schedule.load_state_dict(state["schedule"])
        if self._block_lr:
            self._mirror_blocks()
        self.scaler.load_state_dict(state["scaler"])
        self.skipped = state.get("skipped", 0)
        self.nonfinite = state.get("nonfinite", 0)
        self._reset()


def _no_start(last: bool) -> None:
    raise RuntimeError("FusedStepBundle has no hook object attached")


def _start_offload(hooks, final, last: bool) -> None:
    hooks.start(last, final)


def _start_blocks(blocks, inner, last: bool) -> None:
    blocks.start(last)
    inner(last)
