"""Optimiser step inside backward: each parameter is updated as soon as its gradient is
accumulated, then its gradient is freed, so only one block's gradients are alive.

There is no global gradient norm in this mode; clipping is per parameter. Under a loss
scaler a non-finite gradient zeroes that parameter's update and raises `found_inf`.
The gradient transform (unscale, clip) is composed once at construction; `arm` swaps
the hook body once per micro-batch.
"""

from collections.abc import Callable, Iterable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


class PerParamOptimizer:
    """One optimiser per parameter behind a torch-style `param_groups` view (for
    schedules). `params` is an iterable of parameters or of group dicts; a parameter's
    optimiser takes its group's hyperparameters on every step."""

    def __init__(self, params: Iterable, factory: Callable, **hparams):
        items = list(params)
        if not items or not isinstance(items[0], dict):
            items = [{"params": items}]
        self.param_groups, self.params, self.optims = [], [], {}
        self.group_of: dict[Tensor, dict] = {}
        for item in items:
            own = {k: v for k, v in item.items() if k != "params"}
            params = [p for p in item["params"] if p.requires_grad]
            group = None
            for p in params:
                opt = factory([{**own, "params": [p]}], **hparams)
                if group is None:
                    group = dict(opt.param_groups[0], params=params)
                self.optims[p] = opt
                self.group_of[p] = group
            self.params += params
            self.param_groups.append(group or {**hparams, **own, "params": []})
        self._keys = {
            id(g): tuple(k for k in g if k != "params") for g in self.param_groups
        }

    def step_param(self, p: Tensor) -> None:
        opt = self.optims[p]
        shared, group = self.group_of[p], opt.param_groups[0]
        for k in self._keys[id(shared)]:
            group[k] = shared[k]
        opt.step()

    def step(self) -> None:
        for p in self.params:
            if p.grad is not None:
                self.step_param(p)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            p.grad = None

    def state_dict(self) -> dict:
        return {i: self.optims[p].state_dict() for i, p in enumerate(self.params)}

    def load_state_dict(self, state: dict) -> None:
        for i, p in enumerate(self.params):
            self.optims[p].load_state_dict(state[i])


def _skip(p: Tensor) -> None:
    return None


class FusedStep:
    """Post-accumulate-grad hooks that transform the gradient and call `step_param(p)`.

    `scaled=True` expects `arm(..., inv_scale)` every step; `clip > 0` clips per
    parameter.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        step_param: Callable[[Tensor], None],
        clip: float = 0.0,
        scaled: bool = False,
    ):
        self.params = [p for p in params if p.requires_grad]
        self.step_param = step_param
        self.clip = clip
        self.inv_scale: Tensor | None = None
        self.found_inf: Tensor | None = None
        transforms = []
        if scaled:
            transforms.append(self._unscale)
        if clip > 0:
            transforms.append(self._clip)
        self._transforms = tuple(transforms)
        self._active = _skip
        self.handles = [
            p.register_post_accumulate_grad_hook(self._hook) for p in self.params
        ]

    def arm(self, last_microbatch: bool, inv_scale: Tensor | None = None) -> None:
        """Step on this backward (the last micro-batch of the step) or only accumulate."""
        self._active = self._step if last_microbatch else _skip
        self.inv_scale = inv_scale
        if inv_scale is not None:
            self.found_inf = torch.zeros((), device=inv_scale.device)

    def _hook(self, p: Tensor) -> None:
        self._active(p)

    def _unscale(self, g: Tensor) -> None:
        g.mul_(self.inv_scale.to(g.device))
        finite = torch.isfinite(g).all()
        self.found_inf = torch.maximum(self.found_inf.to(g.device), (~finite).float())
        g.mul_(finite.to(g.dtype))
        g.nan_to_num_(0.0, 0.0, 0.0)

    def _clip(self, g: Tensor) -> None:
        norm = g.float().norm()
        g.mul_(torch.clamp(self.clip / (norm + 1e-6), max=1.0).to(g.dtype))

    @torch.no_grad()
    def _step(self, p: Tensor) -> None:
        g = p.grad
        for transform in self._transforms:
            transform(g)
        self.step_param(p)
        p.grad = None

    def remove(self) -> None:
        for h in self.handles:
            h.remove()


def wrap_optimizer_step_param(opt: Optimizer) -> Callable[[Tensor], None]:
    """`step_param` for an optimiser that has it, else a per-parameter group step."""
    if hasattr(opt, "step_param"):
        return opt.step_param
    raise TypeError("use PerParamOptimizer for fused per-parameter steps")
