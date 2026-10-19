"""Optimiser step inside backward: post-accumulate-grad hooks step each parameter and
free its gradient. See docs/internals/offload.md (fused step)."""

from collections.abc import Callable, Iterable

import torch
from torch import Tensor, nn

from kohakumoss_common.calls import nothing


class FusedStep:
    """Post-accumulate-grad hooks that transform the gradient and call `step_param(p)`.

    `scaled=True` expects `start(..., inv_scale)` every step; `clip > 0` clips per
    parameter. It holds no parameter (their hooks hold it).
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        step_param: Callable[[Tensor], None],
        clip: float = 0.0,
        scaled: bool = False,
    ):
        params = [p for p in params if p.requires_grad]
        self.step_param = step_param
        self.clip = clip
        self.inv_scale: Tensor | None = None
        self.found_inf: Tensor | None = None
        transforms = []
        if scaled:
            transforms.append(FusedStep._unscale)
        if clip > 0:
            transforms.append(FusedStep._clip)
        self._transforms = tuple(transforms)
        self._active = nothing
        self.handles = [
            p.register_post_accumulate_grad_hook(self._hook) for p in params
        ]

    def start(self, last_microbatch: bool, inv_scale: Tensor | None = None) -> None:
        """Step on this backward (the last micro-batch of the step) or only accumulate."""
        self._active = FusedStep._step if last_microbatch else nothing
        self.inv_scale = inv_scale
        if inv_scale is not None:
            self.found_inf = torch.zeros((), device=inv_scale.device)

    def _hook(self, p: Tensor) -> None:
        self._active(self, p)

    def _unscale(self, g: Tensor) -> None:
        g.mul_(self.inv_scale.to(g.device))
        finite = torch.isfinite(g).all()
        self.found_inf = torch.maximum(self.found_inf.to(g.device), (~finite).float())
        g.mul_(finite.to(g.dtype))
        g.nan_to_num_(0.0, 0.0, 0.0)

    def _clip(self, g: Tensor) -> None:
        norm = torch.linalg.vector_norm(g, dtype=torch.float32)
        g.mul_(torch.clamp(self.clip / (norm + 1e-6), max=1.0).to(g.dtype))

    @torch.no_grad()
    def _step(self, p: Tensor) -> None:
        g = p.grad
        for transform in self._transforms:
            transform(self, g)
        self.step_param(p)
        p.grad = None

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
