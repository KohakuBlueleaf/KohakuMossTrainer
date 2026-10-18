"""KohakuMoss's own 8-bit AdamW: m (int8) and sqrt(v) (uint8) in blockwise-quantized
states, stepped by the kernel library's `adamw8bit_step` (bound once per parameter
device and dtype). See docs/concepts/training.md#optimisers."""

import torch
from torch.optim import Optimizer

from kohakumoss_kernels import optim as kernel_optim


class AdamW8bitKernel(Optimizer):
    """AdamW with 2 bytes of state per parameter (`new_8bit_state`); fp32 math in the
    kernel, decoupled weight decay, `rounding` = the parameter write-back
    ("nearest" | "stochastic" | "kahan")."""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        rounding: str = "nearest",
    ):
        if rounding not in kernel_optim.MODES:
            raise ValueError(f"rounding must be one of {kernel_optim.MODES}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.rounding = rounding
        self._ops: dict[tuple, object] = {}

    def _step_op(self, p: torch.Tensor):
        key = (p.device, p.dtype)
        if key not in self._ops:
            self._ops[key] = kernel_optim.resolve(p.device, p.dtype).adamw8bit
        return self._ops[key]

    def _init_state(self, p: torch.Tensor) -> dict:
        state = self.state[p]
        if not state:
            mq, ms, rq, rs = kernel_optim.new_8bit_state(p)
            state.update(step=0, mq=mq, ms=ms, rq=rq, rs=rs)
            if self.rounding == "kahan":
                state["comp"] = torch.zeros_like(p)
        return state

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self._init_state(p)
                state["step"] += 1
                t = state["step"]
                self._step_op(p)(
                    p,
                    p.grad,
                    state["mq"],
                    state["ms"],
                    state["rq"],
                    state["rs"],
                    state.get("comp"),
                    group["lr"],
                    b1,
                    b2,
                    group["eps"],
                    group["weight_decay"],
                    t,
                    self.rounding,
                    t * kernel_optim.SEED_STRIDE & 0xFFFFFFFF,
                )
        return loss
