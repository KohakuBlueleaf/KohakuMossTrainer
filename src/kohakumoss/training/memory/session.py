"""MemorySession: one MemoryPlan applied to a model plus the trainer's optimiser wiring
(`build_memory_optim`), for benches and tests on models outside the trainer. See
docs/internals/offload.md (MemorySession and parameter groups)."""

import torch
from torch import Tensor, nn

from kohakumoss.memory.apply import MemoryEngines, apply_to_backbone
from kohakumoss.offload.optim.cpu_optim import CPUAdamW
from kohakumoss.offload.optim.gpu_adamw import HostStateAdamW, MasterAdamW
from kohakumoss.training.memory.optimizers import build_memory_optim, device_optimizer
from kohakumoss.types.plans import MemoryPlan, PrecisionConfig
from kohakumoss_kernels.registry import settle_guards

CONSTANT = {"mode": "constant"}


class MemorySession:
    """Engines + bundle for one run: `backward(loss, micro)` per micro-batch, `step()`
    per optimiser step. `param_groups` are torch-style groups over the model's
    parameters as given before the session (replaced parameters followed by name).
    `optimizer` is the trainer's OPTIMIZER (default `adamw`, placed by
    `plan.optim_state` exactly as in training)."""

    def __init__(
        self,
        model: nn.Module,
        plan: MemoryPlan,
        device: torch.device | str,
        compute_dtype: torch.dtype,
        grad_acc: int = 1,
        trainable_base: bool = False,
        lr: float = 1e-4,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        quant_min_features: int = 256,
        param_groups: list[dict] | None = None,
        grad_clip: float = 0.0,
        optimizer: str = "adamw",
    ):
        self.plan, self.grad_acc = plan, grad_acc
        self.device = torch.device(device)
        names = {id(p): n for n, p in model.named_parameters()}
        named = (
            None
            if param_groups is None
            else [
                {**g, "params": [names[id(p)] for p in g["params"]]}
                for g in param_groups
            ]
        )
        self.engines: MemoryEngines = apply_to_backbone(
            model,
            plan,
            device,
            compute_dtype,
            training=True,
            trainable_base=trainable_base,
            quant_min_features=quant_min_features,
        )
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        groups = resolve_groups(model, named, defaults)
        prec = PrecisionConfig(compute=compute_dtype, scaler=False)
        self.bundle, self.accumulate = build_memory_optim(
            groups,
            plan,
            self.engines,
            self.device,
            optimizer,
            dict(lr=lr, betas=tuple(betas), eps=eps),
            CONSTANT,
            1 << 30,
            prec,
            grad_clip,
        )

    @property
    def dev_opt(self):
        """The optimiser of the device (non-streamed) parameters."""
        return device_optimizer(self.bundle)

    @property
    def block_opt(self):
        return self.bundle.blocks

    @property
    def grads(self):
        return self.engines.grads

    def backward(self, loss: Tensor, micro: int) -> None:
        """Backward of micro-batch `micro` (0-based) of this step."""
        with self.accumulate(micro == self.grad_acc - 1):
            self.bundle.backward(loss)

    def step(self) -> bool:
        """The optimiser step; the kernels proven in this step bind their plain call
        path (`settle_guards`)."""
        ok = self.bundle.step()
        settle_guards()
        return ok

    def masters(self, model: nn.Module) -> dict[str, Tensor]:
        """fp32 masters of every trainable parameter by name (for checks / checkpoints)."""
        if self.block_opt is not None:
            self.block_opt.synchronize()
        if isinstance(self.dev_opt, HostStateAdamW):
            self.dev_opt.synchronize()
        return {
            name: _master_of(self.dev_opt, p)
            for name, p in model.named_parameters()
            if p.requires_grad
        }


def resolve_groups(
    model: nn.Module, named_groups: list[dict] | None, defaults: dict
) -> list[dict]:
    """Groups over the model's current trainable parameters, hyperparameters filled
    from `defaults`; every trainable parameter must be in a group (None: one group)."""
    current = dict(model.named_parameters())
    if named_groups is None:
        named_groups = [{"params": list(current)}]
    out, seen = [], set()
    for g in named_groups:
        params = [current[n] for n in g["params"] if n in current]
        params = [p for p in params if p.requires_grad]
        seen.update(id(p) for p in params)
        if params:
            out.append({**defaults, **g, "params": params})
    missing = [n for n, p in current.items() if p.requires_grad and id(p) not in seen]
    if missing:
        raise ValueError(f"trainable parameters in no param group: {missing[:4]}")
    return out


def _master_of(opt, p: Tensor) -> Tensor:
    match opt:
        case MasterAdamW() if id(p) in opt.state:
            return opt.state[id(p)][0].detach().float().cpu()
        case HostStateAdamW() if id(p) in opt.host:
            return opt.host[id(p)][0].detach().float().cpu().reshape(p.shape)
        case CPUAdamW() if p in opt.state:
            return opt.state[p]["master"].detach().float().cpu()
    return p.detach().float().cpu()
