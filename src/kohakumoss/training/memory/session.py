"""MemorySession: one MemoryPlan applied to a model plus the optimiser it implies.

Optimiser per `plan.optim_state`:
- `gpu`: GPU AdamW, states resident (`MasterAdamW`; streamed blocks: `StreamedAdamW`
  with resident states).
- `stream`: GPU AdamW, states in host memory streamed per parameter / per block
  (`HostStateAdamW`; streamed blocks: `StreamedAdamW`).
- `cpu`: `CPUAdamW` (host update).
GPU optimisers step inside backward on the last micro-batch whenever `fused_step`,
`grad_offload` or a `swap` stream is on; otherwise at `step()`. The driver calls
`backward(loss, micro)` per micro-batch and `step()` once per optimiser step.

`param_groups` are torch-style groups over the model's parameters (taken before the
session is built; parameters the plan replaces, such as streamed block weights, are
followed by name). Every group's hyperparameters reach every optimiser path: each group
is split into its device part and its streamed part, and each part keeps the group's
lr / betas / eps / weight_decay and extra keys.
"""

import torch
from torch import Tensor, nn

from kohakumoss.offload.cpu_optim import CPUAdamW
from kohakumoss.offload.fused_step import FusedStep
from kohakumoss.offload.gpu_adamw import HostStateAdamW, MasterAdamW
from kohakumoss.offload.grad import GradOffload
from kohakumoss.offload.stream_optim import StreamedAdamW
from kohakumoss.training.memory.apply import MemoryEngines, apply_to_backbone
from kohakumoss.types.plans import MemoryPlan

DEVICE_OPTIMS = {"gpu": MasterAdamW, "stream": HostStateAdamW, "cpu": CPUAdamW}


class MemorySession:
    """Engines + optimisers for one training run; see the module docstring."""

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
    ):
        if plan.optim_state not in DEVICE_OPTIMS:
            raise ValueError(
                f"optim_state {plan.optim_state!r} has no session optimiser"
            )
        self.plan, self.grad_acc = plan, grad_acc
        self.device = torch.device(device)
        hp = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        names = {id(p): n for n, p in model.named_parameters()}
        named_groups = (
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
        stream = self.engines.stream
        streamed = (
            {id(p) for p in stream.params.values()}
            if stream is not None and stream.trainable
            else set()
        )
        dev_groups, block_groups = split_groups(model, named_groups, streamed)
        self.block_opt = None
        self.host_block_opt = None
        if streamed:
            if plan.optim_state == "cpu":
                self.host_block_opt = CPUAdamW(block_groups, **hp)
            else:
                self.block_opt = StreamedAdamW(
                    stream,
                    resident=plan.optim_state == "gpu",
                    groups=block_groups,
                    **hp,
                )
        dev_params = [p for g in dev_groups for p in g["params"]]
        self.dev_opt = DEVICE_OPTIMS[plan.optim_state](
            dev_groups or [{"params": []}], **hp
        )
        gpu_opt = plan.optim_state != "cpu"
        self.in_backward = gpu_opt and (plan.fused_step or plan.grad_offload)
        self.grads = None
        self.fused = None
        if plan.grad_offload:
            self.grads = GradOffload(dev_params, pinned=plan.pinned)
            self.engines.grads = self.grads
            if not gpu_opt:
                self.dev_opt.use_host_grads(self.grads)
        elif self.in_backward:
            self.fused = FusedStep(dev_params, self.dev_opt.step_param)
            self.engines.fused = self.fused

    @property
    def param_groups(self) -> list[dict]:
        groups = list(self.dev_opt.param_groups)
        for opt in (self.block_opt, self.host_block_opt):
            if opt is not None:
                groups += opt.param_groups
        return groups

    def backward(self, loss: Tensor, micro: int) -> None:
        """Backward of micro-batch `micro` (0-based) of this step."""
        last = micro == self.grad_acc - 1
        if self.block_opt is not None:
            self.block_opt.arm(last)
        if self.grads is not None:
            final = self.dev_opt.step_param if self.in_backward else None
            self.grads.arm(last, final)
        if self.fused is not None:
            self.fused.arm(last)
        loss.backward()

    @torch.no_grad()
    def step(self) -> None:
        """Finish the optimiser step (host-side parts and non-fused updates)."""
        stream = self.engines.stream
        if self.host_block_opt is not None:
            stream.flush_grads()
            self.host_block_opt.step()
            self.host_block_opt.zero_grad(set_to_none=True)
        if self.grads is not None and not self.in_backward:
            self.grads.flush()
        if not self.in_backward:
            self.dev_opt.step()
            self.dev_opt.zero_grad(set_to_none=True)
        else:
            if self.grads is not None:
                self.grads.finish()
            if isinstance(self.dev_opt, HostStateAdamW):
                self.dev_opt.finish()
        self.engines.after_step()

    def masters(self, model: nn.Module) -> dict[str, Tensor]:
        """fp32 masters of every trainable parameter by name (for checks / checkpoints)."""
        if self.block_opt is not None:
            self.block_opt.synchronize()
        if isinstance(self.dev_opt, HostStateAdamW):
            self.dev_opt.synchronize()
        out = {}
        block_masters = self.block_opt.masters() if self.block_opt is not None else {}
        stream = self.engines.stream
        by_id = {}
        if stream is not None and stream.trainable:
            for key, p in stream.params.items():
                by_id[id(p)] = block_masters.get(key, p.detach())
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in by_id:
                out[name] = by_id[id(p)].float().cpu()
            else:
                out[name] = _master_of(self.dev_opt, p)
        return out


def split_groups(
    model: nn.Module, named_groups: list[dict] | None, streamed: set[int]
) -> tuple[list[dict], list[dict]]:
    """(device groups, streamed groups) over the model's current trainable parameters;
    each keeps its source group's hyperparameters. Every trainable parameter must be in
    a group (None: one group of all of them)."""
    current = dict(model.named_parameters())
    if named_groups is None:
        named_groups = [{"params": list(current)}]
    dev, blocks, seen = [], [], set()
    for g in named_groups:
        params = [current[n] for n in g["params"] if n in current]
        params = [p for p in params if p.requires_grad]
        seen.update(id(p) for p in params)
        for out, part in (
            (dev, [p for p in params if id(p) not in streamed]),
            (blocks, [p for p in params if id(p) in streamed]),
        ):
            if part:
                out.append({**g, "params": part})
    missing = [n for n, p in current.items() if p.requires_grad and id(p) not in seen]
    if missing:
        raise ValueError(f"trainable parameters in no param group: {missing[:4]}")
    return dev, blocks


def _master_of(opt, p: Tensor) -> Tensor:
    match opt:
        case MasterAdamW():
            return opt.state[id(p)][0].detach().float().cpu()
        case HostStateAdamW():
            return opt.host[id(p)][0].detach().float().cpu().reshape(p.shape)
        case CPUAdamW():
            return opt.state[p]["master"].detach().float().cpu()
    return p.detach().float().cpu()
