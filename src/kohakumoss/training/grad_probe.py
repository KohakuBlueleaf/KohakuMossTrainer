"""STREAM_LOSS_PROBE: on probe steps, the gradient norm over the trainable parameters of
the task loss and of each STREAM_LOSS term (unweighted, at the micro-batch's share), per
parameter group, and each term's cosine with the task gradient. See
docs/concepts/training.md#stream-penalties."""

import re

import torch
from torch import Tensor, nn

from kohakumoss.training.stream_loss import PROBE

GROUPS = (
    ("embed", re.compile(r"^(video_embedder|text_embedder)\.")),
    ("attn_out", re.compile(r"\.proj\.o\.")),
    ("ffn_out", re.compile(r"ffn_down")),
    ("in_proj", re.compile(r"(qkv|ffn_up_gate|ffn_up|ffn_gate)")),
    ("norm", re.compile(r"norm")),
    ("cond", re.compile(r"modulation|time_embedding")),
    ("head", re.compile(r"^final_")),
)
SCALE = 2.0**16
STEP_DOWN = 2.0**8
RETRIES = 3
FIRST = 1


def group_of(name: str) -> str:
    """The parameter group of a parameter name (first match of GROUPS, else 'other')."""
    return next((g for g, pat in GROUPS if pat.search(name)), "other")


class GradProbe:
    """`measure({"task": loss, term: value, ...})` -> metrics `grad_<name>` (total norm),
    `grad_<name>_<group>` (norm per group) and `grad_cos_<term>` (cosine with the task
    gradient), unscaled. Each value's gradient is taken with `autograd.grad` on the
    retained graph, scaled by SCALE (STEP_DOWN-fold lower while any element is not
    finite), so fp16 parameters neither underflow nor overflow."""

    def __init__(self, named: list[tuple[str, nn.Parameter]], every: int):
        named = [(n, p) for n, p in named if p.requires_grad]
        self.params = [p for _, p in named]
        self.groups = [group_of(n) for n, _ in named]
        self.every = every

    def due(self, step: int) -> bool:
        """Every `every` steps from step FIRST: step 0 runs the guarded kernel path, whose
        backward runs once (kohakumoss_kernels.registry)."""
        return step >= FIRST and step % self.every == 0

    def _grads(self, value: Tensor) -> tuple[list[Tensor | None], float]:
        scale = SCALE
        for _ in range(RETRIES + 1):
            grads = torch.autograd.grad(
                value * scale, self.params, retain_graph=True, allow_unused=True
            )
            if all(g is None or bool(torch.isfinite(g).all()) for g in grads):
                return list(grads), scale
            scale /= STEP_DOWN
        raise FloatingPointError(f"{PROBE}: gradient not finite at scale {scale}")

    def _norms(self, name: str, grads, scale: float) -> dict[str, Tensor]:
        sq: dict[str, Tensor] = {}
        for g, group in zip(grads, self.groups):
            if g is not None:
                part = g.float().square().sum()
                sq[group] = sq[group] + part if group in sq else part
        out = {f"grad_{name}_{g}": v.sqrt() / scale for g, v in sq.items()}
        total = torch.stack(list(sq.values())).sum() if sq else torch.zeros(())
        out[f"grad_{name}"] = total.sqrt() / scale
        return out

    def measure(self, values: dict[str, Tensor]) -> dict[str, Tensor]:
        task, task_scale = self._grads(values["task"])
        out = self._norms("task", task, task_scale)
        for name, value in values.items():
            if name == "task":
                continue
            grads, scale = self._grads(value)
            out.update(self._norms(name, grads, scale))
            dots = [
                (a.float() * b.float()).sum()
                for a, b in zip(grads, task)
                if a is not None and b is not None
            ]
            dot = torch.stack(dots).sum() / (scale * task_scale) if dots else 0.0
            denom = (out[f"grad_{name}"] * out["grad_task"]).clamp_min(1e-30)
            out[f"grad_cos_{name}"] = dot / denom
            del grads
        return {k: v.detach() for k, v in out.items()}


def build_probe(k: dict, backbone: nn.Module) -> GradProbe | None:
    """The probe of STREAM_LOSS_PROBE (0 = off) over the backbone's trainable parameters.
    On, compiled backwards are built without donated buffers (they forbid a retained
    graph); set before the first step compiles anything."""
    every = int(k.get(PROBE, 0))
    if every <= 0:
        return None
    torch._functorch.config.donated_buffer = False
    return GradProbe(list(backbone.named_parameters()), every)
