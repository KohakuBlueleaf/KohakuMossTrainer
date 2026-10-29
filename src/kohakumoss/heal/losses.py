"""Heal losses: the student matches the teacher's velocity (soft), optionally the flow
target (hard) and pays for concentrating its block outputs' energy (concentration)."""

from dataclasses import dataclass

import torch
from torch import Tensor

from kohakumoss.training.stream_loss import top_share


def rel_sq(a: Tensor, b: Tensor) -> Tensor:
    """sum ||a - b||^2 / sum ||b||^2 in fp32."""
    a, b = a.float(), b.float()
    return (a - b).square().sum() / b.square().sum().clamp_min(
        torch.finfo(torch.float32).tiny
    )


def concentration(outputs: list[Tensor], k: int) -> Tensor:
    """Mean over the block outputs of their top-k channel energy share."""
    return torch.stack([top_share(h, k) for h in outputs]).mean()


@dataclass(frozen=True)
class HealLoss:
    """loss = soft + hard_weight * hard + conc_weight * concentration."""

    hard_weight: float = 0.0
    conc_weight: float = 0.0
    conc_k: int = 8

    @property
    def needs_blocks(self) -> bool:
        return self.conc_weight > 0

    def __call__(
        self, v_s: Tensor, v_t: Tensor, target: Tensor, blocks: list[Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """(loss, detached parts: soft, hard, conc)."""
        soft = rel_sq(v_s, v_t)
        loss, parts = soft, {"soft": soft.detach()}
        if self.hard_weight > 0:
            hard = rel_sq(v_s, target)
            loss = loss + self.hard_weight * hard
            parts["hard"] = hard.detach()
        if self.conc_weight > 0:
            conc = concentration(blocks, self.conc_k)
            loss = loss + self.conc_weight * conc
            parts["conc"] = conc.detach()
        return loss, parts
