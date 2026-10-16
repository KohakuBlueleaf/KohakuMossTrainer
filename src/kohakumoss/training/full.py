"""Direct full finetune: the backbone's own parameters are trainable (AdapterSet API)."""

import re

import torch
from torch import Tensor, nn

from kohakumoss.model.io.save import save_single
from kohakumoss.training.optim.groups import build_param_groups


class FullTrainable:
    """Trains every backbone parameter except those matching `frozen` patterns."""

    def __init__(
        self,
        model: nn.Module,
        frozen: tuple[str, ...] = (),
        decay_to_one: str | None = None,
    ):
        self.model = model
        patterns = [re.compile(p) for p in frozen]
        self.named = []
        for name, p in model.named_parameters():
            train = not any(pat.match(name) for pat in patterns)
            p.requires_grad_(train)
            if train:
                self.named.append((name, p))
        self.decay_to_one = decay_to_one

    def trainable_params(self):
        return [p for _, p in self.named]

    def named_trainable(self) -> list[tuple[str, nn.Parameter]]:
        return list(self.named)

    def param_groups(self, lr: float, **rules) -> list[dict]:
        return build_param_groups(
            self.named,
            lr,
            rules.get("weight_decay", 0.0),
            rules.get("rules", "moss"),
            decay_to_one=self.decay_to_one,
            use_muon=rules.get("use_muon", False),
        )

    def state_dict(self) -> dict[str, Tensor]:
        return {name: p.detach() for name, p in self.named}

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        for name, p in self.named:
            p.copy_(state[name])

    def export(
        self, path: str, fmt: str, metadata: dict[str, str], row_scales=None
    ) -> None:
        """`merged`: bare-key single safetensors of the base weights, fold undone
        (`row_scales` is for adapter formats and unused here)."""
        save_single(self.model, path, dtype=self.dtype, metadata=metadata)

    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16

    def merge_into(self, multiplier: float = 1.0) -> None:
        return None

    def set_multiplier(self, multiplier: float) -> None:
        return None
