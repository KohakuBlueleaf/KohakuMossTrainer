"""Full finetune: the backbone's own parameters train, behind the AdapterSet API."""

import re

import torch
from torch import Tensor, nn

from kohakumoss.adapters.train.adapter_set import FULL_TAG, fold_metadata
from kohakumoss.model import save_diffusers
from kohakumoss.model.io.save import save_single
from kohakumoss.training.optim.groups import build_param_groups


class FullTrainable:
    """Trains every backbone parameter except those matching `frozen` patterns; the
    trainable parameters are looked up by name on the live model (FSDP2 included)."""

    def __init__(
        self,
        model: nn.Module,
        frozen: tuple[str, ...] = (),
        decay_to_one: str | None = None,
    ):
        self.model = model
        patterns = [re.compile(p) for p in frozen]
        self.names = set()
        for name, p in model.named_parameters():
            train = not any(pat.match(name) for pat in patterns)
            p.requires_grad_(train)
            if train:
                self.names.add(name)
        self.decay_to_one = decay_to_one
        self.prewarm_stats: dict[str, int] = {}

    @property
    def named(self) -> list[tuple[str, nn.Parameter]]:
        return [(n, p) for n, p in self.model.named_parameters() if n in self.names]

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

    def state_metadata(self) -> dict[str, str]:
        """The metadata of a `state_dict()` file: FULL_TAG and the weight fold the
        weights are stored in (FOLD_KEY)."""
        return {FULL_TAG: "1", **fold_metadata(self.model)}

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        for name, p in self.named:
            p.copy_(state[name])

    def export(
        self,
        path: str,
        fmt: str,
        metadata: dict[str, str],
        row_scales=None,
        fused_qkv: bool = False,
    ) -> None:
        """`merged`: bare-key single safetensors of the base weights, fold undone
        (`fused_qkv`: joint q/k/v as one `proj.qkv`, the ComfyUI layout); `diffusers`:
        the transformer folder (`row_scales` is for adapter formats and unused here)."""
        if fmt == "diffusers":
            save_diffusers(self.model, path)
            return
        save_single(
            self.model, path, dtype=self.dtype, fused_qkv=fused_qkv, metadata=metadata
        )

    def begin_step(self, sigma: Tensor, seq, step: int) -> None:
        return None

    def end_step(self) -> None:
        return None

    def step_state(self):
        return None

    def set_step_state(self, state) -> None:
        return None

    def prewarm(
        self, rows: list[int], dtype: torch.dtype, devices: list, budgets=None
    ) -> int:
        return 0

    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16

    def merge_into(self, multiplier: float = 1.0) -> None:
        return None

    def set_multiplier(self, multiplier: float) -> None:
        return None
