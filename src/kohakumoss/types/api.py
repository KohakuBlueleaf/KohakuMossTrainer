"""Cross-owner interfaces: block context, adapter set, optimiser bundle, loader, build order.

Build order (training/entry.py): arch config -> backbone (OpSet + BlockRunner) -> load
weights -> fold (fp16) -> base storage (quantize / stream) -> adapters -> bind_ops ->
parallel wrap -> compile. Every step after `load weights` is a one-time structural change.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from torch import Tensor, nn

from kohakumoss.types.seqinfo import SeqInfo


@dataclass
class BlockCtx:
    """What every joint block needs besides its input: layout, per-sample modulation
    [B, 6, D] fp32 (shift_a, scale_a, gate_a, shift_f, scale_f, gate_f), RoPE bands,
    and the residual scale `r` per block (fold)."""

    seq: SeqInfo
    mod: Tensor
    bands: Tensor
    residual_scale: tuple[float, ...]


class AdapterSet(Protocol):
    """The adapters applied to a model (built by kohakumoss.adapters.apply)."""

    def trainable_params(self) -> Iterable[nn.Parameter]: ...

    def param_groups(self, lr: float, **rules) -> list[dict]: ...

    def state_dict(self) -> dict[str, Tensor]: ...

    def load_state_dict(self, state: dict[str, Tensor]) -> None: ...

    def export(self, path: str, fmt: str, metadata: dict[str, str]) -> None: ...

    def merge_into(self, multiplier: float = 1.0) -> None: ...

    def set_multiplier(self, multiplier: float) -> None: ...


class OptimBundle(Protocol):
    """Optimiser + schedule + scaler as one object; `step_param` serves fused per-block
    steps (called from backward hooks by the memory engine)."""

    def step(self) -> bool: ...

    def zero_grad(self) -> None: ...

    def step_param(self, param: nn.Parameter) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


class StatefulLoader(Protocol):
    """Whole-batch iterator with exact resume state `{seed, start}`.

    Text dropout is applied by the loader (the empty-prompt embedding replaces the
    caption before lengths are known), so `HostBatch.text_len` is final.
    """

    def __iter__(self) -> Iterator[Any]: ...

    def state_dict(self) -> dict[str, int]: ...

    def load_state_dict(self, state: dict[str, int]) -> None: ...
