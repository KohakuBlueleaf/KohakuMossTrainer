"""Protocols the swappable components implement."""

from collections.abc import Iterable, Iterator
from typing import Any, Protocol

import torch
from torch import Tensor, nn

from kohakumoss.types.seqinfo import SeqInfo


class BaseWeight(Protocol):
    """Storage of a frozen or trainable base weight (dense, quantized or streamed)."""

    shape: torch.Size

    def weight(self, dtype: torch.dtype) -> Tensor: ...

    def linear(self, x: Tensor, bias: Tensor | None) -> Tensor: ...

    def materialize(self) -> Tensor: ...


class BlockRunner(Protocol):
    """Executes block i of the backbone loop (plain, checkpointed, streamed, staged);
    `bind(layers)` attaches it to the backbone's blocks and returns the runner."""

    def bind(self, layers) -> "BlockRunner": ...

    def run(self, index: int, block: nn.Module, x: Tensor, ctx: Any) -> Tensor: ...


class AdapterSet(Protocol):
    """What trains on a model (LyCORIS adapters from kohakumoss.adapters.build.apply, or the
    full finetune's own weights); `state_metadata()` is the safetensors metadata of a
    checkpoint's `state_dict()`; `prewarm_stats` the last `prewarm`'s counts (empty
    when it measured nothing)."""

    prewarm_stats: dict[str, int]

    def trainable_params(self) -> Iterable[nn.Parameter]: ...

    def named_trainable(self) -> Iterable[tuple[str, nn.Parameter]]: ...

    def param_groups(self, lr: float, **rules) -> list[dict]: ...

    def state_dict(self) -> dict[str, Tensor]: ...

    def state_metadata(self) -> dict[str, str]: ...

    def load_state_dict(self, state: dict[str, Tensor]) -> None: ...

    def export(
        self,
        path: str,
        fmt: str,
        metadata: dict[str, str],
        row_scales=None,
        fused_qkv: bool = False,
    ) -> None: ...

    def begin_step(self, sigma: Tensor, seq: SeqInfo, step: int) -> None: ...

    def end_step(self) -> None: ...

    def step_state(self) -> object: ...

    def set_step_state(self, state: object) -> None: ...

    def prewarm(
        self, rows: list[int], dtype: torch.dtype, devices: list, budgets=None
    ) -> int: ...

    def merge_into(self, multiplier: float = 1.0) -> None: ...

    def set_multiplier(self, multiplier: float) -> None: ...


class OptimBundle(Protocol):
    """Optimiser + schedule + scaler as one object; `step_param` serves fused per-block
    steps (called from backward hooks by the memory engine). The `last_*` fields and
    the counters describe the last `step()`. `shared` / `optimizer` is the optimiser of
    the device parameters (fused / plain bundle, the other None), `blocks` the
    per-block in-backward optimiser or None; `close()` stops their host workers."""

    scaler: Any
    shared: Any
    optimizer: Any
    blocks: Any
    last_loss: float
    last_norm: float
    last_ok: bool
    skipped: int
    nonfinite: int
    last_nonfinite_params: int
    nonfinite_grads: list[dict]

    @property
    def lr(self) -> float: ...

    def backward(self, loss: Tensor) -> None: ...

    def observe(self, loss: Tensor) -> None: ...

    def step(self) -> bool: ...

    def zero_grad(self) -> None: ...

    def step_param(self, param: nn.Parameter) -> None: ...

    def close(self) -> None: ...

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
