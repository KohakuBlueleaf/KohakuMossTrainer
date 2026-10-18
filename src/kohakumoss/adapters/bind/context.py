"""What a layer's binding is resolved from, and the per-step draws it reads."""

from dataclasses import dataclass


class Draws:
    """One layer's random decisions for the current training step (DyLoRA rank,
    module-dropout skip), set by `LycoAdapterSet.begin_step`."""

    def __init__(self, rank: int = 0):
        self.rank = rank
        self.skip = False


@dataclass(frozen=True)
class BindCtx:
    """One bind: mode, backend, the layer's kernel ops (`bind.kernel_ops`), multiplier,
    DoRA-detach, module dropout, training state, whether the kernel ops apply
    (`use_ops`: no input / rank dropout in training), and the layer's draws."""

    mode: str
    backend: str
    ops: object
    m: float
    detach: bool
    module_p: float
    training: bool
    use_ops: bool
    draws: Draws
