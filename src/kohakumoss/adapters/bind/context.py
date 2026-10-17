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
    """mode / kit / multiplier / flags of one bind, and the layer's draws."""

    mode: str
    backend: str
    kit: object
    m: float
    detach: bool
    module_p: float
    training: bool
    use_fused: bool
    draws: Draws
