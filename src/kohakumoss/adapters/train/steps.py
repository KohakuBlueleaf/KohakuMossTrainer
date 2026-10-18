"""Per-step adapter state: T-LoRA rank masks and the per-layer random draws, drawn
from a generator seeded by (seed, step, microbatch). See docs/concepts/adapters.md."""

import torch

from kohakumoss.adapters.train import tlora

MIX = (1_000_003, 1_009)


def step_seed(seed: int, step: int, micro: int) -> int:
    return (seed * MIX[0] + step * MIX[1] + micro) % (2**63 - 1)


class StepPlan:
    """What `begin_step` sets and `end_step` clears for one adapter set."""

    def __init__(self, targets: list, config):
        self.seed = config.seed
        masked = {t.axis for t in targets if t.record.per_token_mask}
        self.axes = tuple(sorted(masked))
        self.mask_kwargs = config.mask_kwargs()
        self.max_rank = config.dim
        self.drawn = [t for t in targets if t.record.draw_step is not None]
        self.module_p = config.module_dropout
        self.dropped = list(targets) if config.module_dropout > 0 else []
        self.step = None
        self.micro = 0

    def begin(self, sigma, seq, step: int) -> None:
        micro = self.micro if step == self.step else 0
        gen = torch.Generator().manual_seed(step_seed(self.seed, step, micro))
        for t in self.drawn:
            t.record.draw_step(t, gen)
        for t in self.dropped:
            t.draws.skip = bool(torch.rand((), generator=gen) < self.module_p)
        if self.axes:
            tlora.set_masks(self.axes, self.max_rank, self.mask_kwargs, sigma, seq)
        self.step, self.micro = step, micro + 1

    def end(self) -> None:
        if self.axes:
            tlora.clear_masks()
        for t in self.dropped:
            t.draws.skip = False
        self.step, self.micro = None, 0

    def capture(self) -> tuple:
        """The state `begin` installed (masks, DyLoRA ranks, module-dropout skips), to
        `install` again for the same micro-batch's later forward or recompute."""
        masks = tlora.saved_masks() if self.axes else {}
        ranks = tuple(t.draws.rank for t in self.drawn)
        skips = tuple(t.draws.skip for t in self.dropped)
        return masks, ranks, skips

    def install(self, state: tuple) -> None:
        """Put a `capture()` back as the live step state."""
        masks, ranks, skips = state
        if self.axes:
            tlora.restore_masks(masks)
        for t, rank in zip(self.drawn, ranks):
            t.draws.rank = rank
        for t, skip in zip(self.dropped, skips):
            t.draws.skip = skip
