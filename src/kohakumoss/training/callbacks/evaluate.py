"""Evaluation callbacks: sample grids and fixed-noise validation loss."""

import json
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor

from kohakumoss.training.state import TrainState


class FixedSigma:
    """Sampler returning one fixed sigma for every sample."""

    def __init__(self, sigma: float):
        self.sigma = sigma

    def sample(self, batch: int, generator, step: int) -> Tensor:
        return torch.full((batch,), self.sigma, dtype=torch.float32)


class SampleGrid:
    """Every `every` steps: run the inference pipeline on the live training model.

    `pipeline(**settings) -> list[PIL.Image]` is `MossPipeline.from_components(...)` over
    the training backbone (no reload; adapters at the training multiplier). The model is
    put in eval mode and restored to train mode after; images and `settings.json` go to
    `<output>/samples/step_X/` and through the logger.
    """

    def __init__(
        self,
        every: int,
        pipeline: Callable,
        settings: dict,
        output_dir: str,
        logger,
        model: torch.nn.Module,
        modes: tuple[Callable, Callable],
    ):
        self.every = every
        self.pipeline = pipeline
        self.settings = settings
        self.root = Path(output_dir) / "samples"
        self.logger = logger
        self.model = model
        self.eval_mode, self.train_mode = modes

    @torch.no_grad()
    def on_step_end(self, state: TrainState, out) -> None:
        was_training = self.model.training
        self.eval_mode()
        self.model.eval()
        try:
            images = self.pipeline(**self.settings)
        finally:
            self.model.train(was_training)
            self.train_mode()
        folder = self.root / f"step_{state.step:08d}"
        folder.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images):
            image.save(folder / f"{i:03d}.png")
        record = {"step": state.step, **self.settings}
        (folder / "settings.json").write_text(json.dumps(record, indent=2, default=str))
        self.logger.log_images("samples", images, state.step)


class Validation:
    """Mean loss over fixed held-out batches at fixed sigmas and seeded noise."""

    def __init__(
        self,
        every: int,
        make_step: Callable[[object], Callable],
        batches: list,
        sigmas: tuple[float, ...],
        seed: int,
        model: torch.nn.Module,
        modes: tuple[Callable, Callable],
    ):
        self.every = every
        self.batches = batches
        self.sigmas = tuple(sigmas)
        self.seed = seed
        self.steps = [make_step(FixedSigma(s)) for s in self.sigmas]
        self.model = model
        self.eval_mode, self.train_mode = modes

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Per-sigma and mean validation loss with noise reseeded per sigma."""
        out, total = {}, 0.0
        for sigma, step in zip(self.sigmas, self.steps):
            step.noise_gen.manual_seed(self.seed)
            losses = [float(step(b, step.denominator(b)).loss) for b in self.batches]
            value = sum(losses) / max(len(losses), 1)
            out[f"val_loss@{sigma:g}"] = value
            total += value
        out["val_loss"] = total / max(len(self.sigmas), 1)
        return out

    def on_step_end(self, state: TrainState, out) -> None:
        was_training = self.model.training
        self.eval_mode()
        self.model.eval()
        try:
            state.metrics.update(self.evaluate())
        finally:
            self.model.train(was_training)
            self.train_mode()
