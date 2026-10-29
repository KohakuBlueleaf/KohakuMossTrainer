"""Uniform stream rescale: every writer of the residual stream times c and every norm eps
times c^2 leaves every norm output, and so the velocity, unchanged (exact for a power of
two c). See docs/concepts/massive-channels.md#5-rescaling-the-whole-stream."""

import math

import torch

from kohakumoss.heal.channels import writer_names
from kohakumoss.heal.sites import model_sites, norm_path


def power_of_two_scale(block_rms: list[float]) -> float:
    """The power of two nearest 1 / (geometric mean of the per-block stream RMS)."""
    mean_log = sum(math.log2(r) for r in block_rms) / len(block_rms)
    return 2.0 ** -round(mean_log)


@torch.no_grad()
def rescale_stream(model, c: float, eps: bool = True) -> dict[str, float]:
    """Writers times c in place; with `eps` every reader norm's eps times c^2. Returns
    the previous eps per norm path."""
    params = dict(model.named_parameters())
    for name in writer_names(len(model.text_embedder.blocks), model.cfg.depth):
        params[name].mul_(c)
    before = {}
    if eps:
        for site in model_sites(model):
            mod = model.get_submodule(norm_path(site))
            before[norm_path(site)] = mod.eps
            mod.eps = mod.eps * c * c
    return before


def restore_eps(model, before: dict[str, float]) -> None:
    for path, eps in before.items():
        model.get_submodule(path).eps = eps
