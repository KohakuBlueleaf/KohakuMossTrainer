"""The official implementation shipped with the weights, probe prompts, and the relative
error the parity probes report."""

import importlib.util
from pathlib import Path

import torch

PROBE_PROMPTS = (
    "A photograph of a tabby cat sitting on a sunlit windowsill, shallow depth of field.",
    "1girl, solo, long hair, school uniform, cherry blossoms, outdoors, smile, looking at viewer",
    "An isometric illustration of a small floating island with a lighthouse, waterfalls "
    "pouring off the edges into clouds, soft pastel colors, highly detailed, with a "
    "wooden sign that reads 'KOHAKU' in bold letters.",
)


def official_module(weights: str | Path):
    """Import the official `pipeline_moss_image.py` shipped with the weights."""
    path = Path(weights) / "transformer" / "pipeline_moss_image.py"
    spec = importlib.util.spec_from_file_location("pipeline_moss_image", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """||a - b|| / ||b|| in fp64 on CPU."""
    a, b = a.double().cpu(), b.double().cpu()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()
