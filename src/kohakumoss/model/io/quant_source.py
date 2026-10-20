"""The quantization source of a loaded backbone: its checkpoint by model key, and the
output-row multiplier of every fold applied in memory since the load. See
docs/internals/quant-kernels.md#source."""

from pathlib import Path

import torch
from torch import Tensor

from kohakumoss.model.backbone import MossBackbone
from kohakumoss.model.io.load import holds_values, read_fp16_fix
from kohakumoss.model.io.source import CheckpointSource, ModelKeyView
from kohakumoss.model.numerics.fold import output_row_scales, record_row_scales
from kohakumoss.quant.source import SourceWeights


def file_fold_record(fix: dict | None) -> dict | None:
    """An fp16-fix metadata record as a fold record (scales / via / pregate)."""
    if fix is None:
        return None
    return {
        "scales": [float(fix["c"])],
        "via": fix["via"],
        "pregate": [float(fix["s_attn"]), float(fix["s_ffn"])],
    }


def fold_since_load(model: MossBackbone, file_fold: dict | None) -> dict[str, Tensor]:
    """Module path -> per-output-row multiplier from the fold the file's weights carry
    to the fold the model carries now (only modules with a row != 1)."""
    now = output_row_scales(model)
    stored = record_row_scales(model, file_fold)
    out = {}
    for module in set(now) | set(stored):
        a, b = now.get(module), stored.get(module)
        a = torch.ones_like(b) if a is None else a
        b = torch.ones_like(a) if b is None else b
        ratio = a.float() / b.float()
        if bool((ratio != 1).any()):
            out[module] = ratio
    return out


def backbone_source(
    model: MossBackbone,
) -> tuple[SourceWeights | None, dict[str, Tensor]]:
    """(the checkpoint `model` was loaded from, by model weight name, its 16-bit and
    fp32 weights only (stored codes are not values); the fold rows applied since), or
    (None, {}) for a backbone not loaded from a file."""
    if model.weight_path is None:
        return None, {}
    path = Path(model.weight_path)
    view = ModelKeyView(CheckpointSource(path))

    def resolve(name: str) -> str | None:
        return name if holds_values(view, name) else None

    source = SourceWeights(view, resolve)
    return source, fold_since_load(model, file_fold_record(read_fp16_fix(path)))
