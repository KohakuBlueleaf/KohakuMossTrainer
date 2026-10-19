"""Pre-quantized text-encoder files: the quantized linears of a frozen text model as
`<n>.weight` (the q codes) plus `<n>.weight_scale` (and `<n>.bias`), every other tensor
in the save dtype, the precision / recipe / layer list in the safetensors metadata
(`kohakumoss_text_quant`). The streamed text-encoder load (`encoders.text.load`)
attaches the stored bases. See
docs/internals/text-encoder.md#pre-quantized-text-encoder."""

import json
from pathlib import Path

import torch
from torch import Tensor, nn

from kohakumoss.quant.compute import COMPUTE, ComputeWeight
from kohakumoss.quant.convert import QuantLinear
from kohakumoss.quant.stored import SCALE, read_meta_record
from kohakumoss.quant.text import GroupQuantLinear
from kohakumoss.quant.weights import QuantWeight

TEXT_QUANT_KEY = "kohakumoss_text_quant"


def text_of(model: nn.Module) -> nn.Module:
    """The language model of a (possibly multimodal) text-encoder checkpoint."""
    return getattr(model, "language_model", model)


def _quant_linears(text: nn.Module) -> dict[str, QuantLinear]:
    return {p: m for p, m in text.named_modules() if isinstance(m, QuantLinear)}


def quantized_text_state(
    text: nn.Module, precision: str, dtype: torch.dtype
) -> tuple[dict[str, Tensor], dict[str, str]]:
    """(state, metadata) of a text model whose linears were replaced by QuantLinear:
    stored layers as q / scale / bias, every other tensor cast to `dtype`, on the host.
    Per-group quantized linears have no stored layout and raise."""
    grouped = [p for p, m in text.named_modules() if isinstance(m, GroupQuantLinear)]
    if grouped:
        raise ValueError(f"grouped quantized linears cannot be stored: {grouped[:4]}")
    layers = _quant_linears(text)
    recipes = {
        m.qweight.recipe
        for m in layers.values()
        if isinstance(m.qweight, ComputeWeight)
    }
    if len(recipes) > 1:
        raise ValueError(f"one fp8 recipe per file, found {sorted(recipes)}")
    state = {}
    for path, lin in layers.items():
        state[f"{path}.weight"] = lin.qweight.q.contiguous().cpu()
        state[path + SCALE] = lin.qweight.scale.contiguous().cpu()
        if lin.bias is not None:
            state[f"{path}.bias"] = lin.bias.detach().to(dtype).contiguous().cpu()
    for key, t in text.state_dict().items():
        if any(key.startswith(p + ".") for p in layers):
            continue
        t = t.detach()
        state[key] = (t.to(dtype) if t.is_floating_point() else t).contiguous().cpu()
    record = {
        "precision": precision,
        "recipe": recipes.pop() if recipes else None,
        "layers": sorted(layers),
    }
    return state, {TEXT_QUANT_KEY: json.dumps(record)}


def read_text_record(path: str | Path) -> dict | None:
    """The text-quant record of a file, if it carries one."""
    return read_meta_record(path, TEXT_QUANT_KEY)


def _under(key: str, layers: set[str]) -> bool:
    """Whether `key` belongs to one of the stored layers (any depth below it)."""
    parts = key.split(".")
    return any(".".join(parts[:i]) in layers for i in range(1, len(parts)))


def stored_base(record: dict, q: Tensor, scale: Tensor):
    """The base of one stored layer's codes `q` and `scale` under the file's record."""
    if record["precision"] in COMPUTE:
        return ComputeWeight.from_stored(record["recipe"], q, scale)
    return QuantWeight(q, scale, record["precision"])
