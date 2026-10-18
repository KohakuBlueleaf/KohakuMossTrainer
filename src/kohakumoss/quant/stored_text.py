"""Pre-quantized text-encoder files: the quantized linears of a frozen text model as
`<n>.weight` (the q codes) plus `<n>.weight_scale` (and `<n>.bias`), every other tensor
in the save dtype, the precision / recipe / layer list in the safetensors metadata
(`kohakumoss_text_quant`). Loading builds the model from its config without reading the
dense weights and attaches the stored bases, so nothing is quantized at load. See
docs/internals/text-encoder.md#pre-quantized-text-encoder."""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel
from transformers.initialization import no_init_weights

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


def _base(record: dict, q: Tensor, scale: Tensor):
    if record["precision"] in COMPUTE:
        return ComputeWeight.from_stored(record["recipe"], q, scale)
    return QuantWeight(q, scale, record["precision"])


@torch.no_grad()
def load_stored_text(
    model_dir: str | Path, path: str | Path, dtype: torch.dtype
) -> nn.Module:
    """The text-encoder model of `model_dir`'s config with the stored file's quantized
    linears attached and its other tensors loaded (no dense weight is read)."""
    record = read_text_record(path)
    if record is None:
        raise ValueError(f"{path} carries no {TEXT_QUANT_KEY} record")
    config = AutoConfig.from_pretrained(str(model_dir))
    with no_init_weights():
        model = AutoModel.from_config(config, dtype=dtype)
    text = text_of(model)
    with safe_open(str(path), framework="pt") as f:
        for p in record["layers"]:
            parent_path, _, name = p.rpartition(".")
            parent = text.get_submodule(parent_path) if parent_path else text
            dense = getattr(parent, name)
            q = f.get_tensor(f"{p}.weight")
            base = _base(record, q, f.get_tensor(p + SCALE))
            bias = f.get_tensor(f"{p}.bias") if f"{p}.bias" in f.keys() else None
            setattr(
                parent,
                name,
                QuantLinear.from_base(
                    dense.in_features, dense.out_features, base, bias, dtype
                ),
            )
        layers = set(record["layers"])
        rest = {
            k: f.get_tensor(k)
            for k in f.keys()
            if k.rpartition(".")[0] not in layers and not k.endswith(SCALE)
        }
    missing, unexpected = text.load_state_dict(rest, strict=False)
    left = sorted(k for k in missing if not _under(k, layers))
    if left or unexpected:
        raise ValueError(
            f"{path} does not match {model_dir}: missing {left[:4]}, "
            f"unexpected {list(unexpected)[:4]}"
        )
    return model
