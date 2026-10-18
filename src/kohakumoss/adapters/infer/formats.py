"""Read adapter files of any supported key style into per-layer tensor groups."""

import json

from safetensors import safe_open
from torch import Tensor, nn

SUFFIXES = sorted(
    (
        "lora_up.weight",
        "lora_down.weight",
        "lora_mid.weight",
        "alpha",
        "dora_scale",
        "hada_w1_a",
        "hada_w1_b",
        "hada_w2_a",
        "hada_w2_b",
        "hada_t1",
        "hada_t2",
        "lokr_w1",
        "lokr_w1_a",
        "lokr_w1_b",
        "lokr_w2",
        "lokr_w2_a",
        "lokr_w2_b",
        "lokr_t1",
        "lokr_t2",
        "oft_blocks",
        "rescale",
        "a1.weight",
        "a2.weight",
        "b1.weight",
        "b2.weight",
        "bm.weight",
        "on_input",
        "weight",
        "diff",
        "diff_b",
        "w_norm",
        "b_norm",
        "q_layer.weight",
        "p_layer.weight",
        "lambda_layer",
        "base_q",
        "base_p",
        "base_lambda",
    ),
    key=len,
    reverse=True,
)
STEM_PREFIXES = ("lora_unet_", "lycoris_", "diffusion_model.", "transformer.")


def read(path: str) -> tuple[dict[str, Tensor], dict[str, str]]:
    """(tensors, metadata) of a safetensors file."""
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    return tensors, meta


def split_key(key: str) -> tuple[str, str] | None:
    """(stem, tensor name) of an adapter key, or None for an unknown layout."""
    for suffix in SUFFIXES:
        if key.endswith("." + suffix):
            return key[: -len(suffix) - 1], suffix
    return None


def stem_index(
    model: nn.Module, extra_prefixes: tuple[str, ...] = ()
) -> dict[str, str]:
    """Every key stem a layer may be saved under -> its module path."""
    index = {}
    for path, _ in model.named_modules():
        if not path:
            continue
        under = path.replace(".", "_")
        index[path] = path
        for prefix in STEM_PREFIXES:
            index[prefix + (under if prefix.endswith("_") else path)] = path
        for prefix in extra_prefixes:
            index[f"{prefix}_{under}"] = path
    return index


def group(
    tensors: dict[str, Tensor], index: dict[str, str]
) -> tuple[dict[str, dict[str, Tensor]], list[str]]:
    """{module path: {tensor name: tensor}} and the keys that matched no layer."""
    out: dict[str, dict[str, Tensor]] = {}
    unmatched = []
    for key, value in tensors.items():
        parts = split_key(key)
        if parts is None or parts[0] not in index:
            unmatched.append(key)
            continue
        stem, name = parts
        out.setdefault(index[stem], {})[name] = value
    return out, unmatched


def config_from_meta(meta: dict[str, str]) -> dict | None:
    """The adapter configuration fields a training-state file records, or None."""
    raw = meta.get("kohakumoss_spec")
    return None if raw is None else json.loads(raw)
