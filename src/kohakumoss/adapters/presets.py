"""MOSS target presets in LyCORIS preset semantics: `base[+modifier...]` expressions
whose regexes match backbone module paths (`layers.3.proj.q`, ...)."""

import copy
import os

from lycoris.utils.preset import read_preset

JOINT_ATTN = r"layers\.\d+\.proj\.(q|k|v|o)$"
JOINT_MLP = r"layers\.\d+\.proj\.(ffn_up_gate|ffn_down)$"
TEXT_BLOCK = r"text_embedder\.blocks\.\d+\.(qkv|proj|ffn_gate|ffn_up|ffn_down)$"
TEXT_PROJ = r"text_embedder\.proj$"
MODULATION = r"(shared_block_modulation|video_head_modulation|time_embedding\.\d+)$"
EMBED_HEAD = r"(video_embedder|final_linear_video)$"

JOINT_NORMS = r"layers\.\d+\.(norm1|norm2|q_norm|k_norm)$"
TEXT_NORMS = r"text_embedder\.blocks\.\d+\.(norm1|norm2|q_norm|k_norm)$"
FINAL_NORM = r"final_norm_video$"

BASES: dict[str, tuple[list[str], list[str]]] = {
    "attn": ([JOINT_ATTN], [JOINT_NORMS]),
    "attn-mlp": ([JOINT_ATTN, JOINT_MLP], [JOINT_NORMS]),
    "full-block": (
        [JOINT_ATTN, JOINT_MLP, TEXT_BLOCK],
        [JOINT_NORMS, TEXT_NORMS],
    ),
    "all": (
        [JOINT_ATTN, JOINT_MLP, TEXT_BLOCK, TEXT_PROJ, MODULATION, EMBED_HEAD],
        [JOINT_NORMS, TEXT_NORMS, FINAL_NORM],
    ),
}
MODIFIERS: dict[str, tuple[list[str], list[str]]] = {
    "text-refiner": ([TEXT_BLOCK, TEXT_PROJ], [TEXT_NORMS]),
    "modulation": ([MODULATION], []),
    "embed": ([EMBED_HEAD], [FINAL_NORM]),
}
PRESET_KEYS_DEFAULT = {
    "enable_conv": False,
    "target_module": [],
    "target_name": [],
    "module_algo_map": {},
    "name_algo_map": {},
    "use_fnmatch": False,
    "exclude_name": [],
}


def _expression(expr: str) -> tuple[dict, list[str]]:
    base, *mods = expr.split("+")
    if base not in BASES:
        raise KeyError(f"unknown MOSS preset {base!r}; known: {sorted(BASES)}")
    names, norms = (list(v) for v in BASES[base])
    for mod in mods:
        if mod not in MODIFIERS:
            raise KeyError(
                f"unknown preset modifier {mod!r}; known: {sorted(MODIFIERS)}"
            )
        extra_names, extra_norms = MODIFIERS[mod]
        names += [n for n in extra_names if n not in names]
        norms += [n for n in extra_norms if n not in norms]
    return {"target_name": names}, norms


def resolve_preset(preset: str | dict, prefix: str) -> tuple[dict, list[str]]:
    """(full LyCORIS preset dict with every key set, norm path patterns) for an
    expression, toml or dict. See docs/internals/adapter-kinds.md."""
    match preset:
        case dict():
            body, norms = copy.deepcopy(preset), list(preset.get("norm_name", []))
            body.pop("norm_name", None)
        case str() if preset.endswith(".toml") or os.path.isfile(preset):
            body = read_preset(preset)
            if body is None:
                raise ValueError(f"cannot read preset file {preset!r}")
            norms = []
        case str():
            body, norms = _expression(preset)
        case _:
            raise TypeError(f"preset must be str or dict, got {type(preset).__name__}")
    full = copy.deepcopy(PRESET_KEYS_DEFAULT)
    full.update(body)
    full["lora_prefix"] = prefix
    return full, norms
