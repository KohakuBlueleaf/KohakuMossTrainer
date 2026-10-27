"""Sample-script knobs (named and valued as the train script's): parse helpers,
collection from globals and validation into one SystemExit."""

import os
from collections.abc import Sequence

import torch

from kohakumoss.encoders.knob_names import ENCODER_FIELDS, RENAMED_SAMPLE, SAMPLE_NAMES
from kohakumoss.encoders.text.plan import plan_problems as text_problems
from kohakumoss.encoders.text.plan import text_plan
from kohakumoss.encoders.vae import vae_dtype_problems
from kohakumoss.memory.knob_map import (
    MANUAL,
    MEMORY_KNOBS,
    PLACEMENT_KNOBS,
    knob_fields,
    mixing_problems,
    primary_names,
)
from kohakumoss.memory.profiles import PROFILES
from kohakumoss.model.io.load import read_fp16_fix
from kohakumoss.model.numerics import fix_problems
from kohakumoss.offload.encoders.vae import plan_problems as vae_problems
from kohakumoss.offload.encoders.vae import vae_plan
from kohakumoss.types.arch import PRESETS
from kohakumoss.weights import DIT, TEXT, VAE, component_path, is_comfy_folder
from kohakumoss_common.knobs import choice_problems, collect, exit_on, unknown_problems
from kohakumoss_data.resolution.geometry import LATENT_FACTOR
from kohakumoss_kernels.dtypes import BY_NAME
from kohakumoss_kernels.hardware.arch import arch_of
from kohakumoss_kernels.tuning.tuner import MODES as TUNE_MODES

INFERENCE_KNOBS = {name: MEMORY_KNOBS[name] for name in PLACEMENT_KNOBS}
KNOWN = frozenset(
    """WEIGHTS VARIANT TRANSFORMER TEXT_ENCODER VAE PROMPT PROMPTS PROMPTS_FILE NEGATIVE SIZES SEEDS
    SEED_BASE STEPS CFG SHIFT SKIP_LAYERS CFG_MODE BATCH PRECISION FP16_FIX
    FP16_FIX_KWARGS ADAPTERS ADAPTER_MODE DEVICE TEXT_DEVICE TEXT_ENCODE VAE_ENCODE
    VAE_DTYPE TUNE OUT MEMORY_PROFILE MEMORY_OVERRIDES BASE_PRECISION FP8_RECIPE STREAM
    STREAM_RESIDENT PINNED PROGRESS INT8_COMPUTE""".split()
)
BASE_PRECISIONS = ("auto", "compute", "int8", "fp8", "fp8_compute", "mxfp8")
FP8_RECIPES = ("tensor", "row", "block")
TORCH_DTYPES = BY_NAME
DTYPE_NAMES = (*TORCH_DTYPES, "auto")
FIXES = ("none", "online", "fp32_residual")
DEPTH = min(cfg.depth for cfg in PRESETS.values())
CHOICES = {
    "VARIANT": tuple(PRESETS),
    "PRECISION": DTYPE_NAMES,
    "MEMORY_PROFILE": ("", *PROFILES, "auto"),
    "BASE_PRECISION": BASE_PRECISIONS,
    "FP8_RECIPE": FP8_RECIPES,
    "FP16_FIX": FIXES,
    "CFG_MODE": ("packed", "two_pass"),
    "ADAPTER_MODE": ("merge", "bypass"),
    "TUNE": TUNE_MODES,
}
DEFAULT_SIZE = (1024, 1024)
RENAMED = {
    **RENAMED_SAMPLE,
    **{f"SAMPLE_{n}": f"the sample script's name is {n}" for n in SAMPLE_NAMES},
    "SAMPLE_SIZE": "the sample script's name is SIZES",
    "SAMPLE_STEPS": "the sample script's name is STEPS",
}


def compute_dtype(name: str, device) -> torch.dtype:
    """PRECISION -> torch dtype; "auto" is bf16 where `device` has it (the host
    included), else fp16."""
    if name != "auto":
        return TORCH_DTYPES[name]
    device = torch.device(device)
    has_bf16 = not _on_cuda(device) or arch_of(device).has_bf16
    return torch.bfloat16 if has_bf16 else torch.float16


def _on_cuda(device) -> bool:
    """`device` is a CUDA device this host has."""
    return torch.device(device).type == "cuda" and torch.cuda.is_available()


def parse_size(value: str | Sequence[int]) -> tuple[int, int]:
    """`"HxW"` or `(H, W)` -> `(H, W)` in pixels."""
    if isinstance(value, str):
        h, w = value.lower().split("x")
        return int(h), int(w)
    h, w = value
    return int(h), int(w)


def parse_adapter(value: str | Sequence) -> tuple[str, float]:
    """`"path[:multiplier]"` or `(path, multiplier)` -> `(path, multiplier)`; a bare
    path (also `C:\\x.safetensors`) gets multiplier 1.0."""
    if not isinstance(value, str):
        path, mult = value
        return str(path), float(mult)
    path, _, mult = value.rpartition(":")
    if not path or not mult.replace(".", "", 1).lstrip("-").isdigit():
        return value, 1.0
    return path, float(mult)


def skip_layers(value: str | Sequence[int]) -> tuple[int, ...] | None:
    """`"default"` -> None (the pipeline default), `"none"` -> (), `"22,23"` or a
    sequence of ints -> that tuple."""
    match value:
        case "default":
            return None
        case "none":
            return ()
        case str():
            return tuple(int(v) for v in value.split(","))
    return tuple(int(v) for v in value)


def skip_problems(layers: Sequence[int] | None, depth: int = DEPTH) -> list[str]:
    """Skip-layer indices outside the backbone's blocks [0, depth)."""
    bad = [i for i in layers or () if not 0 <= i < depth]
    if not bad:
        return []
    return [f"SKIP_LAYERS {bad} outside the {depth} blocks (0 .. {depth - 1})"]


def text_spec(k: dict) -> dict:
    """The TextEncoderPlan fields of TEXT_ENCODE (memory default 'auto'; TEXT_DEVICE
    'cpu' is memory 'cpu')."""
    spec = {"memory": "auto", **dict(k.get("TEXT_ENCODE") or {})}
    if str(k.get("TEXT_DEVICE") or "").startswith("cpu"):
        spec["memory"] = "cpu"
    return spec


def vae_spec(k: dict) -> dict:
    """The VaePlan fields of VAE_ENCODE (memory default 'auto')."""
    return {"memory": "auto", **dict(k.get("VAE_ENCODE") or {})}


def carries_fix(transformer: str) -> bool:
    """TRANSFORMER is a single file whose metadata records an fp16-fix fold (a file
    whose header does not parse records none; loading it reports that)."""
    if not transformer:
        return False
    try:
        return read_fp16_fix(transformer) is not None
    except (OSError, ValueError):
        return False


def _missing(name: str, path: str) -> list[str]:
    return [] if os.path.exists(path) else [f"{name}={path!r} does not exist"]


COMPONENT_KNOBS = {DIT: "TRANSFORMER", TEXT: "TEXT_ENCODER", VAE: "VAE"}


def component_problems(k: dict) -> list[str]:
    """TRANSFORMER / TEXT_ENCODER / VAE: a set path exists; in a ComfyUI WEIGHTS
    folder a file name (or the default) names a file there."""
    p = []
    comfy = bool(k["WEIGHTS"]) and is_comfy_folder(k["WEIGHTS"])
    for which, name in COMPONENT_KNOBS.items():
        choice = k.get(name) or ""
        if comfy:
            try:
                component_path(k["WEIGHTS"], which, choice, k["VARIANT"])
            except FileNotFoundError as err:
                p.append(f"{name}: {err}")
        elif choice:
            p += _missing(name, choice)
    return p


def size_problems(k: dict) -> list[str]:
    """SIZES entries that are not 'HxW' / (H, W) multiples of the latent factor."""
    p = []
    for entry in k["SIZES"]:
        try:
            h, w = parse_size(entry)
        except (TypeError, ValueError):
            p.append(f"SIZES entry {entry!r} is not 'HxW' or (H, W)")
            continue
        if h <= 0 or w <= 0 or h % LATENT_FACTOR or w % LATENT_FACTOR:
            p.append(
                f"SIZES entry {entry!r}: H and W must be positive multiples of "
                f"{LATENT_FACTOR}"
            )
    return p


def _adapter_problems(k: dict) -> list[str]:
    p = []
    for entry in k["ADAPTERS"]:
        try:
            path, _ = parse_adapter(entry)
        except (TypeError, ValueError):
            p.append(f"ADAPTERS entry {entry!r} is not 'path[:mult]' or (path, mult)")
            continue
        p += _missing("ADAPTERS entry", path)
    return p


def _prompt_problems(k: dict) -> list[str]:
    if not (k["PROMPTS"] or k["PROMPT"] or k["PROMPTS_FILE"]):
        return ["no prompts: set PROMPT, PROMPTS or PROMPTS_FILE"]
    if k["PROMPTS_FILE"]:
        return _missing("PROMPTS_FILE", k["PROMPTS_FILE"])
    return []


def encoder_problems(k: dict) -> list[str]:
    """TEXT_ENCODE (TextEncoderPlan fields) and VAE_ENCODE (VaePlan fields)."""
    p = []
    try:
        plan = text_plan(text_spec(k))
        device = "cpu" if plan.memory == "cpu" else (k["TEXT_DEVICE"] or k["DEVICE"])
        checked = device if _on_cuda(device) or device == "cpu" else None
        p += [f"TEXT_ENCODE: {m}" for m in text_problems(plan, checked)]
    except (TypeError, ValueError) as err:
        p.append(f"TEXT_ENCODE: {err}")
    try:
        p += [f"VAE_ENCODE: {m}" for m in vae_problems(vae_plan(vae_spec(k)))]
    except (TypeError, ValueError) as err:
        p.append(f"VAE_ENCODE: {err}")
    return p


def memory_of(k: dict) -> tuple[str | None, dict]:
    """(DiT memory profile or None, MemoryPlan overrides) of MEMORY_PROFILE /
    BASE_PRECISION / FP8_RECIPE / INT8_COMPUTE / MEMORY_OVERRIDES: a quantized
    BASE_PRECISION places the DiT under the `speed` profile unless MEMORY_PROFILE
    names one; `compute` is
    passed on (a pre-quantized DiT file is dequantized at load); `auto` keeps what the
    file stores."""
    overrides = dict(k.get("MEMORY_OVERRIDES") or {})
    precision = k.get("BASE_PRECISION", "auto")
    memory = k.get("MEMORY_PROFILE") or None
    if memory == MANUAL:
        overrides.update(knob_fields(k, INFERENCE_KNOBS))
    if precision == "compute":
        overrides.setdefault("base_precision", precision)
    elif precision != "auto":
        overrides.setdefault("base_precision", precision)
        overrides.setdefault("fp8_recipe", k.get("FP8_RECIPE", "row"))
        memory = memory or "speed"
    if k.get("INT8_COMPUTE"):
        overrides.setdefault("int8_compute", True)
    return memory, overrides


def validate(k: dict) -> list[str]:
    """Every problem with the knob set (empty list = valid); missing paths are reported
    with the text "does not exist"."""
    p = unknown_problems(k, KNOWN, RENAMED, ENCODER_FIELDS)
    p += choice_problems(k, CHOICES)
    memory = k.get("MEMORY_PROFILE")
    p += mixing_problems(k, memory, "MEMORY_PROFILE", INFERENCE_KNOBS)
    if not k["WEIGHTS"]:
        p.append("WEIGHTS is empty (a weights root or pipeline dir)")
    else:
        p += _missing("WEIGHTS", k["WEIGHTS"])
    p += component_problems(k)
    if not k["OUT"]:
        p.append("OUT is empty (output directory)")
    if k["STEPS"] <= 0:
        p.append(f"STEPS must be > 0, got {k['STEPS']}")
    if k["BATCH"] < 1:
        p.append(f"BATCH must be >= 1, got {k['BATCH']}")
    if k["CFG"] < 0:
        p.append(f"CFG must be >= 0, got {k['CFG']}")
    fp16 = k["PRECISION"] == "fp16" or (
        k["PRECISION"] == "auto" and compute_dtype("auto", k["DEVICE"]) == torch.float16
    )
    p += fix_problems(
        k["FP16_FIX"],
        k["FP16_FIX_KWARGS"],
        fp16,
        carries_fix(k["TRANSFORMER"]),
        "PRECISION='fp16'",
        "TRANSFORMER",
    )
    try:
        p += skip_problems(skip_layers(k["SKIP_LAYERS"]))
    except (TypeError, ValueError):
        p.append(
            f"SKIP_LAYERS={k['SKIP_LAYERS']!r}: 'default', 'none', '22,23' or a tuple"
        )
    p += vae_dtype_problems(k.get("VAE_DTYPE", "auto"), k["DEVICE"])
    p += encoder_problems(k)
    p += size_problems(k) + _adapter_problems(k) + _prompt_problems(k)
    return list(dict.fromkeys(p))


def check(g: dict) -> dict:
    """Knobs from globals (MEMORY_PROFILE in its primary spelling); raises one
    SystemExit listing every problem."""
    k = primary_names(collect(g))
    exit_on(validate(k))
    return k
