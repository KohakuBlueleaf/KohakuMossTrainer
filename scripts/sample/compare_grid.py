"""Comparison grid of the base model then each of CHECKPOINTS over PROMPTS, each prompt
with its seed and the same sampler settings; GRID_LAYOUT puts the prompts on the
columns (default) or the rows. Writes grid.png, every single image
(r<prompt>_c<weight>.png) and metadata.json. The DiT is loaded once.

    kogine run scripts/sample/compare_grid.py --config <grid config>
    kogine run scripts/sample/compare_grid.py --set WEIGHTS=<root> --set OUT=out/grid \
        --set CHECKPOINTS=runs/q/checkpoints/step_00000100/trainable.safetensors
"""

import json
from pathlib import Path
from typing import Annotated

import torch
from kohakuengine import FormatArg

from kohakumoss.encoders.vae import resolve_vae_dtype, vae_dtype_problems
from kohakumoss.inference import MossPipeline, grid, knobs
from kohakumoss.inference.placement_report import forward_tokens
from kohakumoss.logging.log import get_logger, hl, setup
from kohakumoss.memory.knob_map import mixing_problems, primary_names
from kohakumoss.model.numerics import fix_problems
from kohakumoss_common.knobs import check_script, choice_problems, prompt_list

log = get_logger("scripts.compare_grid")
LAYOUTS = {"prompt-cols": "cols", "prompt-rows": "rows"}
GRID_CHOICES = ("VARIANT", "PRECISION", "MEMORY_PROFILE", "BASE_PRECISION", "FP16_FIX")
GRID_CHOICES += ("ADAPTER_MODE", "CFG_MODE", "TUNE")

# ===================== MODEL =====================
WEIGHTS = ""  # ComfyUI models folder | HF weights root (base/, sft/) | one pipeline dir
VARIANT = "base"  # "base" | "sft": the arch preset; WEIGHTS/<VARIANT> is read when it exists (ComfyUI: the DiT file)
TRANSFORMER = ""  # DiT override (file / dir; ComfyUI folder: a file name); "" = WEIGHTS
TEXT_ENCODER = ""  # text encoder as in sample.py ("" = WEIGHTS' own)
VAE = ""  # VAE as in sample.py ("" = WEIGHTS' own)
CHECKPOINTS = (
    []
)  # adapter files, one grid line each after the base: path or (path, multiplier)
LABELS = []  # label per checkpoint; missing = "step N" from the path
MULTIPLIER = 1.0  # multiplier of checkpoints given as a bare path
ADAPTER_MODE = "bypass"  # "bypass" | "merge" (both undone between columns)
PRECISION = "bf16"  # compute dtype: "bf16" | "fp16" | "fp32" | "auto"
FP16_FIX = "none"  # "none" | "online" | "fp32_residual"
FP16_FIX_KWARGS = {}  # e.g. {"c": 0.0625}
TUNE = "off"  # "off" (stored per-card decisions, else fixed defaults; nothing timed) | "table" (missing decisions also tuned on first use, then stored)

# ===================== DEVICES =====================
DEVICE = "cuda"
TEXT_DEVICE = ""  # text encoder device, e.g. "cpu"; "" = DEVICE
TEXT_ENCODE = (
    {}
)  # TextEncoderPlan fields as in sample.py ("free_after_use": True frees it after the one encode)
VAE_ENCODE = {}  # VaePlan fields as in sample.py
VAE_DTYPE = "auto"  # "auto" (bf16, fp16 on fp16-only GPUs) | "bf16" | "fp16" | "fp32"
MEMORY_PROFILE = ""  # DiT memory profile: "" (resident) | a catalog profile | "speed" | "balanced" | "low" | "minimal" | "manual" | "auto" (the fastest that fits)
MEMORY_OVERRIDES = {}  # MemoryPlan fields
BASE_PRECISION = "auto"  # DiT base: "auto" (what the file stores) | "compute" | "int8" | "fp8" | "fp8_compute" | "mxfp8"
FP8_RECIPE = "row"  # fp8_compute scaling: "tensor" | "row" | "block"
# MEMORY_PROFILE="manual": the DiT placement, as in sample.py (None = the default).
STREAM = None  # "off" (resident) | "ring" (block weights streamed through device slots)
STREAM_RESIDENT = None  # device block slots when streaming (>= 2)
PINNED = None  # pinned host memory for the streamed weights

# ===================== PROMPTS =====================
PROMPTS: Annotated[list, FormatArg(prompt_list)] = (
    []
)  # one grid line each; --set takes list syntax ["a, b", "c"], a bare string is one prompt
NEGATIVE = ""  # negative prompt of every prompt
SIZES: list = ["1024x1024"]  # "HxW" or (H, W) per prompt; the last one repeats
SEEDS = []  # per prompt; missing ones are SEED_BASE + prompt index
SEED_BASE = 0

# ===================== SAMPLER =====================
STEPS = 50
CFG = 3.0
SHIFT = 5.0
SKIP_LAYERS = "default"  # "default" | "none" | "22,23"
CFG_MODE = "packed"  # "packed" | "two_pass"
BATCH = 4  # prompts per generate batch

# ===================== OUTPUT =====================
OUT = ""  # output directory
GRID_LAYOUT = "prompt-cols"  # "prompt-cols" (prompts head the columns, weights the rows) | "prompt-rows"
LABEL_LINES = 8  # wrapped lines per label at most
FONT_SIZE = 0  # label font size in pixels; 0 = scaled to the image width


def checkpoints_of(k: dict) -> list[tuple[str, float]]:
    """(path, multiplier) per checkpoint; bare paths get MULTIPLIER."""
    return [
        (c, float(k["MULTIPLIER"])) if isinstance(c, str) else (str(c[0]), float(c[1]))
        for c in k["CHECKPOINTS"]
    ]


def seeds_of(k: dict) -> list[int]:
    seeds = list(k["SEEDS"])
    return [
        seeds[r] if r < len(seeds) else k["SEED_BASE"] + r
        for r in range(len(k["PROMPTS"]))
    ]


def sizes_of(k: dict) -> list[tuple[int, int]]:
    """(H, W) per prompt: SIZES in order, the last one repeating."""
    sizes = [knobs.parse_size(s) for s in k["SIZES"]] or [knobs.DEFAULT_SIZE]
    return [sizes[min(r, len(sizes) - 1)] for r in range(len(k["PROMPTS"]))]


def problems(k: dict) -> list[str]:
    """Every problem with the grid knobs (empty = valid)."""
    p = [f"{name} is empty" for name in ("WEIGHTS", "OUT", "PROMPTS") if not k[name]]
    p += [
        f"CHECKPOINTS entry {c!r} does not exist"
        for c, _ in checkpoints_of(k)
        if not Path(c).exists()
    ]
    valid = k["PRECISION"] in knobs.DTYPE_NAMES
    fp16 = valid and knobs.compute_dtype(k["PRECISION"], k["DEVICE"]) == torch.float16
    p += fix_problems(
        k["FP16_FIX"],
        k["FP16_FIX_KWARGS"],
        fp16,
        knobs.carries_fix(k["TRANSFORMER"]),
        "PRECISION='fp16'",
        "TRANSFORMER",
    )
    p += choice_problems(k, {n: knobs.CHOICES[n] for n in GRID_CHOICES})
    p += mixing_problems(
        k, k["MEMORY_PROFILE"], "MEMORY_PROFILE", knobs.INFERENCE_KNOBS
    )
    if k["GRID_LAYOUT"] not in LAYOUTS:
        p.append(f"GRID_LAYOUT={k['GRID_LAYOUT']!r} not in {tuple(LAYOUTS)}")
    p += knobs.component_problems(k)
    p += vae_dtype_problems(k["VAE_DTYPE"], k["DEVICE"])
    return p + knobs.size_problems(k) + knobs.encoder_problems(k)


def main() -> None:
    setup()
    k = primary_names(check_script(globals(), knobs.RENAMED))
    found = problems(k)
    if found:
        raise SystemExit("invalid configuration:\n  - " + "\n  - ".join(found))
    memory, overrides = knobs.memory_of(k)
    pipe = MossPipeline.from_pretrained(
        k["WEIGHTS"],
        variant=k["VARIANT"],
        transformer=k["TRANSFORMER"] or None,
        text_encoder_file=k["TEXT_ENCODER"],
        vae_file=k["VAE"],
        dtype=knobs.compute_dtype(k["PRECISION"], k["DEVICE"]),
        device=k["DEVICE"],
        text_device=k["TEXT_DEVICE"] or None,
        text_plan=knobs.text_spec(k),
        vae_plan=knobs.vae_spec(k),
        memory=memory,
        memory_overrides=overrides,
        fp16_fix=k["FP16_FIX"],
        fp16_fix_kwargs=dict(k["FP16_FIX_KWARGS"]),
        tune=k["TUNE"],
        vae_dtype=resolve_vae_dtype(k["VAE_DTYPE"], k["DEVICE"]),
        forward_tokens=forward_tokens(
            sizes_of(k), min(k["BATCH"], len(k["PROMPTS"])), k["CFG"], k["CFG_MODE"]
        ),
    )
    cols = grid.columns(checkpoints_of(k), k["LABELS"])
    prompts, seeds = list(k["PROMPTS"]), seeds_of(k)
    images = grid.render_columns(
        pipe,
        cols,
        prompts,
        mode=k["ADAPTER_MODE"],
        negative_prompts=[k["NEGATIVE"]] * len(prompts),
        sizes=sizes_of(k),
        steps=k["STEPS"],
        cfg=k["CFG"],
        shift=k["SHIFT"],
        skip_layers=knobs.skip_layers(k["SKIP_LAYERS"]),
        seeds=seeds,
        batch=k["BATCH"],
        cfg_mode=k["CFG_MODE"],
    )
    out = Path(k["OUT"])
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for c, col in enumerate(cols):
        for r, image in enumerate(images[c]):
            name = f"r{r:02d}_c{c:02d}.png"
            image.save(out / name)
            files.append({"file": name, "row": r, "column": c, "label": col.label})
    prompt_labels = [f"{s}: {p}" for p, s in zip(prompts, seeds)]
    sheet = grid.compose(
        images,
        [c.label for c in cols],
        prompt_labels,
        prompt_axis=LAYOUTS[k["GRID_LAYOUT"]],
        font_size=int(k["FONT_SIZE"]),
        max_lines=int(k["LABEL_LINES"]),
    )
    sheet.save(out / "grid.png")
    meta = {
        "knobs": k,
        "columns": [{"label": c.label, "adapter": c.adapter} for c in cols],
        "rows": [{"prompt": p, "seed": s} for p, s in zip(prompts, seeds)],
        "images": files,
    }
    (out / "metadata.json").write_text(
        json.dumps(meta, indent=1, default=str), encoding="utf-8"
    )
    log.info(
        f"wrote {hl(out / 'grid.png', 'path')}", rows=len(prompts), columns=len(cols)
    )


if __name__ == "__main__":
    main()
