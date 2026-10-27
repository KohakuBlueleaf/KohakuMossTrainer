"""Generate images with MossPipeline; writes PNGs `<index>_<seed>.png` numbered after
the highest index already in OUT, plus metadata-<first index>.json of every setting.
The knobs are named and valued as scripts/train/moss.py's.

    kogine run scripts/sample/sample.py --set WEIGHTS=<root> --set PROMPT="a cat" \
        --set OUT=out/samples
    kogine run scripts/sample/sample.py --set WEIGHTS=<root> --set PROMPTS_FILE=p.json \
        --set ADAPTERS=my_lora.safetensors:0.8 --set PRECISION=fp16 --set FP16_FIX=online \
        --set FP16_FIX_KWARGS={"c":0.0625} --set VAE_ENCODE={"memory":"offload"} \
        --set OUT=out/samples
    kogine run scripts/sample/sample.py --config <sample config> --set TEXT_DEVICE=cpu
    kogine run scripts/sample/sample.py --config <sample config> \
        --set TEXT_ENCODE={"memory":"ring","precision":"int8"} --set MEMORY_PROFILE=low

The text encoder follows TEXT_ENCODE (TextEncoderPlan fields; docs/internals/text-encoder.md),
the VAE VAE_ENCODE (VaePlan fields; docs/internals/offload.md#vae); an unset memory of
either is "auto", the fastest placement that fits next to the DiT. MEMORY_PROFILE places
the DiT under an inference memory profile and BASE_PRECISION / FP8_RECIPE set its base
(weight-only int8 / fp8, or fp8_compute / mxfp8 GEMMs; docs/guides/inference.md). The
metadata file records generate time, peak VRAM and the text-encoder stats.

A prompts file is a .txt (one prompt per line) or a .json list of strings or of objects
{"prompt", "negative", "size": [H, W], "seed"}.
"""

import json
import time
from pathlib import Path
from typing import Annotated

from kohakuengine import FormatArg

from kohakumoss.encoders.vae import resolve_vae_dtype
from kohakumoss.inference import MossPipeline, knobs, outputs
from kohakumoss.inference.placement_report import forward_tokens
from kohakumoss.logging.log import get_logger, hl, setup
from kohakumoss.offload.accel import configure_allocator
from kohakumoss_common.knobs import prompt_list
from kohakumoss_kernels import compile_setup

log = get_logger("scripts.sample")

# ===================== MODEL =====================
WEIGHTS = ""  # ComfyUI models folder (diffusion_models/, text_encoders/, vae/) | HF weights root (base/, sft/) | one pipeline dir
VARIANT = "base"  # "base" | "sft": the arch preset; WEIGHTS/<VARIANT> is read when it exists (ComfyUI: the DiT file moss_image_<VARIANT>_*)
TRANSFORMER = ""  # DiT override (file / dir, e.g. an fp16-fix checkpoint; ComfyUI folder: a file name under diffusion_models/); "" = WEIGHTS (ComfyUI: the int8 ConvRot file, else bf16)
TEXT_ENCODER = ""  # "" = WEIGHTS' text encoder (ComfyUI: qwen3.5_9b_int8_convrot, else _bf16, else _w4a8_gptq) | a file name under text_encoders/ | a path
VAE = ""  # "" = WEIGHTS' VAE (ComfyUI: flux2-vae_small, else flux2-vae) | a file name under vae/ | a path
ADAPTERS = []  # [(path, multiplier), ...] or "path[:multiplier]" strings
ADAPTER_MODE = "merge"  # "merge" | "bypass"
PRECISION = "bf16"  # compute dtype: "bf16" | "fp16" | "fp32" | "auto" (bf16 where supported, else fp16)
FP16_FIX = "none"  # "none" | "online" | "fp32_residual" (an fp16-fix TRANSFORMER carries its fold)
FP16_FIX_KWARGS = {}  # e.g. {"c": 0.0625, "clamp": True}
TUNE = "off"  # "off" (stored per-card decisions, else fixed defaults; nothing timed) | "table" (missing decisions also tuned on first use, then stored)

# ===================== DEVICES =====================
DEVICE = "cuda"
TEXT_DEVICE = ""  # text encoder device, e.g. "cpu"; "" = DEVICE
TEXT_ENCODE = {}  # TextEncoderPlan fields, as in training:
#   "memory": "auto" (default) | "resident" | "offload" | "ring" | "cpu"; "dtype": "auto" | "bf16" | "fp16"
#   "precision": "auto" (what the weights store) | "compute" | "int8" | "fp8" | "fp8_compute" | "mxfp8"; "w4a8": "auto" | "int8" | "packed"; "quant_group", "fp8_recipe", "layout", "stored", ...
#   "free_after_use": True drops the text encoder after its first pass
VAE_ENCODE = (
    {}
)  # VaePlan fields, as in training: "memory": "auto" (default) | "resident" | "offload" | "ring" | "cpu"; "tile": True (tiled decode)
VAE_DTYPE = "auto"  # "auto" (bf16, fp16 on fp16-only GPUs) | "bf16" | "fp16" | "fp32"
MEMORY_PROFILE = ""  # DiT memory profile: "" (resident) | a catalog profile | "speed" | "balanced" | "low" | "minimal" | "manual" | "auto" (the fastest that fits)
MEMORY_OVERRIDES = {}  # MemoryPlan fields
# MEMORY_PROFILE="manual": these set the DiT placement (None = the default in brackets); with a profile they stay None.
STREAM = None  # "off" (resident) | "ring" (block weights streamed through device slots)  [off]
STREAM_RESIDENT = None  # device block slots when streaming (>= 2)  [0]
PINNED = None  # pinned host memory for the streamed weights  [True]
BASE_PRECISION = "auto"  # DiT base: "auto" (what the DiT file stores) | "compute" (16-bit; a quantized file dequantized) | "int8" | "fp8" | "fp8_compute" | "mxfp8"
FP8_RECIPE = "row"  # fp8_compute scaling: "tensor" | "row" | "block"
INT8_COMPUTE = False  # int8 bases run int8 GEMMs (activations quantized per row), not weight-only; CUDA

# ===================== PROMPTS =====================
PROMPT = ""  # one prompt
PROMPTS: Annotated[list, FormatArg(prompt_list)] = (
    []
)  # list of prompts (before PROMPT); --set takes list syntax ["a, b", "c"], a bare string is one prompt
PROMPTS_FILE = ""  # .txt or .json prompts file (after PROMPT)
NEGATIVE = ""  # negative prompt for every item without its own
SIZES: list = [(1024, 1024)]  # (H, W) or "HxW" per prompt; the last one repeats
SEEDS = []  # per prompt; missing ones are SEED_BASE + index
SEED_BASE = 0

# ===================== SAMPLER =====================
STEPS = 50
CFG = 3.0
SHIFT = 5.0
SKIP_LAYERS = "default"  # negative-branch skipped blocks: "default" | "none" | "22,23"
CFG_MODE = "packed"  # "packed" | "two_pass"
BATCH = 4  # prompts per generate batch

# ===================== OUTPUT =====================
OUT = ""  # output directory
PROGRESS = True  # tqdm bars for the sampler steps and the decode (INFO level)


def load_items(k: dict) -> list[dict]:
    """One dict per image: prompt, negative, size, seed (file entries keep their own)."""
    items = [{"prompt": p} for p in k["PROMPTS"]]
    if k["PROMPT"]:
        items.append({"prompt": k["PROMPT"]})
    if k["PROMPTS_FILE"]:
        path = Path(k["PROMPTS_FILE"])
        if path.suffix == ".json":
            for entry in json.loads(path.read_text(encoding="utf-8")):
                items.append(entry if isinstance(entry, dict) else {"prompt": entry})
        else:
            lines = path.read_text(encoding="utf-8").splitlines()
            items += [{"prompt": line} for line in lines if line.strip()]
    if not items:
        raise SystemExit("no prompts: set PROMPT, PROMPTS or PROMPTS_FILE")
    sizes = [knobs.parse_size(s) for s in k["SIZES"]] or [knobs.DEFAULT_SIZE]
    seeds = k["SEEDS"]
    for i, item in enumerate(items):
        item.setdefault("negative", k["NEGATIVE"])
        item["size"] = tuple(item.get("size") or sizes[min(i, len(sizes) - 1)])
        item.setdefault("seed", seeds[i] if i < len(seeds) else k["SEED_BASE"] + i)
    return items


def main() -> None:
    setup()
    k = knobs.check(globals())
    allocator = configure_allocator(k["DEVICE"])
    log.info(f"allocator {hl(allocator)}, compile pool: {hl(compile_setup.apply())}")
    items = load_items(k)
    memory, overrides = knobs.memory_of(k)
    pipe = MossPipeline.from_pretrained(
        k["WEIGHTS"],
        variant=k["VARIANT"],
        transformer=k["TRANSFORMER"] or None,
        text_encoder_file=k["TEXT_ENCODER"],
        vae_file=k["VAE"],
        dtype=knobs.compute_dtype(k["PRECISION"], k["DEVICE"]),
        device=k["DEVICE"],
        adapters=[knobs.parse_adapter(a) for a in k["ADAPTERS"]],
        adapter_mode=k["ADAPTER_MODE"],
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
            [it["size"] for it in items],
            min(k["BATCH"], len(items)),
            k["CFG"],
            k["CFG_MODE"],
        ),
    )
    te = pipe.place.text_encoder
    start = time.perf_counter()
    result = pipe.generate(
        [it["prompt"] for it in items],
        negative_prompts=[it["negative"] for it in items],
        sizes=[it["size"] for it in items],
        steps=k["STEPS"],
        cfg=k["CFG"],
        shift=k["SHIFT"],
        skip_layers=knobs.skip_layers(k["SKIP_LAYERS"]),
        seeds=[it["seed"] for it in items],
        batch=k["BATCH"],
        cfg_mode=k["CFG_MODE"],
        progress=bool(k["PROGRESS"]),
    )
    run = {"generate_s": time.perf_counter() - start}
    if result.settings.get("vram"):
        run["vram"] = result.settings["vram"]
    if te is not None and hasattr(te, "stats"):
        run["text_encoder"] = {"plan": str(te.plan), **te.stats}
    out = Path(k["OUT"])
    out.mkdir(parents=True, exist_ok=True)
    first = outputs.next_index(out)
    files = []
    for i, (image, item) in enumerate(zip(result.images, items), start=first):
        name = outputs.image_name(i, item["seed"])
        image.save(out / name)
        log.info(f"wrote {hl(out / name, 'path')}")
        files.append({"file": name, **item, "size": list(item["size"])})
    meta = {"knobs": k, "settings": result.settings, "run": run, "images": files}
    meta_path = out / outputs.metadata_name(first)
    meta_path.write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    log.info(f"wrote {hl(len(files))} images and {hl(meta_path, 'path')}")


if __name__ == "__main__":
    main()
