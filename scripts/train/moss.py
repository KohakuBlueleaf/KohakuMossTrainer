"""Train MOSS-Image (LyCORIS adapters or full finetune). Every knob is a global below.

kogine run scripts/train/moss.py --config configs/moss/recipes/recipes_lora_24g_bf16.py --set LR=2e-4
"""

from typing import Annotated

from kohakuengine import FormatArg

from kohakumoss.training.entry import run
from kohakumoss_common.knobs import prompt_list

# ===================== MODEL =====================
WEIGHTS = ""  # ComfyUI models folder (diffusion_models/ + text_encoders/ + vae/) | HF dir (transformer/ + text_encoder/ + vae/) | single DiT .safetensors (docs/guides/weights.md)
VARIANT = "base"  # "base" | "sft" (also picks the ComfyUI DiT file)
ARCH_OVERRIDES = {}  # MossArchConfig field overrides
TRANSFORMER = ""  # DiT: "" = WEIGHTS (ComfyUI folder: moss_image_<VARIANT>_int8_convrot, else _bf16; full FT: _bf16 first) | a file name under diffusion_models/ | a path
TEXT_ENCODER = ""  # "" = <WEIGHTS>/text_encoder (ComfyUI folder: qwen3.5_9b_int8_convrot, else _bf16, else _w4a8_gptq) | a file name under text_encoders/ | a path (Qwen3.5 dir or ComfyUI file)
TEXT_DEVICE = ""  # GPU the text encoder runs on ("" = the training device); "cpu" only with TEXT_CACHE="offline"
TEXT_ENCODE = {}  # TextEncoderPlan fields (docs/internals/text-encoder.md):
#   "memory": "auto" (default: fastest that fits next to the DiT step) | "resident" | "offload" (whole TE in per pass) | "ring" (layer ring; "stream" is accepted too)
#   "stream_resident": 2 (ring slots per layer kind), "pinned": True
#   "precision": "auto" (default: what the weights store; a ComfyUI int8 / W4A8 file keeps its bases) | "compute" | "int8" | "fp8" (weight-only) | "fp8_compute" | "mxfp8" (fp8 GEMMs, sm89+); "quant_group": 0 (per channel) | 64 | 128
#   "w4a8": "auto" (int8 decode when it fits resident next to the DiT step, else packed) | "int8" (decoded once) | "packed" (decoded per call): a ComfyUI W4A8 file
#   "stored": "" | a pre-quantized text-encoder file of that precision (scripts/export/text_quant.py; its stored bases attach at load)
#   "fp8_recipe": "row" | "tensor" | "block" (fp8_compute scaling)
#   "dtype": "auto" (bf16 where supported, else fp16) | "bf16" | "fp16"; "batch": 32 prompts / "max_tokens": 16384 tokens per kernel call
#   "layout": "packed" (varlen) | "padded" | "single"; "forward": "kernels" | "hf" (transformers modules; single / padded)
#   "ahead": 1 (TEXT_CACHE "online" / "off": host batches encoded per TE pass); "ahead_device": "host" | "device" (where results wait)
#   "kernels": {} (op -> backend, e.g. {"gated_delta_rule_varlen": "triton"})
# Sample prompts are encoded once at build; with TEXT_CACHE="offline" the TE is freed after that.
VAE = ""  # "" = <WEIGHTS>/vae (ComfyUI folder: flux2-vae_small, else flux2-vae) | a file name under vae/ | a path
VAE_ENCODE = {}  # VaePlan fields (docs/internals/offload.md#vae):
#   "memory": "resident" (default) | "offload" (VAE in per pass) | "ring" (block rings; "stream" is accepted too) | "auto" | "cpu" (LATENT_CACHE="offline" only)
#   "batch": 1 (same-size images per encode call), "ahead": 1 (LATENT_CACHE "online" / "off": host batches encoded per VAE pass), "tile": False (tiled / sliced encode and decode)
#   "compile": "auto" (channels_last + torch.compile on a resident CUDA VAE) | "on" | "off"
VAE_DTYPE = (
    "auto"  # "auto" (bf16 where the card has it, else fp16) | "bf16" | "fp16" | "fp32"
)
TRAIN_TEXT_REFINER = False  # adapters/full FT also on the 2 text-refiner blocks

# ===================== DATA =====================
DATASETS = (
    []
)  # [{"source": "kohya" | "folder" | "jsonl" | "parquet" | "webdataset" | "moss_shard", "path": ..., "weight": 1.0, ...}]
RESOLUTION_MODE = "resize"  # "resize" (native aspect, area ~= TARGET_AREA) | "bucket" | "shifted_square" | "square_crop"
TARGET_AREA = 1024 * 1024  # pixels
SIZE_STEP = 16  # pixel multiple (VAE /8 x patchify 2)
ASPECT_RANGE = (0.25, 4.0)  # min, max width/height
UPSCALE = "never"  # "never" | "to_area"
BATCHING = "tokens"  # "tokens" (greedy concat to MAX_TOKENS) | "tokens_ffd" | "count" (BATCH_SIZE)
MAX_TOKENS = 16384  # target joint tokens (image + text) per batch per rank
OVERLONG = "drop"  # sample longer than MAX_TOKENS: "drop" | "truncate_text" | "drop_tokens" (random image-token drop)
BATCH_SIZE = 4  # BATCHING="count": samples per batch per rank
MAX_SAMPLES = 0  # cap on samples per token batch; 0 = none (pipeline needs > 0)
TOKEN_BUDGET = "estimate"  # "estimate" (caption char columns) | "render" (each sampled caption) | "exact" (cached text lengths only; pipeline needs it)
CAPTION = {"template": "tags", "keep": ["artist"], "shuffle": True, "tag_dropout": 0.0}
TEXT_DROP = 0.1  # probability a sample uses the empty-prompt embedding
LATENT_CACHE = "offline"  # "off" | "offline" | "online"
TEXT_CACHE = "offline"  # "off" | "offline" | "online"
CACHE_DIR = ""  # "" = <dataset>/cache; directory for latent / text vaults
CACHE_DTYPE = "bf16"  # "bf16" | "fp16" | "fp32" | "fp8"
CAPTION_VARIANTS = 4  # cached caption renderings per image
FLIP = False  # horizontal flip augmentation
CROP = "center"  # "center" | "random" (offline latent caches need "center")
PLANNER_KWARGS = (
    {}
)  # override derived planner arguments, e.g. {"window": 64, "drop_last": True}
LATENT_ENCODER_ID = "flux2-vae"  # latent cache key: which VAE encoded the vault
LATENT_DRAW = "mode"  # VAE latent draw, part of the latent cache key: "mode" | "sample"
TEXT_ENCODER_ID = "qwen3.5-9b"  # text cache key: which text encoder encoded the vault
BACKGROUND = "white"  # fill colour for transparent images
RESAMPLE = "lanczos"  # image resize filter
TEXT_CHARS_PER_TOKEN = 3.5  # caption length estimate when no cached length exists
TEXT_OVERHEAD_TOKENS = 12  # chat-template tokens added to every caption estimate
MAX_TEXT_TOKENS = 4096  # cap on a caption's text tokens in planning
PREFETCH = 2  # batches prefetched per loader worker
PIN_MEMORY = True  # pin host batches for async H2D copies
NUM_WORKERS = 4
DATA_RESCAN = False  # catalog cache: False (re-probe changed / new files) | True (re-probe all) | "never" (the cached catalog as is)
DATA_SCAN_WORKERS = 8  # worker processes probing image headers on a (re)scan
DATA_WAIT = 3600.0  # seconds a non-writer rank waits for the catalog writer
SEED = 0

# ===================== ADAPTER =====================
ALGO = "lora"  # "lora" | "locon" | "loha" | "lokr" | "oft" | "boft" | "glora" | "ia3" | "dylora" | "tlora" | "full"
PRESET = "attn-mlp"  # base "attn" | "attn-mlp" | "full-block" | "all", + modifiers "+text-refiner" "+modulation" "+embed"; or a LyCORIS-style toml path
DIM = 32
ALPHA = 32.0
ALGO_KWARGS = (
    {}
)  # e.g. {"factor": 8} for lokr, {"constraint": 0.0} for oft (DIM is the OFT block size)
DORA = False
DORA_DETACH = False  # True detaches the DoRA weight norm in training (the DoRA layers then run LyCORIS's unfused path)
DROPOUT = {"input": 0.0, "rank": 0.0, "module": 0.0}
ADAPTER_MODE = "bypass"  # "bypass" | "merge"
LORAPLUS_RATIO = 1.0  # lr ratio of up / down factors (LoRA+); 1.0 = off
TRAIN_NORMS = False

# ===================== OBJECTIVE =====================
TIMESTEP = "moss"  # "moss" | "uniform" | "logit_normal"
TIMESTEP_KWARGS = {
    "mu": -1.09,
    "s": 0.95,
    "shift": 1.0,
    "cutoff": 0.95,
    "uniform_mix": 0.01,
}
LOSS = "mse"  # "mse" | "huber" | "l1" | "masked_mse" (needs a per-token loss mask)
LOSS_KWARGS = {}
WEIGHTING = "none"  # "none" | "x0" | "eps"

# ===================== OPTIM =====================
OPTIMIZER = "adamw"  # "adamw" | "adamw16" | "adamw8bit" (bitsandbytes) | "adamw8bit_kernel" (KohakuMoss 8-bit kernel) | "muon" | "prodigy" | "schedule_free"
OPTIMIZER_KWARGS = {}
LR = 1e-4
BETAS = (0.9, 0.95)
EPS = 1e-8
WEIGHT_DECAY = 0.01
WD_RULES = "moss"  # "moss" (wd on 2-D weights only, decay-to-one on last q/k norm) | "none" | "all"
SCHEDULE = {"mode": "cosine", "warmup": 100, "end": -1}  # AnySchedule config
GRAD_CLIP = 1.0
EMA = 0.0  # 0 = off; e.g. 0.9999

# ===================== PRECISION =====================
PRECISION = "bf16"  # "bf16" | "fp16" | "auto" (bf16 where supported, else fp16)
KEEP_FP32 = ("time_embedding", "shared_block_modulation", "video_head_modulation")
FP16_FIX = "none"  # "none" (also for an fp16-fix WEIGHTS file) | "online" (runtime factors) | "fp32_residual"
FP16_FIX_KWARGS = (
    {}
)  # e.g. {"c": 1 / 16, "s_attn": 1.0, "s_ffn": 1.0, "eps": "stock"}; base needs c = 1/16

# ===================== MEMORY =====================
MEMORY_PROFILE = "balanced"  # a catalog profile (kohakumoss-config list; docs/concepts/profiles.md) | "speed" | "balanced" | "low" | "minimal" | "auto" | "manual"
MEMORY_AUTO_FORCE = False  # "auto" with no profile that fits: False = error, True = run the last one anyway (WARNING)
STEP_RELEASE = "none"  # between-step release of free cached device memory: "none" (the allocator keeps its cached blocks between steps) | "empty_cache" (every step; synchronises) | N (every N steps)
# MEMORY_PROFILE="manual": the knobs below set the plan (None = the default in brackets); with a profile they must stay None.
CHECKPOINT = (
    None  # "none" | "block" | "sub-block" | "selective" | "partial:<fraction>"  [block]
)
STREAM = None  # block weights: "off" (resident) | "ring" (frozen base streamed) | "swap" (full FT, host masters)  [off]
STREAM_RESIDENT = None  # device block slots when streaming (>= 2)  [0]
PINNED = None  # pinned host memory for every offload (False: pageable, staged copies)  [True]
ACT_OFFLOAD = (
    None  # saved activations to host memory between forward and backward  [False]
)
OPTIM_STATE = None  # AdamW state placement: "gpu" | "stream" (host, streamed per parameter) | "cpu" (host update)  [gpu]
GRAD_OFFLOAD = (
    None  # gradients summed in host fp32 across GRAD_ACC micro-batches  [False]
)
FUSED_STEP = None  # optimiser step per parameter inside backward  [False]
# Base precision, independent of the memory profile:
BASE_PRECISION = None  # frozen base: "auto" (what the DiT file stores: an int8 ConvRot file trains on its int8 bases) | "compute" (16-bit; a quantized file dequantized, the full-FT escape) | "int8" | "fp8" | "fp8_compute" | "mxfp8" | "quant" (fp8 on sm89+, else int8)  [auto]
FP8_RECIPE = None  # fp8_compute scaling: "tensor" | "row" | "block"  [row]
FP8_DGRAD = (
    None  # fp8 bases: "fp8" (W^T copy, fp8 input-gradient GEMM) | "16bit"  [fp8]
)
MEMORY_OVERRIDES = (
    {}
)  # any MemoryPlan field, e.g. {"checkpoint": "sub-block", "stream": "ring"}
# frozen-base precision: BASE_PRECISION / FP8_RECIPE / FP8_DGRAD above (docs/internals/quant-kernels.md)

# ===================== PARALLEL =====================
PARALLEL = "none"  # "none" | "ddp" | "fsdp" | "pipeline" | "threaded"
DIST_BACKEND = "auto"  # "auto" (nccl on CUDA, xccl on XPU, gloo otherwise; xccl only through auto) | "nccl" | "gloo"
DEVICES = (0,)
PIPELINE_SCHEDULE = "1f1b"  # "gpipe" | "1f1b" | "interleaved-1f1b" | "looped-bfs" | "interleaved-zb" | "zbv" | "dualpipe-v"; threaded: "gpipe" | "async"
PIPELINE_SPLIT = "measured"  # stage split: "measured" | "flops" | "even" | layers per stage, e.g. (11, 10, 11, 10)
MICROBATCHES = 4
FSDP_REDUCE = "step"  # fsdp with GRAD_ACC > 1: "step" (one reduce-scatter per step; unsharded gradients held over the micro-batches) | "micro" (reduce-scatter every micro-batch; sharded gradients only)
FSDP_RESHARD = "forward"  # fsdp block parameters: "forward" (ZeRO-3: freed after each block's forward, gathered again in backward) | "backward" (ZeRO-2: held unsharded from the forward gather through backward; one gather per micro-batch)
DDP_BACKEND = "kohakumoss"  # ddp: "kohakumoss" (the project's reducers, every memory plan) | "torch" (stock torch DDP, the A / B baseline; plans that keep gradients on the device until the step)
DDP_REDUCE_DTYPE = (
    "fp32"  # ddp gradient sums: "fp32" | "param" (each parameter's own dtype)
)
DIST_TIMEOUT = (
    600  # seconds a collective of the step loop / the build waits for every rank
)
DIST_GATE_TIMEOUT = 7200  # seconds a store gate waits for rank-0-only or per-rank work (catalog scan, kernel tuning, the auto plan, the data / prewarm barriers, checkpoint saves, the sample grid)
DIST_BIND = "lazy"  # nccl / xccl process group: "lazy" (communicators on first use) | "device" (bound to the rank's device at init; the pipeline stays lazy)

# ===================== KERNELS =====================
KERNELS = (
    {}
)  # op -> backend, e.g. {"attention_varlen": "triton", "lycoris": "torch"}; empty = arch priority
TUNE = "table"  # "table" (stored decisions; missing ones tuned in the prewarm before step 0, then stored) | "off" (stored decisions, else fixed defaults; nothing timed)
RETUNE_STEPS = 3  # TUNE="table": the kernel-vs-compile retune runs after each of the first N steps; after them no kernel-vs-compile shape is queued, and outside the first sample grid's first sampler step a new shape binds untimed
COMPILE = "off"  # "off" | "block" | "model"

# ===================== RUNTIME =====================
DRIVER = "native"  # "native" | "lightning"
STEPS = 1000
GRAD_ACC = 1
OUTPUT_DIR = "runs/moss"
RUN_NAME = ""

# ===================== LOGGING =====================
LOGGER = "console"  # "console" | "wandb"
PROJECT = "kohakumoss"
LOG_EVERY = 10  # console step line every N steps (and the first / last)
METRICS_EVERY: int | None = (
    None  # sink rows every N steps (+ first / last); None = LOG_EVERY
)
PROGRESS = True  # tqdm bar over STEPS on a TTY (or FORCE_COLOR=1), at INFO
NONFINITE_PATIENCE = 8  # stop after N non-finite steps at the loss-scale floor, or N steps with the same non-finite parameters; 0 = off
TRACE_STEPS = ()  # (first, last): torch.profiler over steps first+1..last on every rank -> <output>/ranks/
PROFILE_STEPS = ()  # (first, last): cudaProfilerStart / Stop around steps first+1..last (ncu --profile-from-start off, nsys --capture-range=cudaProfilerApi)

# ===================== SAMPLING =====================
SAMPLE_PROMPTS: Annotated[list, FormatArg(prompt_list)] = (
    []
)  # a bare string is one prompt
SAMPLE_NEGATIVE = ""
SAMPLE_EVERY = 0  # steps; 0 = off
SAMPLE_STEPS = 50
SAMPLE_CFG = 3.0
SAMPLE_SHIFT = 5.0
SAMPLE_SKIP_LAYERS = (22,)
SAMPLE_SIZE = (1024, 1024)
SAMPLE_SEEDS = (0,)
VALIDATE_EVERY = 0  # steps; 0 = off
VALIDATION = {}  # {"datasets": [...], "sigmas": [...], "seed": 0}

# ===================== CHECKPOINT / EXPORT =====================
SAVE_EVERY = 500  # steps; 0 = off
SAVE_MINUTES = (
    0  # also a checkpoint every N minutes (checked every LOG_EVERY steps); 0 = off
)
KEEP_LAST = 3
FINAL_CHECKPOINT = "auto"  # resume checkpoint at the run's end: "auto" = when SAVE_EVERY or SAVE_MINUTES is on | True | False (exports are always written)
RESUME = ""  # checkpoint dir | "latest"
SAVE_FORMAT = "comfy"  # exports: "comfy" (ComfyUI LyCORIS file; full FT: one ComfyUI DiT file) | "hf" (LyCORIS file with diffusers keys; full FT: the diffusers folder) | "both" | "none"
SAVE_MERGED = False  # adapters: also the merged DiT in SAVE_FORMAT (ComfyUI single file / diffusers folder)
SAVE_FUSED_QKV = "auto"  # comfy files: joint q/k/v as one proj.qkv: "auto" (as the DiT file stores them; the Comfy-Org files are fused) | True | False
CHECKPOINT_EXPORT = True  # also write the exports into every checkpoint's export/ (their size is logged at build)
EXPORT_FORMATS = None  # None | the older spelling: ("comfy" | "lycoris" | "merged" | "diffusers", ...) -> SAVE_FORMAT / SAVE_MERGED


def main():
    return run(globals())


if __name__ == "__main__":
    main()
