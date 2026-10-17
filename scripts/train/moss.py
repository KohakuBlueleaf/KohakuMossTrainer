"""Train MOSS-Image (LyCORIS adapters or full finetune). Every knob is a global below.

kogine run scripts/train/moss.py --config configs/moss/smoke/smoke_tiny.py
kogine run scripts/train/moss.py --config configs/moss/recipes/recipes_lora_24g_bf16.py --set LR=2e-4
"""

from kohakumoss.training.entry import run

# ===================== MODEL =====================
WEIGHTS = (
    ""  # HF dir (transformer/ + text_encoder/ + vae/) | single .safetensors | "random"
)
VARIANT = "base"  # "base" | "sft" | "tiny"
ARCH_OVERRIDES = {}  # MossArchConfig field overrides
TEXT_ENCODER = ""  # "" = <WEIGHTS>/text_encoder; path to Qwen3.5 dir
TEXT_DEVICE = (
    ""  # device the text encoder runs on ("" = the training device; e.g. "cpu")
)
VAE = ""  # "" = <WEIGHTS>/vae
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
TOKEN_BUDGET = (
    "estimate"  # "estimate" | "exact" (cached text lengths only; pipeline needs it)
)
CAPTION = {"template": "tags", "keep": ["artist"], "shuffle": True, "tag_dropout": 0.0}
TEXT_DROP = 0.1  # probability a sample uses the empty-prompt embedding
LATENT_CACHE = "offline"  # "off" | "offline" | "online"
TEXT_CACHE = "offline"  # "off" | "offline" | "online"
CACHE_DIR = ""  # "" = <dataset>/cache; directory for latent / text vaults
CACHE_DTYPE = "bf16"  # "bf16" | "fp8"
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
DORA_DETACH = False  # True detaches the DoRA norm (eager path only)
DROPOUT = {"input": 0.0, "rank": 0.0, "module": 0.0}
ADAPTER_MODE = "bypass"  # "bypass" | "merge"
LORAPLUS_RATIO = 1.0  # lr ratio of up / down factors (LoRA+); 1.0 = off
LYCORIS_COMPILE_DYNAMIC = "tokens"  # LyCORIS compile tier: "tokens" (one graph per weight shape) | "static" | "dynamic"
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
LOSS = "mse"  # "mse" | "huber" | "masked_mse"
LOSS_KWARGS = {}
WEIGHTING = "none"  # "none" | "x0" | "eps"

# ===================== OPTIM =====================
OPTIMIZER = (
    "adamw"  # "adamw" | "adamw16" | "adamw8bit" | "muon" | "prodigy" | "schedule_free"
)
OPTIMIZER_KWARGS = {}
FULL_MODE = (
    "direct"  # ALGO="full": "direct" (train base weights) | "lycoris" (LyCORIS diff)
)
LR = 1e-4
BETAS = (0.9, 0.95)
EPS = 1e-8
WEIGHT_DECAY = 0.01
WD_RULES = "moss"  # "moss" (no wd on bias/norm/1-D, decay-to-one on last q/k norm) | "none" | "all"
SCHEDULE = {"mode": "cosine", "warmup": 100, "end": -1}  # AnySchedule config
GRAD_CLIP = 1.0
EMA = 0.0  # 0 = off; e.g. 0.9999

# ===================== PRECISION =====================
PRECISION = "bf16"  # "bf16" | "fp16" | "auto" (bf16 where supported, else fp16; fp32 on CPU) | "fp32" (CPU reference only)
KEEP_FP32 = ("time_embedding", "shared_block_modulation", "video_head_modulation")
FP16_FIX = "none"  # "none" | "weights" (fold baked into weights) | "online" (runtime factors) | "fp32_residual"
FP16_FIX_KWARGS = (
    {}
)  # e.g. {"c": 1 / 16, "s_attn": 1.0, "s_ffn": 1.0, "eps": "stock"}; base needs c = 1/16

# ===================== MEMORY =====================
MEMORY_PROFILE = "balanced"  # "speed" | "balanced" | "low" | "minimal" | "auto"
MEMORY_OVERRIDES = (
    {}
)  # any MemoryPlan field, e.g. {"checkpoint": "sub-block", "stream": "ring"}

# ===================== PARALLEL =====================
PARALLEL = "none"  # "none" | "ddp" | "fsdp" | "pipeline" | "threaded"
DIST_BACKEND = (
    "auto"  # "auto" (nccl on CUDA, xccl on XPU, gloo otherwise) | "nccl" | "gloo"
)
DEVICES = (0,)
PIPELINE_SCHEDULE = "1f1b"  # "gpipe" | "1f1b" | "interleaved-1f1b" | "zb"
MICROBATCHES = 4

# ===================== KERNELS =====================
KERNELS = (
    {}
)  # op -> backend, e.g. {"attention_varlen": "triton", "lycoris": "torch"}; empty = arch priority
TUNE = "auto"  # "auto" | "off" | "prewarm"
COMPILE = "off"  # "off" | "block" | "model"

# ===================== RUNTIME =====================
DRIVER = "native"  # "native" | "lightning"
STEPS = 1000
GRAD_ACC = 1
OUTPUT_DIR = "runs/moss"
RUN_NAME = ""

# ===================== LOGGING =====================
LOGGER = "console"  # "console" | "board" | "wandb"
PROJECT = "kohakumoss"
LOG_EVERY = 10

# ===================== SAMPLING =====================
SAMPLE_PROMPTS = []
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
SAVE_EVERY = 500
SAVE_MINUTES = 0
KEEP_LAST = 3
RESUME = ""  # checkpoint dir | "latest"
EXPORT_FORMATS = ("comfy",)  # "comfy" | "lycoris" | "merged" | "diffusers"


def main():
    return run(globals())


if __name__ == "__main__":
    main()
