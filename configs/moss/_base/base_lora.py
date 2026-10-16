"""Shared LoRA-family defaults (MOSS recipe objective); recipes override from here."""

WEIGHTS = "models/MOSS-Image-Base"
VARIANT = "base"
DATASETS = [
    {"source": "kohya", "path": "datasets/my_dataset", "repeat_prefix": "first"}
]
RESOLUTION_MODE = "resize"
BATCHING = "tokens"
OVERLONG = "drop_tokens"
TARGET_AREA = 1024 * 1024
MAX_TOKENS = 12288
LATENT_CACHE = "offline"
TEXT_CACHE = "offline"
TEXT_DROP = 0.1
ALGO = "lora"
PRESET = "attn-mlp"
DIM = 32
ALPHA = 32.0
ADAPTER_MODE = "bypass"
TIMESTEP = "moss"
TIMESTEP_KWARGS = {
    "mu": -1.09,
    "s": 0.95,
    "shift": 1.0,
    "cutoff": 0.95,
    "uniform_mix": 0.01,
}
OPTIMIZER = "adamw"
LR = 1e-4
WEIGHT_DECAY = 0.01
SCHEDULE = {"mode": "cosine", "warmup": 100, "end": -1, "min_value": 0.1}
GRAD_CLIP = 1.0
PRECISION = "bf16"
MEMORY_PROFILE = "balanced"
STEPS = 4000
SAVE_EVERY = 500
LOG_EVERY = 10
SAMPLE_EVERY = 500
SAMPLE_PROMPTS = ["1girl, solo, looking at viewer, upper body"]
EXPORT_FORMATS = ("comfy", "lycoris")
