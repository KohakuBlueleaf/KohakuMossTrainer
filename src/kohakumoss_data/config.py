"""DataConfig: every data knob (lower-case names of the train-script globals)."""

import os
from dataclasses import dataclass, field

from kohakumoss_data.cache.store import STORAGE
from kohakumoss_data.cache.tensor_cache import MODES
from kohakumoss_data.loader.decode import BACKGROUNDS, RESAMPLE
from kohakumoss_data.registry import PLANNERS
from kohakumoss_data.resolution.batching import BATCHINGS
from kohakumoss_data.resolution.overlong import OVERLONG


@dataclass
class DataConfig:
    """`datasets` items are `{"source": name, **source_kwargs, "weight", "caption",
    "loss_weight", "catalog"}`; `cache_dir` holds `latents.db` / `text.db`.
    `max_samples` caps samples per token batch (0 = none); `token_budget="exact"` takes
    every caption length from the offline text cache, so `max_tokens` is a hard bound.
    `planner_kwargs` override the planner arguments derived from the other fields.
    `latent_draw` names the VAE draw (`mode` / `sample`) the latents were encoded with.
    See docs/concepts/data.md."""

    datasets: list[dict] = field(default_factory=list)
    resolution_mode: str = "resize"
    target_area: int = 1024 * 1024
    size_step: int = 16
    aspect_range: tuple[float, float] = (0.25, 4.0)
    upscale: str = "never"
    crop: str = "center"
    batching: str = "tokens"
    max_tokens: int = 16384
    max_samples: int = 0
    token_budget: str = "estimate"
    overlong: str = "drop"
    batch_size: int = 4
    planner_kwargs: dict = field(default_factory=dict)
    caption: dict = field(default_factory=lambda: {"template": "tags"})
    caption_variants: int = 4
    text_drop: float = 0.1
    latent_cache: str = "offline"
    text_cache: str = "offline"
    cache_dtype: str = "bf16"
    cache_dir: str = ""
    latent_encoder_id: str = "flux2-vae"
    latent_draw: str = "mode"
    text_encoder_id: str = "qwen3.5-9b"
    latent_factor: int = 16
    flip: bool = False
    background: str = "white"
    resample: str = "lanczos"
    text_chars_per_token: float = 3.5
    text_overhead_tokens: int = 12
    max_text_tokens: int = 4096
    num_workers: int = 4
    prefetch: int = 2
    pin_memory: bool = True
    seed: int = 0

    @classmethod
    def from_knobs(cls, knobs: dict) -> "DataConfig":
        """Pick this config's fields from a dict of UPPER_CASE train-script globals;
        `cache_dir` is resolved with `resolved_cache_dir`."""
        names = cls.__dataclass_fields__
        cfg = cls(**{k.lower(): v for k, v in knobs.items() if k.lower() in names})
        cfg.cache_dir = resolved_cache_dir(cfg)
        return cfg


def resolved_cache_dir(cfg: DataConfig) -> str:
    """`cache_dir` when set; "" when both caches are off or no dataset has a path; else
    `<path>/cache` of the first dataset with a `path` (its folder when it is a file)."""
    if cfg.cache_dir:
        return cfg.cache_dir
    if cfg.latent_cache == "off" and cfg.text_cache == "off":
        return ""
    paths = [str(spec["path"]) for spec in cfg.datasets if spec.get("path")]
    if not paths:
        return ""
    path = paths[0]
    is_file = "*" in path or os.path.isfile(path) or bool(os.path.splitext(path)[1])
    root = os.path.dirname(os.path.abspath(path)) if is_file else path
    return os.path.join(root, "cache")


def data_problems(cfg: DataConfig) -> list[str]:
    """Every knob combination that would make `MossDataset(cfg)` (or its first batch)
    fail; empty when the config builds."""
    problems = []
    for name, value, allowed in (
        ("RESOLUTION_MODE", cfg.resolution_mode, PLANNERS.names()),
        ("BATCHING", cfg.batching, sorted(BATCHINGS)),
        ("OVERLONG", cfg.overlong, sorted(OVERLONG)),
        ("LATENT_CACHE", cfg.latent_cache, list(MODES)),
        ("TEXT_CACHE", cfg.text_cache, list(MODES)),
        ("CACHE_DTYPE", cfg.cache_dtype, sorted(STORAGE)),
        ("TOKEN_BUDGET", cfg.token_budget, ["estimate", "exact"]),
        ("UPSCALE", cfg.upscale, ["never", "to_area"]),
        ("CROP", cfg.crop, ["center", "random"]),
        ("LATENT_DRAW", cfg.latent_draw, ["mode", "sample"]),
        ("RESAMPLE", cfg.resample, sorted(RESAMPLE)),
    ):
        if value not in allowed:
            problems.append(f"{name}={value!r} is not one of {allowed}")
    if isinstance(cfg.background, str) and cfg.background not in BACKGROUNDS:
        problems.append(
            f"BACKGROUND={cfg.background!r} is not one of {sorted(BACKGROUNDS)}"
        )
    if not cfg.datasets:
        problems.append("DATASETS is empty")
    caches_on = cfg.latent_cache != "off" or cfg.text_cache != "off"
    if caches_on and not resolved_cache_dir(cfg):
        problems.append(
            "a cache is not 'off' but CACHE_DIR is empty and no dataset has a path: "
            "set CACHE_DIR"
        )
    if cfg.token_budget == "exact" and cfg.text_cache != "offline":
        problems.append("TOKEN_BUDGET='exact' needs TEXT_CACHE='offline'")
    random_crop = cfg.crop == "random" or cfg.resolution_mode == "shifted_square"
    if cfg.latent_cache == "offline" and random_crop:
        problems.append(
            "LATENT_CACHE='offline' needs a centre crop (CROP='center', not "
            "RESOLUTION_MODE='shifted_square')"
        )
    if cfg.size_step % cfg.latent_factor:
        problems.append("SIZE_STEP must be a multiple of the latent factor (16)")
    if cfg.max_tokens < 2:
        problems.append("MAX_TOKENS must be >= 2")
    return problems


def pipeline_problems(cfg: DataConfig) -> list[str]:
    """What keeps every planned batch from fitting a static (max_tokens, max_samples)
    pipeline micro-batch shape; empty when it always fits."""
    problems = []
    if cfg.batching == "count":
        problems.append("pipeline needs BATCHING='tokens' or 'tokens_ffd', not 'count'")
    if cfg.max_samples < 1:
        problems.append(
            "pipeline needs MAX_SAMPLES >= 1 (the micro-batch sample slots)"
        )
    if cfg.token_budget != "exact":
        problems.append(
            "pipeline needs TOKEN_BUDGET='exact' (text lengths from the cache)"
        )
    if cfg.text_cache != "offline":
        problems.append(
            "TOKEN_BUDGET='exact' needs TEXT_CACHE='offline' (all lengths known)"
        )
    return problems
