"""DataConfig: every data knob (lower-case names of the train-script globals)."""

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """`datasets` items are `{"source": name, **source_kwargs, "weight", "caption",
    "loss_weight", "catalog"}`; `cache_dir` holds `latents.db` / `text.db`.
    `max_samples` caps samples per token batch (0 = none); `token_budget="exact"`
    requires every caption length from the text cache, so `max_tokens` is a hard bound
    (static pipeline micro-batch shapes need it)."""

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
        """Pick this config's fields from a dict of UPPER_CASE train-script globals."""
        names = cls.__dataclass_fields__
        return cls(**{k.lower(): v for k, v in knobs.items() if k.lower() in names})


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
