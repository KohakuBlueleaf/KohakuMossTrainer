"""Token lengths the prewarm covers: the longest / shortest joint sample and the largest
batch a data config plans, the power-of-two block buckets, the recorded synthetic
batch and its mapping onto each bucket, and the attention streams to warm."""

import math
from dataclasses import dataclass

from kohakumoss_data.resolution.geometry import Geometry
from kohakumoss_kernels.tuning.buckets import pow2_buckets

ASPECT_STEPS = 64
RECORD_HW = ((8, 8), (6, 10))  # recorded batch: 124 image + 46 text tokens, 2 samples
RECORD_TEXT = (17, 29)
RECORD_TXT = sum(RECORD_TEXT)
RECORD_JOINT = sum(h * w for h, w in RECORD_HW) + RECORD_TXT


def _ceil_to(x: float, step: int) -> int:
    return int(math.ceil(x / step)) * step


def image_tokens_max(cfg) -> int:
    """Largest latent grid (tokens) the resolution planner yields for `cfg`: the area
    target at the aspect extremes and 1, each side rounded up to the size step."""
    best = 0
    for aspect in (*cfg.aspect_range, 1.0):
        w = _ceil_to(math.sqrt(cfg.target_area * aspect), cfg.size_step)
        h = _ceil_to(math.sqrt(cfg.target_area / aspect), cfg.size_step)
        best = max(best, (w // cfg.latent_factor) * (h // cfg.latent_factor))
    return best


def planned_lengths(cfg) -> tuple[int, int]:
    """(longest joint sample, largest joint batch) in tokens for data config `cfg`."""
    sample = min(cfg.max_tokens, image_tokens_max(cfg) + cfg.max_text_tokens)
    batch = (
        cfg.max_tokens if cfg.batching.startswith("tokens") else cfg.batch_size * sample
    )
    return sample, batch


def image_tokens_min(cfg, steps: int = ASPECT_STEPS) -> int:
    """Smallest latent grid (tokens) the resolution planner fits at `cfg`'s area over
    its aspect range (`Geometry.fit` of a source at least the area, `steps` aspects
    from one end of the range to the other)."""
    lo, hi = cfg.aspect_range
    geo = Geometry(
        cfg.target_area, cfg.size_step, tuple(cfg.aspect_range), "to_area",
        "center", cfg.latent_factor,
    )  # fmt: skip
    best = None
    for i in range(steps + 1):
        a = lo * (hi / lo) ** (i / steps)
        w = max(1, round(math.sqrt(cfg.target_area * a)))
        h = max(1, round(math.sqrt(cfg.target_area / a)))
        n = geo.tokens(geo.fit(w, h))
        best = n if best is None else min(best, n)
    return best


def planned_segments(cfg, text_min: int = 0, image_min: int = 0) -> tuple[int, int]:
    """(shortest joint sample, most samples one planned batch holds) for `cfg`: the
    smallest image grid (`image_min`, the dataset's smallest planned grid; 0: the
    smallest grid fitted at the area target) plus the shortest text (`text_min`, the
    dataset's shortest caption estimate; 0: the caption-free template), and BATCH_SIZE
    (count batching) or MAX_TOKENS over the shortest sample, capped by MAX_SAMPLES
    (token batching)."""
    sample, _ = planned_lengths(cfg)
    text = max(text_min, cfg.text_overhead_tokens)
    shortest = min(sample, (image_min or image_tokens_min(cfg)) + text)
    if not cfg.batching.startswith("tokens"):
        return shortest, cfg.batch_size
    most = max(1, cfg.max_tokens // shortest)
    return shortest, min(most, cfg.max_samples) if cfg.max_samples else most


def block_lengths(batch_max: int) -> list[list[int]]:
    """One single-sample batch per power-of-two token bucket up to `batch_max`'s, at
    the bucket's top or `batch_max`, whichever is smaller."""
    return [[n] for n in pow2_buckets(batch_max)]


@dataclass(frozen=True)
class AttentionStream:
    """One attention call site to warm: segments up to `longest` tokens, the rest of
    a batch `shortest` long, forward + backward when `backward`, else forward only."""

    name: str
    longest: int
    shortest: int
    backward: bool


def recorded_attention_grads(calls: list) -> dict[int, bool]:
    """{recorded token rows: whether the call takes a gradient} of the recorded
    `attention_varlen` calls (the pass records under enable_grad)."""
    out = {}
    for call in calls:
        if getattr(call.op, "op", None) == "attention_varlen":
            q = call.args[0]
            out[q.shape[0]] = out.get(q.shape[0], False) or q.requires_grad
    return out


def token_maps(batch_max: int) -> list[dict[int, int]]:
    """Recorded -> planned token counts (joint, image, text) per joint-length bucket."""
    img0 = RECORD_JOINT - RECORD_TXT
    out = []
    for n in pow2_buckets(batch_max):
        img = max(len(RECORD_HW), round(img0 * n / RECORD_JOINT))
        out.append(
            {RECORD_JOINT: n, img0: img, RECORD_TXT: max(len(RECORD_TEXT), n - img)}
        )
    return out
