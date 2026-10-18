"""The release stage of a pool (host `priorities` holding 0): the README rows on the
released code, cached (offline latent and text caches), plus one no-cache row:
LoRA r32 balanced and minimal, LoKr 16 balanced, the low-VRAM LoKr setup, fp8-compute
LoRA r32 balanced (sm89+), the vram_8g / vram_6g layers under their VRAM caps in each of
their `VRAM_CACHE_MODES`. See docs/performance/bench-matrix.md#release.
"""

from pathlib import Path

from kohakumoss_bench.plan.config import Card, Host, Options
from kohakumoss_bench.plan.knobs import script_defaults
from kohakumoss_bench.plan.matrix import (
    CACHE_MODES,
    LOWVRAM_LOKR_PROFILE,
    LOWVRAM_LOKR_TEXT,
    _slug,
    bases,
    precisions,
    speed_run,
)
from kohakumoss_bench.run.filters import keep_names

VRAM_LAYERS = ("8g", "6g")
# cache modes per VRAM layer; the first is the layer's plain row name
VRAM_CACHE_MODES = {"8g": ("latenttext", "nocache", "latent"), "6g": ("latenttext",)}
LAYER_KNOBS = ("MEMORY_PROFILE", "MAX_TOKENS", "VAE_ENCODE", "TEXT_ENCODE")
CONFIGS = Path("configs/moss/vram")


def vram_layer(tree: Path, size: str) -> dict:
    """The training knobs of configs/moss/vram/vram_<size>.py (`LAYER_KNOBS`)."""
    knobs = script_defaults(tree / CONFIGS / f"vram_{size}.py")
    return {k: knobs[k] for k in LAYER_KNOBS if k in knobs}


def setups(card: Card, host: Host, tree: Path) -> list[tuple]:
    """(precision, base, algo, profile, label, extra knobs, VRAM cap) of each row."""
    p = precisions(card)[0]
    lokr = {"TEXT_ENCODE": dict(LOWVRAM_LOKR_TEXT)}
    out = [
        (p, "compute", "lora32", "balanced", "lora32-balanced", {}, 0.0),
        (p, "compute", "lora32", "minimal", "lora32-minimal", {}, 0.0),
        (p, "compute", "lokr16", "balanced", "lokr16-balanced", {}, 0.0),
        (
            "fp16",
            "int8",
            "lokr16",
            LOWVRAM_LOKR_PROFILE,
            "int8-lokr16-lowvramlokr",
            lokr,
            0.0,
        ),
        (p, "compute", "lora32", "balanced", "lora32-balanced-nocache", {}, 0.0),
    ]
    out[-1][5].update(CACHE_MODES["nocache"])
    if "fp8_compute" in bases(card, host):
        out.append(
            (p, "fp8_compute", "lora32", "balanced", "fp8-lora32-balanced", {}, 0.0)
        )
    later = []
    for size in VRAM_LAYERS:
        layer = {**vram_layer(tree, size), "SAVE_MINUTES": 0}
        profile = layer["MEMORY_PROFILE"]
        first = VRAM_CACHE_MODES[size][0]
        for mode in VRAM_CACHE_MODES[size]:
            label = f"lora32-vram{size}" + ("" if mode == first else f"-{mode}")
            row = (
                p,
                "compute",
                "lora32",
                profile,
                label,
                {**layer, **CACHE_MODES[mode]},
            )
            (out if mode == first else later).append((*row, float(size[:-1])))
    return out + later


def cache_mode(knobs: dict) -> str:
    """The `CACHE_MODES` name of a run's LATENT_CACHE / TEXT_CACHE pair."""
    pair = (knobs.get("LATENT_CACHE"), knobs.get("TEXT_CACHE"))
    for name, caches in CACHE_MODES.items():
        if pair == (caches["LATENT_CACHE"], caches["TEXT_CACHE"]):
            return name
    raise ValueError(f"cache pair {pair} is not one of {sorted(CACHE_MODES)}")


def queue(host: Host, opts: Options, card: Card, coverage=(), tree: Path = Path(".")):
    """The release rows of `card`'s pool that match the host's `runs` patterns (all
    when unset), the first with the tuning timeout; `coverage` is ignored (the
    signature of `matrix.card_queue`)."""
    rows = []
    for precision, base, algo, profile, label, extra, cap in setups(card, host, tree):
        run = speed_run(host, opts, card, precision, base, algo, profile, "release")
        run.name = _slug("release", precision, label)
        run.knobs.update(extra)
        run.tags.update(cache=cache_mode(run.knobs), features=["release"])
        if cap:
            run.cap_gib = cap
            run.tags["cap_gib"] = cap
        rows.append(run)
    rows = keep_names(rows, host.runs)
    if rows:
        rows[0].timeout_s = opts.timeout_first_s
    return rows
