"""Cache build settings: the knobs that shape each cache's keys, recorded per build next
to the store (`<cache_dir>/<kind>.settings.json`), and the offline-miss error that
compares a run against them and prints the rebuild command."""

import json
import os
import re
import shlex
import sys

from kohakumoss_common.io import write_json_atomic
from kohakumoss_data.config import DataConfig, resolved_cache_dir

FIELDS = {
    "latents": (
        "resolution_mode",
        "target_area",
        "size_step",
        "aspect_range",
        "upscale",
        "crop",
        "flip",
        "latent_encoder_id",
        "latent_encoder_spec",
        "cache_dtype",
        "background",
        "resample",
        "latent_draw",
    ),
    "text": (
        "caption",
        "caption_variants",
        "seed",
        "text_encoder_id",
        "text_encoder_spec",
        "cache_dtype",
    ),
}
SCRIPTS = {
    "latents": "scripts/data/cache_latents.py",
    "text": "scripts/data/cache_text.py",
}
NAMES = {"latents": "latents", "text": "text embeddings"}


class CacheMiss(KeyError):
    """A KeyError whose message prints as written (multi-line)."""

    def __str__(self) -> str:
        return str(self.args[0])


def settings_of(cfg: DataConfig, kind: str) -> dict:
    """The key-shaping knobs of `cfg` for cache `kind`, as JSON values."""
    return json.loads(json.dumps({f: getattr(cfg, f) for f in FIELDS[kind]}))


def settings_path(cache_dir: str, kind: str) -> str:
    return os.path.join(cache_dir, f"{kind}.settings.json")


def recorded(cache_dir: str, kind: str) -> list[dict]:
    """Every build's settings recorded for cache `kind` (empty when none were)."""
    path = settings_path(cache_dir, kind)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def record(cfg: DataConfig, kind: str) -> None:
    """Add this config's settings to the cache's build list (once per distinct set)."""
    cache_dir = resolved_cache_dir(cfg)
    builds = recorded(cache_dir, kind)
    current = settings_of(cfg, kind)
    if current in builds:
        return
    os.makedirs(cache_dir, exist_ok=True)
    write_json_atomic(settings_path(cache_dir, kind), builds + [current])


SPEC_FIELDS = {"latents": "latent_encoder_spec", "text": "text_encoder_spec"}
SAFE = re.compile(r"[\w%+=:./-]*", re.ASCII)
SRC = "src:"


def spec_knobs(cfg: DataConfig, kind: str) -> dict:
    """The encoder knobs that give `cfg`'s encoder spec in the cache pass of `kind` (the
    pass records the spec of the encoder it builds from them): VAE_DTYPE for latents,
    TEXT_ENCODE's dtype / precision / quant_group / fp8_recipe for text (the inverse of
    kohakumoss.encoders.specs.text_spec; a stored-layout part `src:...` names the
    weights file, not a knob); empty for no spec."""
    spec = getattr(cfg, SPEC_FIELDS[kind])
    if not spec:
        return {}
    if kind == "latents":
        return {"VAE_DTYPE": spec}
    dtype, precision, *rest = [p for p in spec.split("/") if not p.startswith(SRC)]
    fields = {"dtype": dtype, "precision": precision}
    if rest and rest[0].startswith("g") and rest[0][1:].isdigit():
        fields["quant_group"] = int(rest[0][1:])
    elif rest and rest[0] != "mx":
        fields["fp8_recipe"] = rest[0]
    return {"TEXT_ENCODE": fields}


def _knob_values(cfg: DataConfig, kind: str) -> dict:
    """The knobs a rebuild sets: the key-shaping ones, the spec through `spec_knobs`."""
    run = {f.upper(): v for f, v in settings_of(cfg, kind).items()}
    run.pop(SPEC_FIELDS[kind].upper())
    return {**run, **spec_knobs(cfg, kind)}


def powershell_quote(text: str) -> str:
    """`text` as one PowerShell argument: as is when every character is safe (no space,
    comma, quote or other PowerShell metacharacter), else in single quotes with inner
    quotes doubled."""
    if SAFE.fullmatch(text):
        return text
    return "'" + text.replace("'", "''") + "'"


QUOTE = powershell_quote if sys.platform == "win32" else shlex.quote


def _arg(name: str, value, quote=QUOTE) -> str:
    text = value if isinstance(value, str) else repr(value)
    return quote(f"{name.upper()}={text}")


def rebuild_command(cfg: DataConfig, kind: str, quote=QUOTE) -> str:
    """The cache pass for `kind` with every key-shaping knob set to `cfg`'s value (the
    encoder spec through the knobs that give it), quoted for this platform's shell
    (PowerShell on Windows, POSIX elsewhere)."""
    knobs = _knob_values(cfg, kind).items()
    sets = " ".join(f"--set {_arg(f, v, quote)}" for f, v in knobs)
    cache = f"--set {_arg('cache_dir', resolved_cache_dir(cfg), quote)}"
    return f"kogine run {SCRIPTS[kind]} --config <your data config> {sets} {cache}"


def config_lines(cfg: DataConfig, kind: str) -> list[str]:
    """The knobs of `rebuild_command` as data-config lines."""
    lines = [f"{f} = {v!r}" for f, v in _knob_values(cfg, kind).items()]
    return lines + [f"CACHE_DIR = {resolved_cache_dir(cfg)!r}"]


def _differing(run: dict, build: dict) -> list[str]:
    return [f for f in run if build.get(f) != run[f]]


def offline_miss(cfg: DataConfig, kind: str, missing: int, first: str) -> CacheMiss:
    """The error for `missing` offline-cache misses (`first`: the first missing sample):
    recorded builds that lack a key knob said so, the key knobs that differ from the
    nearest recorded build (or the run's knobs when the cache recorded none), and the
    rebuild command."""
    cache_dir = resolved_cache_dir(cfg)
    run = settings_of(cfg, kind)
    builds = recorded(cache_dir, kind)
    lines = [
        f"{missing} {NAMES[kind]} missing from the offline cache {cache_dir} "
        f"(first: {first})."
    ]
    if builds and all(set(run) - set(b) for b in builds):
        lacking = ", ".join(f.upper() for f in sorted(set(run) - set(builds[-1])))
        lines.append(
            f"This cache was written under other key knobs (its builds lack "
            f"{lacking}): every entry misses; rebuild it with the command below."
        )
    if builds:
        near = min(builds, key=lambda b: len(_differing(run, b)))
        diff = _differing(run, near)
        lines.append(
            "Key knobs that differ from the nearest cache build (this run vs cache):"
            if diff
            else "This run's key knobs match a recorded build: the cache lacks these "
            "items (new or changed images, or an interrupted pass)."
        )
        lines += [f"  {f.upper()}: {run[f]!r} vs {near.get(f)!r}" for f in diff]
    else:
        lines.append(
            f"The cache records no build settings ({settings_path(cache_dir, kind)} "
            "is missing); this run's key knobs:"
        )
        lines += [f"  {f.upper()}: {v!r}" for f, v in run.items()]
    lines += ["Rebuild with this run's settings: put these lines in the data config,"]
    lines += [f"  {line}" for line in config_lines(cfg, kind)]
    lines += ["or run:", "  " + rebuild_command(cfg, kind)]
    return CacheMiss("\n".join(lines))
