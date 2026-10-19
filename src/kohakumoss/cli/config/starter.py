"""Starter configs: the preset layers for a choice of hardware, algorithm and data plus
the knobs given on the command line, rendered as a kogine config that `use_config`s
each layer.

Dict knobs named in MERGED that several layers set are merged in layer order and written
into the starter, since `use_config` replaces whole values. See docs/guides/configs.md.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

from kohakumoss.cli.config import presets
from kohakumoss.cli.config.validate import BASES, SAMPLE, TRAIN, compose, load_knobs
from kohakumoss.logging.log import get_logger
from kohakumoss.memory.estimate import estimate
from kohakumoss.memory.profiles import resolve as resolve_profile
from kohakumoss.precision import DTYPES
from kohakumoss.training.build.model import workload
from kohakumoss.training.knobs import memory_overrides
from kohakumoss_common.units import GIB
from kohakumoss_data.dataset import first_captions

log = get_logger(__name__)

MERGED = ("MEMORY_OVERRIDES",)
DATA_SCRIPTS = (
    "scripts/data/inspect_data.py",
    "scripts/data/cache_text.py",
    "scripts/data/cache_latents.py",
)
WIDTH = 88


@dataclass
class TrainChoice:
    weights: str  # ComfyUI models folder or HF model dir
    data: str  # dataset root (or file, per source)
    source: str = "kohya"  # data layer name
    algo: str = "lora"  # algo layer name
    vram: float | str = 24  # GiB (exact, as detected) or "auto"
    precision: str = "bf16"  # precision layer name
    tolerance: float = presets.VRAM_TOLERANCE  # fraction of a preset size
    gpus: int = 1
    device: str = "cuda"  # device the cache encoders run on
    name: str = "my-lora"
    steps: int = 0  # 0 = the base config's STEPS
    prompts: tuple[str, ...] = ()


@dataclass
class SampleChoice:
    weights: str
    adapters: tuple[str, ...] = ()
    prompts: tuple[str, ...] = ()
    vram: float = 24
    precision: str = "bf16"
    tolerance: float = presets.VRAM_TOLERANCE
    device: str = "cuda"
    out: str = "runs/samples"


def size(vram: float | str) -> str:
    """A VRAM size for a docstring: "auto", or the GiB to two decimals at most."""
    return vram if isinstance(vram, str) else f"{vram:.4g}"


@dataclass
class Starter:
    script: str  # repo-relative script the file drives (validation target)
    title: str  # first docstring line
    layers: list[Path]
    own: dict = field(default_factory=dict)


def posix(path: str | Path) -> str:
    """Absolute forward-slash path."""
    return Path(path).expanduser().resolve().as_posix()


def merged(layers: list[Path], names: tuple[str, ...] = MERGED) -> dict:
    """Each knob in `names` set by two or more layers, merged in layer order."""
    out = {}
    for name in names:
        found = [k[name] for k in map(load_knobs, layers) if name in k]
        if len(found) > 1:
            out[name] = {key: v for part in found for key, v in part.items()}
    return out


def train_layers(root: Path, c: TrainChoice) -> list[Path]:
    """The train base, then the algo, data, vram, precision and (several GPUs) parallel
    layers, in that order."""
    full = c.algo == presets.FULL
    picks = [
        ("algo", c.algo),
        ("data", c.source),
        ("vram", presets.vram_layer(root, c.vram, full, c.tolerance)),
        ("precision", c.precision),
    ]
    layers = [presets.layer(root, "moss", group, name) for group, name in picks]
    layers = [root / BASES[TRAIN], *layers]
    parallel = presets.parallel_layer(c.gpus, full)
    if parallel == "fsdp" and not (peer_linked(c.gpus) and fsdp_fits(root, layers, c)):
        parallel = "ddp"
    if parallel is not None:
        layers.append(presets.layer(root, "moss", "parallel", parallel))
    return layers


def peer_linked(gpus: int, cuda=torch.cuda) -> bool:
    """Every pair of the first `gpus` CUDA devices has peer access (NVLink / P2P); False
    when this machine has fewer CUDA devices. See docs/guides/full-finetune.md."""
    if not cuda.is_available() or cuda.device_count() < gpus:
        return False
    return all(
        cuda.can_device_access_peer(i, j)
        for i in range(gpus)
        for j in range(gpus)
        if i != j
    )


def fsdp_fits(root: Path, layers: list[Path], c: TrainChoice) -> bool:
    """The full finetune's per-rank FSDP2 footprint (the fsdp layer's profile, every
    parameter, gradient and optimizer state sharded over `c.gpus`; the memory
    estimate) is at most `c.vram` GiB (the VRAM tolerance matches card sizes to layers
    only); "auto" VRAM never picks FSDP (DDP runs whatever profile the planner picks).
    """
    if c.vram == presets.AUTO:
        return False
    return fsdp_need(root, layers, c.gpus) <= float(c.vram)


def fsdp_need(root: Path, layers: list[Path], gpus: int) -> float:
    """The estimated per-rank GiB of `layers` plus the fsdp layer over `gpus` ranks."""
    fsdp = presets.layer(root, "moss", "parallel", "fsdp")
    k = {**compose(root, TRAIN, [*layers, fsdp], {}), "PARALLEL": "fsdp"}
    plan = resolve_profile(
        k["MEMORY_PROFILE"], memory_overrides(k), device="cuda", trainable_base=True
    )
    compute = DTYPES.get(k["PRECISION"], torch.bfloat16)
    return estimate(workload(k, compute, gpus), plan).total / GIB


def dataset_prompts(knobs: dict) -> list[str]:
    """The first rendered caption of the first dataset of composed train knobs, as the
    sample prompt; empty (with a warning) when the dataset gives none."""
    entry = knobs["DATASETS"][0]
    try:
        found = first_captions(entry, knobs["CAPTION"], 1, int(knobs.get("SEED", 0)))
    except Exception as err:  # any read error: no prompt from the data, logged
        found, why = [], f" ({type(err).__name__}: {err})"
    else:
        why = ""
    if not found:
        log.warning(
            f"no caption found in {entry.get('path')}{why}: SAMPLE_PROMPTS keeps the "
            "base prompt; pass --prompt with a caption of your data"
        )
    return found


def train_starter(root: Path, c: TrainChoice) -> Starter:
    """The train config of choice `c`; without `prompts` the sample prompt is the
    dataset's first caption."""
    layers = train_layers(root, c)
    data_layer = presets.layer(root, "moss", "data", c.source)
    template = load_knobs(data_layer)["DATASETS"][0]
    own = {
        "WEIGHTS": posix(c.weights),
        "DATASETS": [{**template, "path": posix(c.data)}],
        "RUN_NAME": c.name,
    }
    if c.steps:
        own["STEPS"] = c.steps
    prompts = list(c.prompts) or dataset_prompts(compose(root, TRAIN, layers, own))
    if prompts:
        own["SAMPLE_PROMPTS"] = prompts
    if c.gpus > 1:
        own["DEVICES"] = tuple(range(c.gpus))
    own.update(merged(layers, MERGED))
    title = (
        f"{c.name}: {c.algo} on {size(c.vram)} GiB {c.precision}, {c.source} data. "
        f"Drives {TRAIN}."
    )
    return Starter(TRAIN, title, layers, own)


def data_starter(c: TrainChoice, train_path: Path) -> Starter:
    """The data-preparation config of `c`: the train config plus the cache encoders,
    whose dtype and precision come from its VAE_DTYPE / TEXT_ENCODE."""
    weights = posix(c.weights)
    own = {
        "TEXT_ENCODER": "kohakumoss.encoders.text_encoder",
        "TEXT_ENCODER_KWARGS": {"path": weights, "device": c.device},
        "IMAGE_ENCODER": "kohakumoss.encoders.vae_encoder",
        "IMAGE_ENCODER_KWARGS": {"path": weights, "device": c.device},
    }
    title = (
        f"{c.name}: the train config's data knobs plus the cache encoders. Drives "
        + ", ".join(DATA_SCRIPTS)
        + "."
    )
    return Starter(DATA_SCRIPTS[0], title, [train_path], own)


def sample_profile(root: Path, vram: float, tolerance: float) -> str:
    """The DiT memory profile sampling uses on a `vram` GiB card: the MEMORY_PROFILE of
    the card's training VRAM layer ("" resident above the largest)."""
    name = presets.sample_vram_layer(root, vram, tolerance)
    if name == presets.AUTO:
        return ""
    return load_knobs(presets.layer(root, "moss", "vram", name))["MEMORY_PROFILE"]


def sample_starter(root: Path, c: SampleChoice) -> Starter:
    """The sample config of choice `c`: the DiT placed as training places it on the
    card; text-encoder and VAE placement left to "auto"."""
    layers = [
        root / BASES[SAMPLE],
        presets.layer(root, "sample", "precision", c.precision),
    ]
    own = {"WEIGHTS": posix(c.weights), "OUT": c.out, "DEVICE": c.device}
    profile = sample_profile(root, c.vram, c.tolerance)
    if profile:
        own["MEMORY_PROFILE"] = profile
    if c.adapters:
        own["ADAPTERS"] = list(c.adapters)
    if len(c.prompts) == 1:
        own["PROMPT"] = c.prompts[0]
    elif c.prompts:
        own["PROMPTS"] = list(c.prompts)
    title = f"Sampling on {size(c.vram)} GiB {c.precision}. Drives {SAMPLE}."
    return Starter(SAMPLE, title, layers, own)


def literal(value, indent: int = 0, lead: int = 0) -> str:
    """`value` as a Python literal (double-quoted ASCII strings), one item per line when
    the one-line form, after `indent` spaces and `lead` characters, passes WIDTH."""
    flat = _flat(value)
    fits = len(flat) + indent + lead <= WIDTH
    if fits or not isinstance(value, (dict, list, tuple)):
        return flat
    pad = " " * (indent + 4)
    if isinstance(value, dict):
        items = [f"{_flat(k)}: {literal(v, indent + 4)}" for k, v in value.items()]
        open_, close = "{", "}"
    else:
        items = [literal(v, indent + 4) for v in value]
        open_, close = ("[", "]") if isinstance(value, list) else ("(", ")")
    body = "".join(f"{pad}{item},\n" for item in items)
    return f"{open_}\n{body}{' ' * indent}{close}"


def _flat(value) -> str:
    match value:
        case str():
            return json.dumps(value)
        case dict():
            return (
                "{"
                + ", ".join(f"{_flat(k)}: {_flat(v)}" for k, v in value.items())
                + "}"
            )
        case list():
            return "[" + ", ".join(_flat(v) for v in value) + "]"
        case tuple() if len(value) == 1:
            return f"({_flat(value[0])},)"
        case tuple():
            return "(" + ", ".join(_flat(v) for v in value) + ")"
    return repr(value)


def shown(path: Path, root: Path) -> str:
    """`path` relative to the repository root when inside it, else absolute."""
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return posix(path)


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def from_file(target: Path, base: Path, root: Path) -> str:
    """`target` for a use_config call in a file in directory `base`: relative when both
    are inside the repository `root` or `target` is under `base`, else absolute; a file
    of the installed bundle is always absolute."""
    target, base, root = target.resolve(), base.resolve(), root.resolve()
    if root.name == "bundle" and _inside(target, root):
        return target.as_posix()
    same_repo = _inside(target, root) and _inside(base, root)
    if same_repo or _inside(target, base):
        return Path(os.path.relpath(target, base)).as_posix()
    return target.as_posix()


def render(st: Starter, out: Path, commands: list[str], root: Path) -> str:
    """Config source: docstring with the run commands, use_config layers, own knobs."""
    lines = [
        f'"""{st.title}',
        "Written by kohakumoss-config; run from the repo root:",
        "",
    ]
    lines += [f"    {cmd}" for cmd in commands]
    lines += ['"""', "", "from kohakuengine import use_config", ""]
    base = Path(out).parent
    lines += [f'use_config("{from_file(p, base, root)}")' for p in st.layers]
    lines += ["", "# knobs set below win over the layers above"]
    lines += [f"{n} = {literal(v, lead=len(n) + 3)}" for n, v in st.own.items()]
    return "\n".join(lines) + "\n"
