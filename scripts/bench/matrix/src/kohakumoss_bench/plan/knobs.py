"""Knob blocks of bench runs (scripts/train/moss.py, export, compare_grid, sample).

Run knobs hold placeholders the host runner fills at launch: `{self}` (the run's own
directory), `{run:<name>}` (another run's directory in the same card queue) and the
whole-value `"{prompts}"` / `"{sample_prompts}"` (the grid prompts drawn from the
dataset). See docs/performance/bench-matrix.md.
"""

import ast
from pathlib import Path

from kohakumoss_bench.plan.config import Card, Host, Options

TRAIN = "scripts/train/moss.py"
EXPORT = "scripts/export/export.py"
GRID = "scripts/sample/compare_grid.py"
SAMPLE = "scripts/sample/sample.py"
CACHE_LATENTS = "scripts/data/cache_latents.py"
CACHE_TEXT = "scripts/data/cache_text.py"
FIX_C = 1 / 16
CAPTION = {"template": "tags", "keep": ["artist"], "shuffle": True, "tag_dropout": 0.0}

ALGOS: dict[str, dict] = {
    "lora32": {"ALGO": "lora", "DIM": 32, "ALPHA": 32.0, "LR": 1e-4},
    "lora8": {"ALGO": "lora", "DIM": 8, "ALPHA": 8.0, "LR": 5e-4},
    "lokr16": {
        "ALGO": "lokr",
        "DIM": 10000,
        "ALPHA": 1.0,
        "ALGO_KWARGS": {"factor": 16, "full_matrix": True},
        "LR": 2e-3,
    },
    "dora16": {"ALGO": "lora", "DIM": 16, "ALPHA": 16.0, "DORA": True, "LR": 1e-4},
    "loha16": {"ALGO": "loha", "DIM": 16, "ALPHA": 16.0, "LR": 1e-4},
    "oft64": {"ALGO": "oft", "DIM": 64, "ALPHA": 1.0, "LR": 1e-4},
}


def staged(host: Host, key: str) -> str:
    """The staged copy of a weights entry ("weights" / "fix_file") on the host."""
    entry = getattr(host, key)
    return entry.get("stage") or entry.get("source", "")


def fix_mode(host: Host, opts: Options) -> str:
    """How fp16 runs get the residual fix: `opts.fp16_fix` ("file": the host's fp16-fix
    checkpoint, "online": runtime factors c = 1/16); "online" when the host has no file.
    """
    return "file" if opts.fp16_fix == "file" and host.fix_file else "online"


def other_fix(host: Host, opts: Options) -> str | None:
    """The fix the side-by-side quality run uses: the other one, when the host has both."""
    if not host.fix_file:
        return None
    return "online" if fix_mode(host, opts) == "file" else "file"


def weights_knobs(host: Host, precision: str, fix: str = "online") -> dict:
    """WEIGHTS / precision knobs of `precision` ("bf16" | "fp16"); fp16 uses `fix`
    ("file" | "online")."""
    base = staged(host, "weights")
    if precision == "bf16":
        return {"WEIGHTS": base, "PRECISION": "bf16", "FP16_FIX": "none"}
    if fix == "file":
        return {
            "WEIGHTS": staged(host, "fix_file"),
            "TEXT_ENCODER": str(Path(base) / "text_encoder"),
            "VAE": str(Path(base) / "vae"),
            "PRECISION": "fp16",
            "FP16_FIX": "none",
        }
    return {
        "WEIGHTS": base,
        "PRECISION": "fp16",
        "FP16_FIX": "online",
        "FP16_FIX_KWARGS": {"c": FIX_C},
    }


def data_knobs(host: Host, opts: Options) -> dict:
    """The data block every training run and cache build shares (so the offline caches
    built once per host match every run's keys)."""
    return {
        "DATASETS": list(host.datasets),
        "CACHE_DIR": host.cache_dir,
        "RESOLUTION_MODE": "resize",
        "TARGET_AREA": opts.target_area,
        "BATCHING": "tokens",
        "CAPTION": dict(CAPTION),
        "CAPTION_VARIANTS": 4,
        "SEED": 0,
        "LATENT_CACHE": "offline",
        "TEXT_CACHE": "offline",
        "NUM_WORKERS": 4,
    }


def train_knobs(host: Host, opts: Options, precision: str, algo: str) -> dict:
    """A short training run's knobs (training steps only; one metric row per step;
    console only)."""
    return {
        **data_knobs(host, opts),
        **weights_knobs(host, precision, fix_mode(host, opts)),
        **ALGOS[algo],
        "MAX_TOKENS": opts.max_tokens,
        "STEPS": opts.speed_steps,
        "SCHEDULE": {"mode": "constant", "warmup": 5, "end": -1},
        "OPTIMIZER": "adamw",
        "TUNE": "table",
        "LOGGER": "console",
        "PROGRESS": False,
        "LOG_EVERY": 10,
        "METRICS_EVERY": 1,
        "SAVE_EVERY": 0,
        "KEEP_LAST": 0,
        "SAMPLE_EVERY": 0,
        "VALIDATE_EVERY": 0,
        "EXPORT_FORMATS": (),
        "RUN_NAME": "train",
        "OUTPUT_DIR": "{self}",
    }


def quality_knobs(
    host: Host, opts: Options, precision: str, algo: str, fix: str
) -> dict:
    """The quality run (configs/moss/quality): `opts.quality_steps` steps, checkpoints
    every `opts.quality_save`, samples during training, LyCORIS + ComfyUI exports."""
    base = train_knobs(host, opts, precision, algo)
    for name in ("TEXT_ENCODER", "VAE", "FP16_FIX_KWARGS"):
        base.pop(name, None)
    return {
        **base,
        **weights_knobs(host, precision, fix),
        "MAX_TOKENS": 12288,
        "STEPS": opts.quality_steps,
        "SCHEDULE": {"mode": "constant", "warmup": 20, "end": -1},
        "MEMORY_PROFILE": "auto",
        "TEXT_CACHE": "online",
        "TEXT_ENCODE": {"memory": "ring"},
        "SAVE_EVERY": opts.quality_save,
        "SAMPLE_EVERY": opts.quality_steps // 2,
        "SAMPLE_PROMPTS": "{sample_prompts}",
        "SAMPLE_STEPS": 20,
        "SAMPLE_CFG": opts.grid_cfg,
        "EXPORT_FORMATS": ("comfy", "lycoris"),
    }


def export_knobs(train: dict, step: int, source: str) -> dict:
    """scripts/export/export.py of checkpoint `step` of the training run `source`."""
    return {
        "WEIGHTS": train["WEIGHTS"],
        "ADAPTER": f"{{run:{source}}}/train/checkpoints/step_{step:08d}/trainable.safetensors",
        "OUT": f"{{self}}/step_{step:08d}.safetensors",
        "FORMAT": "lycoris",
        "SAVE_DTYPE": "bf16",
        "DEVICE": "cpu",
    }


def inference_weights(host: Host, train: dict) -> dict:
    """WEIGHTS / TRANSFORMER / PRECISION / fix knobs of an inference script for the model
    a training run trained on."""
    out = {"WEIGHTS": staged(host, "weights"), "PRECISION": train["PRECISION"]}
    if train["PRECISION"] == "fp16" and train.get("FP16_FIX") == "online":
        out["FP16_FIX"] = "online"
        out["FP16_FIX_KWARGS"] = {"c": FIX_C}
    elif train["PRECISION"] == "fp16":
        out["TRANSFORMER"] = train["WEIGHTS"]
        out["FP16_FIX"] = "none"
    return out


def grid_knobs(
    host: Host, card: Card, opts: Options, train: dict, exports: list
) -> dict:
    """compare_grid over `exports` [(export run name, step)] of a quality run: the base
    model then one column per step, prompts from the dataset, prompts on the columns."""
    return {
        **inference_weights(host, train),
        "CHECKPOINTS": [
            (f"{{run:{name}}}/step_{step:08d}.safetensors", 1.0)
            for name, step in exports
        ],
        "LABELS": [f"step {step}" for _, step in exports],
        "ADAPTER_MODE": "bypass",
        "DEVICE": card.device_type,
        "VAE_ENCODE": {"memory": "offload"},
        "TEXT_ENCODE": {"memory": "ring"},
        "TUNE": "table",
        "PROMPTS": "{prompts}",
        "SIZES": ["1024x1024"],
        "SEED_BASE": 0,
        "STEPS": opts.grid_steps,
        "CFG": opts.grid_cfg,
        "BATCH": 2,
        "GRID_LAYOUT": "prompt-cols",
        "OUT": "{self}/grid",
    }


def sample_knobs(host: Host, card: Card, opts: Options, train: dict) -> dict:
    """scripts/sample/sample.py over the grid prompts, the run's model, no adapter."""
    return {
        **inference_weights(host, train),
        "DEVICE": card.device_type,
        "PROMPTS": "{prompts}",
        "SIZES": ["1024x1024"],
        "STEPS": opts.grid_steps,
        "CFG": opts.grid_cfg,
        "BATCH": 2,
        "TUNE": "table",
        "PROGRESS": False,
        "OUT": "{self}/images",
    }


def encoder_dtype(card: Card) -> str:
    """The VAE and text-encoder dtype "auto" resolves to on `card` (bf16 where the card
    has it, else fp16): the dtype part of the cache keys a run on that card reads."""
    return "bf16" if card.bf16 else "fp16"


def cache_knobs(host: Host, opts: Options, card: Card, kind: str) -> tuple[str, dict]:
    """(script, knobs) of the host's offline latent (`kind` "latents") or text cache
    build on `card`, matching `data_knobs`, its encoders pinned to `encoder_dtype(card)`
    (the encoder spec the cache keys record)."""
    data = data_knobs(host, opts)
    base = staged(host, "weights")
    dtype = encoder_dtype(card)
    shared = {k: data[k] for k in ("DATASETS", "CACHE_DIR", "SEED")}
    if kind == "latents":
        keys = ("RESOLUTION_MODE", "TARGET_AREA")
        return CACHE_LATENTS, {
            **shared,
            **{k: data[k] for k in keys},
            "MAX_TOKENS": 12288,
            "IMAGE_ENCODER": "kohakumoss.encoders.vae_encoder",
            "IMAGE_ENCODER_KWARGS": {
                "path": str(Path(base) / "vae"),
                "device": card.device_type,
            },
            "VAE_DTYPE": dtype,
        }
    return CACHE_TEXT, {
        **shared,
        "CAPTION": data["CAPTION"],
        "CAPTION_VARIANTS": data["CAPTION_VARIANTS"],
        "TEXT_ENCODER_KWARGS": {
            "path": str(Path(base) / "text_encoder"),
            "device": card.device_type,
            "plan": {"memory": "auto", "dtype": dtype},
        },
    }


def fill(value, run_dir: Path, dirs: dict[str, Path], prompts: list[str]):
    """`value` with the run placeholders replaced (see the module docstring)."""
    match value:
        case "{prompts}":
            return list(prompts)
        case "{sample_prompts}":
            return list(prompts[:2])
        case str():
            out = value.replace("{self}", run_dir.as_posix())
            for name, path in dirs.items():
                out = out.replace(f"{{run:{name}}}", path.as_posix())
            return out
        case dict():
            return {k: fill(v, run_dir, dirs, prompts) for k, v in value.items()}
        case list():
            return [fill(v, run_dir, dirs, prompts) for v in value]
        case tuple():
            return tuple(fill(v, run_dir, dirs, prompts) for v in value)
    return value


def tree_env(env: dict, tree: str, trees: dict[str, str]) -> dict:
    """`env` with `{tree}` (the frozen tree) and `{tree:<name>}` (an extra repo's tree)
    replaced in every value."""
    out = {}
    for key, value in env.items():
        text = str(value).replace("{tree}", tree)
        for name, path in trees.items():
            text = text.replace(f"{{tree:{name}}}", path)
        out[key] = text
    return out


def script_defaults(path: Path) -> dict:
    """The UPPER_CASE knob defaults a script assigns at module level (plain and
    annotated assignments of constant expressions such as `1024 * 1024`)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        match node:
            case ast.Assign(targets=[ast.Name(id=name)], value=value):
                pass
            case ast.AnnAssign(target=ast.Name(id=name), value=value) if value:
                pass
            case _:
                continue
        if name.isupper():
            code = compile(ast.Expression(value), str(path), "eval")
            out[name] = eval(code, {})
    return out


def config_text(knobs: dict, header: str) -> str:
    """A KohakuEngine config file setting every knob (Python literals)."""
    lines = [f'"""{header}"""', ""]
    lines += [f"{name} = {value!r}" for name, value in knobs.items()]
    return "\n".join(lines) + "\n"
