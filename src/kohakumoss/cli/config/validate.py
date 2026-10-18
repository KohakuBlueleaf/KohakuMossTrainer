"""Knob composition and validation of kogine config files against the script they drive.

A config names its script in its docstring (`scripts/<...>.py`, the first one wins). Its
knobs are the script's defaults, then the file. A layer (a file in one of LAYER_GROUPS)
is not runnable alone and is validated on top of the base it names in `_BASE`, else the
script's base config (BASES).
"""

import ast
import functools
import re
from pathlib import Path

from kohakuengine import introspect_schema, load_config_file

import kohakumoss
from kohakumoss.cli.config.presets import LAYER_GROUPS
from kohakumoss.inference import knobs as sample_knobs
from kohakumoss.training import knobs as train_knobs
from kohakumoss_data import DataConfig, data_problems

SCRIPT_RE = re.compile(r"scripts/[\w/]+\.py")
TRAIN = "scripts/train/moss.py"
SAMPLE = "scripts/sample/sample.py"
BASES = {
    TRAIN: "configs/moss/_base/base_lora.py",
    SAMPLE: "configs/sample/_base/base_sample.py",
}
USER_DIR = "configs/user"
MISSING = "does not exist"


def _has_assets(root: Path) -> bool:
    return (root / "configs").is_dir() and (root / "scripts").is_dir()


def repo_root() -> Path:
    """The directory holding configs/ and scripts/: the checkout of an editable install,
    else the wheel's kohakumoss/bundle/."""
    package = Path(kohakumoss.__file__).resolve().parent
    for root in (package.parents[1], package / "bundle"):
        if _has_assets(root):
            return root
    raise SystemExit(
        "configs/ and scripts/ not found: reinstall kohakumoss (pip install kohakumoss) "
        'or install a clone with `uv pip install -e ".[cuda]"`'
    )


def is_checkout(root: Path) -> bool:
    """True when `root` is a repository checkout (not the installed bundle)."""
    return root.name != "bundle"


def script_ref(root: Path, script: str) -> str:
    """How a command names `script`: repo-relative in a checkout, else absolute."""
    return script if is_checkout(root) else (root / script).as_posix()


def target_script(config: Path) -> str:
    """The `scripts/<...>.py` path named first in the config's docstring."""
    doc = ast.get_docstring(ast.parse(Path(config).read_text(encoding="utf-8"))) or ""
    found = SCRIPT_RE.search(doc)
    if found is None:
        raise ValueError(f"{config}: the docstring names no scripts/<...>.py")
    return found.group(0)


def upper(values: dict) -> dict:
    """The UPPER_CASE entries of a globals dict (the knobs)."""
    return train_knobs.collect(values)


def load_knobs(config: Path) -> dict:
    """UPPER_CASE globals of one config file (its use_config layers resolved)."""
    cfg = load_config_file(config)
    if not hasattr(cfg, "globals_dict"):
        raise ValueError(f"{config}: a sweep / generator config is not one knob set")
    return upper(cfg.globals_dict)


@functools.cache
def _defaults(script: str) -> tuple:
    return tuple(upper(introspect_schema(script).defaults).items())


def script_defaults(root: Path, script: str) -> dict:
    """The script's own knob defaults."""
    return dict(_defaults(str(root / script)))


def compose(root: Path, script: str, layers: list[Path], own: dict) -> dict:
    """Script defaults, then `layers` in order, then `own`."""
    k = script_defaults(root, script)
    for path in layers:
        k.update(load_knobs(path))
    k.update(own)
    return k


def is_layer(config: Path, root: Path) -> bool:
    """True for configs/<domain>/<group>/<file> with <group> in LAYER_GROUPS[domain]."""
    try:
        parts = Path(config).resolve().relative_to(root / "configs").parts
    except ValueError:
        return False
    return len(parts) == 3 and parts[1] in LAYER_GROUPS.get(parts[0], ())


def problems(script: str, k: dict) -> list[str]:
    """Every problem the script's own knob checks report for `k`."""
    if script == TRAIN:
        return train_knobs.validate(k)
    if script == SAMPLE:
        return sample_knobs.validate(k)
    if script.startswith("scripts/data/"):
        return data_problems(DataConfig.from_knobs(k))
    return []


def config_knobs(config: Path, root: Path | None = None) -> tuple[str, dict]:
    """(target script, composed knobs) of one config file."""
    root = root or repo_root()
    script = target_script(config)
    layers = [Path(config)]
    if is_layer(config, root) and script in BASES:
        layers.insert(0, layer_base(config) or root / BASES[script])
    return script, compose(root, script, layers, {})


def layer_base(config: Path) -> Path | None:
    """The `_BASE = "<path>"` a layer names (relative to the layer), else None."""
    for node in ast.parse(Path(config).read_text(encoding="utf-8")).body:
        names = [t.id for t in getattr(node, "targets", []) if isinstance(t, ast.Name)]
        if "_BASE" in names and isinstance(node.value, ast.Constant):
            return (Path(config).parent / node.value.value).resolve()
    return None


def config_problems(config: Path, root: Path | None = None) -> list[str]:
    """Every knob problem of one config file under the script it drives."""
    script, k = config_knobs(config, root)
    return problems(script, k)


def blocking(found: list[str]) -> list[str]:
    """The problems a missing path does not explain."""
    return [p for p in found if MISSING not in p]
