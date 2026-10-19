"""The train script's component paths (WEIGHTS + TRANSFORMER / TEXT_ENCODER / VAE, a
ComfyUI folder or the HF layout) and what the DiT checkpoint stores."""

from functools import cache
from pathlib import Path

from safetensors import SafetensorError

from kohakumoss.model.io.load import stored_precision
from kohakumoss.weights import DIT, TEXT, VAE, component_path

KNOBS = {DIT: "TRANSFORMER", TEXT: "TEXT_ENCODER", VAE: "VAE"}


def component(k: dict, which: str) -> Path:
    """The weights of component `which` (DIT / TEXT / VAE) for the knobs
    (`weights.component_path`; a full finetune's ComfyUI DiT default is the bf16
    file); FileNotFoundError when a ComfyUI folder holds none."""
    return component_path(
        k["WEIGHTS"],
        which,
        k.get(KNOBS[which]) or "",
        k.get("VARIANT", "base"),
        k.get("ALGO") == "full",
    )


@cache
def _stored(path: str) -> str:
    return stored_precision(path)


def stored_dit_precision(k: dict) -> str:
    """The base precision the run's DiT file stores (`compute` when it is not
    readable yet: the path checks report that)."""
    if not k.get("WEIGHTS"):
        return "compute"
    try:
        path = component(k, DIT)
    except FileNotFoundError:
        return "compute"
    if not path.exists():
        return "compute"
    try:
        return _stored(str(path.resolve()))
    except (OSError, ValueError, KeyError, SafetensorError):
        return "compute"
