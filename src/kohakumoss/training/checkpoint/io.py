"""Checkpoint directory I/O.

<dir>/trainable.safetensors   AdapterSet.state_dict()
<dir>/ema.safetensors         EMA buffers (when enabled)
<dir>/optim.pt                OptimBundle.state_dict()
<dir>/rng.pt                  global + step RNG states
<dir>/state.json              step, loader {seed, start}, wall time, knob snapshot
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file


def _cpu(sd: dict) -> dict:
    return {k: v.detach().to("cpu").contiguous() for k, v in sd.items()}


def capture_rng(step_rng: dict) -> dict:
    out = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "step": step_rng,
    }
    if torch.cuda.is_available():
        out["cuda"] = torch.cuda.get_rng_state_all()
    return out


def restore_rng(state: dict) -> dict:
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    return state["step"]


def save_checkpoint(
    path: str | Path,
    trainable: dict,
    ema: dict,
    optim: dict,
    rng: dict,
    state: dict,
) -> Path:
    """Write every part to `path` (created); state.json last, so it marks completeness."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_file(_cpu(trainable), str(path / "trainable.safetensors"))
    if ema:
        save_file(_cpu(ema), str(path / "ema.safetensors"))
    torch.save(optim, path / "optim.pt")
    torch.save(rng, path / "rng.pt")
    (path / "state.json").write_text(json.dumps(state, indent=2, default=str))
    return path


def load_checkpoint(path: str | Path) -> dict:
    """All parts of a checkpoint directory (tensors on CPU)."""
    path = Path(path)
    ema_file = path / "ema.safetensors"
    return {
        "trainable": load_file(str(path / "trainable.safetensors")),
        "ema": load_file(str(ema_file)) if ema_file.exists() else {},
        "optim": torch.load(path / "optim.pt", map_location="cpu", weights_only=False),
        "rng": torch.load(path / "rng.pt", map_location="cpu", weights_only=False),
        "state": json.loads((path / "state.json").read_text()),
    }


def is_complete(path: Path) -> bool:
    return (path / "state.json").exists() and (path / "trainable.safetensors").exists()
