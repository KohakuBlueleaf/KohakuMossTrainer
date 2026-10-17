"""Checkpoint directory I/O.

<dir>/trainable.safetensors   AdapterSet.state_dict()
<dir>/ema.safetensors         EMA buffers (when enabled)
<dir>/optim.pt                OptimBundle.state_dict()
<dir>/rng.<rank>.pt           one per rank: global torch / CUDA (this rank's devices) /
                              numpy / python RNG and the step generators
<dir>/state.json              step, loader state, token / sample counters, world
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file


def _cpu(sd: dict) -> dict:
    return {k: v.detach().to("cpu").contiguous() for k, v in sd.items()}


def _cuda_indices(devices) -> list[int]:
    return [d.index for d in devices if d.type == "cuda" and d.index is not None]


def capture_rng(step_rng: dict, devices=()) -> dict:
    """This process's RNG: global torch / numpy / python, the CUDA state of each of
    `devices` (by index), and the step generators."""
    return {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cuda": {i: torch.cuda.get_rng_state(i) for i in _cuda_indices(devices)},
        "step": step_rng,
    }


def restore_rng(state: dict, devices=()) -> dict:
    """Restore `capture_rng` output on this process's `devices`; returns the step
    generators' states. A device with no saved state keeps its own."""
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    saved = state.get("cuda", {})
    for i in _cuda_indices(devices):
        if i in saved:
            torch.cuda.set_rng_state(saved[i], i)
    return state["step"]


def rng_file(path: str | Path, rank: int) -> Path:
    return Path(path) / f"rng.{rank}.pt"


def save_rng(path: str | Path, rank: int, rng: dict) -> None:
    """Write one rank's RNG file into checkpoint directory `path` (created)."""
    Path(path).mkdir(parents=True, exist_ok=True)
    torch.save(rng, rng_file(path, rank))


def load_rng(path: str | Path, rank: int) -> dict:
    file = rng_file(path, rank)
    if not file.exists():
        raise FileNotFoundError(
            f"{file} missing: the checkpoint was written by fewer ranks than this run"
        )
    return torch.load(file, map_location="cpu", weights_only=False)


def save_checkpoint(
    path: str | Path,
    trainable: dict,
    ema: dict,
    optim: dict,
    state: dict,
) -> Path:
    """Write the shared parts to `path` (created); state.json last, so it marks
    completeness. The rank RNG files are written before (`save_rng`)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_file(_cpu(trainable), str(path / "trainable.safetensors"))
    if ema:
        save_file(_cpu(ema), str(path / "ema.safetensors"))
    torch.save(optim, path / "optim.pt")
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
        "state": json.loads((path / "state.json").read_text()),
    }


def is_complete(path: Path) -> bool:
    """state.json (written last) and the trainable state are present."""
    return (path / "state.json").exists() and (path / "trainable.safetensors").exists()
