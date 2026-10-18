"""Checkpoint directory I/O.

<dir>/trainable.safetensors   AdapterSet.state_dict(), metadata AdapterSet.state_metadata()
<dir>/ema.safetensors         EMA buffers (when enabled)
<dir>/optim.pt                OptimBundle.state_dict()
<dir>/optim.<rank>.pt         one per rank when the optimiser state is sharded (DDP
                              block owners): that rank's OptimBundle.state_dict()
<dir>/rng.<rank>.pt           one per rank: global torch / CUDA (this rank's devices) /
                              numpy / python RNG and the step generators
<dir>/state.json              step, loader state, token / sample counters, world
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from kohakumoss.training.checkpoint.stream import save_stream

_MULTIARRAY = np._core.multiarray if hasattr(np, "_core") else np.core.multiarray
NUMPY_RNG_GLOBALS = (
    np.ndarray,
    np.dtype,
    type(np.dtype(np.uint32)),
    _MULTIARRAY._reconstruct,
)


def load_pt(file: Path):
    """torch.load of a checkpoint `.pt` file with `weights_only=True`: tensors, Python
    containers and scalars, plus the numpy types of a numpy RNG state (the only
    non-tensor objects the writer stores); any other global is refused."""
    with torch.serialization.safe_globals(list(NUMPY_RNG_GLOBALS)):
        return torch.load(file, map_location="cpu", weights_only=True)


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
    return load_pt(file)


def optim_rank_file(path: str | Path, rank: int) -> Path:
    return Path(path) / f"optim.{rank}.pt"


def save_optim_rank(path: str | Path, rank: int, optim: dict) -> None:
    """Write one rank's optimiser shard (state sharded across ranks) into `path`."""
    Path(path).mkdir(parents=True, exist_ok=True)
    torch.save(optim, optim_rank_file(path, rank))


def load_optim_rank(path: str | Path, rank: int) -> dict | None:
    """This rank's optimiser shard, or None when the checkpoint has none."""
    file = optim_rank_file(path, rank)
    if not file.exists():
        return None
    return load_pt(file)


def save_checkpoint(
    path: str | Path,
    trainable: dict,
    ema: dict,
    optim: dict,
    state: dict,
    trainable_meta: dict[str, str] | None = None,
) -> Path:
    """Write the shared parts to `path` (created); state.json last, so it marks
    completeness. The rank RNG files are written before (`save_rng`).
    `trainable_meta` (the trainable set's `state_metadata()`) is the trainable file's
    safetensors metadata, which export and inference read."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_stream(trainable, path / "trainable.safetensors", trainable_meta)
    if ema:
        save_stream(ema, path / "ema.safetensors")
    torch.save(optim, path / "optim.pt")
    (path / "state.json").write_text(
        json.dumps(state, indent=2, default=str), encoding="utf-8"
    )
    return path


def load_checkpoint(path: str | Path) -> dict:
    """All parts of a checkpoint directory (tensors on CPU)."""
    path = Path(path)
    ema_file = path / "ema.safetensors"
    return {
        "trainable": load_file(str(path / "trainable.safetensors")),
        "ema": load_file(str(ema_file)) if ema_file.exists() else {},
        "optim": load_pt(path / "optim.pt"),
        "state": json.loads((path / "state.json").read_text()),
    }


def is_complete(path: Path) -> bool:
    """state.json (written last) and the trainable state are present."""
    return (path / "state.json").exists() and (path / "trainable.safetensors").exists()
