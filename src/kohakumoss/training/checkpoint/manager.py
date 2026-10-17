"""Checkpoint directories: save (every rank its RNG, rank 0 the rest), keep-last-K,
latest lookup, resume, EMA snapshots. Cadence lives in the CallbackRunner
(SaveCheckpoint / TimedCheckpoint).

A checkpoint is written into `step_XXXXXXXX.partial/` and renamed to `step_XXXXXXXX/`
once complete; only fully named directories are listed or resumed.
"""

import json
import re
import shutil
from pathlib import Path

import torch.distributed as dist
from safetensors.torch import save_file

from kohakumoss.training.checkpoint.io import (
    capture_rng,
    is_complete,
    load_checkpoint,
    load_rng,
    restore_rng,
    save_checkpoint,
    save_rng,
)

STEP_DIR = "step_{:08d}"
STEP_NAME = re.compile(r"step_\d{8}")
PARTIAL = ".partial"


def _no_snapshot(step: int, ema: dict, gammas: dict) -> None:
    return None


def _no_write(step: int, parts: dict) -> None:
    return None


def _nothing() -> None:
    return None


def _no_clear(partial: Path) -> None:
    return None


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


class CheckpointManager:
    """Owns `<output>/checkpoints/step_XXXXXXXX/` directories; called on every rank.
    `before_save` runs on every rank before each save (e.g. flushing the online cache
    writer); `devices` are this rank's devices whose CUDA RNG state is kept."""

    def __init__(
        self,
        output_dir: str | Path,
        keep_last: int,
        snapshot_ema: bool = False,
        is_main: bool = True,
        before_save=_nothing,
        rank: int = 0,
        devices=(),
    ):
        self.root = Path(output_dir) / "checkpoints"
        self.snap_root = Path(output_dir) / "ema_snapshots"
        self.keep_last = keep_last
        self.snapshot = self._snapshot if snapshot_ema else _no_snapshot
        self.before_save = before_save
        self.rank = rank
        self.devices = tuple(devices)
        self.last_saved = -1
        self._save = self._write if is_main else _no_write
        self._clear = self._clear_partial if is_main else _no_clear

    def saved(self, step: int) -> bool:
        """Whether this manager already wrote the checkpoint of `step`."""
        return step == self.last_saved

    def save(self, step: int, parts: dict) -> Path | None:
        """Every rank: its RNG file; rank 0: the shared parts, then the rename."""
        self.before_save()
        partial = self.root / (STEP_DIR.format(step) + PARTIAL)
        self._clear(partial)
        _barrier()
        save_rng(partial, self.rank, capture_rng(parts["step_rng"], self.devices))
        _barrier()
        path = self._save(step, parts)
        _barrier()
        self.last_saved = step
        return path

    def _write(self, step: int, parts: dict) -> Path:
        """`parts`: trainable, ema, optim, step_rng, state (dicts)."""
        final = self.root / STEP_DIR.format(step)
        partial = save_checkpoint(
            self.root / (STEP_DIR.format(step) + PARTIAL),
            parts["trainable"],
            parts["ema"],
            parts["optim"],
            {**parts["state"], "step": step},
        )
        if final.exists():
            shutil.rmtree(final)
        partial.rename(final)
        self.snapshot(step, parts["ema"], parts.get("ema_gammas", {}))
        self.prune()
        return final

    @staticmethod
    def _clear_partial(partial: Path) -> None:
        shutil.rmtree(partial, ignore_errors=True)

    def _complete(self) -> list[Path]:
        return sorted(
            p
            for p in self.root.glob("step_*")
            if STEP_NAME.fullmatch(p.name) and is_complete(p)
        )

    def _snapshot(self, step: int, ema: dict, gammas: dict) -> None:
        self.snap_root.mkdir(parents=True, exist_ok=True)
        name = f"{STEP_DIR.format(step)}.safetensors"
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in ema.items()},
            str(self.snap_root / name),
        )
        index_file = self.snap_root / "index.json"
        index = json.loads(index_file.read_text()) if index_file.exists() else []
        index.append({"step": step, "file": name, "gammas": gammas})
        index_file.write_text(json.dumps(index, indent=2))

    def prune(self) -> None:
        """Keep the newest `keep_last` complete checkpoints (all when keep_last <= 0)."""
        if self.keep_last <= 0:
            return
        for old in self._complete()[: -self.keep_last]:
            shutil.rmtree(old, ignore_errors=True)

    def latest(self) -> Path | None:
        done = self._complete()
        return done[-1] if done else None

    def resolve(self, resume: str) -> Path | None:
        """RESUME knob: "" -> None; "latest" -> newest complete dir; else that path."""
        match resume:
            case "" | None:
                return None
            case "latest":
                return self.latest()
            case _:
                return Path(resume)

    def load(self, path: Path) -> dict:
        """The checkpoint's parts; this rank's RNG is restored, its step generators'
        states returned as `step_rng`."""
        ckpt = load_checkpoint(path)
        ckpt["step_rng"] = restore_rng(load_rng(path, self.rank), self.devices)
        return ckpt
