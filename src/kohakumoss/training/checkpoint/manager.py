"""Checkpoint directories: save (rank 0 writes), keep-last-K, latest lookup, resume,
EMA snapshots. Cadence lives in the CallbackRunner (SaveCheckpoint / TimedCheckpoint).
"""

import json
import shutil
from pathlib import Path

from safetensors.torch import save_file

from kohakumoss.training.checkpoint.io import (
    capture_rng,
    is_complete,
    load_checkpoint,
    restore_rng,
    save_checkpoint,
)

STEP_DIR = "step_{:08d}"


def _no_snapshot(step: int, ema: dict, gammas: dict) -> None:
    return None


def _no_write(step: int, parts: dict) -> None:
    return None


def _nothing() -> None:
    return None


class CheckpointManager:
    """Owns `<output>/checkpoints/step_XXXXXXXX/` directories. `before_save` runs on
    every rank before each save (e.g. flushing the online cache writer)."""

    def __init__(
        self,
        output_dir: str | Path,
        keep_last: int,
        snapshot_ema: bool = False,
        is_main: bool = True,
        before_save=_nothing,
    ):
        self.root = Path(output_dir) / "checkpoints"
        self.snap_root = Path(output_dir) / "ema_snapshots"
        self.keep_last = keep_last
        self.snapshot = self._snapshot if snapshot_ema else _no_snapshot
        self.before_save = before_save
        self._save = self._write if is_main else _no_write

    def save(self, step: int, parts: dict) -> Path | None:
        self.before_save()
        return self._save(step, parts)

    def _write(self, step: int, parts: dict) -> Path:
        """`parts`: trainable, ema, optim, step_rng, state (dicts)."""
        path = save_checkpoint(
            self.root / STEP_DIR.format(step),
            parts["trainable"],
            parts["ema"],
            parts["optim"],
            capture_rng(parts["step_rng"]),
            {**parts["state"], "step": step},
        )
        self.snapshot(step, parts["ema"], parts.get("ema_gammas", {}))
        self.prune()
        return path

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
        done = sorted(p for p in self.root.glob("step_*") if is_complete(p))
        for old in done[: -self.keep_last]:
            shutil.rmtree(old, ignore_errors=True)

    def latest(self) -> Path | None:
        done = sorted(p for p in self.root.glob("step_*") if is_complete(p))
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

    @staticmethod
    def load(path: Path) -> dict:
        ckpt = load_checkpoint(path)
        ckpt["step_rng"] = restore_rng(ckpt["rng"])
        return ckpt
