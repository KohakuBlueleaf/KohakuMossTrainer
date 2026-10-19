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

from kohakumoss.parallel.launch import ranks_meet
from kohakumoss.training.checkpoint.io import (
    capture_rng,
    is_complete,
    load_checkpoint,
    load_optim_rank,
    load_rng,
    restore_rng,
    save_checkpoint,
    save_optim_rank,
    save_rng,
)
from kohakumoss.training.checkpoint.stream import save_stream
from kohakumoss_common.calls import nothing

STEP_DIR = "step_{:08d}"
STEP_NAME = re.compile(r"step_\d{8}")
PARTIAL = ".partial"


def _barrier() -> None:
    """Every rank meets on the store (`ranks_meet`: DIST_GATE_TIMEOUT)."""
    ranks_meet("checkpoint")


class CheckpointManager:
    """Owns `<output>/checkpoints/step_XXXXXXXX/` directories; called on every rank.
    `before_save` runs on every rank before each save (e.g. flushing the online cache
    writer), `after_save(step, path)` on every rank after it (path: the checkpoint
    directory on rank 0, None elsewhere; e.g. the checkpoint exports); `devices` are
    this rank's devices whose CUDA RNG state is kept."""

    def __init__(
        self,
        output_dir: str | Path,
        keep_last: int,
        final: bool = True,
        snapshot_ema: bool = False,
        is_main: bool = True,
        before_save=nothing,
        rank: int = 0,
        devices=(),
    ):
        self.root = Path(output_dir) / "checkpoints"
        self.snap_root = Path(output_dir) / "ema_snapshots"
        self.keep_last = keep_last
        self._snap = CheckpointManager._snapshot if snapshot_ema else nothing
        self.before_save = before_save
        self.after_save = nothing
        self.rank = rank
        self.devices = tuple(devices)
        self.last_saved = -1
        self._save = CheckpointManager._write if is_main else nothing
        self._clear = self._clear_partial if is_main else nothing
        self._finish = CheckpointManager._final if final else nothing

    def saved(self, step: int) -> bool:
        """Whether this manager already wrote the checkpoint of `step`."""
        return step == self.last_saved

    def finish(self, step: int, parts: dict) -> None:
        """Every rank, at the run's end: the final checkpoint when `final` is on and
        `step` is not saved yet; nothing otherwise."""
        self._finish(self, step, parts)

    def _final(self, step: int, parts: dict) -> None:
        if not self.saved(step):
            self.save(step, parts)

    def save(self, step: int, parts: dict) -> Path | None:
        """Every rank: its RNG file (and its optimiser shard, `parts["optim_rank"]`);
        rank 0: the shared parts, then the rename."""
        self.before_save()
        partial = self.root / (STEP_DIR.format(step) + PARTIAL)
        self._clear(partial)
        _barrier()
        save_rng(partial, self.rank, capture_rng(parts["step_rng"], self.devices))
        if "optim_rank" in parts:
            save_optim_rank(partial, self.rank, parts["optim_rank"])
        _barrier()
        path = self._save(self, step, parts)
        _barrier()
        self.last_saved = step
        self.after_save(step, path)
        return path

    def _write(self, step: int, parts: dict) -> Path:
        """`parts`: trainable, ema, optim, step_rng, state (dicts) and the trainable
        file's metadata (trainable_meta)."""
        final = self.root / STEP_DIR.format(step)
        partial = save_checkpoint(
            self.root / (STEP_DIR.format(step) + PARTIAL),
            parts["trainable"],
            parts["ema"],
            parts["optim"],
            {**parts["state"], "step": step},
            parts.get("trainable_meta"),
        )
        if final.exists():
            shutil.rmtree(final)
        partial.rename(final)
        self._snap(self, step, parts["ema"], parts.get("ema_gammas", {}))
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
        save_stream(ema, self.snap_root / name)
        index_file = self.snap_root / "index.json"
        index = json.loads(index_file.read_text()) if index_file.exists() else []
        index.append({"step": step, "file": name, "gammas": gammas})
        index_file.write_text(json.dumps(index, indent=2), encoding="utf-8")

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
        ckpt["optim_rank"] = load_optim_rank(path, self.rank)
        return ckpt
