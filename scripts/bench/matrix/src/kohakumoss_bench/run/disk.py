"""Disk use of a host's bench runs: the per-row cleanup of the training checkpoints and
exported weights no queued row reads any more, and the free-space floor a row needs
before it starts. See docs/performance/bench-matrix.md#disk.
"""

import json
import re
import shutil
import threading
from collections import defaultdict
from pathlib import Path

from kohakumoss_bench.plan.config import Options, Run

GIB = 1024**3
REF = re.compile(r"\{run:([^}]+)\}")


def reads(run: Run) -> set[str]:
    """Names of the runs whose directory `run`'s knobs read (`{run:<name>}`)."""
    return set(REF.findall(json.dumps(run.knobs, default=str)))


def artifacts(run_dir: Path) -> list[Path]:
    """What a cleanup deletes in a run directory: `train/checkpoints` and every
    `.safetensors` outside it (exports); results, metrics, logs and traces stay."""
    if not run_dir.is_dir():
        return []
    ckpt = run_dir / "train" / "checkpoints"
    out = [ckpt] if ckpt.is_dir() else []
    out += [p for p in run_dir.rglob("*.safetensors") if ckpt not in p.parents]
    return out


def size_gib(paths: list[Path]) -> float:
    total = 0
    for p in paths:
        files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
        total += sum(f.stat().st_size for f in files)
    return total / GIB


class Cleanup:
    """Deletes a run's `artifacts` once it and every queued run that reads its
    directory have finished (one per pool: `{run:<name>}` resolves inside a pool's
    queue). Paths outside `root` are never touched."""

    def __init__(self, queue: list[Run], root: Path, enabled: bool = True):
        self.root = root.resolve()
        self.enabled = enabled
        self.reads = {r.name: reads(r) for r in queue}
        self.readers: dict[str, set[str]] = defaultdict(set)
        for name, targets in self.reads.items():
            for target in targets:
                self.readers[target].add(name)
        self.done: set[str] = set()
        self.dirs: dict[str, Path] = {}
        self.freed_gib = 0.0
        self.lock = threading.Lock()

    def finished(self, name: str, run_dir: Path | None) -> float:
        """Record that `name` finished (any status; `run_dir` None for a skipped run)
        and delete every artifact now unread; the GiB freed."""
        with self.lock:
            self.done.add(name)
            if run_dir is not None:
                self.dirs[name] = Path(run_dir)
            freed = 0.0
            for target in sorted({name} | self.reads.get(name, set())):
                if target in self.dirs and self.readers[target] <= self.done:
                    freed += self._remove(self.dirs[target])
            self.freed_gib += freed
            return freed

    def _remove(self, run_dir: Path) -> float:
        if not self.enabled or self.root not in run_dir.resolve().parents:
            return 0.0
        paths = artifacts(run_dir)
        gib = size_gib(paths)
        for p in paths:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
        return gib


def retire(run_dir: Path) -> Path | None:
    """Move a previous attempt's `train/` aside (`train.attempt-<n>/`, its checkpoints
    deleted) so a rerun's metric rows start empty; the moved directory, or None."""
    train = run_dir / "train"
    if not train.is_dir():
        return None
    n = 1
    while (run_dir / f"train.attempt-{n}").exists():
        n += 1
    moved = run_dir / f"train.attempt-{n}"
    train.rename(moved)
    shutil.rmtree(moved / "checkpoints", ignore_errors=True)
    return moved


def checkpoint_gib(run: Run, opts: Options) -> float:
    """Estimated GiB a run's checkpoints and exports take on disk."""
    full = run.knobs.get("ALGO") == "full"
    return opts.full_checkpoint_gib if full else opts.adapter_checkpoint_gib


def floor_gib(run: Run, opts: Options) -> float:
    """Free GiB on the runs disk `run` needs before it starts."""
    return max(opts.disk_floor_gib, 2 * checkpoint_gib(run, opts))


def free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / GIB
