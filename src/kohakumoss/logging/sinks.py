"""Metric sinks (rank 0 only): console + metrics.jsonl, KohakuBoard, wandb. A fresh run
moves an earlier attempt's files aside (`fresh_run`); a resumed run keeps the rows up
to its step and appends (`resume_at`)."""

import json
import time
from pathlib import Path

from kohakumoss.logging.log import get_logger

METRICS = "metrics.jsonl"
RUN_FILES = (METRICS, "log.txt", "board")
log = get_logger(__name__)

try:
    import wandb
except ImportError:
    wandb = None
try:
    from kohakuboard.client import Board, Media
    from kohakuboard.logger import init_logger_settings as board_log_settings
except ImportError:
    Board = None
    Media = None
    board_log_settings = None


def free_name(path: Path) -> Path:
    """The first `<stem>.<n><suffix>` (n >= 1) next to `path` that does not exist."""
    n = 1
    while True:
        dst = path.with_name(f"{path.stem}.{n}{path.suffix}")
        if not dst.exists():
            return dst
        n += 1


def rotate(path: Path | str) -> Path | None:
    """Move an existing non-empty file or directory `path` to `free_name(path)`; the
    new path, None when there was nothing to move."""
    path = Path(path)
    empty = path.is_file() and path.stat().st_size == 0
    if not path.exists() or empty:
        return None
    dst = free_name(path)
    path.rename(dst)
    return dst


def fresh_run(output_dir: str | Path) -> list[Path]:
    """A run that does not resume: an earlier attempt's metrics.jsonl, log.txt and
    board directory in `output_dir` move aside (WARNING each); the moved paths."""
    moved = []
    for name in RUN_FILES:
        dst = rotate(Path(output_dir) / name)
        if dst is not None:
            log.warning(f"fresh run: an earlier {name} in {output_dir} moved to {dst}")
            moved.append(dst)
    return moved


def _step_of(line: str) -> int:
    try:
        return int(json.loads(line).get("step", -1))
    except (ValueError, AttributeError):
        return -1


class NullLogger:
    """Non-main ranks."""

    def log(self, metrics: dict, step: int) -> None:
        return None

    def resume_at(self, step: int) -> None:
        return None

    def log_images(self, key: str, images: list, step: int) -> None:
        return None

    def close(self) -> None:
        return None


class ConsoleLogger:
    """Appends every metric row to `<output>/metrics.jsonl` (the console step line is
    the StepLine callback's)."""

    def __init__(self, output_dir: str):
        self.path = Path(output_dir) / METRICS
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a", encoding="utf-8")

    def resume_at(self, step: int) -> None:
        """A run resumed at `step`: rows past it (an attempt that went on after that
        checkpoint) move to `free_name(metrics.jsonl)` (WARNING); new rows append."""
        self.file.close()
        lines = self.path.read_text(encoding="utf-8").splitlines(keepends=True)
        kept = [x for x in lines if _step_of(x) <= step]
        later = [x for x in lines if _step_of(x) > step]
        if later:
            dst = free_name(self.path)
            dst.write_text("".join(later), encoding="utf-8")
            self.path.write_text("".join(kept), encoding="utf-8")
            log.warning(
                f"resumed at step {step}: {len(later)} later rows of {self.path} "
                f"moved to {dst}"
            )
        self.file = self.path.open("a", encoding="utf-8")

    def log(self, metrics: dict, step: int) -> None:
        self.file.write(
            json.dumps({"step": step, "time": time.time(), **metrics}) + "\n"
        )
        self.file.flush()

    def log_images(self, key: str, images: list, step: int) -> None:
        return None

    def close(self) -> None:
        self.file.close()


class BoardLogger(ConsoleLogger):
    """KohakuBoard run under `<output>/board`, plus the console/jsonl sink. The console
    shows what it shows under LOGGER='console': KohakuBoard's own log lines go to
    `<output>/board/kohakuboard.log` only, and it captures no stdout / stderr."""

    def __init__(self, output_dir: str, name: str, project: str, config: dict):
        super().__init__(output_dir)
        if Board is None:
            raise SystemExit("LOGGER='board' needs the `kohakuboard` package installed")
        base = Path(output_dir) / "board"
        board_log_settings(log_file=base / "kohakuboard.log", file_only=True)
        self.board = Board(
            name=name,
            project=project,
            base_dir=str(base),
            config=config,
            capture_output=False,
        )

    def log(self, metrics: dict, step: int) -> None:
        super().log(metrics, step)
        self.board.step()
        self.board.log(**{f"train/{k}": v for k, v in metrics.items()})

    def log_images(self, key: str, images: list, step: int) -> None:
        self.board.log(**{key: Media(images)})

    def close(self) -> None:
        super().close()
        self.board.finish()


class WandbLogger(ConsoleLogger):
    """wandb run, plus the console/jsonl sink. `mode` None is wandb's own default
    (WANDB_MODE when set, else online). The console shows what it shows under
    LOGGER='console': wandb runs silent and captures no console output."""

    def __init__(
        self,
        output_dir: str,
        name: str,
        project: str,
        config: dict,
        mode: str | None = None,
    ):
        super().__init__(output_dir)
        if wandb is None:
            raise SystemExit("LOGGER='wandb' needs the `wandb` package installed")
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            dir=output_dir,
            mode=mode,
            settings=wandb.Settings(silent=True, console="off"),
        )

    def log(self, metrics: dict, step: int) -> None:
        super().log(metrics, step)
        self.run.log(metrics, step=step)

    def log_images(self, key: str, images: list, step: int) -> None:
        self.run.log({key: [wandb.Image(im) for im in images]}, step=step)

    def close(self) -> None:
        super().close()
        self.run.finish()
