"""Metric sinks (rank 0 only): console + metrics.jsonl, KohakuBoard, wandb."""

import json
import os
import time
from pathlib import Path

try:
    import wandb
except ImportError:
    wandb = None
try:
    from kohakuboard.client import Board, Media
except ImportError:
    Board = None
    Media = None


class NullLogger:
    """Non-main ranks."""

    def log(self, metrics: dict, step: int) -> None:
        return None

    def log_images(self, key: str, images: list, step: int) -> None:
        return None

    def close(self) -> None:
        return None


class ConsoleLogger:
    """Prints a one-line summary and appends every metric to `<output>/metrics.jsonl`."""

    def __init__(
        self, output_dir: str, keys: tuple[str, ...] = ("loss", "grad_norm", "lr")
    ):
        self.path = Path(output_dir) / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a", encoding="utf-8")
        self.keys = keys

    def log(self, metrics: dict, step: int) -> None:
        self.file.write(
            json.dumps({"step": step, "time": time.time(), **metrics}) + "\n"
        )
        self.file.flush()
        shown = " ".join(
            f"{k}={metrics[k]:.4g}"
            for k in self.keys
            if isinstance(metrics.get(k), float)
        )
        print(f"[step {step}] {shown}", flush=True)

    def log_images(self, key: str, images: list, step: int) -> None:
        return None

    def close(self) -> None:
        self.file.close()


class BoardLogger(ConsoleLogger):
    """KohakuBoard run under `<output>/board`, plus the console/jsonl sink."""

    def __init__(self, output_dir: str, name: str, project: str, config: dict):
        super().__init__(output_dir)
        if Board is None:
            raise SystemExit("LOGGER='board' needs the `kohakuboard` package installed")
        self.board = Board(
            name=name,
            project=project,
            base_dir=str(Path(output_dir) / "board"),
            config=config,
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
    """wandb run, plus the console/jsonl sink. `mode` defaults to WANDB_MODE, else
    "offline"."""

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
        mode = mode or os.environ.get("WANDB_MODE", "offline")
        self.run = wandb.init(
            project=project, name=name, config=config, dir=output_dir, mode=mode
        )

    def log(self, metrics: dict, step: int) -> None:
        super().log(metrics, step)
        self.run.log(metrics, step=step)

    def log_images(self, key: str, images: list, step: int) -> None:
        self.run.log({key: [wandb.Image(im) for im in images]}, step=step)

    def close(self) -> None:
        super().close()
        self.run.finish()
