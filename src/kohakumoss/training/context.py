"""RunContext: what every run builder reads (knobs, the built model, the Layout, this
rank's place, the run directory), built once per rank in `entry.worker` and passed to
the builders instead of their positional arguments."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RunContext:
    """`k` the knobs, `built` the built model (backbone, trainable, plan, encoders,
    engines), `layout` the PARALLEL Layout, `device` this rank's device, `rank` /
    `world` the process ranks, `data_world` the ranks drawing distinct data, `is_main`
    rank 0, `output_dir` the run directory."""

    k: dict
    built: object
    layout: object
    device: torch.device
    rank: int = 0
    world: int = 1
    data_world: int = 1
    is_main: bool = True
    output_dir: str = ""
