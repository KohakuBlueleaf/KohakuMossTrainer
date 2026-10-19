"""Logging sinks, selected through LOGGERS (rank 0; other ranks get NullLogger)."""

from kohakumoss.logging.sinks import ConsoleLogger, NullLogger, WandbLogger, fresh_run
from kohakumoss_common.registry import Registry

LOGGERS = Registry("logger")
LOGGERS.add(
    "console", lambda output_dir, name, project, config: ConsoleLogger(output_dir)
)
LOGGERS.add("wandb", WandbLogger)


def build_logger(
    kind: str, output_dir: str, name: str, project: str, config: dict, is_main: bool
):
    if not is_main:
        return NullLogger()
    return LOGGERS.get(kind)(
        output_dir=output_dir, name=name, project=project, config=config
    )


__all__ = [
    "LOGGERS",
    "ConsoleLogger",
    "NullLogger",
    "WandbLogger",
    "build_logger",
    "fresh_run",
]
