"""KohakuMoss dataset infrastructure: format, sources, catalog, planning, loading, caches."""

from kohakumoss_data import caption, resolution, sources
from kohakumoss_data.config import DataConfig
from kohakumoss_data.dataset import MossDataset
from kohakumoss_data.loader import Loader, MossHostBatch
from kohakumoss_data.types import (
    BatchPlan,
    CatalogRow,
    HostBatch,
    PlanItem,
)


def build_loader(cfg: DataConfig, rank: int = 0, world: int = 1) -> Loader:
    """The StatefulLoader for `cfg` on `rank` of `world`."""
    return Loader(
        MossDataset(cfg),
        rank=rank,
        world=world,
        num_workers=cfg.num_workers,
        prefetch=cfg.prefetch,
        pin_memory=cfg.pin_memory,
    )


__all__ = [
    "BatchPlan",
    "CatalogRow",
    "DataConfig",
    "HostBatch",
    "Loader",
    "MossDataset",
    "MossHostBatch",
    "PlanItem",
    "build_loader",
    "caption",
    "resolution",
    "sources",
]
