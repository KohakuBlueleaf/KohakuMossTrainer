"""KohakuMoss dataset infrastructure: format, sources, catalog, planning, loading, caches."""

from kohakumoss_data import caption, resolution, sources
from kohakumoss_data.cache.writer import CacheWriter
from kohakumoss_data.config import (
    DataConfig,
    data_problems,
    pipeline_problems,
    resolved_cache_dir,
    type_problems,
)
from kohakumoss_data.dataset import MossDataset
from kohakumoss_data.loader import Loader
from kohakumoss_data.types import (
    BatchPlan,
    CatalogRow,
    HostBatch,
    PlanItem,
)


def build_loader(
    cfg: DataConfig, rank: int = 0, world: int = 1, transform=None, writer: bool = True
) -> Loader:
    """The StatefulLoader for `cfg` on `rank` of `world`; `transform` runs on each
    batch in the worker (see `Loader`); `writer`: this process updates the catalog
    files (one writer per run)."""
    return Loader(
        MossDataset(cfg, writer=writer),
        rank=rank,
        world=world,
        num_workers=cfg.num_workers,
        prefetch=cfg.prefetch,
        pin_memory=cfg.pin_memory,
        transform=transform,
    )


__all__ = [
    "BatchPlan",
    "CacheWriter",
    "CatalogRow",
    "DataConfig",
    "HostBatch",
    "Loader",
    "MossDataset",
    "PlanItem",
    "build_loader",
    "caption",
    "data_problems",
    "pipeline_problems",
    "resolved_cache_dir",
    "resolution",
    "sources",
    "type_problems",
]
