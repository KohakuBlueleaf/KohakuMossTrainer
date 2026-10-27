"""Stateless whole-batch iteration: batch i is f(seed, rank, i), resume state is
`{seed, start}`, the stream is the same for any worker count. See
docs/concepts/data.md#loader."""

import multiprocessing
import sys
from collections.abc import Callable
from functools import partial

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from kohakumoss_common.calls import identity
from kohakumoss_common.devices import streamed_type
from kohakumoss_common.imports import import_object

WORKER_START = "spawn" if sys.platform == "win32" else "forkserver"
WORKER_LOOP = "torch.utils.data._utils.worker"


def pinnable() -> bool:
    """The process's accelerator (CUDA / XPU) takes pinned host batches."""
    if not torch.accelerator.is_available():
        return False
    return streamed_type(torch.accelerator.current_accelerator().type)


def _module(obj) -> str:
    return getattr(obj, "__module__", None) or type(obj).__module__


def worker_context(*shipped):
    """The loader workers' multiprocessing context (WORKER_START). Under forkserver the
    server preloads the main module, the worker loop and the modules of the `shipped`
    objects (what each worker unpickles), so every worker forks with them imported. See
    docs/concepts/data.md#loader."""
    ctx = multiprocessing.get_context(WORKER_START)
    if WORKER_START == "forkserver":
        names = ["__main__", WORKER_LOOP, *(_module(obj) for obj in shipped)]
        ctx.set_forkserver_preload(list(dict.fromkeys(names)))
    return ctx


def share_like(strategy: str, worker_id: int) -> None:
    """Loader-worker init: the tensor sharing strategy of the loader's process."""
    torch.multiprocessing.set_sharing_strategy(strategy)


def resolve_transform(transform: Callable | str | None) -> Callable:
    """None -> identity; a dotted path -> the imported callable; a callable as is."""
    match transform:
        case None:
            return identity
        case str():
            return import_object(transform)
        case _:
            return transform


class BatchStream(IterableDataset):
    """Iterable of materialised (and transformed) batches from `start` on (infinite);
    worker w of W serves indices start + w, start + w + W, ..."""

    def __init__(
        self, dataset, rank: int, world: int, start: int = 0, transform=identity
    ):
        super().__init__()
        self.dataset = dataset
        self.rank = rank
        self.world = world
        self.start = start
        self.transform = transform

    def __iter__(self):
        info = get_worker_info()
        worker, workers = (0, 1) if info is None else (info.id, info.num_workers)
        index = self.start + worker
        while True:
            yield self.transform(self.dataset.materialize(index, self.rank, self.world))
            index += workers


class Loader:
    """StatefulLoader over a MossDataset: iterate, `state_dict()`, `load_state_dict()`.
    `transform` (picklable callable or dotted path) runs on each batch in the worker,
    before pinning."""

    def __init__(
        self,
        dataset,
        rank: int = 0,
        world: int = 1,
        num_workers: int = 0,
        prefetch: int = 2,
        pin_memory: bool = True,
        start: int = 0,
        transform: Callable | str | None = None,
    ):
        self.dataset = dataset
        self.rank = rank
        self.world = world
        self.num_workers = num_workers
        self.prefetch = prefetch
        self.pin_memory = pin_memory and pinnable()
        self.start = start
        self.served = 0
        self.transform = resolve_transform(transform)

    def _loader(self) -> DataLoader:
        stream = BatchStream(
            self.dataset, self.rank, self.world, self.start, self.transform
        )
        kwargs = {}
        if self.num_workers:
            kwargs["prefetch_factor"] = self.prefetch
            kwargs["persistent_workers"] = False
            kwargs["multiprocessing_context"] = worker_context(
                self.dataset, self.transform
            )
            strategy = torch.multiprocessing.get_sharing_strategy()
            kwargs["worker_init_fn"] = partial(share_like, strategy)
        return DataLoader(
            stream,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            **kwargs,
        )

    def __iter__(self):
        self.start += self.served
        self.served = 0
        for batch in self._loader():
            self.served += 1
            yield batch

    def state_dict(self) -> dict[str, int]:
        return {"seed": self.dataset.cfg.seed, "start": self.start + self.served}

    def load_state_dict(self, state: dict[str, int]) -> None:
        if state["seed"] != self.dataset.cfg.seed:
            raise ValueError(
                f"resume seed {state['seed']} != data seed {self.dataset.cfg.seed}"
            )
        self.start = int(state["start"])
        self.served = 0
