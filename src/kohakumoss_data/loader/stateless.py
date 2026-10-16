"""Stateless whole-batch iteration: batch i is f(seed, rank, i); resume is `{seed, start}`.

Worker w of W serves indices start + w, start + w + W, ...; DataLoader round-robins
workers in order, so the stream is identical for every worker count.
"""

import sys

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


class BatchStream(IterableDataset):
    """Iterable of materialised batches from `start` on (infinite)."""

    def __init__(self, dataset, rank: int, world: int, start: int = 0):
        super().__init__()
        self.dataset = dataset
        self.rank = rank
        self.world = world
        self.start = start

    def __iter__(self):
        info = get_worker_info()
        worker, workers = (0, 1) if info is None else (info.id, info.num_workers)
        index = self.start + worker
        while True:
            yield self.dataset.materialize(index, self.rank, self.world)
            index += workers


class Loader:
    """StatefulLoader over a MossDataset: iterate, `state_dict()`, `load_state_dict()`."""

    def __init__(
        self,
        dataset,
        rank: int = 0,
        world: int = 1,
        num_workers: int = 0,
        prefetch: int = 2,
        pin_memory: bool = True,
        start: int = 0,
    ):
        self.dataset = dataset
        self.rank = rank
        self.world = world
        self.num_workers = num_workers
        self.prefetch = prefetch
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.start = start
        self.served = 0

    def _loader(self) -> DataLoader:
        stream = BatchStream(self.dataset, self.rank, self.world, self.start)
        kwargs = {}
        if self.num_workers:
            kwargs["prefetch_factor"] = self.prefetch
            kwargs["persistent_workers"] = False
            if sys.platform != "win32":
                kwargs["multiprocessing_context"] = "spawn"
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
