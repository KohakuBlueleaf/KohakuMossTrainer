"""CacheWriter: the trainer's `on_encoded(hb, latents, text, text_len)` hook for online
caches. It splits a batch's fresh encodings per sample, queues them, and a background
thread writes them to the vaults; a busy vault (another rank writing) is retried with
backoff for up to `busy_timeout` seconds.

Written: latents when the batch carried `pixels` (full-grid encodings only; a kept-token
subset is skipped), text when it carried `prompts` (a sample cut to `text_limit` is
skipped). See docs/concepts/data.md.
"""

import queue
import threading
import time

import kohakuvault as kv
from torch import Tensor

from kohakumoss_data.cache.store import TensorStore
from kohakumoss_data.types import HostBatch

FLUSH = "flush"
STOP = "stop"


def _nothing(*args) -> list:
    return []


class CacheWriter:
    """Background writer for the online latent / text caches of `dataset`."""

    def __init__(
        self,
        dataset,
        max_pending: int = 8,
        busy_timeout: float = 600.0,
        store_retries: int = 4,
    ):
        self._paths = {}
        self._take_latents = _nothing
        self._take_text = _nothing
        if dataset.latents.mode == "online":
            self._paths["latent"] = (
                dataset.latents.store.path,
                dataset.latents.storage,
            )
            self._take_latents = self._latent_jobs
        if dataset.texts.mode == "online":
            self._paths["text"] = (dataset.texts.store.path, dataset.texts.storage)
            self._take_text = self._text_jobs
        self.busy_timeout = busy_timeout
        self.store_retries = store_retries
        self._queue: queue.Queue = queue.Queue(maxsize=max_pending)
        self.written = 0
        self.busy_waits = 0
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def __call__(self, hb: HostBatch, latents, text, text_len) -> None:
        jobs = self._take_latents(hb, latents) + self._take_text(hb, text, text_len)
        if jobs:
            self._queue.put(jobs)

    def _latent_jobs(self, hb: HostBatch, latents: Tensor | None) -> list:
        if hb.pixels is None or latents is None:
            return []
        full = (hb.hw[:, 0].long() * hb.hw[:, 1].long()).tolist()
        if latents.shape[0] == sum(full):
            parts = latents.split(full, 0)
            return [("latent", k, p) for k, p in zip(hb.latent_keys, parts)]
        keeps = hb.img_keep or [None] * len(full)
        kept = [n if k is None else int(k.numel()) for n, k in zip(full, keeps)]
        parts = latents.split(kept, 0)
        return [
            ("latent", key, part)
            for key, part, keep in zip(hb.latent_keys, parts, keeps)
            if keep is None
        ]

    def _text_jobs(self, hb: HostBatch, text: Tensor | None, text_len) -> list:
        if hb.prompts is None or text is None:
            return []
        lens = [int(n) for n in text_len.tolist()]
        limits = hb.text_limit or [0] * len(lens)
        parts = text.split(lens, 0)
        return [
            ("text", key, part)
            for key, part, n, limit in zip(hb.text_keys, parts, lens, limits)
            if not limit or n < limit
        ]

    def _run(self) -> None:
        stores = {
            kind: TensorStore(path, storage, retries=self.store_retries)
            for kind, (path, storage) in self._paths.items()
        }
        while True:
            jobs = self._queue.get()
            try:
                if jobs == FLUSH:
                    for store in stores.values():
                        self._retry(store.flush)
                elif jobs == STOP:
                    return
                else:
                    for kind, key, value in jobs:
                        cpu = value.detach().cpu()
                        self._retry(self._put, stores[kind], key, cpu)
                        self.written += 1
            except BaseException as err:
                self.error = err
            finally:
                self._queue.task_done()

    def _retry(self, fn, *args) -> None:
        deadline = time.monotonic() + self.busy_timeout
        delay = 0.05
        while True:
            try:
                fn(*args)
                return
            except kv.DatabaseBusy:
                if time.monotonic() + delay > deadline:
                    raise
                self.busy_waits += 1
                time.sleep(delay)
                delay = min(delay * 2, 2.0)

    def _put(self, store: TensorStore, key: str, value: Tensor) -> None:
        store.put(key, value)

    def pending(self) -> int:
        return self._queue.unfinished_tasks

    def flush(self) -> None:
        """Block until every queued write is on disk; re-raise a writer error."""
        self._queue.put(FLUSH)
        self._queue.join()
        if self.error is not None:
            raise RuntimeError("online cache write-back failed") from self.error

    def close(self) -> None:
        self.flush()
        self._queue.put(STOP)
        self._thread.join()
