"""CacheWriter: the trainer's `on_encoded(hb, latents, text, text_len)` hook for online
caches; a background thread writes full-value encodings to this rank's shard files
(`<kind>.r<rank>.kv`). See docs/concepts/data.md#caches."""

import queue
import threading

import torch
from torch import Tensor

from kohakumoss_common.devices import streamed_type
from kohakumoss_data.cache.store import TensorStore, shard_path
from kohakumoss_data.types import HostBatch

FLUSH = "flush"
STOP = "stop"


def _no_jobs(*args) -> list:
    return []


class _Done:
    """A copy that finished when it was issued (host values)."""

    def synchronize(self) -> None:
        return None


DONE = _Done()


class CacheWriter:
    """Background writer for the online latent / text caches of `dataset`, writing the
    shards of `rank`; latents only at full grid, text only when not cut by
    `text_limit`. Accelerator encodings are copied into pinned host memory on a side
    stream when handed over; the queue holds host tensors and the copy's event only."""

    def __init__(self, dataset, max_pending: int = 8, rank: int = 0):
        self._paths = {}
        self._take_latents = _no_jobs
        self._take_text = _no_jobs
        if dataset.latents.mode == "online":
            self._paths["latent"] = (
                dataset.latents.store.path,
                dataset.latents.storage,
            )
            self._take_latents = CacheWriter._latent_jobs
        if dataset.texts.mode == "online":
            self._paths["text"] = (dataset.texts.store.path, dataset.texts.storage)
            self._take_text = CacheWriter._text_jobs
        self.rank = rank
        self._streams: dict = {}
        self._queue: queue.Queue = queue.Queue(maxsize=max_pending)
        self.written = 0
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def shard(self, kind: str) -> str:
        """The file this writer writes for `kind` ("latent" / "text")."""
        return shard_path(self._paths[kind][0], self.rank)

    def __call__(self, hb: HostBatch, latents, text, text_len) -> None:
        self.raise_error()
        latent_jobs = self._take_latents(self, hb, latents)
        jobs = latent_jobs + self._take_text(self, hb, text, text_len)
        if jobs:
            self._queue.put(self._to_host(jobs))

    def _side(self, device: torch.device):
        stream = self._streams.get(device)
        if stream is None:
            stream = torch.get_device_module(device).Stream(device)
            self._streams[device] = stream
        return stream

    def _to_host(self, jobs: list) -> tuple[list, object]:
        """(jobs with every value in host memory, the event the writer waits for):
        accelerator values copied into pinned buffers on a side stream after the
        current stream's work, others detached as they are."""
        device = next((v.device for *_, v in jobs if v.device.type != "cpu"), None)
        if device is None or not streamed_type(device.type):
            return [(kind, key, v.detach().cpu()) for kind, key, v in jobs], DONE
        dm = torch.get_device_module(device)
        side = self._side(device)
        side.wait_stream(dm.current_stream(device))
        out = []
        with dm.stream(side):
            for kind, key, v in jobs:
                if v.device.type == "cpu":
                    out.append((kind, key, v.detach()))
                    continue
                host = torch.empty(v.shape, dtype=v.dtype, pin_memory=True)
                host.copy_(v.detach(), non_blocking=True)
                v.record_stream(side)
                out.append((kind, key, host))
        done = dm.Event()
        done.record(side)
        return out, done

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

    def _put(self, store: TensorStore, key: str, value: Tensor) -> None:
        """Write one encoding (on the writer thread)."""
        store.put(key, value.detach().cpu())

    def _run(self) -> None:
        stores = {
            kind: TensorStore(path, storage, writes=self.shard(kind))
            for kind, (path, storage) in self._paths.items()
        }
        while True:
            jobs = self._queue.get()
            try:
                if jobs == FLUSH:
                    for store in stores.values():
                        store.flush()
                elif jobs == STOP:
                    return
                else:
                    values, done = jobs
                    done.synchronize()
                    for kind, key, value in values:
                        self._put(stores[kind], key, value)
                        self.written += 1
            except BaseException as err:
                self.error = err
            finally:
                self._queue.task_done()

    def pending(self) -> int:
        """Queued batches not yet written."""
        return self._queue.unfinished_tasks

    def raise_error(self) -> None:
        """Re-raise a failed background write (checked by every hand-over, `flush` and
        `close`, so a failure surfaces at the next encoded batch)."""
        if self.error is not None:
            raise RuntimeError("online cache write-back failed") from self.error

    def flush(self) -> None:
        """Block until every queued write is on disk; re-raise a writer error."""
        self._queue.put(FLUSH)
        self._queue.join()
        self.raise_error()

    def close(self) -> None:
        """Flush, then stop the writer thread (stopped even when the flush raises)."""
        try:
            self.flush()
        finally:
            self._queue.put(STOP)
            self._thread.join()
