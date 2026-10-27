"""The training rank's end of the encode feed: a loader over the rank's queue of ready
batches (text and latents filled in as a cache hit carries them), with the loader state
of the last batch it handed out, the feed's encode statistics for the callbacks and the
seconds it waited. See docs/internals/encode-feed.md."""

import os
import queue
import time

import torch

from kohakumoss.training.encode.feed import BATCH, RankFeed
from kohakumoss.training.encode.service import POLL_S, unpack


class FeedError(RuntimeError):
    """The encode feed failed or its main process is gone (the main process's log
    holds the cause)."""


class FeedStats:
    """Encode statistics of one encoder kind as the callbacks read them: `stats`,
    `flops_per_token`, `device`."""

    def __init__(self, stats: dict, device: str, flops_per_token: float = 0):
        self.stats = stats
        self.device = device
        self.flops_per_token = flops_per_token


def take(feed: RankFeed, q, wait: list[float]):
    """The next message of `q`; blocks, adding the seconds to `wait[0]`, and raises
    FeedError when the feed failed or the main process is gone."""
    start = time.perf_counter()
    try:
        while True:
            try:
                return q.get(timeout=POLL_S)
            except queue.Empty:
                if feed.failed.is_set():
                    raise FeedError("the encode feed failed") from None
                if os.getppid() == 1:
                    raise FeedError("the encode feed's main process is gone") from None
    finally:
        wait[0] += time.perf_counter() - start


class StreamedLoader:
    """The rank's loader under the encode feed: iterates ready host batches (pinned
    when `pin`); `state_dict` is the loader state after the last batch handed out,
    `load_state_dict` asks the feed to stream from there. `text` / `vae` hold the
    feed's encode statistics, `wait_seconds` the time spent waiting on the queue."""

    def __init__(self, feed: RankFeed, dataset, pin: bool, devices: str = ""):
        self.feed = feed
        self.dataset = dataset
        self.pin = pin
        self.gen = 0
        self.state = dict(feed.start)
        self.wait = [0.0]
        self.text = FeedStats(
            {"seconds": 0.0, "tokens": 0, "passes": 0.0, "prompts": 0}, devices
        )
        self.vae = FeedStats({"seconds": 0.0, "images": 0, "passes": 0.0}, devices)
        self.samples: list = []

    @property
    def wait_seconds(self) -> float:
        return self.wait[0]

    def connect(self) -> None:
        """Wait for the encoder processes: their sample-prompt embeddings and the text
        encoder's FLOPs per token."""
        self.feed.control.put(("info", self.feed.rank))
        _, info = take(self.feed, self.feed.reply, [0.0])
        self.samples = [] if info["samples"] is None else unpack(info["samples"])
        self.text.flops_per_token = info["flops_per_token"]

    def _note(self, stats: dict) -> None:
        self.text.stats["seconds"] += stats["te_s"]
        self.text.stats["tokens"] += stats["tokens"]
        self.vae.stats["seconds"] += stats["vae_s"]
        self.vae.stats["images"] += stats["images"]
        if stats["tokens"]:
            self.text.stats["passes"] += stats["passes"]
        if stats["images"]:
            self.vae.stats["passes"] += stats["passes"]

    def _ready(self, hb):
        return hb.pin() if self.pin else hb

    def __iter__(self):
        while True:
            kind, gen, state, hb, stats = take(self.feed, self.feed.batches, self.wait)
            if kind != BATCH or gen != self.gen:
                continue
            self.state = dict(state)
            self._note(stats)
            yield self._ready(hb)

    def validation(self) -> list:
        """The rank's fixed validation batches (encoded once by the feed)."""
        out = []
        for _ in range(self.feed.val_count):
            _, _, _, hb, _ = take(self.feed, self.feed.validation, [0.0])
            out.append(self._ready(hb))
        return out

    def state_dict(self) -> dict:
        return dict(self.state)

    def load_state_dict(self, state: dict) -> None:
        self.feed.control.put(("seek", self.feed.rank, dict(state)))
        _, self.gen = take(self.feed, self.feed.reply, [0.0])
        self.state = dict(state)


def pin_batches(device: torch.device, pin_memory: bool) -> bool:
    """Whether streamed batches are pinned on arrival (a CUDA / XPU consumer)."""
    return pin_memory and device.type in ("cuda", "xpu")
