"""The encode feed of a run with ENCODE_DEVICES, in the main process: it loads the
training batches of every data rank (and each rank's validation batches) far ahead,
sends them in chunks to the encoder processes (one per encoder card, one shared request
queue), finishes each batch as a cache hit carries it (text cut to its limit, latents
gathered to the kept tokens, rounded as an online cache stores them, written to it) and
streams every data rank's batches, in loader order, to its training ranks. A training
process only takes ready batches (`training.encode.streamed`). See
docs/internals/encode-feed.md."""

import itertools
import os
import queue
import threading
import traceback
from dataclasses import dataclass, field, replace

import torch
import torch.multiprocessing as mp

from kohakumoss.logging.log import get_logger, hl
from kohakumoss.model import resolve_config
from kohakumoss.training.build.loop import validation_data_config
from kohakumoss.training.build.model import (
    component_path,
    data_knobs,
    sample_texts,
    training_text_plan,
)
from kohakumoss.training.encode.service import (
    CONTEXT,
    FAILED,
    POLL_S,
    PROCESS_KNOBS,
    READY,
    SHARING,
    serve,
    unpack,
)
from kohakumoss.training.step import LayoutTransform, cap_text, gather_kept
from kohakumoss.weights import TEXT, VAE
from kohakumoss_data import CacheWriter, DataConfig, build_loader

log = get_logger(__name__)
CLOSE_S = 30.0
BATCH = "batch"


@dataclass
class RankFeed:
    """One training rank's end of the feed (picklable, handed to the rank process):
    its batch and validation queues, the control / reply queues, the failure event,
    the data knobs, the loader state its stream starts at and the validation count."""

    rank: int
    batches: object
    validation: object | None
    control: object
    reply: object
    failed: object
    knobs: dict
    start: dict
    val_count: int


@dataclass
class _Entry:
    stream: "_Stream"
    gen: int
    seq: int
    hb: object
    state: dict


@dataclass
class _Stream:
    """One data stream: a loader, the rank queues it fans out to, the in-flight bound
    `sem`, the generation (a seek starts a new one) and the in-order release buffer."""

    name: str
    loader: object
    queues: list
    ranks: list
    sem: threading.Semaphore
    limit: int | None = None
    writer: object | None = None
    text_rounding: bool = False
    latent_rounding: bool = False
    gen: int = 0
    out: int = 0
    ready: dict = field(default_factory=dict)
    fwd: queue.Queue = field(default_factory=queue.Queue)
    lock: threading.Lock = field(default_factory=threading.Lock)
    seeks: dict = field(default_factory=dict)
    start: dict = field(default_factory=dict)
    producer: threading.Thread | None = None

    @property
    def dataset(self):
        return self.loader.dataset


def encode_devices(k: dict, kind: str) -> list[torch.device]:
    """The encoder cards of ENCODE_DEVICES."""
    return [torch.device(kind, int(i)) for i in k["ENCODE_DEVICES"]]


def item_share(payload: dict | None, items: list[dict], i: int) -> dict:
    """Item `i`'s share of a chunk's encode seconds (by tokens / images) and counts."""
    if payload is None:
        return {"te_s": 0.0, "vae_s": 0.0, "tokens": 0, "images": 0, "passes": 0.0}
    tokens = sum(payload["text"][i][1])
    images = len(items[i]["pixels"]) if items[i]["pixels"] is not None else 0
    te = payload["te_s"] * tokens / max(payload["tokens"], 1)
    vae = payload["vae_s"] * images / max(payload["images"], 1)
    return {
        "te_s": te,
        "vae_s": vae,
        "tokens": tokens,
        "images": images,
        "passes": 1.0 / len(items),
    }


def finish_batch(hb, text: tuple | None, full, stream: "_Stream") -> None:
    """`hb` as a cache hit carries it: `text` (packed per-prompt encodings) rounded as
    an online cache stores it and cut to `text_limit`, `full` (full-grid latents)
    rounded and gathered to the kept tokens; the online cache gets the fresh
    encodings; the pixels are dropped."""
    if text is not None and text[0] is not None:
        parts = unpack(text)
        if stream.text_rounding:
            parts = stream.dataset.texts.as_cached(parts)
        parts = cap_text(parts, hb.text_limit)
        hb.text = torch.cat(parts, 0)
        hb.text_len = torch.tensor([p.shape[0] for p in parts], dtype=torch.int32)
    if full is not None:
        rounded = full
        if stream.latent_rounding:
            rounded = stream.dataset.latents.as_cached(full, hb.hw)
        hb.latents = gather_kept(rounded, hb.hw, hb.img_keep)
    if stream.writer is not None:
        stream.writer(hb, full, hb.text, hb.text_len)
    hb.pixels = None


class EncodeFeed:
    """The feed of a run over `world` training ranks of `layout`: build with the
    knobs, `start()` before the training processes start, `handles[rank]` for each,
    `close()` after they end."""

    def __init__(self, k: dict, layout, world: int, kind: str):
        mp.set_sharing_strategy(SHARING)
        self.k = k
        self.ctx = mp.get_context(CONTEXT)
        self.devices = encode_devices(k, kind)
        self.kd = data_knobs(k, self.devices[0])
        self.text_online = k["TEXT_CACHE"] != "offline" and bool(
            component_path(k, TEXT)
        )
        self.latent_online = k["LATENT_CACHE"] != "offline" and bool(
            component_path(k, VAE)
        )
        self.depth = int(k["ENCODE_DEPTH"])
        self.chunk = int(k["ENCODE_CHUNK"])
        self.stop = threading.Event()
        self.failed = self.ctx.Event()
        self.requests = self.ctx.Queue()
        self.results = self.ctx.Queue()
        self.control = self.ctx.Queue()
        self.pending: queue.Queue = queue.Queue()
        self.inflight: dict[int, tuple] = {}
        self.ids = itertools.count()
        self.info: dict = {"samples": None, "flops_per_token": 0}
        self.ready_count = 0
        self.info_waiting: list[int] = []
        self.procs: list = []
        self.threads: list[threading.Thread] = []
        self.specs = self._specs()
        self.slots = threading.Semaphore(2 * max(len(self.specs), 1))
        self.replies = [self.ctx.Queue() for _ in range(world)]
        self.streams, self.rank_stream = self._train_streams(layout, world)
        self.val_streams = self._val_streams(world)
        self.handles = [self._handle(r) for r in range(world)]

    def _specs(self) -> list[dict]:
        sampling = self.k["SAMPLE_EVERY"] > 0
        needed = self.text_online or self.latent_online or sampling
        if not needed:
            return []
        text_path, vae_path = component_path(self.k, TEXT), component_path(self.k, VAE)
        out = []
        for i, device in enumerate(self.devices):
            samples = sampling and i == 0
            plan, _ = training_text_plan(self.k, device)
            out.append(
                {
                    "device": str(device),
                    "knobs": {name: self.k.get(name) for name in PROCESS_KNOBS},
                    "text_path": text_path if (self.text_online or samples) else "",
                    "text_plan": plan,
                    "vae_path": vae_path if self.latent_online else "",
                    "sample_texts": sample_texts(self.k) if samples else [],
                    "parent": os.getpid(),
                }
            )
        return out

    def _loader_cfg(self, cfg: DataConfig) -> DataConfig:
        return replace(cfg, pin_memory=False)

    def _tokens(self) -> LayoutTransform:
        arch = resolve_config(self.k["VARIANT"], self.k["ARCH_OVERRIDES"] or None)
        return LayoutTransform(arch.rope_axes)

    def _train_streams(self, layout, world: int):
        cfg = self._loader_cfg(DataConfig.from_knobs(self.kd))
        tokens = self._tokens()
        data_world = layout.data_world(world)
        streams: dict[int, _Stream] = {}
        rank_stream: dict[int, _Stream] = {}
        for rank in range(world):
            d = layout.data_rank(rank)
            if d not in streams:
                loader = build_loader(cfg, d, data_world, tokens, writer=not streams)
                writes = "online" in (self.k["LATENT_CACHE"], self.k["TEXT_CACHE"])
                streams[d] = _Stream(
                    name=f"data rank {d}",
                    loader=loader,
                    queues=[],
                    ranks=[],
                    sem=threading.Semaphore(self.depth),
                    writer=CacheWriter(loader.dataset, rank=d) if writes else None,
                    text_rounding=self.k["TEXT_CACHE"] == "online",
                    latent_rounding=self.k["LATENT_CACHE"] == "online",
                    start=dict(loader.state_dict()),
                )
            s = streams[d]
            s.queues.append(self.ctx.Queue(maxsize=self.depth))
            s.ranks.append(rank)
            rank_stream[rank] = s
        return list(streams.values()), rank_stream

    def _val_streams(self, world: int) -> dict[int, _Stream]:
        if self.k["VALIDATE_EVERY"] <= 0:
            return {}
        count = int(self.k["VALIDATION"].get("batches", 4))
        cfg = self._loader_cfg(validation_data_config(self.kd))
        tokens = self._tokens()
        out = {}
        for rank in range(world):
            loader = build_loader(cfg, rank, world, tokens, writer=rank == 0)
            out[rank] = _Stream(
                name=f"validation rank {rank}",
                loader=loader,
                queues=[self.ctx.Queue(maxsize=count)],
                ranks=[rank],
                sem=threading.Semaphore(count),
                limit=count,
                start=dict(loader.state_dict()),
            )
        return out

    def _handle(self, rank: int) -> RankFeed:
        s = self.rank_stream[rank]
        val = self.val_streams.get(rank)
        return RankFeed(
            rank=rank,
            batches=s.queues[s.ranks.index(rank)],
            validation=None if val is None else val.queues[0],
            control=self.control,
            reply=self.replies[rank],
            failed=self.failed,
            knobs=self.kd,
            start=dict(s.start),
            val_count=0 if val is None else int(val.limit),
        )

    def start(self) -> None:
        """Start the encoder processes and the feed threads (the streams fill while the
        training processes build)."""
        for spec in self.specs:
            proc = self.ctx.Process(
                target=serve, args=(spec, self.requests, self.results), daemon=False
            )
            proc.start()
            self.procs.append(proc)
        log.info(
            f"encode feed: {hl(len(self.specs))} encoder process(es) on "
            f"{hl(', '.join(s['device'] for s in self.specs) or '-')}, "
            f"{hl(len(self.streams))} data stream(s), depth {hl(self.depth)}, chunk "
            f"{hl(self.chunk)}"
        )
        self._thread(self._collect)
        self._thread(self._dispatch)
        self._thread(self._control)
        for s in [*self.streams, *self.val_streams.values()]:
            self._thread(self._forward, s)
            s.producer = self._thread(self._produce, s, s.gen)

    def _thread(self, fn, *args) -> threading.Thread:
        t = threading.Thread(target=fn, args=args, daemon=True)
        t.start()
        self.threads.append(t)
        return t

    def _fail(self, where: str, tb: str) -> None:
        if not self.failed.is_set():
            log.error(f"encode feed: {where} failed:\n{tb}")
        self.failed.set()
        self.stop.set()

    def _produce(self, s: _Stream, gen: int) -> None:
        """Loader batches of `s` into the pending queue, at most `sem` ahead."""
        it = iter(s.loader)
        try:
            for seq in itertools.count():
                if s.limit is not None and seq >= s.limit:
                    return
                while not s.sem.acquire(timeout=POLL_S):
                    if self.stop.is_set() or s.gen != gen:
                        return
                if self.stop.is_set() or s.gen != gen:
                    s.sem.release()
                    return
                hb = next(it)
                state = dict(s.loader.state_dict())
                self.pending.put(_Entry(s, gen, seq, hb, state))
        except BaseException:  # noqa: BLE001 - fails the run through the feed
            self._fail(s.name, traceback.format_exc())
        finally:
            it.close()

    def _live(self, e: _Entry) -> bool:
        if e.gen == e.stream.gen:
            return True
        e.stream.sem.release()
        return False

    def _request(self, hb) -> dict:
        text = self.text_online and hb.text is None and bool(hb.prompts)
        lat = self.latent_online and hb.latents is None and hb.pixels is not None
        limits = list(hb.text_limit) or [0] * len(hb.prompts or [])
        return {
            "prompts": list(hb.prompts) if text else [],
            "limits": limits if text else [],
            "pixels": hb.pixels if lat else None,
            "hw": hb.hw.tolist() if lat else None,
        }

    def _dispatch(self) -> None:
        """Chunks of pending batches to the encoder processes; a chunk is gathered once
        a request slot is free, so it holds everything that arrived meanwhile."""
        try:
            while not self.stop.is_set():
                if not self.slots.acquire(timeout=POLL_S):
                    continue
                chunk = self._gather()
                if not chunk:
                    self.slots.release()
                    continue
                items = [self._request(e.hb) for e in chunk]
                if not any(it["prompts"] or it["pixels"] is not None for it in items):
                    self.slots.release()
                    self._finish(chunk, items, None)
                    continue
                rid = next(self.ids)
                self.inflight[rid] = (chunk, items)
                self.requests.put((rid, items))
        except BaseException:  # noqa: BLE001 - fails the run through the feed
            self._fail("dispatch", traceback.format_exc())

    def _gather(self) -> list[_Entry]:
        try:
            chunk = [self.pending.get(timeout=POLL_S)]
        except queue.Empty:
            return []
        while len(chunk) < self.chunk:
            try:
                chunk.append(self.pending.get_nowait())
            except queue.Empty:
                break
        return [e for e in chunk if self._live(e)]

    def _collect(self) -> None:
        """Encoder answers: READY infos, failures, chunk results."""
        try:
            while not self.stop.is_set():
                try:
                    kind, payload = self.results.get(timeout=POLL_S)
                except queue.Empty:
                    self._check_procs()
                    continue
                if kind == READY:
                    self._on_ready(payload)
                elif kind == FAILED:
                    self._fail(f"encoder process on {payload[0]}", payload[1])
                else:
                    chunk, items = self.inflight.pop(kind)
                    self.slots.release()
                    self._finish(chunk, items, payload)
        except BaseException:  # noqa: BLE001 - fails the run through the feed
            self._fail("collect", traceback.format_exc())

    def _check_procs(self) -> None:
        for proc, spec in zip(self.procs, self.specs):
            if not proc.is_alive() and not self.stop.is_set():
                self._fail(
                    f"encoder process on {spec['device']}",
                    f"exited with code {proc.exitcode}",
                )

    def _on_ready(self, info: dict) -> None:
        self.ready_count += 1
        if info["samples"] is not None:
            self.info["samples"] = info["samples"]
        self.info["flops_per_token"] = max(
            self.info["flops_per_token"], info["flops_per_token"]
        )
        self._answer_info()

    def _answer_info(self) -> None:
        if self.ready_count < len(self.specs):
            return
        for rank in self.info_waiting:
            self.replies[rank].put(("info", dict(self.info)))
        self.info_waiting.clear()

    def _finish(self, chunk: list[_Entry], items: list[dict], payload) -> None:
        for i, e in enumerate(chunk):
            s = e.stream
            stats = item_share(payload, items, i)
            if e.gen != s.gen:
                s.sem.release()
                continue
            text = None if payload is None else payload["text"][i]
            full = None if payload is None else payload["latents"][i]
            finish_batch(e.hb, text, full, s)
            self._release_in_order(s, e, stats)

    def _release_in_order(self, s: _Stream, e: _Entry, stats: dict) -> None:
        with s.lock:
            if e.gen != s.gen:
                s.sem.release()
                return
            s.ready[e.seq] = (e.gen, e.state, e.hb, stats)
            while s.out in s.ready:
                s.fwd.put(s.ready.pop(s.out))
                s.out += 1

    def _forward(self, s: _Stream) -> None:
        """Ordered finished batches of `s` into its rank queues."""
        try:
            while not self.stop.is_set():
                try:
                    gen, state, hb, stats = s.fwd.get(timeout=POLL_S)
                except queue.Empty:
                    continue
                for q in s.queues:
                    self._put(q, (BATCH, gen, state, hb, stats), s, gen)
                s.sem.release()
        except BaseException:  # noqa: BLE001 - fails the run through the feed
            self._fail(f"{s.name} forward", traceback.format_exc())

    def _put(self, q, msg, s: _Stream, gen: int) -> None:
        while not self.stop.is_set() and s.gen == gen:
            try:
                q.put(msg, timeout=POLL_S)
                return
            except queue.Full:
                continue

    def _control(self) -> None:
        """Rank requests: `("info", rank)` and `("seek", rank, state)`."""
        try:
            while not self.stop.is_set():
                try:
                    msg = self.control.get(timeout=POLL_S)
                except queue.Empty:
                    continue
                match msg:
                    case ("info", rank):
                        self.info_waiting.append(rank)
                        self._answer_info()
                    case ("seek", rank, state):
                        self._seek(rank, state)
        except BaseException:  # noqa: BLE001 - fails the run through the feed
            self._fail("control", traceback.format_exc())

    def _seek(self, rank: int, state: dict) -> None:
        """A rank resumes at `state`; once every rank of its stream asked, the stream
        restarts there (a new generation) unless it already starts there."""
        s = self.rank_stream[rank]
        s.seeks[rank] = dict(state)
        if len(s.seeks) < len(s.ranks):
            return
        if dict(state) != s.start:
            self._restart(s, dict(state))
        for r in s.seeks:
            self.replies[r].put(("gen", s.gen))
        s.seeks.clear()

    def _restart(self, s: _Stream, state: dict) -> None:
        with s.lock:
            s.gen += 1
            s.ready.clear()
            s.out = 0
        if s.producer is not None:
            s.producer.join()
        s.loader.load_state_dict(state)
        s.start = dict(state)
        log.info(f"encode feed: {s.name} resumes at batch {hl(state['start'])}")
        s.producer = self._thread(self._produce, s, s.gen)

    def _written_queues(self) -> list:
        """The queues this process writes: batches nobody reads after the run are
        dropped at exit instead of flushed (`cancel_join_thread`)."""
        streams = [*self.streams, *self.val_streams.values()]
        return [self.requests, *self.replies, *(q for s in streams for q in s.queues)]

    def close(self) -> None:
        """Stop the threads, end the encoder processes, flush the cache writers; the
        unread batches are dropped."""
        self.stop.set()
        for _ in self.procs:
            self.requests.put(None)
        for proc in self.procs:
            proc.join(CLOSE_S)
            if proc.is_alive():
                proc.terminate()
                proc.join(CLOSE_S)
        for q in self._written_queues():
            q.cancel_join_thread()
        for t in self.threads:
            t.join(2 * POLL_S)
        for s in self.streams:
            if s.writer is not None:
                s.writer.close()
