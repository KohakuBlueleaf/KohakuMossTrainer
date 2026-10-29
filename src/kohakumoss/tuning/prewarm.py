"""The record / replay prewarm of a backbone (`ModelPrewarm`) and of a trainable set's
autotuned ops (`AdapterPrewarm`), their per-op log table and the device peaks they
reach (kept for the step-1 report). See docs/internals/tuning.md#prewarm."""

import contextlib
import time

import torch

from kohakumoss.adapters.lycoris.tuning import set_lycoris_tuning
from kohakumoss.logging.log import get_logger, hl, log_table
from kohakumoss.parallel.launch import gate_name, release_ranks, wait_for_rank0
from kohakumoss.tuning.lengths import (
    RECORD_HW,
    RECORD_JOINT,
    RECORD_TEXT,
    RECORD_TXT,
    AttentionStream,
    recorded_attention_grads,
    token_maps,
)
from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss_common.units import GIB
from kohakumoss_kernels.dist import dist_rank, initialized
from kohakumoss_kernels.hardware.current import rng_kept
from kohakumoss_kernels.hardware.memory import (
    allocated,
    release,
    tuning_budget,
    window,
)
from kohakumoss_kernels.recording import recording
from kohakumoss_kernels.tuning import decisions, prewarm_attention
from kohakumoss_kernels.tuning.manifest import Manifest, problem_key
from kohakumoss_kernels.tuning.prewarm import attention_lengths
from kohakumoss_kernels.tuning.replay import OpReport, replay, timed_kernels, unique
from kohakumoss_kernels.tuning.replay import problem as replay_problem
from kohakumoss_kernels.tuning.tuner import broadcast, set_cap
from kohakumoss_kernels.tuning.window import freeze

log = get_logger(__name__)
PREWARM_PEAK: dict[str, tuple[int, int]] = {}


def tunes(k: dict) -> bool:
    """TUNE times kernels ("table"); "off" never times."""
    return k["TUNE"] == "table"


def _dropped(t) -> None:
    return None


def _never_unpacked(packed) -> None:
    raise RuntimeError("the prewarm recording pass keeps no saved tensor")


def keep_nothing():
    """Saved-tensor hooks under which autograd keeps no saved tensor (a recorded pass
    that never runs backward holds no activation or weight copy)."""
    return torch.autograd.graph.saved_tensors_hooks(_dropped, _never_unpacked)


class ModelPrewarm:
    """Tune every tuned op of the backbone and loss: record the `Bound` calls of one
    forward (+ loss when `backward`, no saved tensor kept) of `backbone` (the run's
    weightless twin) on a small synthetic batch, replay them at every planned token
    bucket up to `batch_max`, warm the attention buckets; log one table with each op's
    device peak. `loss_fn=None` / `backward=False` is the forward-only prewarm
    (`kohakumoss-tune` for generate shapes); `input_grad` makes the recorded latents
    require grad (a frozen base whose activations carry adapter gradients). The
    attention buckets cover at most `segments` samples per batch of at least
    `shortest` tokens (`planned_segments`; default `batch_max // sample_max` samples
    of `sample_max`) for the joint attention and captions of `text_shortest` to
    `text_max` tokens for the text refiner's (`text_max` 0: not warmed), each with a
    backward only where its recorded call takes a gradient. See
    docs/internals/tuning.md#prewarm."""

    def __init__(
        self,
        backbone,
        loss_fn,
        autocast,
        dtype,
        sample_max,
        batch_max,
        backward: bool = True,
        segments: int | None = None,
        shortest: int | None = None,
        text_max: int = 0,
        text_shortest: int = 0,
        input_grad: bool = False,
        masked: bool = False,
    ):
        self.backbone = backbone
        self.loss_fn = loss_fn
        self.autocast = autocast
        self.dtype = dtype
        self.sample_max = sample_max
        self.batch_max = batch_max
        self.segments = segments or max(1, batch_max // max(sample_max, 1))
        self.shortest = shortest or sample_max
        self.text_max = text_max
        self.text_shortest = text_shortest
        self.targets = token_maps(batch_max)
        self.backward = backward
        self.input_grad = input_grad
        self.masked = masked
        self.reports: dict[torch.device, dict[str, OpReport]] = {}

    def record(self, device: torch.device) -> list:
        """The backbone (+ loss) calls of one synthetic pass."""
        cfg = self.backbone.cfg
        seq = build_seqinfo(
            torch.tensor(RECORD_HW), torch.tensor(RECORD_TEXT), cfg.rope_axes
        ).to(device)
        img = torch.randn(seq.n_img, cfg.latent_channels, device=device)
        img.requires_grad_(self.input_grad and self.backward)
        text = torch.randn(seq.n_txt, cfg.text_dim, device=device).to(self.dtype)
        sigma = torch.rand(seq.batch, device=device)
        grad = torch.enable_grad() if self.backward else torch.no_grad()
        saved = keep_nothing() if self.backward else contextlib.nullcontext()
        with recording() as calls, grad, saved:
            with self.autocast():
                pred = self.backbone(img, text, sigma, seq)
            if self.backward:
                weight = torch.ones(seq.batch, device=device)
                mask = torch.ones(seq.n_img, device=device) if self.masked else None
                self.loss_fn(pred, torch.randn_like(pred), seq.seg_img, weight, mask)
        return calls

    def __call__(self, devices: list[torch.device]) -> int:
        """Rank 0 measures; every rank then binds from rank 0's broadcast tables (the
        other ranks wait on the store for rank 0's measurement, not in a collective)."""
        distributed = initialized()
        rank = dist_rank()[0]
        gate = gate_name("prewarm") if distributed else None
        if distributed and rank != 0:
            take_working(gate, devices)
            devices = []
        problems, workings = 0, []
        for device in devices:
            start = time.perf_counter()
            baseline = allocated(device)
            with rng_kept([device]):
                calls = self.record(device)
            streams = self.streams(calls)
            reports = replay(calls, self.targets, self.backward)
            reports["attention_varlen (segment lengths)"] = self.attention(
                device, streams
            )
            problems += sum(r.problems for r in reports.values())
            self.reports[device] = reports
            log_prewarm(reports, time.perf_counter() - start, device)
            key = self.working_key(calls, streams)
            workings.append(
                note_prewarm(device, reports, Manifest(device), key, baseline)
            )
        if self.backward:
            self.backbone.zero_grad(set_to_none=True)
        if distributed:
            if rank == 0:
                share_working(gate, workings)
            broadcast(0)
        return problems

    def streams(self, calls: list) -> list[AttentionStream]:
        """The attention streams to warm: the joint one, and the text refiner's when
        `text_max` is known, each taking a gradient where its recorded call does."""
        grads = recorded_attention_grads(calls)
        joint = self.backward and grads.get(RECORD_JOINT, True)
        out = [AttentionStream("joint", self.sample_max, self.shortest, joint)]
        if self.text_max and RECORD_TXT in grads:
            short = min(self.text_shortest or self.text_max, self.text_max)
            text = self.backward and grads[RECORD_TXT]
            out.append(AttentionStream("text", self.text_max, short, text))
        return out

    def attention_key(self, streams: list) -> str:
        """Manifest key of the attention warm over `streams`."""
        a, op = self.backbone.cfg, self.backbone.ops.attention_varlen
        layout = (
            getattr(op, "backend", "-"),
            a.heads,
            a.head_dim,
            str(self.dtype),
            self.sample_max,
            self.backward,
            ("segments", self.segments, self.shortest, self.batch_max, "mean"),
            ("streams", *(tuple(vars(s).values()) for s in streams)),
        )
        return problem_key("attention_varlen", layout)

    def working_key(self, calls: list, streams: list) -> str:
        """Manifest key of this prewarm's working bytes: its replay problems and its
        attention warm."""
        problems = sorted(
            replay_problem(c, t, self.backward)
            for c in unique(calls)
            for t in self.targets
        )
        return problem_key("prewarm_working", (*problems, self.attention_key(streams)))

    def attention(self, device: torch.device, streams: list) -> OpReport:
        """The attention kernels, per stream, at every segment-length bucket up to its
        longest segment and every segment-count bucket the planned batches reach."""
        a, op = self.backbone.cfg, self.backbone.ops.attention_varlen
        backend = getattr(op, "backend", "-")
        buckets = sum(len(attention_lengths(s.longest)) for s in streams)
        manifest = Manifest(device)
        key = self.attention_key(streams)
        if manifest.covered(key):
            report = OpReport(cached=buckets)
        else:
            release(device)
            began, before = time.perf_counter(), timed_kernels()
            with window(device) as w, manifest.running(key):
                for s in streams:
                    prewarm_attention(
                        op,
                        device,
                        self.dtype,
                        a.heads,
                        a.head_dim,
                        s.longest,
                        segments=self.segments,
                        backward=s.backward,
                        max_tokens=self.batch_max,
                        shortest=s.shortest,
                        label=f"attention_varlen ({s.name})",
                    )
            manifest.commit()
            report = OpReport(
                buckets, time.perf_counter() - began, timed_kernels() - before
            )
            report.peak_alloc, report.peak_reserved = w.alloc, w.reserved
            release(device)
        report.bound.add(backend)
        return report


def reset_peaks() -> None:
    """Start a run's prewarm peaks (a process may train more than one run)."""
    PREWARM_PEAK.clear()


def note_peak(device: torch.device, reports: dict) -> None:
    """Keep the prewarm's (allocated, reserved) peak of `device` for the step-1 report."""
    alloc = max((r.peak_alloc for r in reports.values()), default=0)
    held = max((r.peak_reserved for r in reports.values()), default=0)
    old = PREWARM_PEAK.get(str(device), (0, 0))
    PREWARM_PEAK[str(device)] = (max(old[0], alloc), max(old[1], held))


def note_prewarm(
    device: torch.device, reports: dict, manifest: Manifest, key: str, baseline: int
) -> int | None:
    """Keep the kernel prewarm's peak of `device` (`note_peak`): the peaks of the ops
    that ran, and `baseline` (bytes allocated at its start) plus the working bytes
    stored for `key` by an earlier run; the larger working is stored back. A cached
    op's near-empty peak is left out, so nothing ran and nothing stored notes nothing.
    Returns the working bytes (None: nothing ran or stored). See
    docs/internals/tuning.md#adapter-autotuned-op-prewarm."""
    ran = {name: r for name, r in reports.items() if r.problems}
    stored = manifest.working(key)
    working = max([r.peak_alloc - baseline for r in ran.values()] + [stored or 0])
    if ran:
        manifest.keep_working(key, working)
        note_peak(device, ran)
    if stored is not None:
        note_working(device, working, baseline)
    return working if ran or stored is not None else None


def note_working(device: torch.device, working: int, baseline: int) -> None:
    """Keep `baseline` plus `working` bytes as a prewarm peak of `device`."""
    peak = baseline + working
    note_peak(device, {"working": OpReport(peak_alloc=peak, peak_reserved=peak)})


def share_working(gate: str, workings: list, store=None) -> None:
    """Rank 0: release the other ranks from `gate` with its kernel prewarm's working
    bytes (the largest over its devices; none: "done")."""
    known = [w for w in workings if w is not None]
    release_ranks(gate, store, str(max(known)) if known else "done")


def take_working(gate: str, devices: list, store=None) -> None:
    """The other ranks: wait for rank 0 at `gate` and keep the bytes allocated now
    plus rank 0's working bytes as the prewarm peak of each of `devices`."""
    payload = wait_for_rank0(gate, store)
    if payload.isdigit():
        for device in devices:
            note_working(device, int(payload), allocated(device))


def log_prewarm(reports: dict, took: float, device) -> None:
    """The per-op prewarm table and the one-line summary."""
    rows = [
        (
            name,
            r.cached,
            r.problems,
            r.failed,
            r.timed,
            round(r.secs, 2),
            round(r.peak_alloc / GIB, 2),
            round(r.peak_reserved / GIB, 2),
            "/".join(sorted(r.bound)) or "-",
        )
        for name, r in reports.items()
    ]
    total = [sum(r[i] for r in rows) for i in range(1, 5)]
    top = [max((r[i] for r in rows), default=0) for i in (6, 7)]
    rows.append(("total", *total, round(took, 2), *top, ""))
    log_table(
        log,
        f"kernel prewarm on {device}",
        (
            "op",
            "from table",
            "tuned",
            "skipped",
            "timed",
            "secs",
            "peak GiB",
            "reserved GiB",
            "bound",
        ),
        rows,
    )
    log.info(
        f"prewarm: {hl(total[0])} problems from the tuning table, {hl(total[1])} tuned "
        f"in {hl(f'{took:.1f} s')}"
    )


def prewarm_devices(device: torch.device) -> list[torch.device]:
    """Devices the prewarm tunes: the training device (ops on further threaded stage
    devices bind the same card's tables, else their first candidate untimed); none on
    CPU, where every op is the untuned eager reference."""
    if device.type == "cpu":
        return []
    return [device]


def release_workspaces(devices: list[torch.device]) -> None:
    """Return the prewarm's scratch blocks to the driver on every prewarmed device."""
    for device in devices:
        release(device)


class AdapterPrewarm:
    """The trainable set's autotuned-op prewarm (`prewarm(rows, dtype, devices,
    budgets)`) over the adapter token buckets of the planned batches, from and back to
    an empty allocator cache, each problem's timing within the device's tuning budget
    (`tuning_budget` up to the kernel prewarm's peak: larger buckets inherit); its
    device peak is kept for the step-1 report (`note_peak`)."""

    def __init__(self, trainable, dtype: torch.dtype, buckets: list[int]):
        self.trainable = trainable
        self.dtype = dtype
        self.buckets = buckets

    def __call__(self, devices: list[torch.device]) -> int:
        release_workspaces(devices)
        budgets = {
            d: tuning_budget(d, PREWARM_PEAK.get(str(d), (0, 0))[0]) for d in devices
        }
        with contextlib.ExitStack() as stack:
            windows = {d: stack.enter_context(window(d)) for d in devices}
            with rng_kept(devices):
                measured = self.trainable.prewarm(
                    self.buckets, self.dtype, devices, budgets
                )
        release_workspaces(devices)
        peaks = []
        for device, w in windows.items():
            report = OpReport()
            report.peak_alloc, report.peak_reserved = w.alloc, w.reserved
            note_peak(device, {"adapter ops": report})
            peaks.append(f"{device} {w.alloc / GIB:.2f} / {w.reserved / GIB:.2f}")
        stats = self.trainable.prewarm_stats
        if stats:
            log.info(
                f"adapter kernel ops: {hl(stats['ops'])} autotuned, "
                f"{hl(stats['distinct'])} distinct, {hl(measured)} problems measured "
                f"(the rest stored or settled); peak GiB allocated / reserved: "
                f"{hl(', '.join(peaks) or '-')}"
            )
        decisions.flush("after the adapter prewarm")
        end_prewarm(budgets)
        return measured


def end_prewarm(budgets: dict) -> None:
    """The end of the prewarm: kernel configs are fixed (`window.freeze`: a key first
    met later binds untimed), the tuning windows after it (text-encoder prewarm, first
    sampler step, retune steps) time within `budgets` (device -> bytes,
    `tuner.set_cap`), and LyCORIS binds its stored or model-ranked tiles untimed, so
    every LyCORIS tile tune runs before step 0."""
    for device, nbytes in budgets.items():
        set_cap(device, nbytes)
    freeze()
    set_lycoris_tuning(False)
    caps = ", ".join(f"{d} {b / GIB:.2f} GiB" for d, b in budgets.items() if b)
    log.info(
        "after the prewarm: kernel configs fixed (a key first met later binds its "
        "first candidate, untimed); the text-encoder prewarm, the first sampler step "
        f"and the retune steps tune within the tuning budget ({caps or 'none'}); "
        "LyCORIS tile tuning off, an untuned LyCORIS key binds its model-ranked tile, "
        "untimed and unstored"
    )
