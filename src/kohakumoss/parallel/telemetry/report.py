"""Per-rank metric rows and the job summary of a multi-device run:
`<output>/ranks/rank<r>.jsonl` on every rank, `<output>/ranks/summary.json` (rank 0,
every rank gathered) at the last step. See docs/guides/multi-gpu.md (per-rank
report)."""

import json
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.distributed as dist

from kohakumoss.model.flops import forward_flops, peak_tflops, step_flops
from kohakumoss.parallel.telemetry.trace import ProfilerWindow, StepTrace
from kohakumoss_common.units import GIB
from kohakumoss_kernels.hardware.memory import close_window, device_used, open_window

COUNTS = ("step", "samples", "tokens", "text_tokens", "joint_sq")
PEAKS = ("vram_peak_alloc_gib", "vram_peak_reserved_gib", "device_used_gib")
HOST = ("host_rss_gib", "host_private_gib", "host_shared_gib")
MEANS = ("_ms", "_frac", "_mb")
STATUS = {"VmRSS": "host_rss_gib", "RssAnon": "host_private_gib"}
SHARED = ("RssFile", "RssShmem")


def host_memory() -> dict[str, float]:
    """This process's resident host memory in GiB from /proc/self/status: all of it,
    the private part (anonymous) and the mapped shared / file part; {} without
    /proc."""
    try:
        text = Path("/proc/self/status").read_text()
    except OSError:
        return {}
    kib = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        if name in STATUS or name in SHARED:
            kib[name] = float(rest.split()[0])
    out = {key: kib[name] / 2**20 for name, key in STATUS.items() if name in kib}
    out["host_shared_gib"] = sum(kib.get(name, 0.0) for name in SHARED) / 2**20
    return out


class SmiSampler:
    """`nvidia-smi` memory.used and utilization.gpu of one device every `period_ms`
    (a background `nvidia-smi -lms` process); `take()` gives the peak memory (GiB) and
    the mean utilization (%) since the last call."""

    def __init__(self, device: torch.device, period_ms: int = 200):
        uuid = str(torch.cuda.get_device_properties(device).uuid)
        ident = uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"
        self.proc = subprocess.Popen(
            [
                "nvidia-smi",
                f"--id={ident}",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
                "-lms",
                str(period_ms),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self.lock = threading.Lock()
        self.window = [0.0, 0.0, 0]
        self.peak = 0.0
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            fields = line.strip().split(",")
            try:
                mem, util = float(fields[0]) / 1024, float(fields[1])
            except (ValueError, IndexError):
                continue
            with self.lock:
                w = self.window
                w[0], w[1], w[2] = max(w[0], mem), w[1] + util, w[2] + 1
                self.peak = max(self.peak, mem)

    def take(self) -> dict[str, float]:
        with self.lock:
            mem, util, n = self.window
            self.window = [0.0, 0.0, 0]
        if not n:
            return {}
        return {"smi_mem_peak_gib": mem, "smi_util_mean": util / n}

    def close(self) -> None:
        self.proc.terminate()


class NoSmi:
    """No nvidia-smi on this host (or not a CUDA device)."""

    peak = 0.0

    def take(self) -> dict[str, float]:
        return {}

    def close(self) -> None:
        return None


def build_smi(device: torch.device):
    if torch.device(device).type != "cuda":
        return NoSmi()
    try:
        return SmiSampler(torch.device(device))
    except OSError:
        return NoSmi()


class RankReport:
    """Every `every` steps (and the first / last): one row of this rank's throughput
    (samples/s, tokens/s, step time, TFLOP/s, MFU of its share of the model), peak
    VRAM allocated / reserved, device memory in use, nvidia-smi memory / utilization,
    resident host memory (total / private / shared), every `sources()` breakdown and
    the step's metrics so far. At `last`: the steady-state summary (rows after
    `warmup`, outside `exclude`) of every rank and the job's host RAM (private parts
    summed, the shared part once)."""

    edges = True

    def __init__(
        self,
        every: int,
        root: str,
        rank: int,
        world: int,
        data_world: int,
        device,
        flops: Callable[[int, int, int], float],
        share: float,
        sources: list[Callable[[], dict]],
        last: int,
        warmup: int = 3,
        exclude: frozenset[int] = frozenset(),
    ):
        self.every = every
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / f"rank{rank}.jsonl"
        self.rank, self.world, self.data_world = rank, world, data_world
        self.device = torch.device(device)
        self.name = (
            torch.cuda.get_device_name(self.device)
            if self.device.type == "cuda"
            else self.device.type
        )
        self.flops = flops
        self.share = share
        self.peak = peak_tflops(self.device)
        self.sources = sources
        self.last = last
        self.warmup = warmup
        self.exclude = exclude
        self.smi = build_smi(self.device)
        self.window = open_window(self.device)
        self.rows: list[dict] = []
        self.mark = dict.fromkeys(COUNTS, 0)
        self.t0 = time.perf_counter()

    def on_step_end(self, state, out) -> None:
        now = time.perf_counter()
        dt = max(now - self.t0, 1e-9)
        d = {k: getattr(state, k) - self.mark[k] for k in COUNTS}
        flops = self.flops(d["tokens"], d["text_tokens"], d["joint_sq"]) * self.share
        alloc, held = close_window(self.window)
        self.window = open_window(self.device)
        row = {
            "step": state.step,
            "rank": self.rank,
            "steps": d["step"],
            "seconds": dt,
            "samples": d["samples"],
            "tokens": d["tokens"],
            "step_time": dt / max(d["step"], 1),
            "samples_per_s": d["samples"] / dt,
            "tokens_per_s": d["tokens"] / dt,
            "joint_tokens_per_s": (d["tokens"] + d["text_tokens"]) / dt,
            "tflops": flops / dt / 1e12,
            "loss": state.loss,
            "grad_norm": state.grad_norm,
            "vram_peak_alloc_gib": alloc / GIB,
            "vram_peak_reserved_gib": held / GIB,
            "device_used_gib": device_used(self.device) / GIB,
        }
        if self.peak:
            row["mfu"] = row["tflops"] / self.peak
        row.update(self.smi.take())
        row.update(host_memory())
        for source in self.sources:
            row.update(source())
        row.update({k: v for k, v in state.metrics.items() if k not in row})
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=float) + "\n")
        self.rows.append(row)
        self.mark = {k: getattr(state, k) for k in COUNTS}
        self.t0 = time.perf_counter()
        if state.step >= self.last:
            self._summary()

    def _steady(self) -> list[dict]:
        skip = self.exclude
        rows = [
            r for r in self.rows if r["step"] > self.warmup and r["step"] not in skip
        ]
        return rows or self.rows

    def mine(self) -> dict:
        """This rank's steady-state totals, rates, means and peaks."""
        rows = self._steady()
        seconds = max(sum(r["seconds"] for r in rows), 1e-9)
        steps = sum(r["steps"] for r in rows)
        work = sum(r["tflops"] * r["seconds"] for r in rows)
        out = {
            "rank": self.rank,
            "device": self.name,
            "steady_steps": steps,
            "step_time": seconds / max(steps, 1),
            "samples_per_s": sum(r["samples"] for r in rows) / seconds,
            "tokens_per_s": sum(r["tokens"] for r in rows) / seconds,
            "tflops": work / seconds,
            "smi_mem_peak_gib": self.smi.peak,
        }
        if self.peak:
            out["mfu"] = out["tflops"] / self.peak
        for key in PEAKS:
            out[key] = max(r[key] for r in self.rows)
        for key in HOST:
            vals = [r[key] for r in self.rows if key in r]
            if vals:
                out[key] = max(vals)
        means = {k for r in rows for k in r if k.endswith(MEANS)}
        for key in sorted(means):
            vals = [float(r[key]) for r in rows if key in r]
            out[key] = sum(vals) / len(vals)
        return out

    def _summary(self) -> None:
        mine = self.mine()
        every = [mine]
        if self.world > 1:
            every = [None] * self.world
            dist.all_gather_object(every, mine)
        self.smi.close()
        if self.rank != 0:
            return
        scale = self.data_world / self.world
        job = {
            "world": self.world,
            "data_world": self.data_world,
            "samples_per_s": scale * sum(r["samples_per_s"] for r in every),
            "tokens_per_s": scale * sum(r["tokens_per_s"] for r in every),
            "step_time": max(r["step_time"] for r in every),
            "tflops": sum(r["tflops"] for r in every),
        }
        if all("mfu" in r for r in every):
            job["mfu"] = sum(r["mfu"] for r in every) / len(every)
        for key in (*PEAKS, "smi_mem_peak_gib"):
            job[key] = max(r[key] for r in every)
        if all("host_private_gib" in r for r in every):
            shared = max(r.get("host_shared_gib", 0.0) for r in every)
            job["host_ram_gib"] = sum(r["host_private_gib"] for r in every) + shared
        (self.root / "summary.json").write_text(
            json.dumps({"job": job, "ranks": every}, indent=1), encoding="utf-8"
        )


def train_flops(arch, mode: str, recompute: bool) -> Callable[[int, int, int], float]:
    """(image tokens, text tokens, sum of squared joint lengths) -> training FLOPs."""

    def flops(tokens: int, text: int, joint_sq: int) -> float:
        attn = 4 * arch.dim * arch.depth * joint_sq
        return step_flops(forward_flops(arch, tokens, [], text) + attn, mode, recompute)

    return flops


def rank_layout(kind: str, engine, depth: int, devices) -> tuple[float, list, list]:
    """(FLOP share of this rank, the extra devices it traces, its breakdown sources)
    for PARALLEL `kind`: the pipeline's local layers, the threaded pipeline's stage
    devices, the whole model elsewhere."""
    match kind:
        case "pipeline":
            local = sum(m.hi - m.lo for m in engine.modules_)
            return local / depth, [], [engine.metrics]
        case "threaded":
            return 1.0 / len(devices), list(devices[1:]), []
    return 1.0, [], []


def build_rank_callbacks(
    k: dict,
    arch,
    mode: str,
    recompute: bool,
    rank: int,
    world: int,
    data_world: int,
    device,
    output_dir: str,
    sources: list[Callable[[], dict]],
    share: float = 1.0,
    extra_devices=(),
) -> list:
    """The profiled windows (TRACE_STEPS / PROFILE_STEPS = (first, last), empty = none;
    `extra_devices` are traced too) and the per-rank report at the metric-row cadence,
    writing to `<output>/ranks/`. `data_world`: ranks drawing distinct data (world under
    data parallelism, 1 under the pipeline); `share`: this rank's fraction of the
    model's FLOPs."""
    root = str(Path(output_dir) / "ranks")
    every = k["LOG_EVERY"] if k.get("METRICS_EVERY") is None else k["METRICS_EVERY"]
    trace = tuple(k["TRACE_STEPS"])
    window = tuple(k["PROFILE_STEPS"])
    cbs = []
    exclude = frozenset()
    if trace:
        first, last = int(trace[0]), int(trace[1])
        cbs.append(StepTrace(first, last, root, rank, device, extra_devices))
        exclude = frozenset(range(first + 1, last + 1))
    if window and torch.device(device).type == "cuda":
        first, last = int(window[0]), int(window[1])
        cbs.append(ProfilerWindow(first, last, device))
        exclude = exclude | frozenset(range(first + 1, last + 1))
    report = RankReport(
        every,
        root,
        rank,
        world,
        data_world,
        device,
        train_flops(arch, mode, recompute),
        share,
        sources,
        k["STEPS"],
        exclude=exclude,
    )
    return [*cbs, report]
