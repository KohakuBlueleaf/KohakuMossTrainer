"""nvidia-smi on the host: the card list, whether a card is idle, and a 500 ms sampler
of every compute process's memory and every card's used memory, recorded per run.

The sampler runs two long-lived `nvidia-smi ... --loop-ms` processes for the whole host
queue; a run registers its child pid and card and gets its own rows (time, process MiB,
card MiB). Under Windows WDDM the per-process column reads [N/A]; the card's used memory
minus its idle baseline stands in. See docs/performance/bench-matrix.md#vram.
"""

import shutil
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from kohakumoss_bench.plan.config import Card

SMI = "nvidia-smi"
GPU_FIELDS = (
    "index,uuid,name,memory.total,memory.used,utilization.gpu,compute_cap,pci.bus_id"
)
APP_FIELDS = "gpu_uuid,pid,used_memory"
PERIOD_MS = 500


def available() -> bool:
    return shutil.which(SMI) is not None


def _query(fields: str, what: str, prefix: tuple = ()) -> list[list[str]]:
    """nvidia-smi query rows, on this machine or through a command `prefix` (ssh)."""
    out = subprocess.run(
        [*prefix, SMI, f"--query-{what}={fields}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [[c.strip() for c in line.split(",")] for line in out.splitlines() if line]


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def cards(prefix: tuple = ()) -> list[Card]:
    """Every CUDA card nvidia-smi lists (PCI bus order, as CUDA_DEVICE_ORDER=PCI_BUS_ID),
    on this machine or through a command `prefix` (ssh)."""
    out = []
    for idx, uuid, name, total, _, _, cap, bus in _query(GPU_FIELDS, "gpu", prefix):
        major, _, minor = cap.partition(".")
        out.append(
            Card(
                index=int(idx),
                name=name,
                total_gib=float(total) / 1024,
                sm=int(major) * 10 + int(minor or 0),
                uuid=uuid,
                bus=bus,
            )
        )
    return out


def bus_number(bus: str) -> str:
    """The PCI bus number of a bus id, lower case: "00000000:83:00.0" -> "83"; a bare
    number stays as it is ("C3" -> "c3")."""
    parts = bus.strip().lower().split(":")
    return parts[-2] if len(parts) >= 2 else parts[0]


def select(found: list[Card], gpus: list[int] | None, buses: list[str] | None) -> list:
    """The host's bench cards among `found`: those on the PCI `buses` when given
    (indices shift when a card drops off the bus), else those of the `gpus` indices,
    else all. With `buses`, a listed bus without a card, or a card outside `gpus` when
    both are given, raises ValueError: the runner refuses to start."""
    if not buses:
        return [c for c in found if gpus is None or c.index in gpus]
    by_bus = {bus_number(c.bus): c for c in found}
    missing = [b for b in buses if bus_number(b) not in by_bus]
    if missing:
        have = {c.index: c.bus for c in found}
        raise ValueError(f"no card on PCI bus {missing}; found {have}")
    chosen = [by_bus[bus_number(b)] for b in buses]
    if gpus is not None and sorted(c.index for c in chosen) != sorted(gpus):
        raise ValueError(
            f"buses {buses} hold indices {[c.index for c in chosen]}, not gpus {gpus}"
        )
    return chosen


def used(prefix: tuple = ()) -> dict[int, float]:
    """Used MiB per card index, on this machine or through a command `prefix`."""
    return {int(r[0]): float(r[4]) for r in _query(GPU_FIELDS, "gpu", prefix)}


def busy_pids(card: Card) -> set[int]:
    """Compute processes holding memory on `card` now. Under Windows WDDM every desktop
    process with a GPU context is listed with [N/A] memory; those do not count."""
    return {
        int(pid)
        for uuid, pid, used in _query(APP_FIELDS, "compute-apps")
        if uuid == card.uuid and pid.isdigit() and _number(used) is not None
    }


def used_mib(card: Card) -> float:
    for row in _query(GPU_FIELDS, "gpu"):
        if int(row[0]) == card.index:
            return float(row[4])
    return 0.0


def steady(values: list[float]) -> float | None:
    """Median of the second half of `values` (None when empty)."""
    tail = values[len(values) // 2 :]
    return statistics.median(tail) if tail else None


def descends(pid: int, root: int, proc: Path = Path("/proc")) -> bool:
    """Whether `pid` is `root` or one of its descendants (the /proc parent chain; False
    without /proc or once a process in the chain is gone)."""
    seen = 0
    while pid > 1 and seen < 64:
        if pid == root:
            return True
        try:
            stat = (proc / str(pid) / "stat").read_text()
        except OSError:
            return False
        pid = int(stat.rsplit(")", 1)[1].split()[1])
        seen += 1
    return pid == root


@dataclass
class Trace:
    """The rows of one run: (seconds since start, process MiB or None, card MiB)."""

    card_uuid: str
    pid: int
    baseline_mib: float
    t0: float = field(default_factory=time.monotonic)
    rows: list = field(default_factory=list)
    process: float | None = None
    foreign: dict = field(default_factory=dict)  # pid -> first seen (s) on this card

    def summary(self) -> dict:
        """Peak and steady process MiB (None under WDDM), peak card MiB, peak and steady
        card MiB above the idle baseline, samples taken. Steady: the median over the
        second half of the samples where the process held memory (under WDDM: the card
        above its baseline)."""
        procs = [r[1] for r in self.rows if r[1] is not None]
        card = [r[2] for r in self.rows]
        held = [c - self.baseline_mib for c in card if c > self.baseline_mib]
        peak_card = max(card, default=0.0)
        return {
            "smi_process_peak_mib": max(procs) if procs else None,
            "smi_process_steady_mib": steady(procs),
            "smi_card_peak_mib": peak_card,
            "smi_card_above_idle_mib": max(peak_card - self.baseline_mib, 0.0),
            "smi_card_steady_above_idle_mib": steady(held),
            "smi_baseline_mib": self.baseline_mib,
            "smi_samples": len(self.rows),
            "smi_foreign": {str(k): v for k, v in self.foreign.items()},
        }

    def phases(self, split_s: float | None) -> dict:
        """`summary` of the samples before `split_s` seconds (startup and prewarm,
        `prewarm_*`) and of those from it on (the step loop, `step_*`); {} without a
        split."""
        if split_s is None:
            return {}
        out = {}
        for phase, rows in (
            ("prewarm", [r for r in self.rows if r[0] < split_s]),
            ("step", [r for r in self.rows if r[0] >= split_s]),
        ):
            part = Trace(self.card_uuid, self.pid, self.baseline_mib, self.t0, rows)
            out.update({f"{phase}_{k}": v for k, v in part.summary().items()})
        out["split_s"] = split_s
        return out

    def csv(self) -> str:
        lines = ["t_s,process_mib,card_mib"]
        for t, p, c in self.rows:
            lines.append(f"{t:.2f},{'' if p is None else f'{p:.0f}'},{c:.0f}")
        return "\n".join(lines) + "\n"


class Sampler:
    """The host's two sampling processes and the traces registered on them."""

    def __init__(self):
        self.lock = threading.Lock()
        self.traces: dict[int, Trace] = {}
        self.card_used: dict[str, float] = {}
        self.procs: list[subprocess.Popen] = []
        self.threads: list[threading.Thread] = []

    def start(self) -> None:
        for what, fields, handle in (
            ("gpu", "uuid,memory.used", self._gpu_line),
            ("compute-apps", APP_FIELDS, self._app_line),
        ):
            proc = subprocess.Popen(
                [
                    SMI,
                    f"--query-{what}={fields}",
                    "--format=csv,noheader,nounits",
                    f"--loop-ms={PERIOD_MS}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            thread = threading.Thread(
                target=self._read, args=(proc, handle), daemon=True
            )
            thread.start()
            self.procs.append(proc)
            self.threads.append(thread)

    def _read(self, proc: subprocess.Popen, handle) -> None:
        for line in proc.stdout:
            cells = [c.strip() for c in line.split(",")]
            if len(cells) >= 2:
                handle(cells)

    def _gpu_line(self, cells: list[str]) -> None:
        uuid, used = cells[0], _number(cells[1])
        if used is None:
            return
        with self.lock:
            self.card_used[uuid] = used
            for trace in self.traces.values():
                if trace.card_uuid == uuid:
                    t = time.monotonic() - trace.t0
                    trace.rows.append((t, trace.process, used))
                    trace.process = None

    def _app_line(self, cells: list[str]) -> None:
        if len(cells) < 3 or not cells[1].isdigit():
            return
        uuid, pid, used = cells[0], int(cells[1]), _number(cells[2])
        if used is None:
            return
        with self.lock:
            trace = self.traces.get(pid)
            if trace is not None:
                trace.process = max(trace.process or 0.0, used)
            for other in self.traces.values():
                if other.card_uuid == uuid and other is not trace:
                    if pid not in other.foreign and not descends(pid, other.pid):
                        other.foreign[pid] = round(time.monotonic() - other.t0, 1)

    def register(self, pid: int, card: Card, baseline_mib: float) -> Trace:
        trace = Trace(card.uuid, pid, baseline_mib)
        with self.lock:
            self.traces[pid] = trace
        return trace

    def release(self, pid: int) -> Trace | None:
        with self.lock:
            return self.traces.pop(pid, None)

    def stop(self) -> None:
        for proc in self.procs:
            proc.terminate()
        for proc in self.procs:
            proc.wait(timeout=10)


class NoSampler:
    """Hosts without nvidia-smi (XPU): runs record their in-process device use only."""

    def start(self) -> None:
        return None

    def register(self, pid: int, card: Card, baseline_mib: float) -> None:
        return None

    def release(self, pid: int) -> None:
        return None

    def stop(self) -> None:
        return None
