"""Kernel timeline of a profiled window (chrome-trace events of one device): compute,
NCCL communication and copy time as unions of kernel intervals, the communication not
hidden under compute, the device's idle time, and the compute time per step phase (the
innermost phase range open at each kernel's launch). See docs/guides/multi-gpu.md (step
breakdown)."""

import bisect

from kohakumoss.logging.phases import PHASES, PREFIX

COPY_CATS = ("gpu_memcpy", "gpu_memset")
DEVICE_CATS = ("kernel", *COPY_CATS)
LAUNCH_CATS = ("cuda_runtime", "cuda_driver")
TOP_KERNELS = 40


def kind_of(event: dict) -> str:
    """`comm` (an NCCL kernel), `copy` (memcpy / memset) or `compute`."""
    if event.get("cat") in COPY_CATS:
        return "copy"
    return "comm" if event.get("name", "").startswith("nccl") else "compute"


def device_events(events: list[dict], device: int) -> list[dict]:
    """Complete ("X") kernel / copy events of `device`."""
    return [
        e
        for e in events
        if e.get("ph") == "X"
        and e.get("cat") in DEVICE_CATS
        and int(e.get("args", {}).get("device", -1)) == device
    ]


def union(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sorted, merged, non-overlapping intervals covering `spans`."""
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def length(spans: list[tuple[float, float]]) -> float:
    return sum(b - a for a, b in spans)


def overlap(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Length of the intersection of two merged interval lists."""
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def breakdown(events: list[dict], device: int, wall_us: float, steps: int) -> dict:
    """Per-step ms of `device` over a window of `steps` steps and `wall_us` host
    microseconds: compute, communication (all / not overlapped by compute), copies,
    idle (no kernel or copy running), plus the busy and compute fractions."""
    spans: dict[str, list] = {"compute": [], "comm": [], "copy": []}
    for e in device_events(events, device):
        start = float(e["ts"])
        spans[kind_of(e)].append((start, start + float(e.get("dur", 0.0))))
    merged = {k: union(v) for k, v in spans.items()}
    busy = union([s for v in merged.values() for s in v])
    steps = max(steps, 1)
    per = 1e-3 / steps
    compute, comm = length(merged["compute"]), length(merged["comm"])
    exposed = comm - overlap(merged["comm"], merged["compute"])
    wall = max(wall_us, length(busy), 1e-9)
    return {
        "trace_wall_ms": wall * per,
        "trace_compute_ms": compute * per,
        "trace_comm_ms": comm * per,
        "trace_comm_exposed_ms": exposed * per,
        "trace_copy_ms": length(merged["copy"]) * per,
        "trace_idle_ms": (wall - length(busy)) * per,
        "trace_busy_frac": length(busy) / wall,
        "trace_compute_frac": compute / wall,
        "trace_kernels_per_step": sum(len(v) for v in spans.values()) / steps,
    }


def phase_ranges(events: list[dict]) -> list[tuple[float, float, str]]:
    """(start, end, phase) of every phase range on any host thread, sorted by start
    (outer before inner at equal starts)."""
    out = []
    for e in events:
        name = e.get("name", "")
        if e.get("ph") == "X" and e.get("cat") == "user_annotation":
            if name.startswith(PREFIX) and name[len(PREFIX) :] in PHASES:
                start = float(e["ts"])
                out.append(
                    (start, start + float(e.get("dur", 0.0)), name[len(PREFIX) :])
                )
    out.sort(key=lambda r: (r[0], -r[1]))
    return out


def launch_times(events: list[dict]) -> dict[int, float]:
    """Host launch time of each correlation id (runtime / driver launch events)."""
    out = {}
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in LAUNCH_CATS:
            corr = e.get("args", {}).get("correlation")
            if corr is not None:
                out[int(corr)] = float(e["ts"])
    return out


def parents(ranges: list[tuple[float, float, str]]) -> list[int]:
    """Index of the innermost range enclosing each range's start (-1: none)."""
    out, stack = [], []
    for i, (start, end, _) in enumerate(ranges):
        while stack and ranges[stack[-1]][1] <= start:
            stack.pop()
        out.append(stack[-1] if stack else -1)
        stack.append(i)
    return out


def phase_at(ranges, starts: list[float], up: list[int], ts: float) -> str:
    """The innermost phase range containing host time `ts`, or "other": the
    latest-starting range at `ts`, else its nearest enclosing range that has not ended.
    """
    i = bisect.bisect_right(starts, ts) - 1
    while i >= 0:
        if ranges[i][1] >= ts:
            return ranges[i][2]
        i = up[i]
    return "other"


def phase_breakdown(events: list[dict], device: int, steps: int) -> tuple[dict, dict]:
    """Per-step ms of `device`'s compute kernels (union of intervals) by step phase, and
    per phase the TOP_KERNELS kernel names by total time (ms per step, calls per step).
    """
    ranges = phase_ranges(events)
    starts = [r[0] for r in ranges]
    up = parents(ranges)
    launches = launch_times(events)
    spans: dict[str, list] = {p: [] for p in PHASES}
    names: dict[str, dict[str, list]] = {p: {} for p in PHASES}
    for e in device_events(events, device):
        if kind_of(e) != "compute":
            continue
        corr = e.get("args", {}).get("correlation")
        at = launches.get(int(corr)) if corr is not None else None
        p = phase_at(ranges, starts, up, at) if at is not None else "other"
        start, dur = float(e["ts"]), float(e.get("dur", 0.0))
        spans[p].append((start, start + dur))
        row = names[p].setdefault(e.get("name", ""), [0.0, 0])
        row[0] += dur
        row[1] += 1
    steps = max(steps, 1)
    per = 1e-3 / steps
    out = {f"trace_phase_{p}_ms": length(union(spans[p])) * per for p in PHASES}
    top = {}
    for p, rows in names.items():
        ranked = sorted(rows.items(), key=lambda kv: -kv[1][0])[:TOP_KERNELS]
        top[p] = [
            {"kernel": n, "ms": t * per, "calls": c / steps} for n, (t, c) in ranked
        ]
    return out, top
