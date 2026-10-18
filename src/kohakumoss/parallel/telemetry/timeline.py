"""Kernel timeline of a profiled window (chrome-trace events of one device): compute,
NCCL communication and copy time as unions of kernel intervals, the communication not
hidden under compute, and the device's idle time. See docs/guides/multi-gpu.md (step
breakdown)."""

COPY_CATS = ("gpu_memcpy", "gpu_memset")
DEVICE_CATS = ("kernel", *COPY_CATS)


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
