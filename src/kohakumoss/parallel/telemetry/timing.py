"""Device-event intervals of one training step: recorded on the current stream with no
host sync, read back once (`read`) at the metric cadence. See docs/guides/multi-gpu.md
(step breakdown)."""

import torch


class EventClock:
    """Named device intervals of the step between `begin()` and `end()` (CUDA)."""

    def __init__(self, device: torch.device):
        self.device = torch.device(device)
        self.pool: list[torch.cuda.Event] = []
        self.used = 0
        self.marks: list[tuple[str, int, int]] = []
        self.span: tuple[int, int] | None = None
        self._open = 0

    def _event(self) -> int:
        i = self.used
        if i == len(self.pool):
            self.pool.append(torch.cuda.Event(enable_timing=True))
        self.pool[i].record()
        self.used = i + 1
        return i

    def begin(self) -> None:
        """Start a step: the previous step's intervals are dropped."""
        self.used = 0
        self.marks = []
        self.span = None
        self._open = self._event()

    def start(self) -> int:
        """Open an interval; pass the handle to `stop`."""
        return self._event()

    def stop(self, name: str, handle: int) -> None:
        self.marks.append((name, handle, self._event()))

    def end(self) -> None:
        """Close the step (the span `read` reports as `wall_ms`)."""
        self.span = (self._open, self._event())

    def read(self) -> dict[str, float]:
        """`<name>_ms` summed per name and `wall_ms` (begin -> end) of the last closed
        step; {} before the first."""
        if self.span is None:
            return {}
        pool = self.pool
        pool[self.span[1]].synchronize()
        out: dict[str, float] = {}
        for name, a, b in self.marks:
            key = f"{name}_ms"
            out[key] = out.get(key, 0.0) + pool[a].elapsed_time(pool[b])
        out["wall_ms"] = pool[self.span[0]].elapsed_time(pool[self.span[1]])
        return out


class NoClock:
    """Placeholder off CUDA: records nothing, reads {}."""

    def begin(self) -> None:
        return None

    def start(self) -> int:
        return 0

    def stop(self, name: str, handle: int) -> None:
        return None

    def end(self) -> None:
        return None

    def read(self) -> dict[str, float]:
        return {}


def build_clock(device: torch.device | str):
    """EventClock on a CUDA device, else NoClock."""
    device = torch.device(device)
    return EventClock(device) if device.type == "cuda" else NoClock()
