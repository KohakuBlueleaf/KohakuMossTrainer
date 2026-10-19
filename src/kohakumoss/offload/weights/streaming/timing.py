"""Device spans of a weight stream: H2D copies and the compute stream's waits for them,
cut once per step. See docs/internals/offload.md#weight-streaming."""

from kohakumoss.offload.weights.streaming.device import DeviceStreams
from kohakumoss_common.calls import nothing


def read_timing(cut: dict) -> dict[str, float]:
    """`WeightStream.cut_timing()` as seconds copying / waiting, GB moved and host
    syncs (its events must have completed)."""
    return {
        "h2d_gb": cut["h2d_gb"],
        "copy_s": sum(a.elapsed_time(b) for a, b in cut["copies"]) / 1e3,
        "wait_s": sum(a.elapsed_time(b) for a, b in cut["waits"]) / 1e3,
        "host_syncs": cut["host_syncs"],
    }


class SpanTimer:
    """Copy and copy-wait spans recorded on the device while timing is on, plus the
    stream's H2D bytes and host syncs (`stats`) since the last cut. `stamp(stream)` is
    a span's start event while timing is on, else None."""

    def __init__(self, dev: DeviceStreams, stats: dict):
        self.dm = dev.dm
        self.streamed = dev.streamed
        self.stats = stats
        self.copies: list = []
        self.waits: list = []
        self.mark_bytes = self.mark_syncs = 0
        self._stamp = nothing

    def stamp(self, stream):
        """A span's start event on `stream` while timing is on, else None."""
        return self._stamp(self, stream)

    def _event(self, stream):
        ev = self.dm.Event(enable_timing=True)
        ev.record(stream)
        return ev

    def log(self, spans: list, start, stream) -> None:
        """Close the span started at `start` (None: timing off) on `stream`."""
        if start is not None:
            spans.append((start, self._event(stream)))

    def set(self, on: bool) -> None:
        self._stamp = SpanTimer._event if on and self.streamed else nothing

    def cut(self) -> dict:
        """The copy / wait spans, GB moved and host syncs since the last cut, unread
        (no device sync; `read_timing` once their events have completed)."""
        out = {
            "copies": list(self.copies),
            "waits": list(self.waits),
            "h2d_gb": (self.stats["bytes_h2d"] - self.mark_bytes) / 1e9,
            "host_syncs": self.stats["host_syncs"] - self.mark_syncs,
        }
        self.copies.clear()
        self.waits.clear()
        self.mark_bytes = self.stats["bytes_h2d"]
        self.mark_syncs = self.stats["host_syncs"]
        return out
