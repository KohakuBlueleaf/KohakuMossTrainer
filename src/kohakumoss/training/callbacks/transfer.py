"""Host <-> device traffic of the DiT-side offload engines between log points: weight
streaming, gradient offload, activation offload and the host-state optimisers, read
from their byte counters (no device sync). The text encoder and VAE report their own
traffic (`te_h2d_gb`, `vae_h2d_gb`). See docs/guides/logging.md#transfer."""

from functools import partial

from kohakumoss.offload.optim.host_acc import RING_WAIT
from kohakumoss.training.state import TrainState

PACKED = "packed_bytes"  # activation offload: the same bytes go out and come back


def ring_wait(sources: list) -> float:
    """Host seconds backward waited for a busy gradient staging slot, so far, summed
    over `sources`."""
    return sum(src.stats.get(RING_WAIT, 0.0) for src in sources)


def _counters(source) -> tuple[int, int]:
    """(host-to-device, device-to-host) bytes counted so far by `source`."""
    stats = source.stats
    if PACKED in stats:
        return stats[PACKED], stats[PACKED]
    return stats.get("bytes_h2d", 0), stats.get("bytes_d2h", 0)


def _has_counters(obj) -> bool:
    stats = getattr(obj, "stats", None)
    return isinstance(stats, dict) and (
        PACKED in stats or "bytes_h2d" in stats or "bytes_d2h" in stats
    )


def transfer_sources(engines, bundle) -> list:
    """Every object of the run's memory engines and optimiser bundle that counts its
    host <-> device bytes (each once)."""
    candidates = [
        getattr(engines, "weight_stream", None),
        getattr(engines, "grads", None),
        getattr(engines, "offload", None),
    ]
    candidates += [bundle.optimizer, bundle.shared, bundle.blocks]
    opts = getattr(bundle, "opts", None) or {}
    candidates += [entry[0] for entry in opts.values()]
    out, seen = [], set()
    for obj in candidates:
        if obj is not None and id(obj) not in seen and _has_counters(obj):
            seen.add(id(obj))
            out.append(obj)
    return out


class TransferStats:
    """`xfer_h2d_gb` / `xfer_d2h_gb`: DiT-side GB moved per step since the last log
    point, summed over `sources` (`transfer_sources`); with a gradient staging ring
    among them, `grad_offload_wait_s`: host seconds per step backward waited for one."""

    edges = True

    def __init__(self, every: int, sources: list):
        self.every = every
        self.sources = sources
        self.mark = self._total()
        self.step0 = 0
        waits = [s for s in sources if RING_WAIT in s.stats]
        self.waited = partial(ring_wait, waits) if waits else None
        self.wait_mark = ring_wait(waits)

    def _total(self) -> tuple[int, int]:
        h2d = d2h = 0
        for src in self.sources:
            a, b = _counters(src)
            h2d, d2h = h2d + a, d2h + b
        return h2d, d2h

    def on_step_end(self, state: TrainState, out) -> None:
        cur = self._total()
        steps = max(state.step - self.step0, 1)
        state.metrics.update(
            xfer_h2d_gb=(cur[0] - self.mark[0]) / 1e9 / steps,
            xfer_d2h_gb=(cur[1] - self.mark[1]) / 1e9 / steps,
        )
        if self.waited is not None:
            waited = self.waited()
            state.metrics["grad_offload_wait_s"] = (waited - self.wait_mark) / steps
            self.wait_mark = waited
        self.mark, self.step0 = cur, state.step
