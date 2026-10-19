"""Step phases as profiler ranges: the `TRACE_STEPS` kernel timeline
(`kohakumoss.parallel.telemetry.timeline`) attributes each kernel to the innermost phase
range open at its launch. See docs/guides/multi-gpu.md (step breakdown)."""

from torch.profiler import record_function

PREFIX = "kohakumoss."
PHASES = ("data", "forward", "backward", "recompute", "optimizer", "other")
DATA = PREFIX + "data"
FORWARD = PREFIX + "forward"
BACKWARD = PREFIX + "backward"
RECOMPUTE = PREFIX + "recompute"
OPTIMIZER = PREFIX + "optimizer"


def phase(name: str) -> record_function:
    """A profiler range named `name` (one of the labels above)."""
    return record_function(name)
