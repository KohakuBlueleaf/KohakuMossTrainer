"""Device-memory test helpers over kohakumoss_kernels.hardware.memory."""

from kohakumoss_kernels.hardware import memory


def read_window(w: memory.Window) -> tuple[int, int]:
    """(peak allocated, peak reserved) of an open window so far."""
    memory.fold(w.device)
    return w.alloc, w.reserved
