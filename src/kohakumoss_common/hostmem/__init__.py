"""Host memory a process may plan for now (`host_room`, `host_fits`), and the C heap's
free pages returned to the OS (`trim_heap`)."""

from kohakumoss_common.hostmem.available import HostRoom, host_fits, host_room
from kohakumoss_common.hostmem.trim import trim_heap

__all__ = ["HostRoom", "host_fits", "host_room", "trim_heap"]
