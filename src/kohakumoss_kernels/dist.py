"""The default process group as this process sees it: (rank, world), whether a group
of two or more ranks runs, and the device type each backend's collectives use."""

import torch.distributed as dist

COLLECTIVE_DEVICE = {"nccl": "cuda", "xccl": "xpu"}


def initialized() -> bool:
    """Whether a default process group exists."""
    return dist.is_available() and dist.is_initialized()


def dist_rank() -> tuple[int, int]:
    """(rank, world) of the default process group; (0, 1) without one."""
    if initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def grouped(group=None) -> bool:
    """Whether `group` (default: the default group) exists with two or more ranks."""
    return initialized() and dist.get_world_size(group) > 1
