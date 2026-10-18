"""The device a weight stream and its parts run on. See
docs/internals/offload.md#weight-streaming."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceStreams:
    """`device`, its device module `dm` (torch.cuda / torch.xpu / the host stand-in) and
    `cuda`: side streams and events exist there (CUDA / XPU)."""

    device: torch.device
    dm: object
    cuda: bool
