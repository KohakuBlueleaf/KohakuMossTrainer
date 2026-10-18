"""The device a weight stream and its parts run on. See
docs/internals/offload.md#weight-streaming."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceStreams:
    """`device`, its device module `dm` (torch.cuda / torch.xpu / the host stand-in) and
    `streamed`: side streams and events exist there (CUDA / XPU, `accel.streamed`)."""

    device: torch.device
    dm: object
    streamed: bool
