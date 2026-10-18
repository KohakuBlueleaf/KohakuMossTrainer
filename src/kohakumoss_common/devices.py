"""Device-type facts every package shares (no torch import): the accelerator types with
side streams, events and pinned host memory."""

STREAMED = ("cuda", "xpu")


def streamed_type(device_type: str) -> bool:
    """`device_type` ("cuda", "xpu", "cpu", ...) has side streams, events and pinned
    host memory."""
    return device_type in STREAMED
