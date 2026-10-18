"""The device events of a slot ring, shared by the weight stream and the layer streamer.
See docs/internals/offload.md#weight-streaming."""


def recorded(dm, device, n: int) -> list:
    """`n` events already recorded on `device`'s current stream (`dm`: its device
    module)."""
    cur = dm.current_stream(device)
    events = [dm.Event() for _ in range(n)]
    for ev in events:
        ev.record(cur)
    return events


class SlotEvents:
    """Per slot of a ring: `ready[s]`, recorded after slot s's H2D copy, and `free[s]`,
    recorded when the compute stream releases slot s (every slot free at creation)."""

    def __init__(self, dm, device, slots: int):
        self.ready = [dm.Event() for _ in range(slots)]
        self.free = recorded(dm, device, slots)
