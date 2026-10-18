"""The slot-ring events shared by the weight stream and the layer streamer, model-free
(a recording stand-in for the device module): every `free` event and every `recorded`
event is recorded once on the current stream at creation, `ready` events are not."""

import pytest

from kohakumoss.offload.weights.streaming.events import SlotEvents, recorded
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


class _Event:
    def __init__(self):
        self.on = []

    def record(self, stream) -> None:
        self.on.append(stream)


class _Module:
    Event = _Event
    STREAM = "current"

    def current_stream(self, device):
        return self.STREAM


def assert_events(make_slots, make_recorded) -> None:
    dm = _Module()
    events = make_slots(dm, "dev", 3)
    assert len(events.ready) == len(events.free) == 3
    assert all(ev.on == [] for ev in events.ready)
    assert all(ev.on == [dm.STREAM] for ev in events.free)
    assert len({id(ev) for ev in events.ready + events.free}) == 6
    done = make_recorded(dm, "dev", 4)
    assert len(done) == 4 and all(ev.on == [dm.STREAM] for ev in done)


def test_slot_events():
    assert_events(SlotEvents, recorded)


class _Unrecorded(SlotEvents):
    def __init__(self, dm, device, slots):
        self.ready = [dm.Event() for _ in range(slots)]
        self.free = [dm.Event() for _ in range(slots)]


def test_events_check_fails_on_broken_ring():
    must_fail(lambda: assert_events(_Unrecorded, recorded), "free not recorded")
    must_fail(
        lambda: assert_events(SlotEvents, lambda dm, d, n: recorded(dm, d, n)[:-1]),
        "event dropped",
    )
