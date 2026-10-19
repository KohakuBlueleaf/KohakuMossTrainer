"""A saved tensor read more often than it was saved (a custom Function or a recompute
reading its saver twice) gets its own data again, never a recycled host buffer's next
owner, model-free: the device path over fake streams and events on the host."""

import contextlib
import types

import pytest
import torch

from kohakumoss.offload.activation import ActivationOffload, Packet
from kohakumoss_common.calls import nothing
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


class _Event:
    def record(self, stream=None) -> None:
        return None

    def synchronize(self) -> None:
        return None


class _Stream:
    def wait_stream(self, other) -> None:
        return None

    def wait_event(self, event) -> None:
        return None


class _Device:
    Event = _Event

    def stream(self, stream):
        return contextlib.nullcontext()

    def current_stream(self, device) -> _Stream:
        return _Stream()


def _engine(monkeypatch) -> ActivationOffload:
    off = ActivationOffload("cpu", min_bytes=0)
    off.dm, off.h2d_stream, off.d2h_stream = _Device(), _Stream(), _Stream()
    off._load, off._prefetch = ActivationOffload._load_device, nothing
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None)
    return off


def _packet(off: ActivationOffload, value: float) -> Packet:
    """A packet whose host copy holds `value`, in a pooled host buffer."""
    t = torch.full((64,), value)
    flat = off._take(t)
    host = flat[: t.numel() * t.element_size()].view(t.dtype)
    host.copy_(t)
    return Packet(host, 0, d2h=_Event(), flat=flat)


def check_reread(off: ActivationOffload) -> None:
    pk = _packet(off, 1.0)
    first = off._load_device(pk).clone()
    _packet(off, 7.0)  # the next packet takes the recycled host buffer
    second = off._load_device(pk)
    assert torch.equal(first, second) and bool((second == 1.0).all()), second[:4]


def _early_release(self, pk: Packet) -> torch.Tensor:
    """The device copy dropped once `users` reads happened, re-fetched on another."""
    if pk.device_copy is None:
        self.pending.setdefault(pk.block, []).append(pk)
        self._fetch(pk.block)
    copy = pk.device_copy
    pk.users -= 1
    if pk.users == 0:
        pk.device_copy = None
    return copy


def test_second_unpack_reads_the_packet_not_the_recycled_buffer(monkeypatch):
    check_reread(_engine(monkeypatch))
    broken = _engine(monkeypatch)
    broken._load_device = types.MethodType(_early_release, broken)
    must_fail(lambda: check_reread(broken), "device copy released at users == 0")
