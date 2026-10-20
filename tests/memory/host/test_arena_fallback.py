"""A refused host registration, injected (model-free, no device): the runtime's last
error is reset, the arena moves into pinned chunks holding the same values, every view
handed out and every tracked tensor follows it, and one WARNING names the size and the
error; the loaded-libcudart lookup reads /proc maps text."""

import logging

import pytest
import torch

from kohakumoss.offload.host import arena as ha
from kohakumoss.offload.host import cudart
from tests._util.logs import collecting
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
ERROR = "cudaErrorHostMemoryAlreadyRegistered: injected"


@pytest.fixture
def refused(monkeypatch):
    calls = {"reset": 0}
    monkeypatch.setattr(cudart, "register", lambda ptr, n: ERROR)
    monkeypatch.setattr(
        cudart, "reset_last_error", lambda: calls.__setitem__("reset", 1)
    )
    monkeypatch.setattr(ha, "pinned_empty", lambda n: torch.empty(n, dtype=torch.uint8))
    monkeypatch.setattr(ha, "arena_kind", lambda pinned, device: "registered")
    return calls


def assert_fallback(calls, finish) -> None:
    src = [torch.randn(64, 32), torch.arange(100, dtype=torch.int16)]
    arena = ha.HostArena([(t.shape, t.dtype) for t in src], True, "cuda")
    views = [arena.fill(i, t) for i, t in enumerate(src)]
    owner = torch.empty(0)
    owner.data = views[0]
    arena.track(0, owner)
    with collecting(logging.WARNING) as logs:
        finish(arena)
    assert calls["reset"] == 1, "the runtime's last error left set"
    assert arena.kind == "chunks" and arena.pinned
    chunk = arena.buffers[arena.where[0][0]]
    lo, hi = chunk.data_ptr(), chunk.data_ptr() + chunk.numel()
    for v, t in zip(views, src):
        assert torch.equal(v, t)
    assert lo <= views[0].data_ptr() < hi, "a handed-out view left behind"
    assert owner.data_ptr() == views[0].data_ptr(), "a tracked tensor left behind"
    text = "\n".join(logs.messages)
    assert "cudaHostRegister refused" in text and "injected" in text, text


def _no_repoint(arena) -> None:
    arena.tracked = [[] for _ in arena.specs]
    arena.finish()


def test_refused_registration_moves_into_chunks(refused):
    assert_fallback(refused, ha.HostArena.finish)
    refused["reset"] = 0
    must_fail(lambda: assert_fallback(refused, _no_repoint), "tracked not re-pointed")

    def no_reset(arena) -> None:
        arena.finish()
        refused["reset"] = 0

    must_fail(lambda: assert_fallback(refused, no_reset), "error not reset")


MAPS = """\
7f00-7f10 r-xp 00000000 08:01 123 /venv/lib/nvidia/cu13/lib/libcudart.so.13
7f10-7f20 r--p 00010000 08:01 123 /venv/lib/nvidia/cu13/lib/libcudart.so.13
7f20-7f30 r-xp 00000000 08:01 124 /usr/lib/libc.so.6
7f30-7f40 rw-p 00000000 00:00 0
"""


def test_mapped_cudart_is_found():
    found = cudart.mapped_libraries(MAPS, "libcudart.so")
    assert found == ["/venv/lib/nvidia/cu13/lib/libcudart.so.13"], found
    must_fail(
        lambda: _equal(cudart.mapped_libraries(MAPS, "libcuda.so"), found),
        "another library taken",
    )


def _equal(a, b) -> None:
    assert a == b, (a, b)
