"""Split-backward pipeline support (no model): `retained_backward` turns donated
buffers off as the config default, seen by every thread, for the split-backward
schedules only; `_owned` copies a stage output only when it is a view, so the
schedule's in-place detach works and owned outputs are not copied."""

import threading
from collections.abc import Callable
from functools import partial

import pytest
import torch
import torch._functorch.config as functorch_config

from kohakumoss.parallel.pipeline.distributed import (
    SCHEDULES,
    SPLIT_BACKWARD,
    _owned,
    _same,
    retained_backward,
)
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
ENTRY = functorch_config._config["donated_buffer"]


@pytest.fixture
def donated_default():
    """Restore the donated_buffer config default after the test."""
    before = ENTRY.default
    yield
    ENTRY.default = before


def _in_thread(fn: Callable[[], object]) -> object:
    out = []
    t = threading.Thread(target=lambda: out.append(fn()))
    t.start()
    t.join()
    return out[0]


def _read() -> bool:
    return functorch_config.donated_buffer


def _thread_override() -> None:
    functorch_config.donated_buffer = False


def _every_schedule(_: str) -> None:
    ENTRY.default = False


def check_retained(setter: Callable[[str], None]) -> None:
    """For every schedule: `setter(schedule)` on one thread, then a second thread
    reads donated_buffer False exactly for the split-backward schedules."""
    for schedule in SCHEDULES:
        ENTRY.default = True
        _in_thread(partial(setter, schedule))
        want = schedule not in SPLIT_BACKWARD
        assert _in_thread(_read) is want, (schedule, want)


def test_retained_backward_is_the_config_default(donated_default):
    assert set(SPLIT_BACKWARD) == {"interleaved-zb", "zbv", "dualpipe-v"}
    check_retained(retained_backward)

    def per_thread(schedule: str) -> None:
        if schedule in SPLIT_BACKWARD:
            _thread_override()

    must_fail(lambda: check_retained(per_thread), "a per-thread config override")
    must_fail(lambda: check_retained(_every_schedule), "set for every schedule")


def check_owned(own: Callable[[torch.Tensor], torch.Tensor]) -> None:
    """A view output comes back detachable in place with its values; an output that
    owns its storage comes back as the same tensor."""
    base = torch.randn(4, 6).requires_grad_() * 2
    view = base[:, :3]
    out = own(view)
    assert out._base is None, "view passed through"
    assert torch.equal(out, view)
    out.detach_()
    full = torch.randn(4, 3).requires_grad_() * 2
    assert own(full) is full, "owned output copied"


def _always_clone(t: torch.Tensor) -> torch.Tensor:
    return t.clone()


def test_owned_copies_only_views():
    check_owned(_owned)
    must_fail(lambda: check_owned(_same), "views passed through")
    must_fail(lambda: check_owned(_always_clone), "every output copied")
    with pytest.raises(RuntimeError):
        (torch.randn(4, 6).requires_grad_() * 2)[:, :3].detach_()
