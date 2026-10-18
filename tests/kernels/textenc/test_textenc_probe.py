"""The text-encoder op probe: an out-of-memory retries once after the device's free
blocks go back to the driver, so a transient OOM does not bind the eager reference for
the run; a second OOM still propagates (the eager fallback of `TextOps`)."""

import pytest
import torch

from kohakumoss.encoders.text import forward
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
CPU = torch.device("cpu")


def _flaky(failures: int):
    """An op that runs out of memory on its first `failures` calls."""
    calls = []

    def fn(*args):
        calls.append(1)
        if len(calls) <= failures:
            raise torch.OutOfMemoryError("CUDA out of memory (test)")
        return args[0]

    return fn, calls


def check_retry(probe, monkeypatch) -> None:
    released = []
    monkeypatch.setattr(forward, "release", released.append)
    fn, calls = _flaky(1)
    try:
        probe(fn, "rms_norm", CPU, torch.float32, {})
    except torch.OutOfMemoryError as err:
        raise AssertionError("a single out-of-memory escaped the probe") from err
    assert len(calls) == 2 and released == [CPU], (calls, released)
    fn, calls = _flaky(2)
    with pytest.raises(torch.OutOfMemoryError):
        probe(fn, "rms_norm", CPU, torch.float32, {})
    assert len(calls) == 2, "retried more than once"


def _no_retry(fn, name, device, dtype, opts):
    """The broken variant: one attempt, an OOM binds eager."""
    with torch.no_grad():
        fn(*forward.probe_args(name, device, dtype, opts))


def test_probe_retries_an_oom_once(monkeypatch):
    check_retry(forward.probe, monkeypatch)


def test_negative_probe_without_retry(monkeypatch):
    must_fail(lambda: check_retry(_no_retry, monkeypatch), "no retry")
