"""HttpShard retries transient read failures (timeout, connection error, 5xx, short
body) and raises at once on a 4xx (model-free: a recorded session, no network)."""

import pytest
import requests

from kohakumoss_data.images.kc import HttpShard
from tests._util.negative import must_fail

pytestmark = [pytest.mark.cpu, pytest.mark.budget(5)]

DATA = bytes(range(16))


class Reply:
    def __init__(self, status: int, body: bytes = b"", length: int = 16):
        self.status_code = status
        self.content = body
        self.headers = {"Content-Length": str(length)}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            err = requests.HTTPError(f"HTTP {self.status_code}")
            err.response = self
            raise err

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


class Session:
    """`get` plays `script` in order (an exception instance is raised)."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def head(self, url, timeout):
        return Reply(200)

    def get(self, url, headers, timeout, stream):
        self.calls += 1
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def _read(script, attempts: int = 5) -> tuple[bytes, int]:
    session = Session(script)
    shard = HttpShard("http://store/shard", session, attempts=attempts, backoff=0.0)
    return shard.read(0, 16), session.calls


def check_recovers(read) -> None:
    script = [
        requests.Timeout("read timed out"),
        requests.ConnectionError("reset"),
        Reply(503),
        Reply(206, DATA[:8]),
        Reply(206, DATA),
    ]
    try:
        data, calls = read(script)
    except OSError as err:
        raise AssertionError(f"the read did not recover: {err}") from err
    assert data == DATA and calls == 5


def test_transient_failures_are_retried():
    check_recovers(_read)
    must_fail(lambda: check_recovers(lambda s: _read(s, attempts=1)), "no retries")


def test_client_error_raises_at_once():
    with pytest.raises(requests.HTTPError):
        _read([Reply(404), Reply(206, DATA)])


def test_retries_exhausted_raise():
    with pytest.raises(OSError, match="3 attempts in"):
        _read([requests.Timeout("t")] * 3, attempts=3)


def test_deadline_bounds_the_wait():
    session = Session([requests.Timeout("t")] * 4 + [Reply(206, DATA)])
    shard = HttpShard("http://s/x", session, deadline=0.0, backoff=1.0)
    with pytest.raises(OSError, match="1 attempts in"):
        shard.read(0, 16)
