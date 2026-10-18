"""Encode-ahead for the online text cache: one encoder call per `ahead` batches, the
batcher gets every batch's own encodings, and the checkpoint state resumes exactly."""

from types import SimpleNamespace

import pytest
import torch

from kohakumoss.training.textenc import TextAhead
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


class FakeLoader:
    """Batch i holds prompts ["i:0", "i:1"] (text cached on every 3rd batch); state is
    {seed, start} like kohakumoss_data's Loader."""

    def __init__(self):
        self.start, self.served = 0, 0

    def __iter__(self):
        self.start += self.served
        self.served = 0
        i = self.start
        while True:
            cached = torch.zeros(1, 4) if i % 3 == 2 else None
            self.served += 1
            yield SimpleNamespace(index=i, prompts=[f"{i}:0", f"{i}:1"], text=cached)
            i += 1

    def state_dict(self):
        return {"seed": 0, "start": self.start + self.served}

    def load_state_dict(self, state):
        self.start, self.served = int(state["start"]), 0


class Counter:
    def __init__(self):
        self.calls = 0

    def __call__(self, prompts):
        self.calls += 1
        return [torch.full((1, 4), float(hash(p) % 997)) for p in prompts]


def _run(n: int, take: int):
    enc = Counter()
    ahead = TextAhead(FakeLoader(), enc, n)
    it = iter(ahead)
    got = []
    for _ in range(take):
        hb = next(it)
        parts = None if hb.text is not None else ahead.take(hb)
        got.append((hb.index, parts))
    return ahead, enc, got


def test_one_pass_per_ahead_batches():
    ahead, enc, got = _run(4, 8)

    def check(calls):
        assert calls == 2

    check(enc.calls)
    for index, parts in got:
        if parts is not None:
            want = [float(hash(f"{index}:{j}") % 997) for j in range(2)]
            assert [p[0, 0].item() for p in parts] == want
    must_fail(lambda: check(_run(1, 8)[1].calls), "one encoder call per batch")


def test_state_resumes_at_the_first_unconsumed_batch():
    ahead, _, got = _run(4, 5)
    state = ahead.state_dict()

    def check(start):
        assert start == 5

    check(state["start"])
    resumed = TextAhead(FakeLoader(), Counter(), 4)
    resumed.load_state_dict(state)
    assert next(iter(resumed)).index == 5
    must_fail(lambda: check(ahead.loader.state_dict()["start"]), "loader read-ahead")
