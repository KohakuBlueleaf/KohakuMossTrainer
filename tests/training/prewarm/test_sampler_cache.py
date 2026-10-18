"""The step-1 stack sampler matches a frame against CATEGORIES once per (module,
function) and looks it up after, with the same categories as `classify`
(model-free: stand-in frames)."""

import pytest

from kohakumoss.training.callbacks import sampler
from kohakumoss.training.callbacks.sampler import StackSampler, classify
from tests._util.negative import must_fail
from tests.training.prewarm.test_step_one_breakdown import STACKS, _chain

pytestmark = pytest.mark.cpu


class Uncached(StackSampler):
    """The broken variant: every frame matched against CATEGORIES on every sample."""

    def _rank(self, frame):
        return sampler.rank_of(sampler._module_of(frame), frame.f_code.co_name)


def check_cached(kind, monkeypatch_ctx) -> None:
    s = kind()
    calls = []
    real = sampler.rank_of
    assert all(classify(stack) == want for want, stack in STACKS.items())

    def spy(module, func):
        calls.append((module, func))
        return real(module, func)

    with monkeypatch_ctx() as m:
        m.setattr(sampler, "rank_of", spy)
        for _ in range(3):
            for want, stack in STACKS.items():
                assert s.category(_chain(stack)) == want
    distinct = {frame for stack in STACKS.values() for frame in stack}
    assert len(calls) <= len(distinct), (len(calls), len(distinct))


def test_frames_matched_once(monkeypatch):
    check_cached(StackSampler, monkeypatch.context)
    must_fail(lambda: check_cached(Uncached, monkeypatch.context), "every sample")
