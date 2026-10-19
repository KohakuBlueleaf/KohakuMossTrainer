"""Weak hooks and callbacks fail loudly, model-free (small real parameters on the host):
each kind fires while its owner lives (the GradOffload and GradReducer post-accumulate
hooks, the AsyncReducer hook body, the bundle's per-parameter step a FusedStep calls, a
bare `weak_call`), and a call after the owner was dropped raises ReferenceError instead
of doing nothing. A hook that silently skips once its owner is gone fails the check. A
GradReducer (one gloo rank) is freed by reference counting alone; a strong hook put back
keeps it."""

from functools import partial

import pytest
import torch
import torch.distributed as dist

from kohakumoss.offload.optim.grad import GradOffload
from kohakumoss.parallel.ddp import AsyncReducer, GradReducer
from kohakumoss.types.plans import MemoryPlan
from kohakumoss_common import weak
from tests._util.negative import must_fail
from tests._util.release import check_released
from tests.memory.offload.test_optim_release import (
    _backward,
    _params,
    _train_step,
    _wired,
)

pytestmark = pytest.mark.cpu


@pytest.fixture
def group(tmp_path):
    store = dist.FileStore(str(tmp_path / "store"), 1)
    dist.init_process_group("gloo", store=store, rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def check_loud(call) -> None:
    """`call()` (a weak hook whose owner is gone) raises ReferenceError."""
    with pytest.raises(ReferenceError):
        call()


def _silent(fn, ref, args, *more, **kwargs):
    owner = ref()
    return None if owner is None else fn(owner, *args, *more, **kwargs)


def test_weak_call_fires_then_raises(monkeypatch):
    class Owner:
        def __init__(self):
            self.seen = []

        def note(self, x):
            self.seen.append(x)

    owner = Owner()
    call = weak.weak_method(owner.note)
    call(1)
    assert owner.seen == [1]
    del owner
    check_loud(lambda: call(2))
    monkeypatch.setattr(weak, "_call", _silent)
    quiet = Owner()
    hook = weak.weak_method(quiet.note)
    del quiet
    must_fail(lambda: check_loud(lambda: hook(3)), "a hook that skips silently")


def test_grad_offload_hook_fires_then_raises():
    params = _params()
    off = GradOffload(params, pinned=False)
    off.start(False)
    sum((p * 2.0).sum() for p in params).backward()
    off.drain()
    assert off.lanes[id(params[0])].filled
    del off
    for p in params:
        p.grad = None
    check_loud(lambda: sum((p * 2.0).sum() for p in params).backward())


def _reduced(params) -> GradReducer:
    """A GradReducer over `params` after one reducing backward."""
    reducer = GradReducer(params)
    reducer.start(True)
    _backward(params)
    assert reducer.next == len(reducer.buckets), "the weak hook did not reach its owner"
    reducer.wait()
    return reducer


def test_grad_reducer_hook_fires_then_raises(group):
    params = _params()
    reducer = _reduced(params)
    assert torch.equal(params[0].grad, torch.full((3, 4), 2.0))
    del reducer
    check_loud(lambda: _backward(params))


def _reducer(back=None):
    """`build()` of a reduced GradReducer; `back(reducer)` returns a hook that is put
    on a parameter."""

    def build() -> tuple:
        params = _params()
        reducer = _reduced(params)
        if back is not None:
            params[0].register_post_accumulate_grad_hook(back(reducer))
        return reducer, params[0], reducer.buckets[0].flat

    return build


def _holds_bucket(bucket, p) -> None:
    return None


def test_grad_reducer_is_released(group):
    check_released(_reducer())
    strong = _reducer(lambda r: partial(GradReducer._hook, r, 0))
    must_fail(lambda: check_released(strong), "a strong reducer hook")
    bucket = _reducer(lambda r: partial(_holds_bucket, r.buckets[0]))
    must_fail(lambda: check_released(bucket), "a hook holding its bucket")


def test_async_reducer_body_fires_then_raises():
    params = _params()
    reducer = AsyncReducer("cpu")
    body = reducer.reduced(params, lambda p: None)
    params[0].grad = torch.ones_like(params[0])
    body(params[0])
    assert id(params[0]) in reducer.slab_of[id(params[0])].arrived
    del reducer
    check_loud(lambda: body(params[1]))


def test_bundle_step_fires_then_raises():
    lin, engines, bundle, accumulate = _wired(MemoryPlan(fused_step=True))
    _train_step(lin, bundle, accumulate)
    step = engines.fused.step_param
    del bundle, accumulate
    lin.weight.grad = torch.ones_like(lin.weight)
    check_loud(lambda: step(lin.weight))
