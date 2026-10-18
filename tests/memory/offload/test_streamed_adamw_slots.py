"""StreamedAdamW's slot ring, model-free (blocks of one linear each, on the host): the
device bytes it allocates are `slot_bytes` (no master / moment rows with device
states), the training estimate counts exactly that, a slot still holding an unstepped
block is never handed out again, and the host gradient sums are replaced, not zeroed,
between steps."""

import importlib

import pytest
import torch
from torch import nn

from kohakumoss.memory.profiles import resolve
from kohakumoss.model import resolve_config
from kohakumoss.model.linear import MossLinear
from kohakumoss.offload.optim import stream_optim
from kohakumoss.offload.optim.stream_optim import StreamedAdamW, slot_bytes
from kohakumoss.offload.weights.stream import WeightStream
from tests._util.negative import must_fail

E = importlib.import_module("kohakumoss.memory.estimate")
pytestmark = pytest.mark.cpu
OUT, IN = 8, 16
BLOCKS = 4


def _blocks() -> list[nn.Module]:
    out = []
    for _ in range(BLOCKS):
        block = nn.Module()
        block.lin = MossLinear(IN, OUT, bias=False)
        nn.init.normal_(block.lin.weight)
        out.append(block)
    return out


def _optimizer(device_states: bool) -> StreamedAdamW:
    stream = WeightStream(
        _blocks(), "cpu", slots=2, pinned=False, trainable=True, dtype=torch.bfloat16
    )
    return StreamedAdamW(stream, lr=1e-3, device_states=device_states)


def _ring_bytes(opt: StreamedAdamW) -> int:
    tensors = [t for slot in opt.slots for t in slot.values()]
    tensors += list(opt.grad_flat)
    tensors += [t for slot in opt.weight_slots for t in slot.values()]
    return sum(t.numel() * t.element_size() for t in tensors)


def _matches(opt: StreamedAdamW, device_states: bool) -> None:
    assert _ring_bytes(opt) == slot_bytes(OUT * IN, 2, device_states), _ring_bytes(opt)


@pytest.mark.parametrize("device_states", [True, False])
def test_ring_bytes_are_slot_bytes(device_states):
    opt = _optimizer(device_states)
    _matches(opt, device_states)
    assert (opt.slots == []) == device_states
    must_fail(lambda: _matches(opt, not device_states), "the other state placement")


def test_device_state_ring_holds_no_state_rows(monkeypatch):
    real = StreamedAdamW.__init__

    def always_rows(self, stream, *args, **kwargs):
        real(self, stream, *args, **{**kwargs, "device_states": False})
        self.device_states = True

    monkeypatch.setattr(StreamedAdamW, "__init__", always_rows)
    must_fail(
        lambda: _matches(_optimizer(True), True), "rows allocated with device states"
    )


@pytest.mark.parametrize("state", ["gpu", "stream"])
def test_estimate_counts_the_ring(state):
    arch = resolve_config("base")
    overrides = {"stream": "swap", "stream_resident": 2, "optim_state": state}
    plan = resolve("manual", overrides, "cpu", trainable_base=True)
    work = E.Workload(arch, img_tokens=4096, txt_tokens=300, full_ft=True)
    got = E.estimate(work, plan).parts["optim_slots"]
    want = slot_bytes(E.block_params(arch), 2, state == "gpu")
    assert got == want, (got, want)
    old = max(plan.stream_resident, 2) * E.block_params(arch) * 16
    must_fail(lambda: _equal(old if state == "stream" else 0, want), "old slot term")


def _equal(a, b) -> None:
    assert a == b, (a, b)


def _backward_out_of_order(opt: StreamedAdamW) -> None:
    """Every block's backward starts before block 0 got its gradient: the slot ring
    wraps onto block 0's slot."""
    opt.start(True)
    for index in reversed(range(BLOCKS)):
        opt._prefetch(index)


def _raises(fn, words: str = "still held") -> None:
    try:
        fn()
    except RuntimeError as err:
        assert words in str(err), err
        return
    raise AssertionError(f"no RuntimeError ({words})")


def test_held_slot_is_never_reused(monkeypatch):
    _raises(lambda: _backward_out_of_order(_optimizer(False)))
    in_order = _optimizer(False)
    in_order.start(True)
    for index in reversed(range(BLOCKS)):
        in_order._prefetch(index)
        for key in in_order.keys[index]:
            in_order._on_grad(key, torch.ones(OUT, IN, dtype=torch.bfloat16))
    in_order.finish()
    unchecked = _optimizer(False)
    monkeypatch.setattr(unchecked, "_take_slot", _round_robin.__get__(unchecked))
    must_fail(lambda: _raises(lambda: _backward_out_of_order(unchecked)), "no check")


def _round_robin(self, index: int) -> int:
    slot = self.next_slot
    self.next_slot = (slot + 1) % self.n_slots
    self.slot_of[index] = slot
    return slot


def test_unstepped_block_fails_the_step(monkeypatch):
    def unstepped(opt):
        opt.start(True)
        opt._prefetch(BLOCKS - 1)
        opt.finish()

    _raises(lambda: unstepped(_optimizer(False)), "never stepped")
    quiet = _optimizer(False)
    monkeypatch.setattr(quiet, "slot_of", _Empty())
    must_fail(lambda: _raises(lambda: unstepped(quiet), "never stepped"), "unchecked")


class _Empty(dict):
    """A slot map that never reports a held block (the unchecked variant)."""

    def __setitem__(self, key, value) -> None:
        return None


def _sums_after_two_steps(stream: WeightStream) -> dict:
    for step in (1.0, 2.0):
        stream.start(last=False)
        for _ in range(2):
            for key, master in stream.params.items():
                stream.on_grad(key, torch.full(master.shape, step))
        if step == 1.0:
            stream.zero_grads()
            kept = {k: v.clone() for k, v in stream.host_grad.items()}
    return kept, {k: v.clone() for k, v in stream.host_grad.items()}


def _check_sums(stream: WeightStream) -> None:
    kept, sums = _sums_after_two_steps(stream)
    for key, value in sums.items():
        assert torch.equal(value, torch.full_like(value, 4.0)), value
        assert torch.equal(kept[key], torch.full_like(value, 2.0)), "sum was zeroed"


def test_host_sums_replaced_not_zeroed(monkeypatch):
    _check_sums(_optimizer(False).weight_stream)
    adding = _optimizer(False).weight_stream
    sink = adding.grads
    monkeypatch.setattr(sink, "accumulate", _always_add.__get__(sink))
    must_fail(lambda: _check_sums(adding), "a step's first gradient added to old sums")
    assert stream_optim.STATE_SLOTS == 3


def _always_add(sink, key, grad) -> None:
    sink.buffer(key).add_(grad.detach())
    sink.filled[key] = True
