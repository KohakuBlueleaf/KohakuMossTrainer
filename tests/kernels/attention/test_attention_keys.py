"""Attention problem keys: the tile keys of both backends carry the segment-count
bucket, and the implementation choice keys the mean segment length and keys and times
whether a backward runs."""

import pytest
import torch

from kohakumoss_kernels.backends import varlen_choice
from kohakumoss_kernels.tuning import tuner
from kohakumoss_kernels.tuning.buckets import length_bucket, segment_bucket
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
Q = torch.empty(8, 4, 64, dtype=torch.float16, device="meta")


def check_segment_buckets(key) -> None:
    assert key(Q, 2, 100) != key(Q, 8, 100), "2 and 8 segments share a key"
    assert key(Q, 9, 100) == key(Q, 16, 100), "8 and above are one bucket"
    assert key(Q, 3, 100) == key(Q, 4, 100)


@pytest.mark.parametrize("backend", ["triton", "tilelang"])
def test_tile_keys_carry_the_segment_bucket(backend):
    pytest.importorskip(backend)
    module = pytest.importorskip(
        f"kohakumoss_kernels.backends.{backend}.attention.varlen"
    )
    check_segment_buckets(module._key)
    assert module._key(Q, 3, 100).endswith("-L128-S4")


def test_negative_key_without_segments():
    no_segments = lambda q, segs, max_len: f"{q.dtype}-L{max_len}"  # noqa: E731
    must_fail(lambda: check_segment_buckets(no_segments), "segment count dropped")


BACKWARDS: list[int] = []


class _Counted(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q):
        return q.clone()

    @staticmethod
    def backward(ctx, g):
        BACKWARDS.append(1)
        return g


def _kernel(q, k, v, cu, max_len, scale):
    return _Counted.apply(q)


@pytest.fixture
def entry(tmp_path, monkeypatch):
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    monkeypatch.setattr(tuner, "_TABLES", {})
    monkeypatch.setattr(tuner, "TIMINGS", {"events": lambda fn, dev: (fn(), 1.0)[1]})
    tuner.reset()
    tuner.set_mode("table")
    BACKWARDS.clear()
    choice = varlen_choice.VarlenChoice("test", _kernel, lambda q, s, n: f"k-S{s}")
    yield choice
    tuner.reset()


def _inputs(grad: bool):
    q, k, v = (torch.randn(8, 2, 16, requires_grad=grad) for _ in range(3))
    return q, k, v, torch.tensor([0, 4, 8], dtype=torch.int32), 4, 0.25


def check_grad_keys(choice) -> None:
    with torch.no_grad():
        choice(*_inputs(False))
    assert not BACKWARDS, "the inference problem was timed with a backward"
    choice(*_inputs(True))
    assert BACKWARDS, "the training problem was timed without a backward"
    keys = set(tuner.table(choice.table, "cpu").rows)
    assert keys == {"k-S2-M64-g0", "k-S2-M64-g1"}, keys


def test_choice_keys_and_times_the_backward(entry):
    check_grad_keys(entry)


def check_mean_keys(problem) -> None:
    """12 segments of ~3k and one 4096 segment with seven 64-token ones share the
    longest-segment and segment-count buckets, not the mean bucket."""
    packed = torch.empty(36000, 4, 64, dtype=torch.float16, device="meta")
    short = torch.empty(4096 + 7 * 64, 4, 64, dtype=torch.float16, device="meta")
    assert problem(packed, 12, 3400, True) != problem(short, 8, 4096, True)


def _bucketed(q, segs, max_len) -> str:
    return f"L{length_bucket(max_len)}-S{segment_bucket(segs)}"


def test_choice_keys_carry_the_mean_length():
    choice = varlen_choice.VarlenChoice("test", _kernel, _bucketed)
    check_mean_keys(choice.problem)
    no_mean = lambda q, s, n, b: f"{_bucketed(q, s, n)}-g{int(b)}"  # noqa: E731
    must_fail(lambda: check_mean_keys(no_mean), "mean bucket dropped")


def test_negative_backward_always_timed(entry, monkeypatch):
    real = varlen_choice._launch
    monkeypatch.setattr(varlen_choice, "_launch", lambda *a: real(*a[:-1], True))
    must_fail(lambda: check_grad_keys(entry), "forward-only problem timed fwd + bwd")


def test_negative_one_key_for_both(entry, monkeypatch):
    monkeypatch.setattr(varlen_choice, "needs_backward", lambda *ts: True)
    must_fail(lambda: check_grad_keys(entry), "one key for training and inference")
