"""HostArena, model-free: packed views never overlap, are aligned and hold what was
filled; host copies re-point their tensors so the sources are freed as the arena
fills; `arena_bytes` is what an arena holds; `host_tensors` zeroes or copies each
spec. The CUDA properties (exact size,
asynchronous copies, reuse safety) are in test_arena_cuda.py."""

import pytest
import torch

from kohakumoss.offload.host import arena as ha
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
MIB = 2**20
SIZES = [96 * MIB, 48 * MIB, 3 * MIB + 5, 200 * MIB, 64 * MIB, 17, 300 * MIB]


def assert_packed(pack_fn, sizes, chunk) -> None:
    where, caps = pack_fn(sizes, chunk)
    for cap in caps:
        assert cap & (cap - 1) == 0 and cap >= min(chunk, max(sizes)), cap
    spans: dict[int, list[tuple[int, int]]] = {}
    for (c, start), n in zip(where, sizes):
        assert start % ha.ALIGN == 0 and start + n <= caps[c], (c, start, n)
        spans.setdefault(c, []).append((start, start + n))
    for runs in spans.values():
        runs.sort()
        for (_, end), (start, _) in zip(runs, runs[1:]):
            assert end <= start, runs


def _stacked(sizes, chunk):
    where, caps = ha.pack(sizes, chunk)
    return [(0, 0) for _ in where], caps


def test_pack_is_disjoint_and_aligned():
    assert_packed(ha.pack, SIZES, ha.CHUNK)
    assert_packed(ha.pack, [5, 7, 64, 1], 256)
    must_fail(lambda: assert_packed(_stacked, SIZES, ha.CHUNK), "views overlap")


def test_pageable_arena_holds_the_values():
    src = [torch.randn(5, 7), torch.arange(12, dtype=torch.int8), torch.randn(3)]
    arena = ha.HostArena([(t.shape, t.dtype) for t in src], pinned=False)
    for i, t in enumerate(src):
        arena.fill(i, t)
    assert all(torch.equal(arena.view(i), t) for i, t in enumerate(src))
    assert arena.nbytes == arena.payload == sum(t.nbytes for t in src)


def assert_repointed(copies_fn) -> None:
    ts = [torch.randn(16, 16), torch.randn(4, dtype=torch.float64)]
    want = [t.clone() for t in ts]
    arena, hosts = copies_fn(ts, False, "cpu")
    for t, h, w in zip(ts, hosts, want):
        assert t.data_ptr() == h.data_ptr() and torch.equal(t, w)


def _no_repoint(tensors, pinned, device):
    arena = ha.HostArena([(t.shape, t.dtype) for t in tensors], pinned, device)
    return arena, [arena.fill(i, t) for i, t in enumerate(tensors)]


def test_host_copies_repoint_their_tensors():
    assert_repointed(ha.host_copies)
    must_fail(lambda: assert_repointed(_no_repoint), "sources kept alive")


def test_host_copies_cast_to_the_asked_dtypes():
    ts = [torch.randn(4, 4, dtype=torch.bfloat16), torch.randn(3)]
    want = [t.to(torch.float16) for t in ts]
    _, hosts = ha.host_copies(ts, False, "cpu", [torch.float16, torch.float16])
    assert all(
        h.dtype == torch.float16 and torch.equal(h, w) for h, w in zip(hosts, want)
    )


def assert_arena_bytes(count) -> None:
    sizes = [3 * MIB + 7, 1536 * 4096 * 2, 17]
    for kind in ("pageable", "registered", "chunks"):
        arena = ha.HostArena([((n,), torch.uint8) for n in sizes], False)
        match kind:
            case "registered":
                want = ha.aligned(sum(ha.aligned(n) for n in sizes), ha.PAGE)
            case "chunks":
                arena._chunks()
                want = arena.nbytes
            case _:
                want = arena.nbytes
        assert count(sizes, kind) == want, (kind, count(sizes, kind), want)
    assert count(sizes, "registered") < sum(ha.pow2(n) for n in sizes)


def _rounded(sizes, kind):
    return (
        sum(ha.pow2(n) for n in sizes)
        if kind == "registered"
        else ha.arena_bytes(sizes, kind)
    )


def test_arena_bytes_is_what_an_arena_holds():
    assert_arena_bytes(ha.arena_bytes)
    must_fail(lambda: assert_arena_bytes(_rounded), "per-tensor size classes")


def assert_host_tensors(make) -> None:
    """Zeros where no fill is given, the fill's values cast to the spec dtype else."""
    src = torch.randn(4, 6, dtype=torch.bfloat16)
    specs = [((4, 6), torch.float32), ((5,), torch.float32), ((3, 3), torch.float16)]
    arena, views = make(specs, False, "cpu", [src, None, None])
    assert torch.equal(views[0], src.float())
    assert not views[1].any() and not views[2].any()
    assert [v.dtype for v in views] == [d for _, d in specs]
    assert arena.payload == sum(ha.spec_bytes(s) for s in specs)


def _uninitialised(specs, pinned, device, fill):
    arena = ha.HostArena(specs, pinned, device)
    views = [arena.view(i).fill_(1) for i in range(len(arena))]
    views[0].copy_(fill[0])
    return arena, views


def test_host_tensors_zero_or_copy_each_spec():
    assert_host_tensors(ha.host_tensors)
    must_fail(lambda: assert_host_tensors(_uninitialised), "states left unzeroed")
