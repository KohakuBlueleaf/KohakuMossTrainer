"""DDP over streamed blocks with one shared host copy of the masters: the block owner
plan of the host optimiser shards, the shared region's view layout, and the owner step
over shared masters against the all-reduce reference (two processes, gloo on CPU)."""

import sys

import pytest
import torch

from kohakumoss.offload.weights.shared import aligned, block_owners, nbytes_of, owned
from kohakumoss.parallel.telemetry import selftest
from kohakumoss.types.plans import ParallelPlan
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
TOL = 1e-6


def check_shards(owners: list[int], n: int, world: int) -> None:
    """Every block has one owner; the ranks' shard sizes differ by at most one."""
    assert len(owners) == n and set(owners) <= set(range(world)), owners
    sizes = [len(owned(owners, r)) for r in range(world)]
    assert sum(sizes) == n and max(sizes) - min(sizes) <= 1, sizes


@pytest.mark.parametrize("n, world", [(42, 2), (42, 4), (5, 3)])
def test_block_owners_balance(n, world):
    check_shards(block_owners(n, world), n, world)
    must_fail(lambda: check_shards([0] * n, n, world), "every block on rank 0")


def test_region_layout_is_aligned():
    assert aligned(1) == 64 and aligned(64) == 64 and aligned(65) == 128
    assert nbytes_of((3, 5), torch.float32) == 60
    assert nbytes_of((3, 5), torch.bfloat16) == 30


def _run(*args):
    if sys.platform == "win32":
        pytest.skip("shared host masters are Linux-only (multi-GPU)")
    plan = ParallelPlan(kind="test", backend="gloo", devices=(0, 1))
    return selftest.spawn_results(selftest.zero_worker, plan, *args, device_type="cpu")


def check_owner_step(results: list[dict]) -> None:
    """The shared masters equal the all-reduce reference on every rank; the two ranks
    own disjoint blocks covering all of them."""
    assert all(r["max_err"] < TOL for r in results), results
    blocks = sorted(b for r in results for b in r["owned"])
    assert blocks == list(range(results[0]["blocks"])), results


@pytest.mark.budget(30)  # spawns its rank processes
def test_owner_step_over_shared_masters():
    check_owner_step(_run())
    broken = _run(2, True)
    must_fail(lambda: check_owner_step(broken), "owners stepping unreduced gradients")
