"""Memory placement on the real backbone and two encoded dataset samples (the card's
16-bit compute dtype: bf16 where the card has it, else fp16; a forward + backward per
step through the plan the trainer installs). Activation
offload: every packed activation comes back in its step's backward, every host buffer
returns to the slab after the step, and later steps allocate none. Ring weight
streaming: a streamed forward + backward takes no host sync beyond the resident one's,
though every block is copied in."""

import warnings

import pytest
import torch

from kohakumoss.memory.apply import apply_to_backbone
from kohakumoss.memory.runners import build_runner
from kohakumoss.model import load_backbone
from kohakumoss.offload.activation import ActivationOffload
from kohakumoss.types.plans import MemoryPlan
from kohakumoss_kernels.hardware.arch import arch_of
from tests._util.negative import must_fail

pytestmark = [pytest.mark.real_weights, pytest.mark.cuda, pytest.mark.slow]
STEPS = 3


@pytest.fixture(scope="module")
def dtype():
    """The card's 16-bit compute dtype: bf16 where it has bf16, else fp16."""
    return torch.bfloat16 if arch_of(torch.device("cuda")).has_bf16 else torch.float16


@pytest.fixture(scope="module")
def model(weights_root, dtype):
    m, report = load_backbone(weights_root / "base", device="cuda", dtype=dtype)
    assert report.ok
    return m.requires_grad_(False)


@pytest.fixture(scope="module")
def batch(real_batch, dtype):
    return real_batch.on("cuda", dtype)


def fwd_bwd(model, batch) -> None:
    """One training-style step on a frozen backbone (autocast to the batch's dtype):
    the loss gradient reaches the latents through every block."""
    img, text, sigma, seq = batch
    x = img.detach().requires_grad_(True)
    with torch.autocast("cuda", dtype=x.dtype):
        out = model(x, text, sigma, seq)
    out.float().square().mean().backward()


def _plain(model) -> None:
    model.set_runner(build_runner("none", model.layers))


def offload_step(model, batch, engines) -> dict:
    """Run a step and the trainer's after-step hook; what the offload holds."""
    fwd_bwd(model, batch)
    torch.cuda.synchronize()
    off = engines.offload
    pending = sum(len(p) for p in off.pending.values())
    engines.after_step()
    return {
        "packed": off.stats["packed"],
        "pending": pending,
        "out": off.slab.out,
        "grown": off.slab.grown,
        "chunks": len(off.slab.chunks),
    }


def check_released(reports: list[dict]) -> None:
    """Each step packed activations and got all of them back in its backward (no
    host buffer out after it); from the second step on the slab is one chunk and
    allocates nothing."""
    assert reports[0]["packed"] > 0, "nothing was offloaded"
    for i, r in enumerate(reports, start=1):
        assert r["pending"] == 0 and r["out"] == 0, (i, r)
    for r in reports[2:]:
        assert r["grown"] == reports[1]["grown"] and r["chunks"] == 1, reports


@pytest.mark.parametrize("checkpoint", ["none", "block"])
def test_activation_offload_pools_released(model, batch, dtype, checkpoint):
    plan = MemoryPlan(checkpoint=checkpoint, act_offload=True, stream="off")
    engines = apply_to_backbone(model, plan, "cuda", dtype, training=True)
    model.bind_ops()
    try:
        reports = [offload_step(model, batch, engines) for _ in range(STEPS)]
        check_released(reports)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(ActivationOffload, "_give", lambda self, flat, done: None)
            leaked = reports + [offload_step(model, batch, engines)]
        must_fail(lambda: check_released(leaked), "buffers never returned to the slab")
    finally:
        _plain(model)


class SyncCount:
    """Counts host syncs: explicit synchronize calls on the CUDA module, events and
    streams, and synchronizing CUDA ops (the sync debug mode's warnings)."""

    OWNERS = (torch.cuda, torch.cuda.Event, torch.cuda.Stream)

    def __init__(self, mp: pytest.MonkeyPatch):
        self.armed = False
        self.calls: list[str] = []
        for owner in self.OWNERS:
            self._wrap(mp, owner, "synchronize")

    def _wrap(self, mp, owner, name: str) -> None:
        original = getattr(owner, name)
        label = f"{getattr(owner, '__name__', owner)}.{name}"

        def counted(*args, **kwargs):
            if self.armed:
                self.calls.append(label)
            return original(*args, **kwargs)

        mp.setattr(owner, name, counted)

    def __call__(self, run) -> int:
        """Host syncs `run()` takes."""
        torch.cuda.synchronize()
        start = len(self.calls)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            self.armed = True
            try:
                run()
            finally:
                self.armed = False
                torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
        implicit = sum("synchroniz" in str(w.message) for w in caught)
        return len(self.calls) - start + implicit


def stream_step(model, batch, engine, count: SyncCount) -> dict:
    """Host syncs and engine counters of one warm streamed step."""
    fwd_bwd(model, batch)
    loads, syncs = engine.stats["loads"], engine.stats["host_syncs"]
    host = count(lambda: fwd_bwd(model, batch))
    return {
        "host_syncs": host,
        "loads": engine.stats["loads"] - loads,
        "engine_syncs": engine.stats["host_syncs"] - syncs,
    }


def check_no_block_syncs(streamed: dict, resident: int, depth: int) -> None:
    """Every block was copied in, and the streamed step took no host sync the resident
    step does not take."""
    assert streamed["loads"] >= depth, streamed
    assert streamed["engine_syncs"] == 0, streamed
    assert streamed["host_syncs"] <= resident, (streamed, resident)


def test_ring_stream_takes_no_host_sync_per_block(model, batch, dtype, monkeypatch):
    count = SyncCount(monkeypatch)
    _plain(model)
    fwd_bwd(model, batch)
    resident = count(lambda: fwd_bwd(model, batch))
    plan = MemoryPlan(checkpoint="none", stream="ring", stream_resident=2)
    engine = apply_to_backbone(model, plan, "cuda", dtype, training=True).weight_stream
    model.bind_ops()
    depth = model.cfg.depth
    check_no_block_syncs(stream_step(model, batch, engine, count), resident, depth)
    engine.copier.copy = engine.copier.copy_staged
    staged = stream_step(model, batch, engine, count)
    must_fail(
        lambda: check_no_block_syncs(staged, resident, depth),
        "a ring that waits on the host for every block copy",
    )
