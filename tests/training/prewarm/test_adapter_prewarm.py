"""Adapter autotuned-op prewarm: the token buckets of the planned batches, every
distinct op settled per bucket and grad pattern, one problem's scratch inputs alive at
a time, its device peak in the step-1 report, no gradient left behind (one linear, no
model); per-group settings of the offload AdamWs."""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from lycoris.kernels.plans.tune import tuning_enabled
from lycoris.modules import LoConModule
from torch import nn

from kohakumoss.adapters import AdapterConfig, apply
from kohakumoss.adapters.bind import autotuned, bodies, ops
from kohakumoss.adapters.bind.autotuned import TOKEN_FLOOR, token_buckets
from kohakumoss.adapters.lycoris.tuning import (
    restore_lycoris_tuning,
    set_lycoris_tuning,
)
from kohakumoss.model.linear import MossLinear
from kohakumoss.precision import build_precision
from kohakumoss.training.build.kernels import build_adapter_prewarm
from kohakumoss.training.optim import build_bundle
from kohakumoss.tuning import prewarm
from kohakumoss.tuning.prewarm import AdapterPrewarm
from kohakumoss_kernels.hardware import memory
from kohakumoss_kernels.tuning import tuner, window
from kohakumoss_kernels.tuning.buckets import pow2
from kohakumoss_kernels.tuning.manifest import Manifest
from kohakumoss_kernels.tuning.replay import OpReport
from tests._util.negative import must_fail
from tests.kernels.tuning.test_tuner_working import GIB, Allocator
from tests.training.knobs.test_knobs import runnable

DEV = torch.device("cpu")


def _covers(buckets, batch_max: int) -> None:
    """Every row count 1..batch_max falls in the token bucket of exactly one listed
    value, and no listed value exceeds batch_max (the largest planned batch)."""
    keys = [pow2(b, TOKEN_FLOOR) for b in buckets]
    assert len(keys) == len(set(keys)), keys
    assert max(buckets) <= batch_max, buckets
    want = {pow2(n, TOKEN_FLOOR) for n in range(1, batch_max + 1)}
    assert want <= set(keys), sorted(want - set(keys))


@pytest.mark.cpu
@pytest.mark.parametrize("batch_max", [1, 7, 1000, 4096, 9000, 16384])
def test_token_buckets_cover_every_count(batch_max):
    buckets = token_buckets(batch_max)
    _covers(buckets, batch_max)
    if len(buckets) > 1:
        must_fail(lambda: _covers(buckets[:-1], batch_max), "top bucket missing")
    must_fail(lambda: _covers([b + 1 for b in buckets], batch_max), "off-bucket rows")
    if pow2(batch_max, TOKEN_FLOOR) != batch_max:
        tops = [pow2(b, TOKEN_FLOOR) for b in buckets]
        must_fail(lambda: _covers(tops, batch_max), "bucket tops above the plan")


class _Spy:
    def __init__(self):
        self.calls = []
        self.prewarm_stats = {}

    def prewarm(self, rows, dtype, devices, budgets=None):
        self.calls.append((list(rows), dtype, list(devices)))
        return len(rows)


@pytest.mark.cpu
def test_prewarm_hands_the_planned_buckets_to_the_trainable_set():
    k = runnable(MAX_TOKENS=9000, TUNE="table")
    spy = _Spy()
    warm = build_adapter_prewarm(k, SimpleNamespace(trainable=spy), torch.float16)
    assert isinstance(warm, AdapterPrewarm)
    warm([DEV])
    ((rows, dtype, devices),) = spy.calls
    _covers(rows, 9000)
    assert dtype is torch.float16 and devices == [DEV]
    off = build_adapter_prewarm(
        {**k, "TUNE": "off"}, SimpleNamespace(trainable=spy), torch.float16
    )
    assert not isinstance(off, AdapterPrewarm)
    must_fail(lambda: _covers(rows[:3], 9000), "buckets truncated")


@pytest.fixture
def timing(tmp_path, monkeypatch):
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    monkeypatch.setattr(tuner, "_TABLES", {})

    def unit(fn, dev):
        fn()
        return 1.0

    monkeypatch.setattr(tuner, "TIMINGS", {"events": unit, "burst": unit})
    tuner.reset()
    tuner.set_mode("table")
    yield
    tuner.reset()


def _with_base_op() -> autotuned.AutotunedOp:
    """A LoRA bypass-with-base op on the host, both candidates the eager body."""
    lyco = LoConModule("l", nn.Linear(64, 48), 1.0, 4, 4.0)
    op = autotuned.AutotunedOp(ops.lora_bypass_with_base(lyco, 0.5), "triton", "tokens")
    op.candidates = {name: _body for name in op.candidates}
    return op


def _body(operands):
    return bodies.lora_with_base(operands, 0.5)


def _hoarding(runtime):
    """The broken variant: every bucket's inputs built up front and kept."""
    problems = runtime.problems

    def hoard(rows):
        kept = [(build(torch.float32, DEV), grad) for build, grad in problems(rows)]
        for t, grad in kept:
            yield (lambda dtype, device, t=t: t), grad

    runtime.problems = hoard


def check_one_problem_alive(monkeypatch, broken=None) -> None:
    """While a problem is timed only its own two scratch inputs are alive, and each
    measured problem returns the free blocks to the driver."""
    op = _with_base_op()
    if broken is not None:
        broken(op.op.runtime)
    live: list = []
    alive: list[int] = []
    released: list = []
    randn, select = autotuned._randn, autotuned.select

    def tracked(*args, **kwargs):
        t = randn(*args, **kwargs)
        live.append(weakref.ref(t))
        return t

    def counting(*args, **kwargs):
        gc.collect()
        alive.append(sum(r() is not None for r in live))
        return select(*args, **kwargs)

    monkeypatch.setattr(autotuned, "_randn", tracked)
    monkeypatch.setattr(autotuned, "select", counting)
    monkeypatch.setattr(autotuned, "fusable", lambda operands: True)
    monkeypatch.setattr(autotuned, "release", released.append)
    measured = op.warm([256, 512, 1024], torch.float32, DEV)
    assert measured == 3 * len(autotuned.TOKEN_PATTERNS), measured
    assert alive and max(alive) <= 2, alive
    assert len(released) == measured, released


@pytest.mark.cpu
@pytest.mark.budget(20)
def test_prewarm_keeps_one_problem_alive(timing, monkeypatch):
    monkeypatch.setattr(tuner.TuneTable, "get", _nothing_stored)
    check_one_problem_alive(monkeypatch)
    tuner.reset()
    must_fail(
        lambda: check_one_problem_alive(monkeypatch, _hoarding),
        "every bucket's inputs alive together",
    )


MIB = 1 << 20
ROWS = [256, 1024, 4096, 16384]
BUDGET = 2000 * MIB
PER_ROW = {
    "kernel": 0.4,
    "compile": 0.15,
}  # MiB of working memory per row; 16384 > 2000
SPEED = {"kernel": 0.5, "compile": 1.0}  # ms: the kernel wins wherever it is timed


def _using(fake: Allocator, impl: str, seen: list, clock: dict):
    """Candidate `impl` on the fake device: working memory linear in the rows."""

    def run(operands):
        rows = operands[0].shape[0]
        fake.use(int(PER_ROW[impl] * rows * MIB))
        seen.append(rows)
        clock["ms"] = SPEED[impl]
        return _body(operands)

    return run


def check_inherits(fake, monkeypatch, budget, predict: bool = True) -> None:
    """Under the budget every bucket up to 4096 rows is timed (the kernel wins); the
    16384-row bucket would pass it, so it is never timed and inherits the 4096 decision
    (`predict` off: the cap stops its timing before it allocates)."""
    tuner.reset()
    seen, clock = [], {"ms": 1.0}
    op = _with_base_op()
    op.candidates = {impl: _using(fake, impl, seen, clock) for impl in op.candidates}

    def clocked(fn, d):
        fn()
        return clock["ms"]

    monkeypatch.setattr(tuner, "TIMINGS", {"events": clocked, "burst": clocked})
    if not predict:
        monkeypatch.setattr(autotuned.AutotunedOp, "_over", lambda *a: False)
    fake.top = 0
    op.warm(ROWS, torch.float32, DEV, budget)
    assert fake.top <= BUDGET, fake.top / MIB
    assert 16384 not in seen, "the over-budget bucket was timed"
    tab = tuner.table(op.name, torch.device("cuda", 0))
    big = [k for k in tab.rows if k.startswith("16384x")]
    assert len(big) == len(autotuned.TOKEN_PATTERNS), list(tab.rows)
    for key in big:
        assert tab.rows[key]["inherited"] == key.replace("16384x", "4096x"), key
        assert tab.rows[key]["config"] == {"impl": "kernel"}, tab.rows[key]
    assert fake.fraction == 1.0, fake.fraction


@pytest.mark.cpu
def test_over_budget_buckets_inherit(timing, monkeypatch, tmp_path):
    fake = Allocator()
    monkeypatch.setattr(memory, "MODULES", {"cuda": fake})
    monkeypatch.setattr(memory, "_OPEN", {})
    tables = {}

    def table(kernel, device):
        return tables.setdefault(kernel, tuner.TuneTable(tmp_path / f"{kernel}.json"))

    monkeypatch.setattr(tuner, "table", table)
    monkeypatch.setattr(tuner.TuneTable, "get", _nothing_stored)
    monkeypatch.setattr(autotuned, "fusable", lambda operands: True)
    monkeypatch.setattr(autotuned, "_device", lambda operands: torch.device("cuda", 0))
    check_inherits(fake, monkeypatch, BUDGET)
    tables.clear()
    must_fail(
        lambda: check_inherits(fake, monkeypatch, None), "the over-budget bucket timed"
    )
    tables.clear()
    check_inherits(fake, monkeypatch, BUDGET, predict=False)


def _budget(device, reports: dict, manifest: Manifest, key: str) -> int:
    """The adapter tuning budget after a kernel prewarm that left `reports`."""
    prewarm.reset_peaks()
    prewarm.note_prewarm(device, reports, manifest, key, memory.allocated(device))
    peak = prewarm.PREWARM_PEAK.get(str(device), (0, 0))[0]
    return memory.tuning_budget(device, peak)


def check_cached_budget(fake, device, manifest: Manifest) -> None:
    """A fully cached kernel prewarm (its ops' peaks near the baseline) gives the same
    budget as the run that measured it; a key never measured gives the free-memory
    term alone."""
    base = fake.alloc
    peak = base + 3 * GIB
    ran = {"op": OpReport(problems=3, peak_alloc=peak, peak_reserved=peak)}
    cached = {"op": OpReport(cached=3, peak_alloc=base + MIB, peak_reserved=base)}
    uncached = _budget(device, ran, manifest, "prewarm_working|a")
    assert uncached == 3 * GIB, uncached / GIB
    again = _budget(device, cached, manifest, "prewarm_working|a")
    assert again == uncached, (again / GIB, uncached / GIB)
    free = int(memory.BUDGET_FRACTION * fake.mem_get_info(device)[0])
    assert _budget(device, cached, manifest, "prewarm_working|old") == free


@pytest.mark.cpu
def test_cached_prewarm_keeps_the_budget(monkeypatch, tmp_path):
    fake = Allocator()
    monkeypatch.setattr(memory, "MODULES", {"cuda": fake})
    monkeypatch.setattr(memory, "_OPEN", {})
    tables = {}

    def table(kernel, device):
        return tables.setdefault(kernel, tuner.TuneTable(tmp_path / f"{kernel}.json"))

    monkeypatch.setattr(tuner, "table", table)
    device = torch.device("cuda", 0)
    check_cached_budget(fake, device, Manifest(device))
    tables.clear()
    monkeypatch.setattr(Manifest, "working", lambda self, key: None)
    must_fail(
        lambda: check_cached_budget(fake, device, Manifest(device)),
        "the stored working bytes ignored",
    )


def check_ranks_share_the_budget(fake, device, manifest: Manifest, store) -> None:
    """Rank 1 (no kernel prewarm of its own) gets rank 0's adapter tuning budget
    through the prewarm gate's payload."""
    peak = fake.alloc + 3 * GIB
    ran = {"op": OpReport(problems=3, peak_alloc=peak, peak_reserved=peak)}
    prewarm.reset_peaks()
    working = prewarm.note_prewarm(
        device, ran, manifest, "prewarm_working|r", memory.allocated(device)
    )
    rank0 = memory.tuning_budget(device, prewarm.PREWARM_PEAK[str(device)][0])
    prewarm.share_working("prewarm-r", [working], store)
    prewarm.reset_peaks()
    prewarm.take_working("prewarm-r", [device], store)
    rank1 = memory.tuning_budget(
        device, prewarm.PREWARM_PEAK.get(str(device), (0, 0))[0]
    )
    assert rank1 == rank0 == 3 * GIB, (rank1 / GIB, rank0 / GIB)


class _Store:
    """An in-process key-value store with the c10d store calls the gate uses."""

    def __init__(self):
        self.rows: dict[str, bytes] = {}

    def set(self, key, value):
        self.rows[key] = value.encode() if isinstance(value, str) else value

    def get(self, key):
        return self.rows[key]

    def wait(self, keys, timeout=None):
        missing = [k for k in keys if k not in self.rows]
        assert not missing, f"waited on unset keys {missing}"


def _no_payload(gate, workings, store=None):
    """The broken variant: rank 0 releases the gate without its working bytes."""
    prewarm.release_ranks(gate, store)


@pytest.mark.cpu
def test_ranks_share_the_budget(monkeypatch, tmp_path):
    fake = Allocator()
    monkeypatch.setattr(memory, "MODULES", {"cuda": fake})
    monkeypatch.setattr(memory, "_OPEN", {})
    tables = {}

    def table(kernel, device):
        return tables.setdefault(kernel, tuner.TuneTable(tmp_path / f"{kernel}.json"))

    monkeypatch.setattr(tuner, "table", table)
    device = torch.device("cuda", 0)
    check_ranks_share_the_budget(fake, device, Manifest(device), _Store())
    tables.clear()
    monkeypatch.setattr(prewarm, "share_working", _no_payload)
    must_fail(
        lambda: check_ranks_share_the_budget(fake, device, Manifest(device), _Store()),
        "rank 1 left on the free-memory term",
    )


def check_capped_after_prewarm(go) -> None:
    """The end of the adapter prewarm freezes the kernel configs, caps every later tune
    (inside the tuning windows) and stops LyCORIS's tile tuning (no LyCORIS tune inside
    a step)."""
    device = torch.device("cuda", 0)
    previous = set_lycoris_tuning(True)
    try:
        go({device: 3 * GIB})
        assert tuner._CAPS.get(str(device)) == 3 * GIB, tuner._CAPS
        assert window.frozen(), "kernel configs still tuned in the steps"
        assert not tuning_enabled(), "LyCORIS still tunes in the steps"
    finally:
        restore_lycoris_tuning(previous)
        tuner.set_cap(device, None)


def _cap_only(budgets):
    """The broken variant: the cap set, LyCORIS left tuning."""
    for device, nbytes in budgets.items():
        tuner.set_cap(device, nbytes)


@pytest.mark.cpu
def test_prewarm_end_caps_every_later_tune():
    check_capped_after_prewarm(prewarm.end_prewarm)
    must_fail(lambda: check_capped_after_prewarm(_cap_only), "LyCORIS left tuning")


class _Heavy:
    """A trainable set whose prewarm takes `nbytes` on the fake device."""

    def __init__(self, fake: Allocator, nbytes: int):
        self.fake, self.nbytes = fake, nbytes
        self.prewarm_stats = {}

    def prewarm(self, rows, dtype, devices, budgets=None):
        self.fake.use(self.nbytes)
        return 1


def check_reported(fake, device) -> None:
    """The step-1 prewarm peak holds the adapter prewarm's device peak."""
    prewarm.reset_peaks()
    AdapterPrewarm(_Heavy(fake, 5 * GIB), torch.float16, [256])([device])
    alloc, _ = prewarm.PREWARM_PEAK.get(str(device), (0, 0))
    assert alloc >= 7 * GIB, alloc / GIB


@pytest.mark.cpu
def test_adapter_prewarm_peak_is_reported(monkeypatch):
    fake = Allocator()
    monkeypatch.setattr(memory, "MODULES", {"cuda": fake})
    monkeypatch.setattr(memory, "_OPEN", {})
    device = torch.device("cuda", 0)
    check_reported(fake, device)
    monkeypatch.setattr(prewarm, "note_peak", lambda device, reports: None)
    must_fail(lambda: check_reported(fake, device), "adapter prewarm left out")


def _one_linear(device) -> nn.Module:
    """One linear at a MOSS module path (a container, no forward)."""
    block = nn.Module()
    block.proj = nn.ModuleDict({"q": MossLinear(192, 128, bias=True)})
    root = nn.Module()
    root.layers = nn.ModuleList([block])
    return root.to(device, torch.float16).requires_grad_(False)


def check_untouched(aset, before: dict) -> None:
    """Every factor keeps its value and the gradient it held before the timing."""
    for name, p in aset.named_trainable():
        value, grad = before[name]
        assert torch.equal(p, value), name
        assert p.grad is not None and torch.equal(p.grad, grad), name


def _nothing_stored(self, key):
    return None


def _fresh(device):
    """A fresh LoRA set, every factor holding a gradient, and that state."""
    tuner.reset()
    config = AdapterConfig(algo="lora", preset="attn", dim=4, alpha=4.0)
    aset = apply(_one_linear(device), config, dtype=torch.float16, device=device)
    for _, p in aset.named_trainable():
        p.grad = torch.full_like(p, 0.25)
    before = {
        n: (p.detach().clone(), p.grad.clone()) for n, p in aset.named_trainable()
    }
    return aset, before


def _warmed(device):
    """Every autotuned op of a fresh set warmed, every problem timed (no stored
    winner); the problems measured."""
    aset, before = _fresh(device)
    measured = sum(
        op.warm([256, 512], torch.float16, device)
        for t in aset.targets
        for op in t.autotuned
    )
    return aset, before, measured


def _attached(a, grad: bool):
    """The broken leaf: the live operand itself, still on the factors' graph."""
    return a


@pytest.mark.cuda
@pytest.mark.budget(35)
def test_prewarm_leaves_factors_untouched(monkeypatch, tmp_path):
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    monkeypatch.setattr(tuner.TuneTable, "get", _nothing_stored)
    tuner.set_mode("table")
    device = torch.device("cuda", 0)
    aset, before, measured = _warmed(device)
    assert measured > 0
    check_untouched(aset, before)
    warm, _ = _fresh(device)
    AdapterPrewarm(warm, torch.float16, [256, 512])([device])
    assert warm.prewarm_stats["measured"] > 0, warm.prewarm_stats
    assert all(p.grad is None for _, p in warm.named_trainable())
    monkeypatch.setattr(autotuned, "_leaf", _attached)
    broken, before, _ = _warmed(device)
    must_fail(lambda: check_untouched(broken, before), "timed on the live factors")
    tuner.reset()


def _groups(params):
    a, b, c = params
    return [
        {"params": [a], "lr": 1e-2, "weight_decay": 0.1},
        {"params": [b], "lr": 4e-2, "weight_decay": 0.0},
        {"params": [c], "lr": 1e-2, "weight_decay": 0.0},
    ]


def _leaves(seed: int):
    g = torch.Generator().manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(8, 8, generator=g)) for _ in range(3)]


def _collapsed(params):
    return [{"params": list(params), "lr": 1e-2, "weight_decay": 0.1}]


def _run(optimizer: str, steps: int = 3, groups=_groups):
    params = _leaves(0)
    bundle = build_bundle(
        groups(params),
        DEV,
        optimizer,
        {"lr": 1e-2, "betas": (0.9, 0.95), "eps": 1e-8},
        {"mode": "cosine", "warmup": 1, "end": 6},
        6,
        build_precision("fp32", ()),
        0.0,
    )
    lrs = []
    for i in range(steps):
        loss = sum(((p - i) ** 2).sum() for p in params)
        bundle.backward(loss)
        assert bundle.step()
        lrs.append([float(g["lr"]) for g in _live_groups(bundle)])
    return params, lrs


def _live_groups(bundle):
    return bundle.optimizer.param_groups


@pytest.mark.cpu
def test_offload_adamw_keeps_group_settings():
    try:
        got, got_lrs = _run("master_adamw")
    except RuntimeError as err:  # fused AdamW unavailable for CPU tensors
        pytest.skip(str(err))
    ref, ref_lrs = _run("adamw")
    assert sum(got_lrs, []) == pytest.approx(sum(ref_lrs, []))
    assert all(lr[1] == pytest.approx(4 * lr[0]) for lr in got_lrs if lr[0] > 0)
    for a, b in zip(got, ref):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    def collapsed():
        flat, _ = _run("master_adamw", groups=_collapsed)
        for a, b in zip(flat, ref):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    must_fail(collapsed, "groups collapsed to one (max wd, base lr)")
