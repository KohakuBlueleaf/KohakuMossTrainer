"""Bench matrix planning and reporting, model-free: the catalog frontier, the card
queue's fit filter and dependencies, its units and pool order, run placeholders, failure
classes, the bound result, the coverage status, the ncu CSV summary, the nvidia-smi
steady value, the profiled-step exclusion, the plan's wall time and the train script's
knob defaults."""

import json
import threading
from pathlib import Path

import pytest
import torch

pytest.importorskip(
    "kohakumoss_bench", reason="not installed: uv pip install -e scripts/bench/matrix"
)

from kohakumoss_bench.measure import ncu
from kohakumoss_bench.measure.smi import steady
from kohakumoss_bench.plan import dry_run, hostmem, matrix
from kohakumoss_bench.plan.config import Card, Host, Options, Run
from kohakumoss_bench.plan.coverage import run_status
from kohakumoss_bench.plan.knobs import TRAIN, fill, script_defaults
from kohakumoss_bench.report import results
from kohakumoss_bench.report.build import bound_result, mma_peaks
from kohakumoss_bench.run.child import classify
from kohakumoss_bench.run.pool import Pool
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
REPO = Path(__file__).resolve().parents[3]


def host() -> Host:
    return Host(
        name="h",
        root="/r",
        python="python",
        weights={"source": "/w/base", "stage": "/s/base"},
        fix_file={"source": "/w/fix.safetensors", "stage": "/s/fix.safetensors"},
        datasets=[{"source": "kohya", "path": "/d"}],
        cache_dir="/c",
    )


def test_frontier_drops_offload_that_buys_nothing():
    gib = {"ckpt-stream8": 10.0, "ckpt-stream8-gradoff": 9.9, "ckpt-stream2": 6.0}
    kept = matrix.frontier(list(gib), gib, 0.03)
    assert kept == ["ckpt-stream8", "ckpt-stream2"]

    def check(keep):
        assert "ckpt-stream8-gradoff" not in keep

    must_fail(lambda: check(list(gib)), "no frontier: dominated entry kept")
    assert matrix.frontier(
        list(gib), {**gib, "ckpt-stream8-gradoff": 8.0}, 0.03
    ) == list(gib)


def test_queue_fits_card_and_orders_dependencies():
    opts = Options()
    small = Card(0, "small", 12.0, 86)
    big = Card(0, "big", 80.0, 90)
    q_small = matrix.card_queue(host(), opts, small, [])
    q_big = matrix.card_queue(host(), opts, big, [])
    speed = lambda q: {r.tags["profile"] for r in q if r.kind == "speed"}  # noqa: E731
    assert "speed" in speed(q_big) and "speed" not in speed(q_small)
    assert any(r.tags.get("base") == "fp8_compute" for r in q_big)
    assert not any(r.tags.get("base") == "fp8_compute" for r in q_small)
    names = [r.name for r in q_big]
    for i, r in enumerate(q_big):
        if r.after:
            assert names.index(r.after) < i, (r.name, r.after)
    assert q_big[0].timeout_s == opts.timeout_first_s
    assert q_big[-1].kind == "restart" and q_big[-1].knobs == q_big[0].knobs
    caps = [r for r in q_big if r.kind == "cap"]
    assert {r.cap_gib for r in caps} == set(opts.vram_targets)

    def ordered(q):
        for i, r in enumerate(q):
            if r.after:
                assert [x.name for x in q].index(r.after) < i

    must_fail(lambda: ordered(list(reversed(q_big))), "dependency after its user")


def test_queue_rows_bases_and_ncu():
    opts = Options()
    ada = Card(0, "ada", 24.0, 89)
    turing = Card(0, "turing", 22.0, 75)
    q = matrix.card_queue(host(), opts, ada, [])
    bases = {r.tags.get("base") for r in q if r.kind == "speed"}
    assert bases == {"compute", "int8", "fp8_compute", "mxfp8"}, bases
    assert {r.tags["algo"] for r in q if r.kind == "speed"} >= set(matrix.ROW_ALGOS)
    t_bases = {r.tags.get("base") for r in matrix.card_queue(host(), opts, turing, [])}
    assert "mxfp8" not in t_bases and "int8" in t_bases
    assert not any(r.kind == "ncu" for r in q)
    with_ncu = Host(**{**host().__dict__, "ncu": ["ncu"]})
    nq = [r for r in matrix.card_queue(with_ncu, opts, ada, []) if r.kind == "ncu"]
    assert len(nq) == 1
    first, last = nq[0].knobs["PROFILE_STEPS"]
    assert nq[0].knobs["STEPS"] == last and last - first == opts.ncu_steps
    must_fail(lambda: _same(len(q), len(q) + 1), "count")


def test_queue_priority_order():
    q = matrix.card_queue(host(), Options(), Card(0, "big", 32.0, 120), [])
    ranks = [matrix.priority(r) for r in q[1:]]
    assert ranks == sorted(ranks) and q[-1].kind == "restart"
    first_quality = next(i for i, r in enumerate(q) if r.kind == "quality")
    assert all(r.kind != "batch" for r in q[first_quality:])
    catalog = [r for r in q if r.kind == "speed" and matrix.priority(r) == 5]
    assert catalog and all(q.index(r) > first_quality for r in catalog)
    must_fail(lambda: _same(ranks, sorted(ranks, reverse=True)), "reversed order")


def check_ranks(queue: list, ranks: set[int]) -> None:
    names = {r.name for r in queue}
    for r in queue[1:]:
        assert matrix.priority(r) in ranks or any(
            x.after == r.name for x in queue
        ), f"{r.name} (rank {matrix.priority(r)}) outside {ranks}"
        assert not r.after or r.after in names, f"{r.name} lost {r.after}"


def test_queue_kept_by_priority():
    card = Card(0, "v100", 16.0, 70)
    full = matrix.card_queue(host(), Options(), card, [])
    for ranks in ({1}, {3}, {3, 4, 5, 6}):
        h = Host(**{**host().__dict__, "priorities": sorted(ranks)})
        queue = matrix.card_queue(h, Options(), card, [])
        assert queue[0].name == full[0].name and len(queue) < len(full)
        check_ranks(queue, ranks)
    assert {matrix.priority(r) for r in full} >= {1, 3, 5, 6}
    must_fail(lambda: check_ranks(full, {1}), "every rank kept")


def test_mma_peaks_from_measurement_or_sane_probe():
    probe = {
        "name": "NVIDIA GeForce RTX 5090",
        "gemm_tflops": {"bf16": 250.0, "fp16": 245.0},
        "mma_tflops": {"bf16": 275.0, "fp16": 100.0},
    }
    assert mma_peaks(probe, {"5090": {"bf16": 420.0}}) == {"bf16": 420.0}
    assert mma_peaks(probe, {}) == {"bf16": 275.0}
    must_fail(lambda: _same(mma_peaks(probe, {}).get("fp16"), 100.0), "loop below GEMM")


def test_run_patterns_keep_dependencies():
    q = matrix.card_queue(host(), Options(), Card(0, "big", 80.0, 90), [])
    grid = next(r for r in q if r.kind == "grid")
    kept = matrix.keep_names(q, [grid.name, "speed-*-low"])
    names = {r.name for r in kept}
    assert grid.name in names and grid.after in names
    assert all(r.kind in ("speed", "quality", "export", "grid") for r in kept)
    assert any(r.name.endswith("-low") for r in kept)
    assert matrix.keep_names(q, None) == q
    must_fail(lambda: _same(len(matrix.keep_names(q, ["nothing"])), 1), "kept none")


def test_units_keep_chains_on_one_card():
    q = matrix.card_queue(host(), Options(), Card(0, "big", 80.0, 90), [])
    units, final = matrix.units(q)
    assert [r.kind for r in final] == ["restart"]
    assert units[0] == [q[0]]
    where = {r.name: i for i, u in enumerate(units) for r in u}
    for r in q:
        if r.after:
            assert where[r.after] == where[r.name], (r.name, r.after)

    def chained(us):
        loc = {r.name: i for i, u in enumerate(us) for r in u}
        for u in us:
            for r in u:
                if r.after:
                    assert loc[r.after] == loc[r.name]

    flat = [[r] for u in units for r in u]
    must_fail(lambda: chained(flat), "a dependency split from its user")


def _run(name: str, kind: str = "speed") -> Run:
    return Run(name, kind, TRAIN, {}, 1.0)


def test_pool_first_alone_final_last():
    stop = threading.Event()
    pool = Pool(
        "m", [[_run("a")], [_run("b")], [_run("c")]], [_run("r", "restart")], stop
    )
    unit, first = pool.take()
    assert first and unit[0].name == "a" and not pool.gate.is_set()
    got: list = []
    other = threading.Thread(target=lambda: got.append(pool.take()))
    other.start()
    other.join(0.2)
    assert other.is_alive(), "a second card started before the tuning run finished"
    pool.done(True)
    other.join(10)
    assert got[0][0][0].name == "b"
    unit, _ = pool.take()
    assert unit[0].name == "c"
    pool.done(False)
    pool.done(False)
    unit, _ = pool.take()
    assert unit[0].name == "r"
    pool.done(False)
    assert pool.take() is None
    must_fail(lambda: _same(unit[0].name, "c"), "final before the others")


def _ncu_csv() -> str:
    t = ncu.TENSOR_NAMES[1] + ncu.PCT
    head = ["ID", "Kernel Name", ncu.DURATION, t, ncu.SM, ncu.DRAM]
    units = ["", "", "nsecond", "%", "%", "%"]
    rows = [
        ["0", "gemm(int)", "8,000", "80", "85", "30"],
        ["1", "gemm(int)", "8000", "70", "75", "30"],
        ["2", "norm(float)", "4000", "0", "20", "90"],
    ]
    lines = ["==PROF== Disconnected"] + [
        ",".join(f'"{c}"' for c in r) for r in [head, units, *rows]
    ]
    return "\n".join(lines)


def test_ncu_summary_and_result():
    rows = ncu.parse_csv(_ncu_csv())
    assert [r["ns"] for r in rows] == [8000.0, 8000.0, 4000.0]
    s = ncu.summary(rows, ncu.TENSOR_NAMES[1], steps=2, bound=60.0)
    assert s["kernels"] == 3 and abs(s["kernel_ms_per_step"] - 0.01) < 1e-12
    gemm = next(k for k in s["top"] if k["kernel"] == "gemm")
    assert gemm["class"] == "compute" and abs(gemm["tensor_pct"] - 75.0) < 1e-9
    assert abs(s["tensor_pct"] - 0.8 * 75.0) < 1e-9
    assert s["class_share"] == {"compute": 0.8, "memory": 0.2, "latency": 0.0}
    v = ncu.bound_result(s, step_s=0.01 / 1e3 / 0.8, transfer_share=0.1)
    assert v["bound"] == "compute-bound" and abs(v["kernel_busy_share"] - 0.8) < 1e-9
    assert ncu.bound_result(s, 0.01 / 1e3 / 0.8, 0.9)["bound"] == "PCIe-bound"
    assert ncu.bound_result(s, 1.0, 0.1)["bound"] == "host-bound"
    must_fail(
        lambda: _same(ncu.kernel_class(20.0, 90.0, 60.0), "compute"),
        "memory as compute",
    )


def _replay_is(cmd: list[str], mode: str) -> None:
    assert cmd[cmd.index("--replay-mode") + 1] == mode
    assert ("--profile-from-start" in cmd) == (mode != "app-range")


def test_ncu_command_carries_the_replay_knob():
    target = ["python", "-u", "train.py"]
    for mode in ("kernel", "application", "app-range"):
        cmd = ncu.command(["ncu"], "t", Path("r"), target, mode)
        _replay_is(cmd, mode)
        assert cmd[0] == "ncu" and cmd[-len(target) :] == target
    default = ncu.command(["ncu"], "t", Path("r"), target)
    _replay_is(default, Options().ncu_replay)
    _replay_is(default, "app-range")
    with pytest.raises(ValueError):
        ncu.command(["ncu"], "t", Path("r"), target, "range")
    kernel = ncu.command(["ncu"], "t", Path("r"), target, "kernel")
    must_fail(lambda: _replay_is(kernel, "app-range"), "kernel replay as app-range")
    ranged = [*default[:1], "--profile-from-start", "off", *default[1:]]
    must_fail(lambda: _replay_is(ranged, "app-range"), "app-range with from-start")


def test_ncu_csv_range_rows():
    t = ncu.TENSOR_NAMES[1] + ncu.PCT
    head = ["ID", "Process ID", "Range Name", ncu.DURATION, t, ncu.SM, ncu.DRAM]
    units = ["", "", "", "msecond", "%", "%", "%"]
    row = ["0", "7", "range", "1,500", "41", "52", "33"]
    text = "\n".join(",".join(f'"{c}"' for c in r) for r in [head, units, row])
    rows = ncu.parse_csv("==PROF== Report\n" + text)
    assert rows == [
        {"name": "range", "ns": 1.5e9, t: 41.0, ncu.SM: 52.0, ncu.DRAM: 33.0}
    ]
    s = ncu.summary(rows, ncu.TENSOR_NAMES[1], steps=1, bound=60.0)
    assert s["kernel_ms_per_step"] == 1500.0 and s["tensor_pct"] == 41.0
    must_fail(lambda: _same(ncu.parse_csv(text.replace(ncu.DURATION, "x")), rows))


def test_smi_steady_is_the_tail_median():
    assert steady([100.0, 9000.0, 5000.0, 5000.0, 5100.0, 4900.0]) == 5000.0
    assert steady([]) is None
    must_fail(lambda: _same(steady([1.0, 1.0, 9.0, 9.0]), 1.0), "head median")


def test_profiled_steps_left_out_of_medians(tmp_path):
    rows = [
        {"step": i, "step_time": 9.0 if i in (37, 38, 39, 40) else 1.0}
        for i in range(1, 41)
    ]
    path = tmp_path / "train" / "metrics.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    knobs = {"TRACE_STEPS": (36, 39), "PROFILE_STEPS": ()}
    profiled = results.profiled_steps(knobs)
    assert profiled == {37, 38, 39, 40}
    s = results.training_summary(tmp_path, 10, profiled)
    assert s["step_time_median"] == 1.0 and s["steady_steps"] == 26
    must_fail(
        lambda: _same(results.training_summary(tmp_path, 34)["step_time_median"], 1.0),
        "profiled steps in the median",
    )


def test_nonfinite_steps_listed_with_scale():
    rows = [
        {"step": 3, "grad_norm": 0.1, "loss_scale": 65536.0},
        {"step": 4, "grad_norm": float("nan"), "loss_scale": 32768.0},
        {"step": 5, "grad_norm": float("inf"), "loss_scale": 16384.0},
    ]
    assert results.nonfinite_at(rows) == [(4, 32768.0), (5, 16384.0)]
    must_fail(lambda: _same(results.nonfinite_at(rows[:1]), [(3, 65536.0)]), "finite")


def test_host_ram_estimate_and_budget():
    base = {"MAX_TOKENS": 8192, "BASE_PRECISION": "compute"}
    resident = Run("a", "speed", TRAIN, {**base, "MEMORY_PROFILE": "ckpt-resident"}, 1)
    stream = Run("b", "speed", TRAIN, {**base, "MEMORY_PROFILE": "ckpt-stream8"}, 1)
    acts = {**base, "MEMORY_PROFILE": "nockpt-stream16-actoff"}
    heavy = Run("c", "speed", TRAIN, acts, 1)
    small, mid, big = (hostmem.run_gib(r) for r in (resident, stream, heavy))
    assert small == hostmem.BASELINE_GIB and mid > small + 8 and big > mid + 30
    stop = threading.Event()
    live = hostmem.HostBudget(1e6, stop, reserve_gib=16.0, available=lambda: 80.0)
    assert live.acquire(10.0) and live.free() == 80.0 - 16.0 - 10.0
    assert live.free() < big, "a run past MemAvailable would start"
    live.release(10.0)
    budget = hostmem.HostBudget(100.0, stop, available=lambda: None)
    assert budget.acquire(big)
    got: list = []
    other = threading.Thread(target=lambda: got.append(budget.acquire(big)))
    other.start()
    other.join(0.3)
    assert other.is_alive(), "two runs oversubscribed the host"
    budget.release(big)
    other.join(15)
    assert got == [True]
    stop.set()
    assert budget.acquire(500.0) is False
    must_fail(lambda: _same(small, big), "act offload not counted")


def test_plan_wall_spreads_units():
    opts = Options()
    card = Card(0, "NVIDIA GeForce RTX 5090", 32.0, 120)
    q = matrix.card_queue(host(), opts, card, [])
    units, final = matrix.units(q)
    one = dry_run.wall_s(units, final, card, 1, opts)
    four = dry_run.wall_s(units, final, card, 4, opts)
    serial = sum(dry_run.run_s(r, card, opts) for r in q) + opts.plan_tuning_s
    assert abs(one - serial) < 1e-6 * serial and four < 0.5 * one
    must_fail(lambda: _same(four, one), "four cards as slow as one")


def test_fill_replaces_placeholders():
    knobs = {
        "OUT": "{self}/grid",
        "CHECKPOINTS": [("{run:a}/x.safetensors", 1.0)],
        "PROMPTS": "{prompts}",
        "SAMPLE_PROMPTS": "{sample_prompts}",
    }
    out = fill(knobs, Path("/runs/me"), {"a": Path("/runs/a")}, ["p0", "p1", "p2"])
    assert out == {
        "OUT": "/runs/me/grid",
        "CHECKPOINTS": [("/runs/a/x.safetensors", 1.0)],
        "PROMPTS": ["p0", "p1", "p2"],
        "SAMPLE_PROMPTS": ["p0", "p1"],
    }
    must_fail(
        lambda: _same(fill(knobs, Path("/x"), {}, []), out), "placeholders not filled"
    )


def _same(a, b):
    assert a == b


def test_failure_classes():
    assert classify(torch.OutOfMemoryError("CUDA out of memory")) == "oom"
    assert classify(SystemExit("invalid configuration:\n  - X")) == "config"
    assert classify(ValueError("invalid memory plan: y")) == "config"
    assert classify(RuntimeError("index out of range")) == "crash"
    assert classify(AssertionError("shape")) == "crash"
    must_fail(lambda: _same(classify(KeyError("k")), "config"), "crash read as config")


def test_bound_result():
    probe = {
        "gemm_tflops": {"bf16": 200.0},
        "link": {"pinned": {"h2d_gbps": 20.0, "d2h_gbps": 20.0}},
    }

    def rec(step, tflops, h2d):
        return {
            "tags": {"precision": "bf16", "base": "compute"},
            "summary": {
                "step_time_median": step,
                "tflops_median": tflops,
                "xfer_h2d_gb_median": h2d,
            },
        }

    assert (
        bound_result(rec(1.0, 150.0, 0.0), probe, 0.5, 0.7)["bound"] == "compute-bound"
    )
    assert (
        bound_result(rec(1.0, 50.0, 18.0), probe, 0.5, 0.7)["bound"] == "transfer-bound"
    )
    assert (
        bound_result(rec(1.0, 50.0, 2.0), probe, 0.5, 0.7)["bound"] == "software-bound"
    )
    must_fail(
        lambda: _same(
            bound_result(rec(1.0, 50.0, 2.0), probe, 0.5, 0.7)["bound"], "compute-bound"
        ),
        "slow step read as compute-bound",
    )


def test_coverage_status():
    assert run_status([]) == "not run"
    assert run_status([{"status": "oom"}]) == "does not fit"
    assert run_status([{"status": "crash"}, {"status": "ok"}]) == "working"
    assert run_status([{"status": "nonfinite"}]) == "broken"
    must_fail(lambda: _same(run_status([{"status": "crash"}]), "working"), "crash ok")


def test_train_script_defaults():
    k = script_defaults(REPO / TRAIN)
    assert k["TARGET_AREA"] == 1024 * 1024
    assert "METRICS_EVERY" in k and k["MEMORY_PROFILE"] == "balanced"
    must_fail(lambda: _same(k["TARGET_AREA"], 1024), "default misread")
