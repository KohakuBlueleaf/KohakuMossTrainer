"""The bench's host runner: runs ON the host, from the frozen tree.

    python scripts/bench/matrix/host.py --config <root>/configs/<bench>-<host>.json

Prepare once (stage the weights and copies, probe every card, build the offline caches,
draw the grid prompts), then the idle cards of one model form a pool that shares one
queue (`matrix.card_queue` split by `matrix.units`): each card runs one unit at a time,
each run in its own process (`run.child`, the ncu pass under `sudo ncu`) under the
nvidia-smi sampler. Cards listed later in `<out>/join.json` join their pool once idle.
A run that crashes, times out or is rejected goes to `review-queue.jsonl` with its
traceback and the queue goes on (nothing is retried); only a runner failure around a
run (`runner-error`) stops the host: the other cards finish the run they are on and
start nothing new. Results land under `<root>/runs/<bench>/<host>/`.
See docs/performance/bench-matrix.md#host-runner.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

from kohakumoss_bench.measure import ncu, smi
from kohakumoss_bench.plan import matrix
from kohakumoss_bench.plan.config import Card, Host, Options, Run, read_host_config
from kohakumoss_bench.plan.coverage_runs import coverage_runs
from kohakumoss_bench.plan.hostmem import HostBudget, run_gib
from kohakumoss_bench.plan.knobs import (
    TRAIN,
    cache_knobs,
    config_text,
    data_knobs,
    encoder_dtype,
    fill,
    script_defaults,
    tree_env,
    weights_knobs,
)
from kohakumoss_bench.plan.prompts import grid_prompts
from kohakumoss_bench.report import results, tables
from kohakumoss_bench.run import disk, release
from kohakumoss_bench.run.ncu_pass import ncu_command
from kohakumoss_bench.run.pool import Pool
from kohakumoss_bench.run.proc import (
    end,
    nonfinite,
    reap,
    retuned,
    same_card,
    session,
    visible,
)
from kohakumoss_bench.run.stage import copy_tree, stage
from kohakumoss_kernels.hardware.device import cache_dir as kernel_cache_dir

STOPS = ("runner-error",)
REVIEW = (*STOPS, "crash", "timeout", "config", "nonfinite", "retuned")
REVIEW += ("profile-failed",)
FINAL = ("ok", "oom", "config", "nonfinite", "retuned", "profile-failed")
CACHE_LAYOUT = 2  # cache keys with the encoder spec (VAE / text-encoder dtype)
JOIN = "join.json"
JOIN_POLL_S = 30.0


class HostRunner:
    """State of one host's bench: cards, pools, results, the stop flag."""

    def __init__(self, bench: str, tree: str, opts: Options, host: Host, trees: dict):
        self.bench = bench
        self.tree = Path(tree)
        self.opts = opts
        self.host = host
        self.env = tree_env(host.env, tree, trees)
        self.out = Path(host.root) / "runs" / bench / host.name
        self.out.mkdir(parents=True, exist_ok=True)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.status: dict = {"host": host.name, "cards": {}, "stopped": None}
        self.sampler = smi.Sampler() if smi.available() else smi.NoSampler()
        self.prompts: list[str] = []
        self.probes: dict[str, dict] = {}
        self.active: set[int] = set()
        self.cleanups: dict[str, disk.Cleanup] = {}
        ram = host.ram_gib * opts.host_ram_share if host.ram_gib else float("inf")
        self.budget = HostBudget(ram, self.stop, opts.host_ram_reserve_gib)

    # ----------------------------------------------------------------- bookkeeping
    def _write(self, name: str, data) -> None:
        path = self.out / name
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
        tmp.replace(path)

    def _status(self, card: Card, **fields) -> None:
        with self.lock:
            self.status["cards"].setdefault(card.slug, {}).update(fields)
            self._write("status.json", self.status)

    def _review(self, card: Card, run: Run, result: dict) -> None:
        entry = {
            "host": self.host.name,
            "card": card.slug,
            "run": run.name,
            "kind": run.kind,
            "status": result["status"],
            "failure": result.get("failure"),
            "summary": result.get("summary"),
            "dir": result["dir"],
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with self.lock:
            with open(self.out / "review-queue.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")

    # ----------------------------------------------------------------------- cards
    def cards(self) -> list[Card]:
        if self.host.device_type == "xpu":
            return [
                Card(c["index"], c["name"], float(c["total_gib"]), 0, "xpu")
                for c in self.host.xpu_cards
            ]
        return smi.select(smi.cards(), self.host.gpus, self.host.gpu_bus)

    def idle(self, card: Card) -> bool:
        return card.device_type == "xpu" or not smi.busy_pids(card)

    def wait_idle(self, card: Card) -> bool:
        deadline = time.monotonic() + self.opts.busy_wait_s
        while not self.idle(card):
            if time.monotonic() > deadline or self.stop.is_set():
                return False
            time.sleep(10)
        return True

    def kernel_root(self) -> Path:
        cache = self.env.get("KOHAKUMOSS_KERNEL_CACHE")
        return Path(cache) if cache else kernel_cache_dir()

    # ------------------------------------------------------------------- one run
    def child_command(self, run: Run, card: Card, cfg: Path, failure: Path) -> list:
        cmd = [
            self.host.python,
            "-m",
            "kohakumoss_bench.run.child",
            "--script",
            str(self.tree / run.script),
            "--config",
            str(cfg),
            "--failure",
            str(failure),
            "--device-type",
            card.device_type,
        ]
        if run.cap_gib:
            cmd += [
                "--cap-gib",
                str(run.cap_gib),
                "--allow-gib",
                str(self.opts.cap_allowance_gib),
            ]
        return cmd

    def execute(self, run: Run, card: Card, dirs: dict[str, Path]) -> dict:
        """Run `run` on `card` in its own process; the result record."""
        run_dir = dirs[run.name]
        run_dir.mkdir(parents=True, exist_ok=True)
        disk.retire(run_dir)
        knobs = fill(run.knobs, run_dir, dirs, self.prompts)
        cfg = run_dir / "config.py"
        cfg.write_text(config_text(knobs, f"bench run {run.name}"), encoding="utf-8")
        failure = run_dir / "failure.json"
        failure.unlink(missing_ok=True)
        cmd = self.child_command(run, card, cfg, failure)
        env = {**os.environ, **self.env, **visible(card), "PYTHONUNBUFFERED": "1"}
        meta = None
        if run.kind == "ncu":
            cmd, meta = ncu_command(
                self.host.ncu,
                self.opts.ncu_replay,
                self.env,
                self.kernel_root(),
                card,
                run_dir,
                cmd,
            )
        baseline = smi.used_mib(card) if card.device_type == "cuda" else 0.0
        sudo = self.host.ncu[:2] if meta is not None else []
        started = time.time()
        with open(run_dir / "log.txt", "w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                cmd,
                cwd=self.tree,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                **session(),
            )
            trace = self.sampler.register(proc.pid, card, baseline)
            status = "ok"
            try:
                code = proc.wait(timeout=run.timeout_s)
            except subprocess.TimeoutExpired:
                end(proc, sudo)
                code, status = None, "timeout"
            reap(proc, sudo)
            self.sampler.release(proc.pid)
        record = {
            "run": run.name,
            "kind": run.kind,
            "card": card.slug,
            "bus": card.bus,
            "model": card.model,
            "dir": str(run_dir),
            "script": run.script,
            "tags": run.tags,
            "cap_gib": run.cap_gib,
            "exit": code,
            "seconds": time.time() - started,
        }
        if trace is not None:
            record["vram_smi"] = trace.summary()
            (run_dir / "vram_smi.csv").write_text(trace.csv(), encoding="utf-8")
        if failure.is_file():
            record["failure"] = json.loads(failure.read_text(encoding="utf-8"))
            status = status if status == "timeout" else record["failure"]["class"]
        elif code not in (0, None):
            status = "profile-failed" if meta is not None else "crash"
            record["failure"] = {"class": status, "message": f"exit code {code}"}
        record["summary"] = self.summarize(run, run_dir)
        if trace is not None and run.script == TRAIN:
            start = results.loop_start(run_dir)
            split = None if start is None else start - started
            record["vram_smi_phases"] = trace.phases(split)
        if meta is not None:
            record["ncu"] = ncu.report_summary(
                self.host.ncu,
                run_dir,
                meta,
                self.opts.ncu_steps,
                self.opts.ncu_bound_pct,
            )
        if status == "ok" and run.script == TRAIN and nonfinite(record["summary"]):
            status = "nonfinite"
        if status == "ok" and run.kind == "restart" and retuned(record["summary"]):
            status = "retuned"
        if status == "ok" and (record.get("vram_smi") or {}).get("smi_foreign"):
            status = "contended"
        record["status"] = status
        (run_dir / "result.json").write_text(
            json.dumps(record, indent=1, default=str), encoding="utf-8"
        )
        return record

    def summarize(self, run: Run, run_dir: Path) -> dict:
        out = results.log_summary(run_dir / "log.txt")
        if run.script == TRAIN:
            profiled = results.profiled_steps(run.knobs)
            out.update(results.training_summary(run_dir, self.opts.warmup, profiled))
            out["exports"] = results.exports_of(run_dir)
        else:
            out.update(results.sample_summary(run_dir))
        return out

    # --------------------------------------------------------------------- prepare
    def prepare(self, cards: list[Card], cover: Card) -> None:
        """Stage weights and copies, probe every card, build the caches, draw the grid
        prompts."""
        report = {"weights": stage(self.host.weights)}
        if self.host.fix_file:
            report["fix_file"] = stage(self.host.fix_file)
        report["copies"] = [stage(entry) for entry in self.host.copies]
        self._write("staging.json", report)
        for card in cards:
            self.probes[card.slug] = self.probe(card)
        self._write("probes.json", self.probes)
        self.caches(cards)
        defaults = script_defaults(self.tree / TRAIN)
        data = {
            **defaults,
            **data_knobs(self.host, self.opts),
            **weights_knobs(self.host, "bf16"),
        }
        self.prompts = grid_prompts(
            data, self.opts.grid_artists, self.opts.grid_captions
        )
        self._write("prompts.json", self.prompts)

    def probe(self, card: Card) -> dict:
        path = self.out / card.slug / "probe.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.host.python,
            "-m",
            "kohakumoss_bench.measure.probe",
            "--out",
            str(path),
        ]
        env = {**os.environ, **self.env, **visible(card)}
        done = subprocess.run(
            cmd, cwd=self.tree, env=env, capture_output=True, text=True, timeout=900
        )
        if done.returncode != 0:
            raise RuntimeError(f"probe of {card.slug} failed:\n{done.stderr[-4000:]}")
        return json.loads(path.read_text(encoding="utf-8"))

    def caches(self, cards: list[Card]) -> None:
        """The offline latent and text caches every run reads: one build per encoder
        dtype the host's cards resolve to (`encoder_dtype`: the encoder spec in the cache
        keys), on a card of that dtype (or copied from `cache_from`); a marker per dtype
        records the data block, the dtype and `CACHE_LAYOUT` they match."""
        for dtype, card in {encoder_dtype(c): c for c in reversed(cards)}.items():
            self._cache(card, dtype)

    def _cache(self, card: Card, dtype: str) -> None:
        root = Path(self.host.cache_dir)
        want = {**data_knobs(self.host, self.opts), "dtype": dtype}
        want["layout"] = CACHE_LAYOUT
        mark = root / f"bench-cache-{dtype}.json"
        if mark.is_file() and json.loads(
            mark.read_text(encoding="utf-8")
        ) == json.loads(json.dumps(want)):
            return
        if self.host.cache_from:
            copy_tree(Path(self.host.cache_from), root)
        else:
            dirs = {}
            for kind in ("latents", "text"):
                script, knobs = cache_knobs(self.host, self.opts, card, kind)
                run = Run(f"prepare-cache-{kind}", "prepare", script, knobs, 6 * 3600.0)
                dirs[run.name] = self.out / card.slug / run.name
                result = self.execute(run, card, dirs)
                if result["status"] != "ok":
                    self._review(card, run, result)
                    raise RuntimeError(f"cache build {kind} failed: {result['status']}")
        root.mkdir(parents=True, exist_ok=True)
        mark.write_text(json.dumps(want, indent=1), encoding="utf-8")

    # --------------------------------------------------------------------- queues
    def pools(self, idle: list[Card]) -> dict[str, Pool]:
        """One pool per card model; the coverage runs go to the largest model. Runs a
        previous start finished go to their pool's cleanup at once, so a disk hold at
        the head of the queue cannot keep what they no longer need."""
        models: dict[str, Card] = {}
        for card in idle:
            models.setdefault(card.model, card)
        cover = max(models.values(), key=lambda c: c.total_gib)
        pools, queues = {}, {}
        for model, card in models.items():
            cov = coverage_runs(self.host, self.opts, card) if card is cover else []
            if 0 in (self.host.priorities or ()):
                queue = release.queue(self.host, self.opts, card, cov, self.tree)
            else:
                queue = matrix.card_queue(self.host, self.opts, card, cov)
            queues[model] = [asdict(r) for r in queue]
            units, final = matrix.units(queue)
            pools[model] = Pool(model, units, final, self.stop)
            self.cleanups[model] = disk.Cleanup(queue, self.out, self.opts.cleanup)
            self.sweep(model, queue)
        self._write("queues.json", queues)
        return pools

    def sweep(self, model: str, queue: list[Run]) -> None:
        """Hand every run of `queue` a previous start finished to the pool's cleanup."""
        for run in queue:
            if (previous := self.finished(run.name, model)) is not None:
                self._finished(model, run.name, Path(previous["dir"]))

    def _finished(self, model: str, name: str, run_dir) -> None:
        """Hand a finished row to its pool's cleanup; the GiB freed go to status.json."""
        freed = self.cleanups[model].finished(name, run_dir)
        if freed:
            with self.lock:
                d = self.status.setdefault("disk", {})
                d["freed_gib"] = round(d.get("freed_gib", 0.0) + freed, 2)
                self._write("status.json", self.status)

    def wait_disk(self, card: Card, run: Run) -> bool:
        """Hold the card while the runs disk has less free space than `run` needs
        (`disk.floor_gib`), reported once per hold in status.json and the log; False
        when the host stopped while holding."""
        need = disk.floor_gib(run, self.opts)
        reported = False
        while (free := disk.free_gib(self.out)) < need:
            if not reported:
                held = f"{run.name}: {free:.1f} GiB free < {need:.0f} GiB floor"
                print(f"[{card.slug}] disk hold: {held}", flush=True)
                self._status(card, held=held)
                reported = True
            if self.stop.wait(self.opts.disk_poll_s):
                return False
        if reported:
            self._status(card, held=None)
        return True

    def worker(self, card: Card, pool: Pool) -> None:
        base = self.out / card.slug
        records: list[dict] = []
        done: dict[str, str] = {}
        self._status(card, pool=pool.model, done=0, current=None, failures=0)
        while (taken := pool.take()) is not None:
            unit, first = taken
            dirs = {r.name: base / r.name for r in unit}
            for run in unit:
                if self.stop.is_set():
                    break
                previous = self.finished(run.name, pool.model)
                if previous is not None:
                    dirs[run.name] = Path(previous["dir"])
                    done[run.name] = previous["status"]
                    records.append({**previous, "resumed": True})
                    self._finished(pool.model, run.name, dirs[run.name])
                    continue
                why = None
                if run.after and done.get(run.after) != "ok":
                    why = "after"
                elif not self.wait_disk(card, run):
                    why = "stopped"
                elif not self.wait_idle(card):
                    why = "busy"
                if why is not None:
                    done[run.name] = "skipped"
                    records.append({"run": run.name, "status": "skipped", "why": why})
                    self._finished(pool.model, run.name, None)
                    continue
                self._status(card, current=run.name, done=len(records))
                result = self.guarded(run, card, dirs)
                done[run.name] = result["status"]
                records.append(result)
                self._finished(pool.model, run.name, result.get("dir"))
                if result["status"] in REVIEW:
                    self._review(card, run, result)
                    fails = sum(r["status"] in REVIEW for r in records)
                    self._status(card, failures=fails)
                if result["status"] in STOPS:
                    self.stop.set()
                    with self.lock:
                        self.status["stopped"] = {"card": card.slug, "run": run.name}
            pool.done(first)
            base.mkdir(parents=True, exist_ok=True)
            (base / "results.json").write_text(
                json.dumps(records, indent=1, default=str), encoding="utf-8"
            )
        self._status(card, current=None, done=len(records))

    def finished(self, name: str, model: str) -> dict | None:
        """With `opts.resume`: the record of `name` from a previous start of this
        bench on a card of `model` (pools of other models share run names) when it
        ended in a `FINAL` status, else None; a previous record of another status is
        kept as `result.superseded.json`."""
        if not self.opts.resume:
            return None
        for path in sorted(self.out.glob(f"*-{model}/{name}/result.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("status") in FINAL:
                return record
            path.replace(path.with_name("result.superseded.json"))
        return None

    def guarded(self, run: Run, card: Card, dirs: dict[str, Path]) -> dict:
        """`execute` inside the host-RAM budget; a runner failure (a full disk, a lost
        file) becomes a "runner-error" record instead of ending the card's worker."""
        gib = run_gib(run)
        if not self.budget.acquire(gib):
            return {"run": run.name, "status": "skipped", "why": "stopped"}
        try:
            return self.execute(run, card, dirs)
        except Exception as err:  # noqa: BLE001
            return {
                "run": run.name,
                "kind": run.kind,
                "card": card.slug,
                "dir": str(dirs[run.name]),
                "status": "runner-error",
                "failure": {"type": type(err).__name__, "message": str(err)[:2000]},
            }
        finally:
            self.budget.release(gib)

    def start_worker(self, card: Card, pool: Pool, threads: list) -> None:
        thread = threading.Thread(target=self.worker, args=(card, pool))
        with self.lock:
            self.active.add(card.index)
            threads.append(thread)
        thread.start()

    def joiner(self, pools: dict[str, Pool], threads: list) -> None:
        """Start a worker for each card `join.json` lists ({"gpus": [index, ...]}) once
        it is idle, in the pool of its model; a card of another model is refused."""
        path = self.out / JOIN
        refused: set[int] = set()
        while not self.stop.is_set() and any(t.is_alive() for t in list(threads)):
            time.sleep(JOIN_POLL_S)
            if not path.is_file():
                continue
            try:
                want = json.loads(path.read_text(encoding="utf-8")).get("gpus", [])
            except json.JSONDecodeError:
                continue
            found = {c.index: c for c in smi.cards()}
            for index in want:
                card = found.get(index)
                if index in self.active or index in refused or card is None:
                    continue
                if card.model not in pools:
                    refused.add(index)
                    self._status(card, joined=False, why="no pool of this model")
                    continue
                if not self.idle(card):
                    continue
                try:
                    self.probes[card.slug] = self.probe(card)
                except RuntimeError as err:
                    refused.add(index)
                    self._status(card, joined=False, why=str(err)[-500:])
                    continue
                self._write("probes.json", self.probes)
                self._status(card, joined=time.strftime("%Y-%m-%d %H:%M:%S"))
                self.start_worker(card, pools[card.model], threads)

    def kernel_report(self, models: list[str]) -> None:
        """Copy each model's tuning tables into the results and summarize them."""
        root = self.kernel_root() / "tuning"
        names = [p for p in root.iterdir() if p.is_dir()] if root.is_dir() else []
        report = {}
        for model in models:
            mine = [p for p in names if same_card(p.name, model)]
            for p in mine:
                copy_tree(p, self.out / "tuning" / model / p.name)
            report[model] = [row for p in mine for row in tables.card_report(p)]
        self._write("kernels.json", report)

    def run(self) -> int:
        cards = self.cards()
        idle = [c for c in cards if self.idle(c)]
        found = {"found": [asdict(c) for c in cards], "idle": [c.slug for c in idle]}
        self._write("cards.json", found)
        if not idle:
            return 1
        cover = max(idle, key=lambda c: c.total_gib)
        self.sampler.start()
        try:
            self.prepare(idle, cover)
            pools = self.pools(idle)
            threads: list[threading.Thread] = []
            for card in idle:
                self.start_worker(card, pools[card.model], threads)
            joiner = threading.Thread(
                target=self.joiner, args=(pools, threads), daemon=True
            )
            joiner.start()
            while alive := [t for t in list(threads) if t.is_alive()]:
                alive[0].join(5.0)
            self.kernel_report(list(pools))
        finally:
            self.sampler.stop()
        return 1 if self.stop.is_set() else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    bench, tree, trees, opts, host = read_host_config(args.config)
    runner = HostRunner(bench, tree, opts, host, trees)
    try:
        return runner.run()
    except Exception as err:  # noqa: BLE001
        runner._write(
            "host-failure.json", {"type": type(err).__name__, "message": str(err)}
        )
        raise


if __name__ == "__main__":
    sys.exit(main())
