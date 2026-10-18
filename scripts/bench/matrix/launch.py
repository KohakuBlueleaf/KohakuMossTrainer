"""Real-GPU bench matrix driver (run on the machine that has the working tree).

    python scripts/bench/matrix/launch.py --hosts <hosts.json> --bench c1 --phase plan
    python scripts/bench/matrix/launch.py --hosts <hosts.json> --bench c1 --phase start
    python scripts/bench/matrix/launch.py --hosts <hosts.json> --bench c1 --phase status
    python scripts/bench/matrix/launch.py --hosts <hosts.json> --bench c1 --phase join --only blackwell --gpus 2,3
    python scripts/bench/matrix/launch.py --hosts <hosts.json> --bench c1 --phase collect
    python scripts/bench/matrix/launch.py --hosts <hosts.json> --bench c1 --phase report

`plan` is the dry run (cards, pools, runs, expected hours, read-only preflight); `start`
freezes, ships, installs and starts; `join` hands more cards to a running host;
`--option key=value` overrides a bench option. See docs/performance/bench-matrix.md.
"""

import argparse
import json
import shlex
import threading
from dataclasses import asdict
from pathlib import Path

from kohakumoss_bench.measure import smi
from kohakumoss_bench.plan import dry_run
from kohakumoss_bench.plan.config import Card, host_config, load_hosts, override
from kohakumoss_bench.report.build import build, junit_map
from kohakumoss_bench.run import remote

REPO = Path(__file__).resolve().parents[3]


def freeze(repo: Path) -> tuple[list[str], str]:
    files = remote.tree_files(repo)
    return files, remote.tree_identity(repo, files)


def start_host(host, opts, bench: str, frozen: dict) -> None:
    files, identity = frozen[""]
    trees = {}
    for name, path in opts.repos.items():
        extra_files, extra_id = frozen[name]
        trees[name] = remote.push(host, Path(path), extra_files, extra_id, name)
        remote.install(host, trees[name])
    tree = remote.push(host, REPO, files, identity)
    remote.install(host, tree)
    config = host_config(bench, tree, opts, host, trees)
    path = remote.put_config(host, bench, config)
    remote.start(host, tree, path, bench)
    print(f"[{host.name}] started from tree {identity} (+ {sorted(trees)})")


def host_cards(host) -> list[tuple[Card, float]]:
    """(card, MiB in use now) of every card on the host (no filter)."""
    if host.device_type == "xpu":
        return [
            (Card(c["index"], c["name"], float(c["total_gib"]), 0, "xpu"), 0.0)
            for c in host.xpu_cards
        ]
    prefix = tuple(host.ssh) if host.transport == "ssh" else ()
    found = smi.cards(prefix)
    used = smi.used(prefix)
    return [(c, used.get(c.index, 0.0)) for c in found]


def checks(host) -> dict[str, str]:
    """Read-only preflight commands of an ssh host."""
    q = shlex.quote
    out = {
        "python": f"test -x {q(host.python)} && {q(host.python)} --version",
        "weights": f"ls -d {q(host.weights.get('source', ''))}",
        "dev/shm free": "df -h /dev/shm | tail -1",
    }
    if host.fix_file:
        out["fix file"] = f"ls -l {q(host.fix_file.get('source', ''))}"
    for i, d in enumerate(host.datasets):
        out[f"dataset {i}"] = f"ls -d {q(d.get('path', ''))}"
    for i, c in enumerate(host.copies):
        out[f"copy {i} source"] = f"ls -d {q(c.get('source', ''))}"
    if host.ncu:
        out["ncu"] = " ".join(q(x) for x in host.ncu) + " --version | tail -1"
    return out


def plan_host(host, opts) -> dict:
    try:
        found = host_cards(host)
    except Exception as err:  # noqa: BLE001
        return {"error": f"{type(err).__name__}: {err}", "cards": [], "pools": []}
    try:
        use = smi.select([c for c, _ in found], host.gpus, host.gpu_bus)
    except ValueError as err:
        return {"error": str(err), "cards": [], "pools": []}
    cards = [
        {**asdict(c), "used_mib": u, "use": c in use, "model": c.model}
        for c, u in found
    ]
    return {
        "cards": cards,
        "preflight": remote.preflight(host, checks(host)),
        "pools": dry_run.pool_plan(host, opts, use),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hosts", required=True)
    ap.add_argument("--bench", required=True)
    phases = ("plan", "start", "status", "join", "collect", "report")
    ap.add_argument("--phase", choices=phases, required=True)
    ap.add_argument("--only", default="", help="comma-separated host names")
    ap.add_argument("--gpus", default="", help="join: comma-separated card indices")
    ap.add_argument("--option", action="append", default=[], help="key=value")
    ap.add_argument("--junit", default="", help="pytest --junitxml of the test run")
    ap.add_argument(
        "--mma-peaks", default="", help='JSON {"<card name key>": {"bf16": TFLOP/s}}'
    )
    ap.add_argument("--compute-bound", type=float, default=0.5)
    ap.add_argument("--transfer-bound", type=float, default=0.7)
    args = ap.parse_args()
    opts, hosts = load_hosts(args.hosts)
    opts = override(opts, args.option)
    if args.only:
        keep = set(args.only.split(","))
        hosts = [h for h in hosts if h.name in keep]
    local_root = REPO / "runs" / "bench" / args.bench
    match args.phase:
        case "plan":
            plans = {h.name: plan_host(h, opts) for h in hosts}
            local_root.mkdir(parents=True, exist_ok=True)
            md = dry_run.markdown(plans)
            (local_root / "plan.md").write_text(md, encoding="utf-8")
            (local_root / "plan.json").write_text(
                json.dumps(plans, indent=1, default=str), encoding="utf-8"
            )
            print(md.split("\n## ", 1)[0])
            for name, p in plans.items():
                if "error" in p:
                    print(f"[{name}] {p['error']}")
            print(f"wrote {local_root / 'plan.md'}")
        case "start":
            frozen = {"": freeze(REPO)}
            for name, path in opts.repos.items():
                frozen[name] = freeze(Path(path))
            for name, (files, identity) in frozen.items():
                print(f"tree {name or 'main'} {identity}: {len(files)} files")
            far = [h for h in hosts if h.transport != "local"]
            threads = [
                threading.Thread(target=start_host, args=(h, opts, args.bench, frozen))
                for h in far
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            for h in hosts:
                if h.transport == "local":
                    start_host(h, opts, args.bench, frozen)
        case "status":
            for h in hosts:
                print(h.name, json.dumps(remote.status(h, args.bench), indent=1))
        case "join":
            gpus = [int(x) for x in args.gpus.split(",") if x]
            for h in hosts:
                print(f"[{h.name}] {remote.join(h, args.bench, gpus)}")
        case "collect":
            for h in hosts:
                print(f"[{h.name}] -> {remote.collect(h, args.bench, local_root)}")
        case "report":
            junit = junit_map(Path(args.junit) if args.junit else None)
            peaks = (
                json.loads(Path(args.mma_peaks).read_text(encoding="utf-8"))
                if args.mma_peaks
                else {}
            )
            md, data = build(
                local_root, REPO, junit, args.compute_bound, args.transfer_bound, peaks
            )
            (local_root / "report.md").write_text(md, encoding="utf-8")
            (local_root / "report.json").write_text(
                json.dumps(data, indent=1, default=str), encoding="utf-8"
            )
            print(f"wrote {local_root / 'report.md'}")


if __name__ == "__main__":
    main()
