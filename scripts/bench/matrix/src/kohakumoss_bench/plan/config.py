"""Bench matrix configuration: hosts, cards and runs (JSON on disk).

A hosts file is `{"bench": {...}, "hosts": [{...}, ...]}`; each host entry fills
`Host`, the bench block fills `Options`. A `Run` is one script call with its knobs
on one card. See docs/performance/bench-matrix.md.
"""

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class Options:
    """Bench matrix-wide settings (the hosts file's "bench" block)."""

    speed_steps: int = 40  # steps per speed run; medians over steps after `warmup`
    warmup: int = 10
    quality_steps: int = 400
    quality_save: int = 100
    max_tokens: int = 8192  # joint tokens per speed batch
    target_area: int = 1024 * 1024
    catalog: str = "frontier"  # "frontier" | "all" | "none": catalog points per card
    frontier_slack: float = 0.03  # see matrix.frontier
    overhead_gib: float = (
        2.5  # device use outside the estimate (context, kernels, cache)
    )
    cap_allowance_gib: float = 1.5  # allocator cap = VRAM target - this
    vram_targets: tuple[float, ...] = (8.0, 6.0)
    grid_artists: int = 3
    grid_captions: int = 2
    grid_steps: int = 24
    grid_cfg: float = 5.0
    timeout_first_s: float = 3600.0  # first run of a card (kernel tuning)
    timeout_speed_s: float = 1800.0
    timeout_quality_s: float = 6 * 3600.0
    timeout_other_s: float = 3600.0
    timeout_ncu_s: float = 4 * 3600.0
    busy_wait_s: float = 600.0  # wait for a GPU another process uses, then skip
    resume: bool = False  # skip runs a previous start of this bench finished
    host_ram_share: float = (
        0.6  # without /proc/meminfo: share of `Host.ram_gib` runs may hold
    )
    host_ram_reserve_gib: float = 16.0  # MemAvailable kept free of bench runs
    cleanup: bool = True  # delete checkpoints / exports no queued run reads any more
    disk_floor_gib: float = 50.0  # free GiB on the runs disk a row needs to start
    full_checkpoint_gib: float = 60.0  # disk estimate of a full-finetune run
    adapter_checkpoint_gib: float = 2.0  # disk estimate of an adapter run
    disk_poll_s: float = 60.0  # recheck interval while a card waits for disk
    fp16_fix: str = (
        "online"  # "online" | "file" (the host's fix file; online without one)
    )
    trace_steps: int = 2  # torch.profiler steps at the end of each speed run; 0 = none
    batch_scales: tuple[int, ...] = (2, 4, 8)  # batch rows: multiples of max_tokens
    ncu_steps: int = 1  # steps the ncu pass records (after `warmup`)
    ncu_bound_pct: float = 60.0  # a kernel is compute / memory-bound at this % of peak
    ncu_replay: str = (
        "app-range"  # "app-range" | "kernel" | "application" (ncu.REPLAYS)
    )
    repos: dict = field(
        default_factory=dict
    )  # extra repos shipped and installed: name -> local path
    plan_mfu: float = 0.4  # time estimate: model FLOP/s as a share of the nominal peak
    plan_tflops: float = (
        50.0  # time estimate: nominal peak of a card missing from flops.py
    )
    plan_link_gbps: float = (
        20.0  # time estimate: host <-> device GB/s of streamed weights
    )
    plan_overhead_s: float = (
        120.0  # time estimate: process start, weight load, kernel binding
    )
    plan_tuning_s: float = 1800.0  # time estimate: the first run's kernel tuning
    plan_ncu_step_s: float = 300.0  # time estimate: one step under ncu kernel replay


@dataclass
class Host:
    """One machine of the bench."""

    name: str
    root: str  # bench root on the host: frozen trees, runs, logs
    python: str  # interpreter of the venv that has the frozen tree installed
    transport: str = "ssh"  # "ssh" | "local" | "manual"
    ssh: list[str] = field(default_factory=list)  # e.g. ["ssh", "user@host"]
    device_type: str = "cuda"  # "cuda" | "xpu"
    gpus: list[int] | None = None  # allowed indices (None: every idle one)
    gpu_bus: list[str] | None = None  # PCI buses of the cards ("83" / full id); wins
    # over `gpus`, which then must name the same cards
    xpu_cards: list[dict] = field(
        default_factory=list
    )  # [{"index", "name", "total_gib"}]
    ram_gib: int = 0
    weights: dict = field(
        default_factory=dict
    )  # {"source": path | "hf:<repo>", "stage": path}
    fix_file: dict = field(
        default_factory=dict
    )  # {"source", "stage"}; empty: online fix
    datasets: list[dict] = field(default_factory=list)  # DATASETS entries on the host
    cache_dir: str = ""  # offline latent / text caches (fast local storage)
    cache_from: str = ""  # copy an existing cache dir instead of building it
    install: str = (
        "{python} -m pip install --no-deps -e {tree} -e {tree}/scripts/bench/matrix"
    )
    env: dict = field(default_factory=dict)
    quality: list[str] | None = None  # quality runs to keep (names), None: all
    kinds: list[str] | None = None  # run kinds to keep, None: all
    profiles: list[str] | None = (
        None  # speed-run profiles (no catalog points), None: all
    )
    bases: list[str] | None = (
        None  # base precisions to keep, None: every one the card has
    )
    catalog: str = ""  # catalog points: "all" | "frontier" | "none"; "": the bench's
    runs: list[str] | None = None  # run-name patterns (fnmatch) to keep, None: all
    priorities: list[int] | None = None  # queue ranks (matrix.priority) to keep
    quality_steps: int = 0  # quality-run steps on this host; 0: the bench's
    copies: list[dict] = field(
        default_factory=list
    )  # more {"source", "stage"} (datasets)
    ncu: list[str] = field(
        default_factory=list
    )  # ncu command, e.g. ["sudo", "-n", ".../ncu"]; empty: no ncu pass


@dataclass
class Card:
    """One GPU as the host runner found it."""

    index: int
    name: str
    total_gib: float
    sm: int  # compute capability x10 (89 = 8.9); 0 on XPU
    device_type: str = "cuda"
    uuid: str = ""
    bus: str = ""  # PCI bus id (nvidia-smi pci.bus_id)

    @property
    def bf16(self) -> bool:
        return self.device_type == "xpu" or self.sm >= 80

    @property
    def fp8(self) -> bool:
        return self.device_type == "cuda" and self.sm >= 89

    @property
    def model(self) -> str:
        """The card model with filesystem-safe characters: cards of one model on a host
        form one pool that shares a queue."""
        return "".join(c if c.isalnum() else "_" for c in self.name).strip("_")

    @property
    def slug(self) -> str:
        """`<index>-<model>`."""
        return f"{self.index}-{self.model}"


@dataclass
class Run:
    """One script call on one card."""

    name: str  # unique in the card's queue, filesystem-safe
    kind: str  # "speed" | "online" | "quality" | "coverage" | "cap" | "ncu" | "restart" | "export" | "grid" | "infer"
    script: str  # repo-relative script
    knobs: dict
    timeout_s: float
    cap_gib: float = 0.0  # VRAM target (allocator cap below it); 0 = none
    after: str = ""  # a run of the same queue that must have succeeded first
    tags: dict = field(
        default_factory=dict
    )  # precision / base / algo / profile / features


def _fill(cls, data: dict):
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {unknown}")
    return cls(**data)


def load_hosts(path: str | Path) -> tuple[Options, list[Host]]:
    """(bench options, hosts) of a hosts file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    opts = data.get("bench", {})
    if "vram_targets" in opts:
        opts = {**opts, "vram_targets": tuple(opts["vram_targets"])}
    return _fill(Options, opts), [_fill(Host, h) for h in data["hosts"]]


def override(opts: Options, pairs: list[str]) -> Options:
    """`opts` with `key=value` pairs applied (values parsed as JSON, else kept as text)."""
    names = {f.name for f in fields(Options)}
    out = asdict(opts)
    for pair in pairs:
        key, _, text = pair.partition("=")
        if key not in names:
            raise ValueError(f"Options: unknown key {key!r}")
        try:
            out[key] = json.loads(text)
        except json.JSONDecodeError:
            out[key] = text
    out["vram_targets"] = tuple(out["vram_targets"])
    return Options(**out)


def host_config(
    bench: str, tree: str, opts: Options, host: Host, trees: dict | None = None
) -> dict:
    """The JSON the host runner reads: bench name, frozen tree dir, the extra repos'
    tree dirs (name -> dir), options, host."""
    return {
        "bench": bench,
        "tree": tree,
        "trees": dict(trees or {}),
        "options": asdict(opts),
        "host": asdict(host),
    }


def read_host_config(path: str | Path) -> tuple[str, str, dict, Options, Host]:
    """(bench, tree, extra trees, options, host) of a file `host_config` wrote."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    opts = dict(data["options"])
    opts["vram_targets"] = tuple(opts["vram_targets"])
    return (
        data["bench"],
        data["tree"],
        data["trees"],
        _fill(Options, opts),
        _fill(Host, data["host"]),
    )
