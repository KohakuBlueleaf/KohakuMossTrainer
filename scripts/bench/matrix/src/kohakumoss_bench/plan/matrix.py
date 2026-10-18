"""The run queue of one card pool: speed runs (precision x base x algo x memory profile,
one row each for DoRA / LoHa / OFT), VRAM-target runs, the ncu pass, a restart run,
quality runs with their exports, grids and inference checks; and its split into units
the pool's cards take. Profiles are kept where their estimated peak plus
`Options.overhead_gib` fits the card. See docs/performance/bench-matrix.md#matrix.
"""

from dataclasses import replace

from kohakumoss.memory.catalog_table import lora_params
from kohakumoss.memory.estimate import GIB, Workload, estimate
from kohakumoss.memory.profiles import (
    CATALOG,
    ORDER,
    PROFILES,
    catalog_name,
    resolve,
)
from kohakumoss.types.arch import PRESETS, MossArchConfig
from kohakumoss_bench.plan.config import Card, Host, Options, Run
from kohakumoss_bench.plan.knobs import (
    ALGOS,
    EXPORT,
    GRID,
    SAMPLE,
    TRAIN,
    export_knobs,
    fix_mode,
    grid_knobs,
    other_fix,
    quality_knobs,
    sample_knobs,
    train_knobs,
)
from kohakumoss_bench.run.filters import keep_kinds, keep_names, keep_priorities

NAMED = (*ORDER, "auto")
CHECKPOINT_RANK = {"nockpt": 0, "ckpt": 1, "subckpt": 2}
WEIGHT_RANK = {"resident": 0, "ring16": 1, "ring8": 2, "ring4": 3, "ring2": 4}
CAP_PROFILE = "minimal"
MAIN_ALGOS = ("lora32", "lokr16")
ROW_ALGOS = ("dora16", "loha16", "oft64")
ROW_PROFILE = "auto"
BATCH_PROFILES = ("balanced", "ckpt-resident")
USUAL = {
    "lora32": ("balanced", "auto", "low", "minimal"),
    "lokr16": ("balanced", "low"),
}
CACHE_MODES = {
    "nocache": {"LATENT_CACHE": "off", "TEXT_CACHE": "off"},
    "latent": {"LATENT_CACHE": "offline", "TEXT_CACHE": "off"},
    "latenttext": {"LATENT_CACHE": "offline", "TEXT_CACHE": "offline"},
}
LOWVRAM_LOKR_PROFILE = "ckpt-ring2-actoff-optstream-gradoff"
LOWVRAM_LOKR_TEXT = {"memory": "ring", "ahead": 8}


def lokr_params(arch: MossArchConfig, factor: int = 16) -> int:
    """LoKr full-matrix parameters on every block linear (w1 factor^2, w2 out/f x in/f)."""
    d, f = arch.dim, arch.ffn_dim
    one = factor * factor
    attn = 4 * (one + (d // factor) * (d // factor))
    ffn = 3 * (one + (d // factor) * (f // factor))
    return arch.depth * (attn + ffn)


def workload(algo: str, opts: Options, arch: MossArchConfig = PRESETS["base"]):
    """The micro-step the memory estimate sizes for `algo` at `opts.max_tokens`."""
    img = int(opts.max_tokens * 0.94)
    trainable = (
        lokr_params(arch)
        if ALGOS[algo]["ALGO"] == "lokr"
        else lora_params(arch, int(ALGOS[algo]["DIM"]))
    )
    return Workload(
        arch,
        img_tokens=img,
        txt_tokens=opts.max_tokens - img,
        trainable_params=trainable,
    )


def estimated_gib(profile: str, base: str, algo: str, opts: Options) -> float | None:
    """Estimated peak GiB of `profile` with base precision `base`; None when the
    profile does not resolve (an invalid combination)."""
    if profile == "auto":
        return 0.0
    try:
        plan = resolve(profile, device="cpu")
    except ValueError:
        return None
    plan = replace(plan, base_precision=base)
    return estimate(workload(algo, opts), plan).total / GIB


def fits(card: Card, gib: float | None, opts: Options) -> bool:
    return gib is not None and gib + opts.overhead_gib <= card.total_gib


def _parts(name: str) -> tuple[int, int, frozenset]:
    parts = name.split("-")
    ck = next(p for p in parts if p in CHECKPOINT_RANK)
    w = next(p for p in parts if p in WEIGHT_RANK)
    rest = frozenset(p for p in parts if p not in (ck, w))
    return CHECKPOINT_RANK[ck], WEIGHT_RANK[w], rest


def _cheaper(a: str, b: str) -> bool:
    """`a` moves / recomputes no more than `b` on every axis (and differs)."""
    ca, wa, ra = _parts(a)
    cb, wb, rb = _parts(b)
    return a != b and ca <= cb and wa <= wb and ra <= rb


def frontier(names: list[str], gib: dict[str, float], slack: float) -> list[str]:
    """Drop every catalog entry for which a cheaper entry (`_cheaper`) needs at most
    `slack` more estimated memory: adding that entry's extra offload buys nothing."""
    keep = []
    for b in names:
        dominated = any(
            _cheaper(a, b) and gib[a] <= gib[b] * (1 + slack) for a in names
        )
        if not dominated:
            keep.append(b)
    return keep


def precisions(card: Card) -> list[str]:
    return (["bf16"] if card.bf16 else []) + ["fp16"]


def bases(card: Card, host: Host | None = None) -> list[str]:
    """Base precisions of the card: compute and int8 weight-only everywhere, fp8 and
    MXFP8 compute on sm89+; `host.bases` keeps a subset."""
    have = ["compute", "int8"] + (["fp8_compute", "mxfp8"] if card.fp8 else [])
    keep = None if host is None else host.bases
    return have if keep is None else [b for b in have if b in keep]


def _slug(*parts: str) -> str:
    return "-".join(p.replace("_", "") for p in parts if p)


def trace_window(opts: Options) -> tuple:
    """TRACE_STEPS of a speed run: the last `trace_steps` steps before the final one."""
    if opts.trace_steps <= 0:
        return ()
    last = opts.speed_steps - 1
    return (last - opts.trace_steps, last)


def speed_run(host, opts, card, precision, base, algo, profile, kind="speed") -> Run:
    knobs = train_knobs(host, opts, precision, algo)
    knobs.update(
        MEMORY_PROFILE=profile, BASE_PRECISION=base, TRACE_STEPS=trace_window(opts)
    )
    tags = {"precision": precision, "base": base, "algo": algo, "profile": profile}
    if precision == "fp16":
        tags["fp16_fix"] = fix_mode(host, opts)
    return Run(
        name=_slug(kind, precision, base, algo, profile),
        kind=kind,
        script=TRAIN,
        knobs=knobs,
        timeout_s=opts.timeout_speed_s,
        tags=tags,
    )


def catalog_points(
    card: Card, opts: Options, base: str, algo: str, catalog: str = ""
) -> list[str]:
    """Catalog entries that fit `card` (all, or their `frontier`), largest first;
    `catalog` ("all" | "frontier" | "none", "" = `opts.catalog`) picks which."""
    catalog = catalog or opts.catalog
    if catalog == "none":
        return []
    gib = {n: estimated_gib(n, base, algo, opts) for n in CATALOG}
    names = [n for n in CATALOG if fits(card, gib[n], opts)]
    if catalog == "frontier":
        names = frontier(names, gib, opts.frontier_slack)
    return sorted(names, key=lambda n: -gib[n])


def speed_runs(host: Host, opts: Options, card: Card) -> list[Run]:
    """Every named profile per (precision, base, algo) that fits, one `ROW_PROFILE` row
    per `ROW_ALGOS` entry, the reference combination over the catalog points, one
    online-cache row."""
    runs: list[Run] = []
    ref = (precisions(card)[0], "compute", "lora32")
    named_profiles = NAMED if host.profiles is None else tuple(host.profiles)
    for precision in precisions(card):
        for base in bases(card, host):
            for algo in MAIN_ALGOS:
                for profile in named_profiles:
                    if fits(card, estimated_gib(profile, base, algo, opts), opts):
                        runs.append(
                            speed_run(host, opts, card, precision, base, algo, profile)
                        )
    for algo in ROW_ALGOS:
        runs.append(speed_run(host, opts, card, ref[0], ref[1], algo, ROW_PROFILE))
    named = {r.name for r in runs}
    same = {catalog_name(p) for p in named_profiles if p in PROFILES}
    points = (
        catalog_points(card, opts, ref[1], ref[2], host.catalog)
        if host.profiles is None
        else []
    )
    for profile in points:
        run = speed_run(host, opts, card, *ref, profile)
        if run.name not in named and profile not in same:
            runs.append(run)
    online = speed_run(host, opts, card, *ref, "auto", kind="online")
    online.knobs.update(
        LATENT_CACHE="online", TEXT_CACHE="online", CACHE_DIR="{self}/cache"
    )
    online.tags["features"] = ["online caches"]
    runs.append(online)
    return runs


def cap_runs(host: Host, opts: Options, card: Card) -> list[Run]:
    """`CAP_PROFILE` under each VRAM target below the card's memory (lora8, lokr16;
    every base precision)."""
    runs = []
    precision = precisions(card)[0]
    for target in opts.vram_targets:
        if target >= card.total_gib:
            continue
        for base in bases(card, host):
            for algo in ("lora8", "lokr16"):
                run = speed_run(host, opts, card, precision, base, algo, CAP_PROFILE)
                run.kind = "cap"
                run.name = _slug("cap", f"{target:g}g", precision, base, algo)
                run.cap_gib = target
                run.tags["cap_gib"] = target
                runs.append(run)
    return runs


def restart_run(first: Run) -> Run:
    """The queue's first speed run again, at the end: a restart must tune nothing."""
    return Run(
        name="restart-" + first.name,
        kind="restart",
        script=first.script,
        knobs=dict(first.knobs),
        timeout_s=first.timeout_s,
        tags={**first.tags, "features": ["tuning reuse"]},
    )


def quality_names(card: Card, host: Host, opts: Options) -> list[tuple[str, str, str]]:
    """(precision, algo, fp16 fix) of the card's quality runs: lora8 + lokr16 in bf16
    (where the card has it) and fp16 (`fix_mode`), plus lora8 fp16 with the other fix
    when the host has the fix file (the two fixes side by side)."""
    out = []
    for precision in precisions(card):
        for algo in ("lora8", "lokr16"):
            out.append((precision, algo, fix_mode(host, opts)))
    other = other_fix(host, opts)
    if other is not None:
        out.append(("fp16", "lora8", other))
    return out


def quality_runs(host: Host, opts: Options, card: Card) -> list[Run]:
    """Each quality run, its checkpoint exports and its comparison grid; inference
    checks over the first lora8 run's final export."""
    runs = []
    total = host.quality_steps or opts.quality_steps
    save = min(opts.quality_save, total)
    steps = list(range(save, total + 1, save))
    infer_from = None
    for precision, algo, fix in quality_names(card, host, opts):
        name = _slug("quality", precision, algo, "" if precision == "bf16" else fix)
        if host.quality is not None and name not in host.quality:
            continue
        train = quality_knobs(host, opts, precision, algo, fix)
        train.update(STEPS=total, SAVE_EVERY=save, SAMPLE_EVERY=max(total // 2, 1))
        tags = {"precision": precision, "algo": algo, "fp16_fix": fix}
        runs.append(
            Run(name, "quality", TRAIN, train, opts.timeout_quality_s, tags=tags)
        )
        exports = []
        for step in steps:
            ex = Run(
                f"{name}-export{step}",
                "export",
                EXPORT,
                export_knobs(train, step, name),
                opts.timeout_other_s,
                after=name,
                tags=dict(tags),
            )
            runs.append(ex)
            exports.append((ex.name, step))
        runs.append(
            Run(
                f"{name}-grid",
                "grid",
                GRID,
                grid_knobs(host, card, opts, train, exports),
                opts.timeout_other_s,
                after=exports[-1][0],
                tags=dict(tags),
            )
        )
        if infer_from is None and algo == "lora8":
            infer_from = (train, exports[-1], tags)
    if infer_from is not None:
        runs += infer_runs(host, opts, card, *infer_from)
    return runs


def infer_runs(host, opts, card, train, export, tags) -> list[Run]:
    """sample.py checks: base model, the adapter merged and in bypass, the fp8 base
    (sm89+), the low-VRAM placement (memory profile, streamed text encoder, VAE offload).
    """
    name, step = export
    adapter = (f"{{run:{name}}}/step_{step:08d}.safetensors", 1.0)
    base = sample_knobs(host, card, opts, train)
    variants = {
        "infer-base": {},
        "infer-merge": {"ADAPTERS": [adapter], "ADAPTER_MODE": "merge"},
        "infer-bypass": {"ADAPTERS": [adapter], "ADAPTER_MODE": "bypass"},
        "infer-lowmem": {
            "ADAPTERS": [adapter],
            "ADAPTER_MODE": "bypass",
            "MEMORY_PROFILE": "low",
            "TEXT_ENCODE": {"memory": "ring"},
            "VAE_ENCODE": {"memory": "offload"},
        },
    }
    if card.fp8:
        variants["infer-fp8"] = {
            "ADAPTERS": [adapter],
            "ADAPTER_MODE": "bypass",
            "BASE_PRECISION": "fp8_compute",
            "TEXT_ENCODE": {"precision": "fp8_compute"},
        }
    return [
        Run(
            n,
            "infer",
            SAMPLE,
            {**base, **extra},
            opts.timeout_other_s,
            after=name,
            tags={**tags, "variant": n},
        )
        for n, extra in variants.items()
    ]


def batch_runs(host: Host, opts: Options, card: Card) -> list[Run]:
    """Batch scaling: `BATCH_PROFILES` of the first precision (plus fp16 balanced on a
    bf16 card), LoRA r32, at `opts.batch_scales` x `opts.max_tokens` tokens per step
    (MAX_TOKENS, the BatchLimits budget); a setting past the card's memory ends `oom`.
    """
    combos = [(precisions(card)[0], p) for p in BATCH_PROFILES]
    if card.bf16:
        combos.append(("fp16", "balanced"))
    runs = []
    for precision, profile in combos:
        for scale in opts.batch_scales:
            tokens = opts.max_tokens * scale
            run = speed_run(host, opts, card, precision, "compute", "lora32", profile)
            run.kind = "batch"
            run.name = _slug("batch", precision, "lora32", profile, f"t{tokens}")
            run.knobs["MAX_TOKENS"] = tokens
            run.tags["max_tokens"] = tokens
            runs.append(run)
    return runs


def flagship_setups(host: Host, card: Card) -> list[tuple]:
    """(precision, base, algo, profile, extra knobs) of the flagship setups: LoRA r32
    balanced in the first precision and in fp16, LoRA r32 minimal, LoKr 16 balanced,
    fp8-compute LoRA r32 balanced (sm89+), the low-VRAM LoKr setup (fp16, int8 base,
    `LOWVRAM_LOKR_PROFILE`, text encoder streamed 8 batches ahead); bases outside
    `bases(card, host)` dropped, duplicates (fp16-only cards) once."""
    first = precisions(card)[0]
    setups = [
        (first, "compute", "lora32", "balanced", {}),
        ("fp16", "compute", "lora32", "balanced", {}),
        (first, "compute", "lora32", "minimal", {}),
        (first, "compute", "lokr16", "balanced", {}),
        (first, "fp8_compute", "lora32", "balanced", {}),
        (
            "fp16",
            "int8",
            "lokr16",
            LOWVRAM_LOKR_PROFILE,
            {"TEXT_ENCODE": LOWVRAM_LOKR_TEXT},
        ),
    ]
    have = bases(card, host)
    out, seen = [], set()
    for setup in setups:
        if setup[1] in have and setup[:4] not in seen:
            seen.add(setup[:4])
            out.append(setup)
    return out


def cache_runs(host: Host, opts: Options, card: Card) -> list[Run]:
    """Every fitting flagship setup under each of `CACHE_MODES`: no cache (VAE and text
    encoder every step), the latent cache, latent and text caches (offline). The
    encoder placements are the profile's (TEXT_ENCODE / VAE_ENCODE unset) unless the
    setup names its own."""
    runs = []
    for precision, base, algo, profile, extra in flagship_setups(host, card):
        if not fits(card, estimated_gib(profile, base, algo, opts), opts):
            continue
        short = "lowvramlokr" if profile == LOWVRAM_LOKR_PROFILE else profile
        for mode, caches in CACHE_MODES.items():
            run = speed_run(host, opts, card, precision, base, algo, profile, "cache")
            run.name = _slug("cache", precision, base, algo, short, mode)
            run.knobs.update(caches, **extra)
            run.tags["cache"] = mode
            run.tags["features"] = ["cache modes"]
            runs.append(run)
    return runs


def placement_setups(card: Card) -> list[tuple]:
    """(precision, base, algo, profile, TEXT_ENCODE extra, [(text, VAE memory), ...]) of
    the encoder-placement rows: LoRA r32 balanced (text auto / resident / offload, text
    ring with the VAE resident and offloaded), LoRA r32 minimal and the low-VRAM LoKr
    setup (text auto / resident / offload / ring; the profile's VAE)."""
    first = precisions(card)[0]
    text = [("auto", None), ("resident", None), ("offload", None)]
    return [
        (
            first,
            "compute",
            "lora32",
            "balanced",
            {},
            text + [("ring", "resident"), ("ring", "offload")],
        ),
        (first, "compute", "lora32", "minimal", {}, text + [("ring", None)]),
        (
            "fp16",
            "int8",
            "lokr16",
            LOWVRAM_LOKR_PROFILE,
            {"ahead": LOWVRAM_LOKR_TEXT["ahead"]},
            text + [("ring", None)],
        ),
    ]


def placement_runs(host: Host, opts: Options, card: Card) -> list[Run]:
    """No-cache runs (`CACHE_MODES["nocache"]`) of each `placement_setups` entry under
    each text-encoder / VAE placement; a placement past the card's memory ends `oom`."""
    runs = []
    for precision, base, algo, profile, extra, placements in placement_setups(card):
        if base not in bases(card, host):
            continue
        short = "lowvramlokr" if profile == LOWVRAM_LOKR_PROFILE else profile
        for te, vae in placements:
            run = speed_run(
                host, opts, card, precision, base, algo, profile, "placement"
            )
            parts = ("placement", precision, base, algo, short, f"te{te}")
            run.name = _slug(*parts, f"vae{vae}" if vae else "")
            run.knobs.update(
                CACHE_MODES["nocache"], TEXT_ENCODE={"memory": te, **extra}
            )
            if vae is not None:
                run.knobs["VAE_ENCODE"] = {"memory": vae}
            run.tags.update(
                cache="nocache", text_memory=te, vae_memory=vae or "profile"
            )
            run.tags["features"] = ["encoder placement"]
            runs.append(run)
    return runs


def ncu_run(host: Host, opts: Options, card: Card) -> Run | None:
    """The ncu pass of the pool: the reference combination under `ROW_PROFILE`, steps
    `warmup + 1 .. warmup + ncu_steps` inside the CUDA profiler window; None without an
    ncu command or on a non-CUDA card."""
    if not host.ncu or card.device_type != "cuda":
        return None
    precision = precisions(card)[0]
    run = speed_run(
        host, opts, card, precision, "compute", "lora32", ROW_PROFILE, "ncu"
    )
    first, last = opts.warmup, opts.warmup + opts.ncu_steps
    run.knobs.update(STEPS=last, TRACE_STEPS=(), PROFILE_STEPS=(first, last))
    run.timeout_s = opts.timeout_ncu_s
    run.tags["features"] = ["ncu efficiency"]
    return run


def units(queue: list[Run]) -> tuple[list[list[Run]], list[Run]]:
    """(units, final) of a pool queue: each unit is a run with every run that depends on
    it (in order, on one card); `final` is the restart run, taken when every other unit
    has finished."""
    out: list[list[Run]] = []
    final: list[Run] = []
    unit_of: dict[str, list[Run]] = {}
    for run in queue:
        if run.kind == "restart":
            final.append(run)
            continue
        unit = unit_of.get(run.after) if run.after else None
        if unit is None:
            unit = []
            out.append(unit)
        unit.append(run)
        unit_of[run.name] = unit
    return out, final


def priority(run: Run) -> int:
    """Queue rank: 1 the flagship cache-mode rows and the encoder-placement rows
    (first: `card_queue` builds them ahead of the other rank-1 runs), the usual setups
    (LoRA r32 balanced / auto / low / minimal, LoKr 16 balanced / low, every base, the
    DoRA / LoHa / OFT rows, VRAM targets, batch scaling), 2 feature coverage, 3
    quality with its exports / grids / inference, 4 the ncu pass, 5 the rest (other
    named profiles, catalog points), 6 the restart run."""
    profile = run.tags.get("profile", "")
    algo = run.tags.get("algo", "")
    match run.kind:
        case "cache" | "placement" | "batch" | "cap":
            return 1
        case "speed" if algo in USUAL and profile in USUAL[algo]:
            return 1
        case "speed" if algo in ROW_ALGOS:
            return 1
        case "online" | "coverage":
            return 2
        case "quality" | "export" | "grid" | "infer":
            return 3
        case "ncu":
            return 4
        case "restart":
            return 6
    return 5


def card_queue(host: Host, opts: Options, card: Card, coverage: list[Run]) -> list:
    """The pool's whole queue: the first speed run (it tunes and gets the long timeout),
    then every other run by `priority` (stable: the build order within a rank, which
    keeps a quality run before its exports / grid / inference)."""
    speed = speed_runs(host, opts, card)
    if speed:
        speed[0].timeout_s = opts.timeout_first_s
    runs = speed[:1] + cache_runs(host, opts, card) + placement_runs(host, opts, card)
    runs += batch_runs(host, opts, card) + speed[1:]
    runs += cap_runs(host, opts, card) + coverage
    runs += quality_runs(host, opts, card)
    ncu = ncu_run(host, opts, card)
    if ncu is not None:
        runs.append(ncu)
    if speed:
        runs.append(restart_run(speed[0]))
    runs = runs[:1] + sorted(runs[1:], key=priority)
    runs = keep_names(keep_kinds(runs, host.kinds), host.runs)
    return keep_priorities(runs, host.priorities, priority)
