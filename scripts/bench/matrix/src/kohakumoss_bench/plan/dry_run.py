"""The bench plan (dry run): the cards each host would use, its card pools, every
run of each pool's queue (the runner's own `matrix.card_queue` / `matrix.units`) and an
expected duration per run, per pool and per host. Nothing trains.

Durations are estimates: model FLOPs per step (`model/flops.py`) at `plan_mfu` of the
card's nominal peak, streamed weights at `plan_link_gbps`, `plan_overhead_s` per
process, `plan_tuning_s` on a pool's first run, `plan_ncu_step_s` per ncu-replayed
step. See docs/performance/bench-matrix.md#plan.
"""

import heapq

from kohakumoss.memory.estimate import block_params, total_params
from kohakumoss.memory.profiles import PROFILES, checkpoint_mode, profile_name
from kohakumoss.model.flops import PEAK_TFLOPS, forward_flops, step_flops
from kohakumoss.types.arch import PRESETS
from kohakumoss_bench.plan import matrix
from kohakumoss_bench.plan.config import Card, Host, Options, Run
from kohakumoss_bench.plan.coverage_runs import coverage_runs

ARCH = PRESETS["base"]
SAMPLE_TOKENS = 4096 + 300
TRAINING = ("speed", "online", "cap", "coverage", "quality", "ncu", "restart")


def peak(card: Card, opts: Options) -> float:
    for key, value in PEAK_TFLOPS:
        if key in card.name:
            return value
    return opts.plan_tflops


def _rate(card: Card, opts: Options) -> float:
    return peak(card, opts) * 1e12 * opts.plan_mfu


def train_step_s(card: Card, opts: Options, knobs: dict) -> float:
    """Estimated seconds of one training step: compute at `plan_mfu`, or the streamed
    weight and activation traffic over `plan_link_gbps` when that is longer."""
    tokens = int(knobs.get("MAX_TOKENS", opts.max_tokens))
    samples = max(1, round(tokens / SAMPLE_TOKENS))
    img = int(tokens * 0.94)
    lens = [tokens // samples] * samples
    profile = PROFILES.get(profile_name(knobs.get("MEMORY_PROFILE", "balanced")), {})
    mode, _ = checkpoint_mode(profile.get("checkpoint", "block"))
    full = knobs.get("ALGO") == "full"
    fwd = forward_flops(ARCH, img, lens, tokens - img)
    flops = step_flops(fwd, "full" if full else "adapter", mode != "none")
    compute = flops / _rate(card, opts)
    moved = 0.0
    if profile.get("stream", "off") != "off":
        resident = profile.get("stream_resident", 2) * block_params(ARCH)
        passes = 3 if mode != "none" else 2
        moved += (total_params(ARCH) - resident) * 2 * passes
    if profile.get("act_offload"):
        moved += 2 * tokens * ARCH.dim * 2 * ARCH.depth
    return max(compute, moved / (opts.plan_link_gbps * 1e9))


def image_s(card: Card, opts: Options, steps: int) -> float:
    """Estimated seconds of one 1 MP image: `steps` sampler steps with CFG."""
    fwd = forward_flops(ARCH, 4096, [SAMPLE_TOKENS], 300)
    return steps * 2 * fwd / _rate(card, opts)


def run_s(run: Run, card: Card, opts: Options) -> float:
    """Estimated seconds of one run (process overhead included)."""
    k = run.knobs
    out = opts.plan_overhead_s
    if run.kind in TRAINING:
        steps = int(k.get("STEPS", opts.speed_steps))
        out += steps * train_step_s(card, opts, k)
        if run.kind == "ncu":
            out += opts.ncu_steps * opts.plan_ncu_step_s
        every = int(k.get("SAMPLE_EVERY", 0) or 0)
        if every:
            prompts = 2 * (steps // every)
            out += prompts * image_s(card, opts, int(k.get("SAMPLE_STEPS", 20)))
        return out
    match run.kind:
        case "export":
            return out
        case "grid":
            images = (len(k.get("CHECKPOINTS", [])) + 1) * opts.grid_artists
            images *= opts.grid_captions
            return out + images * image_s(card, opts, int(k.get("STEPS", 24)))
        case "infer":
            images = opts.grid_artists * opts.grid_captions
            return out + images * image_s(card, opts, int(k.get("STEPS", 24)))
    return out


def unit_s(unit: list[Run], card: Card, opts: Options) -> float:
    return sum(run_s(r, card, opts) for r in unit)


def wall_s(units: list[list[Run]], final: list[Run], card: Card, n: int, opts) -> float:
    """Pool wall time on `n` cards: the first unit alone (plus tuning), the rest
    longest-first onto the least loaded card, then the final unit."""
    if not units:
        return unit_s(final, card, opts)
    first = unit_s(units[0], card, opts) + opts.plan_tuning_s
    loads = [0.0] * max(n, 1)
    heapq.heapify(loads)
    for t in sorted((unit_s(u, card, opts) for u in units[1:]), reverse=True):
        heapq.heappush(loads, heapq.heappop(loads) + t)
    return first + max(loads) + unit_s(final, card, opts)


def pool_plan(host: Host, opts: Options, cards: list[Card]) -> list[dict]:
    """One entry per card model among `cards`, as the host runner would build it."""
    models: dict[str, list[Card]] = {}
    for card in cards:
        models.setdefault(card.model, []).append(card)
    if not models:
        return []
    cover = max((c[0] for c in models.values()), key=lambda c: c.total_gib)
    out = []
    for model, group in models.items():
        card = group[0]
        cov = coverage_runs(host, opts, card) if card is cover else []
        queue = matrix.card_queue(host, opts, card, cov)
        units, final = matrix.units(queue)
        runs = [
            {
                "name": r.name,
                "kind": r.kind,
                "tags": r.tags,
                "minutes": run_s(r, card, opts) / 60,
            }
            for r in queue
        ]
        kinds: dict[str, int] = {}
        for r in queue:
            kinds[r.kind] = kinds.get(r.kind, 0) + 1
        serial = sum(run_s(r, card, opts) for r in queue) + opts.plan_tuning_s
        out.append(
            {
                "model": model,
                "cards": [c.index for c in group],
                "total_gib": card.total_gib,
                "sm": card.sm,
                "runs": len(queue),
                "units": len(units) + (1 if final else 0),
                "kinds": kinds,
                "serial_hours": serial / 3600,
                "wall_hours": wall_s(units, final, card, len(group), opts) / 3600,
                "queue": runs,
            }
        )
    return out


def markdown(plans: dict[str, dict]) -> str:
    """The plan of every host as markdown: cards, pools, run counts and hours, then
    every run of every pool."""
    lines = ["# Bench plan", ""]
    lines += [
        "| host | pool | cards | GiB | runs | speed | cap | coverage | quality+ | ncu "
        "| serial h | wall h |",
        "|" + " --- |" * 12,
    ]
    for name, p in plans.items():
        for pool in p["pools"]:
            k = pool["kinds"]
            quality = sum(k.get(x, 0) for x in ("quality", "export", "grid", "infer"))
            speed = sum(k.get(x, 0) for x in ("speed", "online", "restart"))
            lines.append(
                f"| {name} | {pool['model']} | {pool['cards']} | {pool['total_gib']:.0f} "
                f"| {pool['runs']} | {speed} | {k.get('cap', 0)} | {k.get('coverage', 0)} "
                f"| {quality} | {k.get('ncu', 0)} | {pool['serial_hours']:.1f} "
                f"| {pool['wall_hours']:.1f} |"
            )
    for name, p in plans.items():
        lines += ["", f"## {name}", ""]
        for card in p["cards"]:
            mark = "use" if card["use"] else "not used"
            lines.append(
                f"- card {card['index']} {card['name']} {card['total_gib']:.0f} GiB "
                f"sm{card['sm']}, {card['used_mib']:.0f} MiB in use now: {mark}"
            )
        for check, result in p.get("preflight", {}).items():
            lines.append(f"- preflight {check}: {result}")
        for pool in p["pools"]:
            lines += ["", f"### {pool['model']} x{len(pool['cards'])}", ""]
            lines += ["| run | kind | minutes |", "| --- | --- | --- |"]
            lines += [
                f"| {r['name']} | {r['kind']} | {r['minutes']:.1f} |"
                for r in pool["queue"]
            ]
    return "\n".join(lines) + "\n"
