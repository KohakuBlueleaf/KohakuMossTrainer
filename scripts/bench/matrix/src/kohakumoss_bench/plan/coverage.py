"""Feature coverage of the bench matrix against `FEATURES`: each feature names the
runs (a predicate over result records) or the model-free tests (repo globs) that
exercise it; the report marks it working / broken / does not fit / not run, or test
present / passed / failed. See docs/performance/bench-matrix.md#coverage.
"""

from collections.abc import Callable
from pathlib import Path


def kind(*kinds: str) -> Callable[[dict], bool]:
    return lambda r: r.get("kind") in kinds


def named(prefix: str) -> Callable[[dict], bool]:
    return lambda r: r.get("run", "").startswith(prefix)


def tag(key: str, *values) -> Callable[[dict], bool]:
    return lambda r: r.get("tags", {}).get(key) in values


def both(*preds) -> Callable[[dict], bool]:
    return lambda r: all(p(r) for p in preds)


TRAIN_KINDS = kind("speed", "online", "cap", "coverage", "quality", "restart")

FEATURES: list[tuple[str, str, object]] = [
    (
        "training",
        "LoRA (bypass over a frozen base)",
        both(TRAIN_KINDS, tag("algo", "lora32", "lora8", "lora")),
    ),
    (
        "training",
        "LoKr full matrix, factor 16",
        both(TRAIN_KINDS, tag("algo", "lokr16")),
    ),
    ("training", "LoCon", named("cov-algo-locon")),
    ("training", "LoHa", both(TRAIN_KINDS, tag("algo", "loha16"))),
    ("training", "LoKr decomposed", named("cov-algo-lokr-decompose")),
    ("training", "DoRA", both(TRAIN_KINDS, tag("algo", "dora16"))),
    ("training", "OFT", both(TRAIN_KINDS, tag("algo", "oft64"))),
    ("training", "BOFT", named("cov-algo-boft")),
    ("training", "GLoRA", named("cov-algo-glora")),
    ("training", "IA3", named("cov-algo-ia3")),
    ("training", "DyLoRA", named("cov-algo-dylora")),
    ("training", "T-LoRA", named("cov-algo-tlora")),
    ("training", "input / rank / module dropout", named("cov-algo-dropouts")),
    (
        "training",
        "full finetune (direct, no LyCORIS modules)",
        named("cov-training-full-direct"),
    ),
    ("training", "var-len packing at native resolution", kind("speed", "quality")),
    (
        "training",
        "bucket / shifted-square / square-crop planners",
        named("cov-data-res-"),
    ),
    ("training", "tokens-FFD / count batching", named("cov-data-batching-")),
    ("training", "flip + random crop", named("cov-data-flip-random-crop")),
    (
        "training",
        "kohya / folder / jsonl / parquet / webdataset / moss_shard sources",
        "tests/data/**/test_*source*.py",
    ),
    (
        "training",
        "single-file cached catalog (lazy, single writer)",
        "tests/data/**/test_*catalog*.py",
    ),
    ("training", "MOSS objective + timestep sampler", kind("speed", "quality")),
    ("training", "uniform / logit-normal timesteps", named("cov-training-timestep-")),
    (
        "training",
        "huber loss, x0 weighting",
        lambda r: r.get("run", "")
        in ("cov-training-loss-huber", "cov-training-weighting-x0"),
    ),
    ("training", "AdamW", kind("speed")),
    ("training", "8-bit AdamW", named("cov-training-adamw8bit")),
    ("training", "Prodigy", named("cov-training-prodigy")),
    ("training", "schedule-free", named("cov-training-schedule-free")),
    ("training", "Muon + AdamW", named("cov-training-muon")),
    ("training", "EMA", named("cov-training-ema")),
    (
        "training",
        "AnySchedule (constant, cosine)",
        lambda r: r.get("run", "") == "cov-training-cosine-schedule",
    ),
    (
        "training",
        "fp16 with the fix file",
        both(TRAIN_KINDS, tag("precision", "fp16"), tag("fp16_fix", "file")),
    ),
    (
        "training",
        "fp16 with the online fix",
        both(TRAIN_KINDS, tag("precision", "fp16"), tag("fp16_fix", "online")),
    ),
    ("training", "bf16", both(TRAIN_KINDS, tag("precision", "bf16"))),
    ("training", "offline latent / text caches", kind("speed")),
    ("training", "online caches (write-back to disk)", kind("online")),
    (
        "training",
        "sample images during training",
        lambda r: r.get("kind") == "quality"
        or r.get("run") == "cov-training-sample-during-training",
    ),
    ("training", "validation", named("cov-training-validation")),
    ("training", "checkpoint + resume", named("cov-training-resume")),
    (
        "training",
        "export LyCORIS + ComfyUI",
        lambda r: r.get("kind") in ("quality", "export"),
    ),
    (
        "training",
        "logging: colours, tables, progress bar, sinks",
        "tests/logging/**/test_*.py",
    ),
    (
        "memory",
        "named profiles (speed / balanced / low / minimal / auto)",
        both(
            kind("speed"), tag("profile", "speed", "balanced", "low", "minimal", "auto")
        ),
    ),
    (
        "memory",
        "catalog profiles",
        both(kind("speed"), lambda r: "-" in str(r.get("tags", {}).get("profile", ""))),
    ),
    ("memory", "manual knobs", named("cov-memory-manual-mode")),
    (
        "memory",
        "weight streaming without checkpointing",
        named("cov-memory-stream-no-checkpoint"),
    ),
    ("memory", "pageable host memory", named("cov-memory-pageable")),
    ("memory", "int8 weight-only base", both(TRAIN_KINDS, tag("base", "int8"))),
    ("memory", "fp8 weight-only base", named("cov-memory-base-fp8-weight-only")),
    ("memory", "fp8_compute base (DiT)", both(TRAIN_KINDS, tag("base", "fp8_compute"))),
    ("memory", "MXFP8 base (DiT)", both(TRAIN_KINDS, tag("base", "mxfp8"))),
    ("memory", "fp8 16-bit input gradient", named("cov-fp8-base-fp8-dgrad16")),
    ("memory", "text encoder fp8_compute", named("cov-fp8-text-fp8-compute")),
    (
        "memory",
        "TE / VAE offload and streaming (online encode)",
        named("cov-memory-online-encoders-"),
    ),
    ("memory", "8 GB training (caches)", both(kind("cap"), tag("cap_gib", 8.0))),
    ("memory", "6 GB training (caches)", both(kind("cap"), tag("cap_gib", 6.0))),
    ("memory", "tuning once, restart never re-tunes", kind("restart")),
    ("memory", "ncu tensor-pipe / throughput pass", kind("ncu")),
    ("memory", "batch scaling (MAX_TOKENS x2 / x4 / x8)", kind("batch")),
    ("memory", "step breakdown (profiled window)", kind("speed")),
    (
        "memory",
        "kernels vs PyTorch / compile (I/O + error stats)",
        "tests/kernels/**/test_*.py",
    ),
    ("memory", "torch.compile block mode", named("cov-training-compile-block")),
    ("memory", "LyCORIS kernels (torch backend)", named("cov-training-lycoris-torch")),
    ("inference", "pipeline: base model", named("infer-base")),
    ("inference", "own adapters merged", named("infer-merge")),
    ("inference", "own adapters in bypass", named("infer-bypass")),
    ("inference", "fp8 DiT + TE", named("infer-fp8")),
    ("inference", "memory profile + streamed TE + VAE offload", named("infer-lowmem")),
    ("inference", "comparison grids", kind("grid")),
]


def run_status(records: list[dict]) -> str:
    """working (one ran clean) / broken / does not fit (only out of memory) / not run."""
    if not records:
        return "not run"
    statuses = {r.get("status") for r in records}
    if "ok" in statuses:
        return "working"
    if statuses <= {"oom", "skipped"}:
        return "does not fit" if "oom" in statuses else "not run"
    return "broken"


def test_status(pattern: str, repo: Path, junit: dict[str, str]) -> tuple[str, list]:
    files = sorted(repo.glob(pattern))
    if not files:
        return "no test", []
    rel = [str(f.relative_to(repo)).replace("\\", "/") for f in files]
    seen = [junit[f] for f in rel if f in junit]
    if not seen:
        return "test present (not run)", rel
    return ("tests failed" if "failed" in seen else "tests passed"), rel


def coverage(records: list[dict], repo: Path, junit: dict[str, str]) -> list[dict]:
    """One row per feature: area, feature, evidence kind, status, evidence."""
    rows = []
    for area, feature, check in FEATURES:
        if isinstance(check, str):
            status, files = test_status(check, repo, junit)
            rows.append(
                {
                    "area": area,
                    "feature": feature,
                    "by": "test",
                    "status": status,
                    "evidence": files,
                }
            )
            continue
        hits = [r for r in records if check(r)]
        rows.append(
            {
                "area": area,
                "feature": feature,
                "by": "run",
                "status": run_status(hits),
                "evidence": [
                    f"{r.get('host')}/{r.get('card')}/{r.get('run')}: {r.get('status')}"
                    for r in hits
                ],
            }
        )
    return rows
