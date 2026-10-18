"""Feature-coverage runs that the speed / quality matrix does not already exercise: the
other LyCORIS algorithms (DoRA / LoHa / OFT are speed rows), optimisers, objectives,
data modes, memory modes (int8 / MXFP8 bases are speed rows), drivers, checkpoint /
resume, online caches with offloaded encoders. Short training runs on the host's
coverage pool (its largest card). See docs/performance/bench-matrix.md#coverage.
"""

from kohakumoss_bench.plan.config import Card, Host, Options, Run
from kohakumoss_bench.plan.knobs import TRAIN, train_knobs
from kohakumoss_bench.plan.matrix import estimated_gib, fits, precisions

ALGO_CASES = {
    "locon": {"ALGO": "locon", "DIM": 16, "ALPHA": 16.0},
    "lokr-decompose": {
        "ALGO": "lokr",
        "DIM": 16,
        "ALPHA": 16.0,
        "ALGO_KWARGS": {"factor": 8, "decompose_both": True},
    },
    "boft": {"ALGO": "boft", "DIM": 64, "ALPHA": 1.0},
    "glora": {"ALGO": "glora", "DIM": 16, "ALPHA": 16.0},
    "ia3": {"ALGO": "ia3", "LR": 5e-3},
    "dylora": {"ALGO": "dylora", "DIM": 32, "ALPHA": 32.0},
    "tlora": {"ALGO": "tlora", "DIM": 16, "ALPHA": 16.0},
    "dropouts": {"DROPOUT": {"input": 0.1, "rank": 0.1, "module": 0.1}},
}

TRAINING_CASES = {
    "adamw8bit": {"OPTIMIZER": "adamw8bit"},
    "prodigy": {"OPTIMIZER": "prodigy", "LR": 1.0},
    "schedule-free": {"OPTIMIZER": "schedule_free"},
    "muon": {"OPTIMIZER": "muon", "LR": 2e-4},
    "ema": {"EMA": 0.999},
    "timestep-uniform": {"TIMESTEP": "uniform", "TIMESTEP_KWARGS": {}},
    "timestep-logitnormal": {"TIMESTEP": "logit_normal", "TIMESTEP_KWARGS": {}},
    "loss-huber": {"LOSS": "huber"},
    "weighting-x0": {"WEIGHTING": "x0"},
    "cosine-schedule": {"SCHEDULE": {"mode": "cosine", "warmup": 5, "end": -1}},
    "lightning": {"DRIVER": "lightning"},
    "compile-block": {"COMPILE": "block"},
    "lycoris-torch": {"KERNELS": {"lycoris": "torch"}},
    "export-formats": {"EXPORT_FORMATS": ("comfy", "lycoris")},
    "sample-during-training": {
        "SAMPLE_EVERY": 10,
        "SAMPLE_PROMPTS": "{sample_prompts}",
        "SAMPLE_STEPS": 8,
    },
    "validation": {"VALIDATE_EVERY": 10, "VALIDATION": {"sigmas": [0.5], "seed": 0}},
}

DATA_CASES = {
    "res-bucket": {"RESOLUTION_MODE": "bucket", "LATENT_CACHE": "off"},
    "res-shifted-square": {"RESOLUTION_MODE": "shifted_square", "LATENT_CACHE": "off"},
    "res-square-crop": {"RESOLUTION_MODE": "square_crop", "LATENT_CACHE": "off"},
    "batching-ffd": {"BATCHING": "tokens_ffd"},
    "batching-count": {"BATCHING": "count", "BATCH_SIZE": 2},
    "flip-random-crop": {"FLIP": True, "CROP": "random", "LATENT_CACHE": "off"},
    "token-budget-render": {"TOKEN_BUDGET": "render"},
}

MEMORY_CASES = {
    "manual-mode": {
        "MEMORY_PROFILE": "manual",
        "CHECKPOINT": "block",
        "STREAM": "ring",
        "STREAM_RESIDENT": 4,
        "ACT_OFFLOAD": True,
        "OPTIM_STATE": "stream",
        "FUSED_STEP": True,
    },
    "stream-no-checkpoint": {
        "MEMORY_PROFILE": "manual",
        "CHECKPOINT": "none",
        "STREAM": "ring",
        "STREAM_RESIDENT": 8,
    },
    "pageable": {"MEMORY_PROFILE": "low", "MEMORY_OVERRIDES": {"pinned": False}},
    "base-fp8-weight-only": {"BASE_PRECISION": "fp8"},
    "online-encoders-offloaded": {
        "LATENT_CACHE": "online",
        "TEXT_CACHE": "online",
        "CACHE_DIR": "{self}/cache",
        "VAE_ENCODE": {"memory": "offload"},
        "TEXT_ENCODE": {"memory": "offload"},
    },
    "online-encoders-streamed": {
        "LATENT_CACHE": "online",
        "TEXT_CACHE": "online",
        "CACHE_DIR": "{self}/cache",
        "VAE_ENCODE": {"memory": "ring"},
        "TEXT_ENCODE": {"memory": "ring"},
    },
}

FP8_CASES = {
    "base-fp8-dgrad16": {"BASE_PRECISION": "fp8_compute", "FP8_DGRAD": "16bit"},
    "text-fp8-compute": {
        "TEXT_CACHE": "online",
        "CACHE_DIR": "{self}/cache",
        "TEXT_ENCODE": {"memory": "ring", "precision": "fp8_compute"},
    },
}

GROUPS = {
    "algo": ALGO_CASES,
    "training": TRAINING_CASES,
    "data": DATA_CASES,
    "memory": MEMORY_CASES,
}
RESUME_STEP = 10


def _profile(card: Card, opts: Options) -> str:
    for name in ("balanced", "low", "minimal"):
        if fits(card, estimated_gib(name, "compute", "lora32", opts), opts):
            return name
    return "minimal"


def _run(host, opts, card, group, case, extra, profile) -> Run:
    precision = precisions(card)[0]
    knobs = train_knobs(host, opts, precision, "lora32")
    knobs.update(MEMORY_PROFILE=profile, BASE_PRECISION="compute")
    knobs.update(extra)
    if group == "training" and case == "validation":
        knobs["VALIDATION"] = {**knobs["VALIDATION"], "datasets": list(host.datasets)}
    return Run(
        name=f"cov-{group}-{case}",
        kind="coverage",
        script=TRAIN,
        knobs=knobs,
        timeout_s=opts.timeout_speed_s,
        tags={
            "precision": precision,
            "base": knobs["BASE_PRECISION"],
            "algo": knobs.get("ALGO", "lora"),
            "profile": knobs["MEMORY_PROFILE"],
            "group": group,
            "case": case,
        },
    )


def full_finetune_run(host: Host, opts: Options, card: Card) -> Run | None:
    """Direct full finetune under `minimal` (masters and states on the host) where the
    host RAM holds them (fp32 masters + AdamW states, 16 bytes a parameter)."""
    params = 5.1e9
    if host.ram_gib and host.ram_gib * 1024**3 < 1.5 * 16 * params:
        return None
    precision = precisions(card)[0]
    knobs = train_knobs(host, opts, precision, "lora32")
    for name in ("DIM", "ALPHA"):
        knobs.pop(name)
    knobs.update(ALGO="full", LR=1e-5, MEMORY_PROFILE="minimal", OPTIMIZER="adamw")
    return Run(
        name="cov-training-full-direct",
        kind="coverage",
        script=TRAIN,
        knobs=knobs,
        timeout_s=opts.timeout_other_s,
        tags={
            "precision": precision,
            "base": "compute",
            "algo": "full",
            "profile": "minimal",
            "group": "training",
            "case": "full-direct",
        },
    )


def resume_runs(host: Host, opts: Options, card: Card, profile: str) -> list:
    """A run saving at `RESUME_STEP` and its resume to `2 * RESUME_STEP`."""
    first = _run(
        host,
        opts,
        card,
        "training",
        "checkpoint",
        {"STEPS": 2 * RESUME_STEP, "SAVE_EVERY": RESUME_STEP},
        profile,
    )
    second = _run(
        host,
        opts,
        card,
        "training",
        "resume",
        {
            "STEPS": 2 * RESUME_STEP,
            "RESUME": f"{{run:{first.name}}}/train/checkpoints/step_{RESUME_STEP:08d}",
        },
        profile,
    )
    second.after = first.name
    return [first, second]


def coverage_runs(host: Host, opts: Options, card: Card) -> list[Run]:
    """Every coverage run for the host's coverage card."""
    profile = _profile(card, opts)
    runs = [
        _run(host, opts, card, group, case, extra, profile)
        for group, cases in GROUPS.items()
        for case, extra in cases.items()
    ]
    if card.fp8:
        runs += [
            _run(host, opts, card, "fp8", case, extra, profile)
            for case, extra in FP8_CASES.items()
        ]
    runs += resume_runs(host, opts, card, profile)
    full = full_finetune_run(host, opts, card)
    if full is not None:
        runs.append(full)
    return runs
