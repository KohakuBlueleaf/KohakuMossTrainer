"""Heal the massive channels of a MOSS-Image DiT by distillation from the original model,
one process per device in DEVICES (data parallel). Model, kernel and memory knobs come
from TRAIN_CONFIG; the heal knobs are below. Writes OUT/heal.json, metrics.jsonl,
evals.jsonl, grads.json, step_<n>/ snapshots and final/ (scripts/heal/merge.py makes a
DiT file of a parts directory). See docs/concepts/massive-channels.md.

    kogine run scripts/heal/heal.py --config configs/heal/recipes/recipes_anneal_5k_4x24g.py \
        --set TRAIN_CONFIG=my_run.py --set TEACHER=moss_image_base_bf16.safetensors
"""

import json
from pathlib import Path

from kohakumoss.heal.config import from_knobs, problems
from kohakumoss.heal.student import heal_knobs
from kohakumoss.heal.trainer import run
from kohakumoss.logging.log import get_logger, setup
from kohakumoss.parallel.launch import spawn
from kohakumoss.types.plans import ParallelPlan
from kohakumoss_common.knobs import check_script

log = get_logger("scripts.heal")

# ===================== MODELS =====================
TRAIN_CONFIG = ""  # training config: VARIANT, KERNELS, KEEP_FP32 and the memory knobs
TEACHER = ""  # the original DiT file
STUDENT = ""  # the DiT the student starts from ("" = TEACHER)
REPORT = ""  # calibration report (scripts/heal/calibrate.py); "" = no shrink
TAU = 16.0  # the report's tau whose shrink the student takes
TEACHER_MEMORY = {"STREAM": "ring", "STREAM_RESIDENT": 2}  # memory knobs of the teacher
STUDENT_MEMORY = {"STREAM": "off", "CHECKPOINT": "block"}  # memory knobs of the student

# ===================== ANNEAL =====================
ANNEAL_STEPS = 3000  # steps over which t goes 0 -> 1 (0: plain norms from the start)
ANNEAL_RBAR = True  # gains follow w / (D r^t), r the calibrated divisor ratio

# ===================== TRAINABLES =====================
MODE = "full_writers"  # "full_writers" (o / ffn_down full + LoKr readers) | "lokr" (LoKr on LOKR_TARGET)
LOKR_TARGET = "readers"  # MODE lokr: "readers" | "writers" | "all"
LOKR_FACTOR = 8
LR_NORM = 2e-3  # norm-gain deltas (AdamW)
LR_LOKR = 5e-4  # LoKr (AdamW)
LR_FULL = 3e-5  # full writers (AdamW16, stochastic rounding)

# ===================== LOSS =====================
HARD_WEIGHT = 0.0  # + weight x relative error against the flow target
CONC_WEIGHT = 0.0  # + weight x the top-CONC_K channel share of the block outputs
CONC_K = 8

# ===================== SCHEDULE =====================
STEPS = 5000
WARMUP = 100
MICRO = 1  # records per micro-batch per rank
GRAD_ACC = 2
TEXT_DROP = 0.1
MAX_TEXT_TOKENS = 1280  # caption rows kept per record (0 = all)
SEED = 0

# ===================== DATA / EVAL / OUTPUT =====================
RECORDS = ""  # heal records (scripts/heal/cache.py)
HELD = 32  # rows below HELD are the held-out set
EVAL_EVERY = 500
SAVE_EVERY = 1000
GRAD_MAP = True  # gradient map of student vs teacher at the start and the end
OUT = ""
DEVICES = (0,)
TRAIN_SCRIPT = Path(__file__).resolve().parent.parent / "train" / "moss.py"


def main() -> None:
    setup()
    g = check_script(globals())
    c = from_knobs(g)
    found = problems(c)
    if found:
        raise SystemExit("invalid heal configuration:\n  - " + "\n  - ".join(found))
    k, k_t = heal_knobs(TRAIN_SCRIPT, TRAIN_CONFIG, c)
    out = Path(c.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "heal.json").write_text(json.dumps(c.record(), indent=1), encoding="utf-8")
    spawn(
        run, ParallelPlan(kind="ddp", devices=tuple(int(d) for d in DEVICES)), c, k, k_t
    )
    log.info(f"heal done: {out}")


if __name__ == "__main__":
    main()
