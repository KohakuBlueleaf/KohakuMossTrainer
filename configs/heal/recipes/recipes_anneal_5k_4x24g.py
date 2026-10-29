"""The annealed heal on four 24 GB cards: the tau 16 shrink of a calibration report, the
norms annealed exact -> plain over 3,000 steps with the calibrated gain ratios, o /
ffn_down trained in full and LoKr factor 8 on the readers, distillation only, 5,000
steps of 8 records at 512^2 (1 per rank x 2 micro-batches x 4 ranks), the teacher
streamed through a block ring, the student resident with block checkpointing. Drives
scripts/heal/heal.py; set TRAIN_CONFIG, TEACHER, REPORT, RECORDS and OUT.

    kogine run scripts/heal/heal.py --config configs/heal/recipes/recipes_anneal_5k_4x24g.py \
        --set TRAIN_CONFIG=my_run.py --set TEACHER=moss_image_base_bf16.safetensors \
        --set REPORT=heal/calibration.json --set RECORDS=heal/records --set OUT=heal/runs/heal5k
"""

TAU = 16.0
ANNEAL_STEPS = 3000
ANNEAL_RBAR = True
MODE = "full_writers"
LR_NORM = 2e-3
LR_LOKR = 5e-4
LR_FULL = 3e-5
STEPS = 5000
WARMUP = 100
MICRO = 1
GRAD_ACC = 2
MAX_TEXT_TOKENS = 1280
HELD = 32
EVAL_EVERY = 500
SAVE_EVERY = 1000
DEVICES = (0, 1, 2, 3)
