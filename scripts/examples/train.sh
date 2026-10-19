#!/usr/bin/env bash
# Example training run: a short LoRA / LoKr run on your dataset, from a generated config.
#
#   bash scripts/examples/train.sh --weights models/MOSS-Image-Base --dataset /data/my_set
#   bash scripts/examples/train.sh --algo lokr --precision fp16 --base-precision int8 --memory minimal
#   bash scripts/examples/train.sh --steps 30 --sample-every 10    # quick run with sample images
#   bash scripts/examples/train.sh --device xpu --index 1           # second Intel Arc GPU
#   bash scripts/examples/train.sh --tune-only                      # tune the training shapes, no weights loaded
#   bash scripts/examples/train.sh --dry-run                        # write and check the config, do not train
#
# Run it from the activated venv (kogine, kohakumoss-tune and python on PATH). --weights and
# --dataset default to $KOHAKUMOSS_WEIGHTS / $KOHAKUMOSS_DATASET. Every option of train.ps1 is a
# long flag here (--help lists them). The device variables are set for the child processes only.
set -euo pipefail

usage() {
  cat <<'EOF'
options (defaults in brackets; see train.ps1 for each one's meaning):
  --device cuda|xpu [cuda]   --index N [0]
  --weights DIR [$KOHAKUMOSS_WEIGHTS]   --dataset DIR [$KOHAKUMOSS_DATASET]
  --fix-file   --fix-checkpoint FILE    --source kohya|folder [kohya]
  --algo lora|lokr [lora]   --dim N [8]   --factor N [16]   --lr X [0 = algo default]
  --steps N [100]   --precision bf16|fp16 [bf16]
  --base-precision auto|compute|int8|fp8|fp8_compute|mxfp8 [auto]
  --memory PROFILE|auto|manual [auto]
  manual plan: --checkpoint MODE --stream off|ring --stream-resident N --pinned on|off
               --act-offload on|off --optim-state gpu|stream|cpu --grad-offload on|off
               --fused-step on|off
  --fp8-recipe row|tensor|block [row]   --fp8-dgrad fp8|16bit [16bit]
  --target-area PIXELS [1048576]   --max-tokens N [8192]
  --text-memory auto|ring|offload|resident [auto] ("stream" is accepted for ring)
  --cache off|online|offline [off]   --cache-dir DIR
  --sample-every N [0]   --sample-prompt TEXT   --save-every N [0]
  --tune table|off [table]   --tune-only   --kernel OP=BACKEND (repeatable)
  --seed N [0]   --log-every N [10]   --logger console|wandb [console]   --project NAME
  --out DIR [runs/examples/train]   --dry-run
EOF
}

# one allowed value list per choice option
choice() {
  local name=$1 value=$2
  shift 2
  for ok in "$@"; do [ "$value" = "$ok" ] && return 0; done
  echo "--$name: '$value' is not one of: $*" >&2
  exit 2
}

# Python string literal for the generated config
pystr() {
  local s=${1//\\/\\\\}
  s=${s//\'/\\\'}
  s=${s//$'\r'/\\r}
  s=${s//$'\n'/\\n}
  printf "'%s'" "$s"
}
# a path as Python sees it: Git Bash on Windows rewrites /c/x to C:/x, elsewhere unchanged
native() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s' "$1"; fi; }
pybool() { if [ "$1" = on ]; then printf True; else printf False; fi; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "$1 not on PATH: activate the KohakuMoss venv first" >&2; exit 1; }; }

device=cuda index=0
weights=${KOHAKUMOSS_WEIGHTS:-} fix_checkpoint="" dataset=${KOHAKUMOSS_DATASET:-}
source=kohya algo=lora dim=8 factor=16 lr=0 steps=100 precision=bf16 fix_file=0
base_precision=auto memory=auto
checkpoint="" stream="" stream_resident=0 pinned="" act_offload="" optim_state=""
grad_offload="" fused_step="" fp8_recipe=row fp8_dgrad=16bit
target_area=1048576 max_tokens=8192 text_memory=auto cache=off cache_dir=""
sample_every=0 sample_prompt="a lighthouse on a rocky coast at sunset" save_every=0
tune=table tune_only=0 kernels=() seed=0 log_every=10 logger=console project=kohakumoss
out=runs/examples/train dry_run=0

while [ $# -gt 0 ]; do
  case $1 in
    --fix-file | --tune-only | --dry-run | -h | --help) ;;
    *) [ $# -ge 2 ] || { echo "$1 needs a value" >&2; exit 2; } ;;
  esac
  case $1 in
    --device) choice device "$2" cuda xpu; device=$2; shift ;;
    --index) index=$2; shift ;;
    --weights) weights=$2; shift ;;
    --fix-checkpoint) fix_checkpoint=$2; shift ;;
    --dataset) dataset=$2; shift ;;
    --source) choice source "$2" kohya folder; source=$2; shift ;;
    --algo) choice algo "$2" lora lokr; algo=$2; shift ;;
    --dim) dim=$2; shift ;;
    --factor) factor=$2; shift ;;
    --lr) lr=$2; shift ;;
    --steps) steps=$2; shift ;;
    --precision) choice precision "$2" bf16 fp16; precision=$2; shift ;;
    --fix-file) fix_file=1 ;;
    --base-precision) choice base-precision "$2" auto compute int8 fp8 fp8_compute mxfp8; base_precision=$2; shift ;;
    --memory) memory=$2; shift ;;
    --checkpoint) checkpoint=$2; shift ;;
    --stream) choice stream "$2" off ring; stream=$2; shift ;;
    --stream-resident) stream_resident=$2; shift ;;
    --pinned) choice pinned "$2" on off; pinned=$2; shift ;;
    --act-offload) choice act-offload "$2" on off; act_offload=$2; shift ;;
    --optim-state) choice optim-state "$2" gpu stream cpu; optim_state=$2; shift ;;
    --grad-offload) choice grad-offload "$2" on off; grad_offload=$2; shift ;;
    --fused-step) choice fused-step "$2" on off; fused_step=$2; shift ;;
    --fp8-recipe) choice fp8-recipe "$2" row tensor block; fp8_recipe=$2; shift ;;
    --fp8-dgrad) choice fp8-dgrad "$2" fp8 16bit; fp8_dgrad=$2; shift ;;
    --target-area) target_area=$2; shift ;;
    --max-tokens) max_tokens=$2; shift ;;
    --text-memory) choice text-memory "$2" auto ring stream offload resident; text_memory=$2; shift ;;
    --cache) choice cache "$2" off online offline; cache=$2; shift ;;
    --cache-dir) cache_dir=$2; shift ;;
    --sample-every) sample_every=$2; shift ;;
    --sample-prompt) sample_prompt=$2; shift ;;
    --save-every) save_every=$2; shift ;;
    --tune) choice tune "$2" table off; tune=$2; shift ;;
    --tune-only) tune_only=1 ;;
    --kernel) kernels+=("$2"); shift ;;
    --seed) seed=$2; shift ;;
    --log-every) log_every=$2; shift ;;
    --logger) choice logger "$2" console wandb; logger=$2; shift ;;
    --project) project=$2; shift ;;
    --out) out=$2; shift ;;
    --dry-run) dry_run=1 ;;
    -h | --help) usage; exit 0 ;;
    *) echo "unknown option: $1 (--help lists them)" >&2; exit 2 ;;
  esac
  shift
done

repo=$(native "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)")
script=$repo/scripts/train/moss.py
cfg_root=$repo/configs/moss

# device selection for the child processes only (never exported into this shell)
case $device in
  cuda) run_env=(env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$index") ;;
  xpu) run_env=(env ZE_AFFINITY_MASK="$index" PYTHONUTF8=1) ;;
esac

if [ "$tune_only" = 1 ]; then
  need kohakumoss-tune
  # one square image of --target-area per sample (16 px per token), as many samples as --max-tokens packs
  side=$(python -c "import math; print(int(math.isqrt($target_area) // 16 * 16))")
  sample=$(((side / 16) * (side / 16) + 512))
  batch=$((max_tokens / sample))
  [ "$batch" -ge 1 ] || batch=1
  echo "== tune $device $precision ${side}x${side}, $batch per batch (forward + backward)"
  "${run_env[@]}" kohakumoss-tune --device "$device" --dtype "$precision" --sizes "${side}x${side}" \
    --batch "$batch" --backward --trainable adapter
  exit 0
fi

need kogine
need python
[ -n "$weights" ] || { echo "no weights: pass --weights DIR or set KOHAKUMOSS_WEIGHTS" >&2; exit 1; }
[ -n "$dataset" ] || { echo "no dataset: pass --dataset DIR or set KOHAKUMOSS_DATASET" >&2; exit 1; }
[ -e "$weights" ] || { echo "weights root not found: $weights" >&2; exit 1; }
[ -e "$dataset" ] || { echo "dataset not found: $dataset" >&2; exit 1; }
if [ "$fix_file" = 1 ]; then
  [ "$precision" = fp16 ] || { echo "--fix-file needs --precision fp16" >&2; exit 1; }
  [ -n "$fix_checkpoint" ] || { echo "--fix-file needs --fix-checkpoint FILE" >&2; exit 1; }
  [ -e "$fix_checkpoint" ] || { echo "fp16-fix checkpoint not found: $fix_checkpoint" >&2; exit 1; }
  fix_checkpoint=$(native "$(realpath -m "$fix_checkpoint")")
fi
weights=$(native "$(realpath -m "$weights")")
dataset=$(native "$(realpath -m "$dataset")")
out=$(native "$(realpath -m "$out")")

manual=()
[ -n "$checkpoint" ] && manual+=("CHECKPOINT = $(pystr "$checkpoint")")
[ -n "$stream" ] && manual+=("STREAM = '$stream'")
[ "$stream_resident" -gt 0 ] && manual+=("STREAM_RESIDENT = $stream_resident")
[ -n "$pinned" ] && manual+=("PINNED = $(pybool "$pinned")")
[ -n "$act_offload" ] && manual+=("ACT_OFFLOAD = $(pybool "$act_offload")")
[ -n "$optim_state" ] && manual+=("OPTIM_STATE = '$optim_state'")
[ -n "$grad_offload" ] && manual+=("GRAD_OFFLOAD = $(pybool "$grad_offload")")
[ -n "$fused_step" ] && manual+=("FUSED_STEP = $(pybool "$fused_step")")
if [ ${#manual[@]} -gt 0 ] && [ "$memory" != manual ]; then
  echo "--checkpoint / --stream / ... set the plan only with --memory manual (got --memory $memory)" >&2
  exit 1
fi

run_id=$(date +%Y%m%d-%H%M%S)
tag=$run_id-$device-$algo-$precision
[ "$fix_file" = 1 ] && tag=$tag-fixfile
run_dir=$out/$tag
mkdir -p "$run_dir"

if [ "$lr" != 0 ]; then lr_value=$lr; elif [ "$algo" = lokr ]; then lr_value=2e-3; else lr_value=5e-4; fi
cache_line=""
[ -n "$cache_dir" ] && cache_line="CACHE_DIR = $(pystr "$(native "$(realpath -m "$cache_dir")")")"
if [ "$source" = kohya ]; then
  dataset_line="{'source': 'kohya', 'path': $(pystr "$dataset"), 'repeat_prefix': 'first', 'artist_from_folder': 'name', 'txt_as': 'tags'}"
else
  dataset_line="{'source': 'folder', 'path': $(pystr "$dataset")}"
fi
if [ "$algo" = lokr ]; then
  algo_lines="use_config($(pystr "$cfg_root/algo/algo_lokr.py"))
DIM = 10000
ALPHA = 1.0
ALGO_KWARGS = {'factor': $factor, 'full_matrix': True}"
else
  algo_lines="use_config($(pystr "$cfg_root/algo/algo_lora.py"))
DIM = $dim
ALPHA = float($dim)"
fi
if [ "$precision" = fp16 ] && [ "$fix_file" = 1 ]; then
  precision_lines="PRECISION = 'fp16'
FP16_FIX = 'none'
WEIGHTS = $(pystr "$weights")
TRANSFORMER = $(pystr "$fix_checkpoint")"
else
  precision_lines="use_config($(pystr "$cfg_root/precision/precision_$precision.py"))
WEIGHTS = $(pystr "$weights")"
fi
override_lines="BASE_PRECISION = '$base_precision'
FP8_RECIPE = '$fp8_recipe'
FP8_DGRAD = '$fp8_dgrad'"
for line in ${manual[@]+"${manual[@]}"}; do override_lines+=$'\n'"$line"; done
kernel_line=""
if [ ${#kernels[@]} -gt 0 ]; then
  pairs=""
  for k in "${kernels[@]}"; do
    [[ $k == *=* ]] || { echo "--kernel takes op=backend, got $k" >&2; exit 1; }
    pairs+="${pairs:+, }$(pystr "${k%%=*}"): $(pystr "${k#*=}")"
  done
  kernel_line="KERNELS = {$pairs}"
fi
if [ "$sample_every" -gt 0 ]; then
  sample_lines="SAMPLE_EVERY = $sample_every
SAMPLE_PROMPTS = [$(pystr "$sample_prompt")]
SAMPLE_STEPS = 20
SAMPLE_SIZE = (1024, 1024)"
else
  sample_lines="SAMPLE_EVERY = 0"
fi
warmup=$((steps / 10))
[ "$warmup" -ge 1 ] || warmup=1
[ "$warmup" -le 20 ] || warmup=20
[ "$warmup" -le $((steps - 1)) ] || warmup=$((steps - 1))

cfg_path=$run_dir/config.py
cat >"$cfg_path" <<EOF
"""Generated by scripts/examples/train.sh ($tag); drives scripts/train/moss.py."""

from kohakuengine import use_config

use_config($(pystr "$cfg_root/_base/base_lora.py"))
$algo_lines
$precision_lines

DATASETS = [$dataset_line]
$cache_line
LATENT_CACHE = '$cache'
TEXT_CACHE = '$cache'
TEXT_ENCODE = {'memory': '$text_memory', 'ahead': 8}
MEMORY_PROFILE = $(pystr "$memory")
$override_lines
$kernel_line
TARGET_AREA = $target_area
MAX_TOKENS = $max_tokens
LR = $lr_value
OPTIMIZER = 'adamw'
STEPS = $steps
SCHEDULE = {'mode': 'constant', 'warmup': $warmup, 'end': -1}
SAVE_EVERY = $save_every
KEEP_LAST = 0
SEED = $seed
TUNE = '$tune'
LOG_EVERY = $log_every
LOGGER = '$logger'
PROJECT = $(pystr "$project")
$sample_lines
OUTPUT_DIR = $(pystr "$out")
RUN_NAME = '$tag'
EOF

suffix=""
[ "$fix_file" = 1 ] && suffix=" (fix file)"
echo "== $device $algo $precision$suffix -> $run_dir"
echo "   config: $cfg_path"
# validate every knob against the trainer's rules before loading any weights
(cd "$repo" && "${run_env[@]}" python -m kohakumoss.cli.config.main check "$cfg_path") ||
  { echo "config check failed for $cfg_path" >&2; exit 1; }
if [ "$dry_run" = 1 ]; then
  echo "   dry run: kogine run $script --config $cfg_path"
  exit 0
fi
start=$SECONDS
code=0
"${run_env[@]}" kogine run "$script" --config "$cfg_path" || code=$?
echo "   wall $((SECONDS - start)) s, exit $code"
[ "$code" = 0 ] || { echo "training run failed ($tag), exit $code" >&2; exit "$code"; }
