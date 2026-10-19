#!/usr/bin/env bash
# Example comparison grid: the base model vs trained adapters (LoRA / LoKr exports or checkpoints).
# Columns = prompts, rows = base, then each adapter at each multiplier (--layout prompt-rows swaps
# them). Same seed per prompt across rows.
#
#   bash scripts/examples/grid.sh --adapter runs/examples/train/<run>/export/<run>-lycoris.safetensors --prompt "a lighthouse at dusk"
#   bash scripts/examples/grid.sh --adapter a.safetensors --adapter b.safetensors --multiplier 0.5 --multiplier 1.0 --prompt "a cat"
#   bash scripts/examples/grid.sh --run <run> --prompt "a cat"                  # the train example run's lycoris export
#   bash scripts/examples/grid.sh --run <run> --checkpoints --prompt "a cat"    # every saved checkpoint of the run
#   bash scripts/examples/grid.sh --run <run> --dataset <dir> --artist <name>   # captions of that artist's folder
#
# Run it from the activated venv (kogine and python on PATH). --weights defaults to
# $KOHAKUMOSS_WEIGHTS, --dataset to $KOHAKUMOSS_DATASET. Every option of grid.ps1 is a long
# flag here (--help lists them). The device variables are set for the child processes only.
set -euo pipefail

usage() {
  cat <<'EOF'
options (defaults in brackets; see grid.ps1 for each one's meaning):
  --device cuda|xpu [cuda]   --index N [0]   --weights DIR [$KOHAKUMOSS_WEIGHTS]
  --variant base|sft [base]   --transformer FILE
  --adapter FILE (repeatable)   --run NAME   --checkpoints   --train-out DIR [runs/examples/train]
  --multiplier X (repeatable) [1.0]   --adapter-mode bypass|merge [bypass]
  --prompt TEXT (repeatable)   --dataset DIR [$KOHAKUMOSS_DATASET]   --artist NAME (repeatable; - = none)
  --captions N [1]   --suffix TEXT   --negative TEXT [blurry, distortion, lowres]
  --size HxW [1024x1024]   --steps N [30]   --cfg X [3.0]   --seed N [0]
  --precision bf16|fp16 [bf16]   --fp16-fix online|fp32_residual|none [online]
  --base-precision auto|compute|int8|fp8|fp8_compute|mxfp8 [auto]   --fp8-recipe row|tensor|block [row]
  --memory auto|PROFILE|"" [auto; "" = whole DiT resident]   --text-memory auto|resident|offload|ring|cpu [auto]
  --text-precision compute|int8|fp8|fp8_compute|mxfp8 [compute]   --free-text-encoder
  --vae-memory auto|resident|offload|ring|cpu [auto]   --vae-dtype auto|bf16|fp16|fp32 [auto]
  --batch N [4]   --tune off|table [off]   --layout prompt-cols|prompt-rows [prompt-cols]
  --font-size N [0]   --out DIR [runs/examples/grid]   --dry-run
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
need() { command -v "$1" >/dev/null 2>&1 || { echo "$1 not on PATH: activate the KohakuMoss venv first" >&2; exit 1; }; }

device=cuda index=0 weights=${KOHAKUMOSS_WEIGHTS:-} variant=base transformer=""
adapters=() run="" checkpoints=0 train_out=runs/examples/train multipliers=() adapter_mode=bypass
prompts=() dataset=${KOHAKUMOSS_DATASET:-} artists=() captions=1 suffix=""
negative="blurry, distortion, lowres" size=1024x1024 steps=30 cfg=3.0 seed=0
precision=bf16 fp16_fix=online base_precision=auto fp8_recipe=row memory=auto
text_memory=auto text_precision=compute free_text=0 vae_memory=auto vae_dtype=auto
batch=4 tune=off layout=prompt-cols font_size=0 out=runs/examples/grid dry_run=0

while [ $# -gt 0 ]; do
  case $1 in
    --checkpoints | --free-text-encoder | --dry-run | -h | --help) ;;
    *) [ $# -ge 2 ] || { echo "$1 needs a value" >&2; exit 2; } ;;
  esac
  case $1 in
    --device) choice device "$2" cuda xpu; device=$2; shift ;;
    --index) index=$2; shift ;;
    --weights) weights=$2; shift ;;
    --variant) choice variant "$2" base sft; variant=$2; shift ;;
    --transformer) transformer=$2; shift ;;
    --adapter) adapters+=("$2"); shift ;;
    --run) run=$2; shift ;;
    --checkpoints) checkpoints=1 ;;
    --train-out) train_out=$2; shift ;;
    --multiplier) multipliers+=("$2"); shift ;;
    --adapter-mode) choice adapter-mode "$2" bypass merge; adapter_mode=$2; shift ;;
    --prompt) prompts+=("$2"); shift ;;
    --dataset) dataset=$2; shift ;;
    --artist) artists+=("$2"); shift ;;
    --captions) captions=$2; shift ;;
    --suffix) suffix=$2; shift ;;
    --negative) negative=$2; shift ;;
    --size) size=$2; shift ;;
    --steps) steps=$2; shift ;;
    --cfg) cfg=$2; shift ;;
    --seed) seed=$2; shift ;;
    --precision) choice precision "$2" bf16 fp16; precision=$2; shift ;;
    --fp16-fix) choice fp16-fix "$2" online fp32_residual none; fp16_fix=$2; shift ;;
    --base-precision) choice base-precision "$2" auto compute int8 fp8 fp8_compute mxfp8; base_precision=$2; shift ;;
    --fp8-recipe) choice fp8-recipe "$2" row tensor block; fp8_recipe=$2; shift ;;
    --memory) memory=$2; shift ;;
    --text-memory) choice text-memory "$2" auto resident offload ring stream cpu; text_memory=$2; shift ;;
    --text-precision) choice text-precision "$2" compute int8 fp8 fp8_compute mxfp8; text_precision=$2; shift ;;
    --free-text-encoder) free_text=1 ;;
    --vae-memory) choice vae-memory "$2" auto resident offload ring stream cpu; vae_memory=$2; shift ;;
    --vae-dtype) choice vae-dtype "$2" auto bf16 fp16 fp32; vae_dtype=$2; shift ;;
    --batch) batch=$2; shift ;;
    --tune) choice tune "$2" off table; tune=$2; shift ;;
    --layout) choice layout "$2" prompt-cols prompt-rows; layout=$2; shift ;;
    --font-size) font_size=$2; shift ;;
    --out) out=$2; shift ;;
    --dry-run) dry_run=1 ;;
    -h | --help) usage; exit 0 ;;
    *) echo "unknown option: $1 (--help lists them)" >&2; exit 2 ;;
  esac
  shift
done
[ ${#multipliers[@]} -gt 0 ] || multipliers=(1.0)

repo=$(native "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)")
script=$repo/scripts/sample/compare_grid.py
helper=$repo/scripts/examples/grid_prompts.py

# device selection for the child processes only (never exported into this shell)
case $device in
  cuda) run_env=(env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$index") ;;
  xpu) run_env=(env ZE_AFFINITY_MASK="$index" PYTHONUTF8=1) ;;
esac

need kogine
need python
[ -n "$weights" ] || { echo "no weights: pass --weights DIR or set KOHAKUMOSS_WEIGHTS" >&2; exit 1; }
[ -e "$weights" ] || { echo "weights root not found: $weights" >&2; exit 1; }
if [ -n "$transformer" ]; then
  [ -e "$transformer" ] || { echo "transformer not found: $transformer" >&2; exit 1; }
  transformer=$(native "$(realpath -m "$transformer")")
fi
weights=$(native "$(realpath -m "$weights")")
out=$(native "$(realpath -m "$out")")

# grid lines: explicit adapters, then the run's export or its checkpoints
files=() labels=()
for a in ${adapters[@]+"${adapters[@]}"}; do
  [ -e "$a" ] || { echo "adapter not found: $a" >&2; exit 1; }
  files+=("$(native "$(realpath -m "$a")")")
  name=${a##*/}
  labels+=("${name%.*}")
done
if [ -n "$run" ]; then
  run_dir=$(realpath -m "$train_out")/$run
  [ -d "$run_dir" ] || { echo "run not found: $run_dir" >&2; exit 1; }
  if [ "$checkpoints" = 1 ]; then
    found=0
    for c in "$run_dir"/checkpoints/step_*/; do
      [ -e "$c/trainable.safetensors" ] || continue
      step=${c%/}
      step=${step##*/step_}
      files+=("$(native "${c}trainable.safetensors")")
      labels+=("step $((10#$step))")
      found=1
    done
    [ "$found" = 1 ] || { echo "no checkpoints in $run_dir (the run saved none: --save-every 0)" >&2; exit 1; }
  else
    f=$run_dir/export/$run-lycoris.safetensors
    [ -e "$f" ] || { echo "export not found: $f" >&2; exit 1; }
    files+=("$(native "$f")")
    label=$run
    [[ $run =~ ^[0-9]{8}-[0-9]{6}-(.+)$ ]] && label=${BASH_REMATCH[1]}
    labels+=("$label")
  fi
fi
[ ${#files[@]} -gt 0 ] || { echo "no adapters: pass --adapter or --run" >&2; exit 1; }

# one grid line per (adapter, multiplier); labels carry the multiplier when there are several
cols="" col_labels=""
for i in "${!files[@]}"; do
  for m in "${multipliers[@]}"; do
    cols+="${cols:+, }($(pystr "${files[$i]}"), $m)"
    label=${labels[$i]}
    [ ${#multipliers[@]} -gt 1 ] && label="$label x$m"
    col_labels+="${col_labels:+, }$(pystr "$label")"
  done
done

run_id=$(date +%Y%m%d-%H%M%S)
dir=$out/$run_id-$device-grid
mkdir -p "$dir"

# prompts: per artist, the --prompt bodies or --captions captions of its dataset folder
artist_list="" body_list=""
for a in ${artists[@]+"${artists[@]}"}; do
  [ "$a" = - ] && a=""
  artist_list+="${artist_list:+, }$(pystr "$a")"
done
for p in ${prompts[@]+"${prompts[@]}"}; do body_list+="${body_list:+, }$(pystr "$p")"; done
dataset_path=""
[ -n "$dataset" ] && dataset_path=$(native "$(realpath -m "$dataset")")
prompts_json=$dir/prompts.json
prompts_cfg=$dir/prompts_config.py
cat >"$prompts_cfg" <<EOF
"""Generated by scripts/examples/grid.sh ($run_id); drives scripts/examples/grid_prompts.py."""

DATASET = $(pystr "$dataset_path")
ARTISTS = [$artist_list]
PROMPTS = [$body_list]
CAPTIONS = $captions
SEED = $seed
SUFFIX = $(pystr "$suffix")
OUT = $(pystr "$prompts_json")
EOF
kogine run "$helper" --config "$prompts_cfg" || { echo "no prompts (grid_prompts.py failed)" >&2; exit 1; }
rows=$(<"$prompts_json")

fix=none fix_kw="{}"
if [ "$precision" = fp16 ] && [ -z "$transformer" ]; then fix=$fp16_fix; fi
[ "$fix" = online ] && fix_kw="{'c': 0.0625}"
text="{'memory': '$text_memory', 'precision': '$text_precision'"
if [ "$free_text" = 1 ]; then text+=", 'free_after_use': True}"; else text+="}"; fi

cfg_path=$dir/config.py
cat >"$cfg_path" <<EOF
"""Generated by scripts/examples/grid.sh ($run_id); drives scripts/sample/compare_grid.py."""

WEIGHTS = $(pystr "$weights")
VARIANT = '$variant'
TRANSFORMER = $(pystr "$transformer")
CHECKPOINTS = [$cols]
LABELS = [$col_labels]
ADAPTER_MODE = '$adapter_mode'
PRECISION = '$precision'
FP16_FIX = '$fix'
FP16_FIX_KWARGS = $fix_kw
TUNE = '$tune'

DEVICE = '$device'
TEXT_ENCODE = $text
VAE_ENCODE = {'memory': '$vae_memory'}
VAE_DTYPE = '$vae_dtype'
MEMORY_PROFILE = $(pystr "$memory")
BASE_PRECISION = '$base_precision'
FP8_RECIPE = '$fp8_recipe'

PROMPTS = $rows
NEGATIVE = $(pystr "$negative")
SIZES = [$(pystr "$size")]
SEED_BASE = $seed

STEPS = $steps
CFG = $cfg
BATCH = $batch

OUT = $(pystr "$dir")
GRID_LAYOUT = '$layout'
FONT_SIZE = $font_size
EOF

lines=$(( ${#files[@]} * ${#multipliers[@]} + 1 ))
echo "== $device grid: $lines weight lines -> $dir"
(cd "$repo" && "${run_env[@]}" python -m kohakumoss.cli.config.main check "$cfg_path") ||
  { echo "config check failed for $cfg_path" >&2; exit 1; }
if [ "$dry_run" = 1 ]; then
  echo "   dry run: kogine run $script --config $cfg_path"
  exit 0
fi
start=$SECONDS
code=0
"${run_env[@]}" kogine run "$script" --config "$cfg_path" || code=$?
[ "$code" = 0 ] || { echo "grid run failed, exit $code" >&2; exit "$code"; }
echo "   wall $((SECONDS - start)) s; grid at $dir/grid.png"
