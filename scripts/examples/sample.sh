#!/usr/bin/env bash
# Example sampling run: images from the base weights (and adapters) through a generated config.
#
#   bash scripts/examples/sample.sh --weights models/MOSS-Image-Base
#   bash scripts/examples/sample.sh --prompt "a fox in snow" --size 832x1216 --steps 30 --count 4
#   bash scripts/examples/sample.sh --adapter runs/examples/train/<run>/export/<run>-lycoris.safetensors:0.8
#   bash scripts/examples/sample.sh --precision fp16 --transformer <fp16-fix .safetensors> --compare
#   bash scripts/examples/sample.sh --memory "" --base-precision fp8_compute  # whole DiT resident, fp8 GEMMs (sm89+)
#   bash scripts/examples/sample.sh --tune-only                                 # tune the sampling shapes, no weights loaded
#
# Run it from the activated venv (kogine, kohakumoss-tune and python on PATH). --weights
# defaults to $KOHAKUMOSS_WEIGHTS. Every option of sample.ps1 is a long flag here (--help
# lists them). The device variables are set for the child processes only.
set -euo pipefail

usage() {
  cat <<'EOF'
options (defaults in brackets; see sample.ps1 for each one's meaning):
  --device cuda|xpu [cuda]   --index N [0]   --weights DIR [$KOHAKUMOSS_WEIGHTS]
  --variant base|sft [base]   --transformer FILE   --compare
  --prompt TEXT   --negative TEXT [blurry, distortion]   --size HxW [1024x1024]
  --steps N [50]   --cfg X [3.0]   --seed N [0]   --count N [4]   --batch N [4]   --prompts-file FILE
  --precision bf16|fp16 [bf16]   --fp16-fix online|fp32_residual|none [online]
  --text-memory auto|resident|offload|ring|cpu [auto]
  --text-precision compute|int8|fp8|fp8_compute|mxfp8 [compute]   --text-fp8-recipe row|tensor|block [row]
  --free-text-encoder
  --base-precision auto|compute|int8|fp8|fp8_compute|mxfp8 [auto]   --fp8-recipe row|tensor|block [row]
  --memory auto|PROFILE|"" [auto; "" = whole DiT resident]   --vae-memory auto|resident|offload|ring|cpu [auto]
  --vae-dtype auto|bf16|fp16|fp32 [auto]
  --adapter PATH[:MULT] (repeatable)   --adapter-mode merge|bypass [bypass]
  --tune off|table [off]   --tune-only   --out DIR [runs/examples/sample]   --dry-run
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

device=cuda index=0 weights=${KOHAKUMOSS_WEIGHTS:-} variant=base transformer="" compare=0
prompt="a lighthouse on a rocky coast at sunset" negative="blurry, distortion" size=1024x1024
steps=50 cfg=3.0 seed=0 count=4 batch=4 prompts_file=""
precision=bf16 fp16_fix=online text_memory=auto text_precision=compute text_fp8_recipe=row
free_text=0 base_precision=auto fp8_recipe=row memory=auto vae_memory=auto vae_dtype=auto
adapters=() adapter_mode=bypass tune=off tune_only=0 out=runs/examples/sample dry_run=0

while [ $# -gt 0 ]; do
  case $1 in
    --compare | --free-text-encoder | --tune-only | --dry-run | -h | --help) ;;
    *) [ $# -ge 2 ] || { echo "$1 needs a value" >&2; exit 2; } ;;
  esac
  case $1 in
    --device) choice device "$2" cuda xpu; device=$2; shift ;;
    --index) index=$2; shift ;;
    --weights) weights=$2; shift ;;
    --variant) choice variant "$2" base sft; variant=$2; shift ;;
    --transformer) transformer=$2; shift ;;
    --compare) compare=1 ;;
    --prompt) prompt=$2; shift ;;
    --negative) negative=$2; shift ;;
    --size) size=$2; shift ;;
    --steps) steps=$2; shift ;;
    --cfg) cfg=$2; shift ;;
    --seed) seed=$2; shift ;;
    --count) count=$2; shift ;;
    --batch) batch=$2; shift ;;
    --prompts-file) prompts_file=$2; shift ;;
    --precision) choice precision "$2" bf16 fp16; precision=$2; shift ;;
    --fp16-fix) choice fp16-fix "$2" online fp32_residual none; fp16_fix=$2; shift ;;
    --text-memory) choice text-memory "$2" auto resident offload ring stream cpu; text_memory=$2; shift ;;
    --text-precision) choice text-precision "$2" compute int8 fp8 fp8_compute mxfp8; text_precision=$2; shift ;;
    --text-fp8-recipe) choice text-fp8-recipe "$2" row tensor block; text_fp8_recipe=$2; shift ;;
    --free-text-encoder) free_text=1 ;;
    --base-precision) choice base-precision "$2" auto compute int8 fp8 fp8_compute mxfp8; base_precision=$2; shift ;;
    --fp8-recipe) choice fp8-recipe "$2" row tensor block; fp8_recipe=$2; shift ;;
    --memory) memory=$2; shift ;;
    --vae-memory) choice vae-memory "$2" auto resident offload ring stream cpu; vae_memory=$2; shift ;;
    --vae-dtype) choice vae-dtype "$2" auto bf16 fp16 fp32; vae_dtype=$2; shift ;;
    --adapter) adapters+=("$2"); shift ;;
    --adapter-mode) choice adapter-mode "$2" merge bypass; adapter_mode=$2; shift ;;
    --tune) choice tune "$2" off table; tune=$2; shift ;;
    --tune-only) tune_only=1 ;;
    --out) out=$2; shift ;;
    --dry-run) dry_run=1 ;;
    -h | --help) usage; exit 0 ;;
    *) echo "unknown option: $1 (--help lists them)" >&2; exit 2 ;;
  esac
  shift
done

repo=$(native "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)")
script=$repo/scripts/sample/sample.py

# device selection for the child processes only (never exported into this shell)
case $device in
  cuda) run_env=(env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$index") ;;
  xpu) run_env=(env ZE_AFFINITY_MASK="$index" PYTHONUTF8=1) ;;
esac

if [ "$tune_only" = 1 ]; then
  # DiT block ops + attention for this device, --size, --batch images (CFG packed); no weights loaded
  need kohakumoss-tune
  dtypes=("$precision")
  [ "$compare" = 1 ] && [ "$precision" != bf16 ] && dtypes+=(bf16)
  for dt in "${dtypes[@]}"; do
    echo "== tune $device $dt $size"
    "${run_env[@]}" kohakumoss-tune --device "$device" --dtype "$dt" --sizes "$size" --batch "$batch"
  done
  exit 0
fi

need kogine
need python
[ -n "$weights" ] || { echo "no weights: pass --weights DIR or set KOHAKUMOSS_WEIGHTS" >&2; exit 1; }
[ -e "$weights" ] || { echo "weights root not found: $weights" >&2; exit 1; }
if [ -n "$transformer" ]; then
  [ -e "$transformer" ] || { echo "transformer not found: $transformer" >&2; exit 1; }
  transformer=$(native "$(realpath -m "$transformer")")
fi
if [ "$compare" = 1 ] && [ -z "$transformer" ]; then
  echo "--compare renders the stock transformer next to --transformer; pass --transformer" >&2
  exit 1
fi
weights=$(native "$(realpath -m "$weights")")
out=$(native "$(realpath -m "$out")")

# one run id per invocation, so repeated runs never overwrite
run_id=$(date +%Y%m%d-%H%M%S)

# prompts: a given file, or --count copies of --prompt in a JSON file next to the outputs
# (a file keeps prompts with commas / newlines intact; seeds follow SEED_BASE + index)
if [ -n "$prompts_file" ]; then
  [ -e "$prompts_file" ] || { echo "prompts file not found: $prompts_file" >&2; exit 1; }
  prompts_file=$(native "$(realpath -m "$prompts_file")")
else
  mkdir -p "$out"
  prompts_file=$out/$run_id-prompts.json
  python -c 'import json, sys; print(json.dumps([sys.argv[1]] * max(int(sys.argv[2]), 1), indent=1))' \
    "$prompt" "$count" >"$prompts_file"
fi

adapter_list=""
for a in ${adapters[@]+"${adapters[@]}"}; do
  # path[:multiplier]: the path made absolute, the multiplier kept
  path=$a mult=""
  if [[ $a =~ ^(.+):([0-9.]+)$ ]]; then path=${BASH_REMATCH[1]} mult=:${BASH_REMATCH[2]}; fi
  adapter_list+="${adapter_list:+, }$(pystr "$(native "$(realpath -m "$path")")$mult")"
done

# one config per rendering: <out>/<run_id>-<device>-<tag>/config.py, checked, then run
sample_run() {
  local prec=$1 dit=$2 tag=$3
  local dir=$out/$run_id-$device-$tag fix=none fix_kw="{}"
  mkdir -p "$dir"
  if [ "$prec" = fp16 ] && [ -z "$dit" ]; then fix=$fp16_fix; fi
  [ "$fix" = online ] && fix_kw="{'c': 0.0625}"
  local text="{'memory': '$text_memory', 'precision': '$text_precision', 'fp8_recipe': '$text_fp8_recipe'"
  if [ "$free_text" = 1 ]; then text+=", 'free_after_use': True}"; else text+="}"; fi
  local cfg_path=$dir/config.py
  cat >"$cfg_path" <<EOF
"""Generated by scripts/examples/sample.sh ($run_id-$device-$tag); drives scripts/sample/sample.py."""

WEIGHTS = $(pystr "$weights")
VARIANT = '$variant'
TRANSFORMER = $(pystr "$dit")
ADAPTERS = [$adapter_list]
ADAPTER_MODE = '$adapter_mode'
PRECISION = '$prec'
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

PROMPTS_FILE = $(pystr "$prompts_file")
NEGATIVE = $(pystr "$negative")
SIZES = [$(pystr "$size")]
SEED_BASE = $seed
STEPS = $steps
CFG = $cfg
BATCH = $batch

OUT = $(pystr "$dir")
EOF
  echo "== $device $tag -> $dir"
  # validate every knob against the sample script's rules before loading any weights
  (cd "$repo" && "${run_env[@]}" python -m kohakumoss.cli.config.main check "$cfg_path") ||
    { echo "config check failed for $cfg_path" >&2; exit 1; }
  if [ "$dry_run" = 1 ]; then
    echo "   dry run: kogine run $script --config $cfg_path"
    return 0
  fi
  local start=$SECONDS code=0
  "${run_env[@]}" kogine run "$script" --config "$cfg_path" || code=$?
  [ "$code" = 0 ] || { echo "sample run failed ($tag), exit $code" >&2; exit "$code"; }
  echo "   wall $((SECONDS - start)) s; see $dir"
}

tag=$precision
[ -n "$transformer" ] && tag=$precision-transformer
sample_run "$precision" "$transformer" "$tag"
# optional reference: bf16 with the stock transformer, same prompts / seeds
if [ "$compare" = 1 ]; then sample_run bf16 "" bf16-stock; fi
