#!/usr/bin/env bash
# Linux install: torch from the index for this GPU / driver, kohakumoss with its default extra, the doctor.
# Usage: bash scripts/install/install.sh (a clone installs editable; elsewhere kohakumoss from PyPI).
set -euo pipefail

TORCH_SPEC="${KM_TORCH:-torch>=2.13,<2.15}"
PYTHON=3.12
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if command -v nvidia-smi >/dev/null 2>&1; then
  # "CUDA Version: X.Y", or "CUDA UMD Version: X.Y" on newer drivers
  if ! read -r major minor < <(nvidia-smi | sed -n 's/.*CUDA \(UMD \)\{0,1\}Version: *\([0-9]*\)\.\([0-9]*\).*/\2 \3/p' | head -1); then
    echo "could not read the CUDA version from nvidia-smi; install by hand: docs/guides/install.md#manual-install" >&2; exit 1
  fi
  ver=$((major * 100 + minor))
  if [ "$ver" -ge 1300 ]; then tag=cu130
  elif [ "$ver" -ge 1208 ]; then tag=cu128
  elif [ "$ver" -ge 1206 ]; then tag=cu126
  else echo "driver CUDA $major.$minor < 12.6: update the NVIDIA driver" >&2; exit 1; fi
  extra="[cuda]"
elif grep -qs 0x8086 /sys/class/drm/card*/device/vendor && command -v sycl-ls >/dev/null 2>&1; then
  tag=xpu; extra="[xpu]"
else
  echo "no NVIDIA driver (nvidia-smi) or Intel GPU runtime (sycl-ls) found: CPU build"
  tag=cpu; extra=""
fi
index="https://download.pytorch.org/whl/$tag"
echo "torch index: $index"

if [ -f "$HERE/pyproject.toml" ]; then pkg=(-e "$HERE$extra"); else pkg=("kohakumoss$extra"); fi

if command -v uv >/dev/null 2>&1; then
  [ -d .venv ] || uv venv -p "$PYTHON"
  uv pip install "$TORCH_SPEC" torchvision --index-url "$index"
  uv pip install "${pkg[@]}" "$TORCH_SPEC" --extra-index-url "$index" --index-strategy unsafe-best-match
else
  # python3.12 when installed, else the system python3
  [ -d .venv ] || "$(command -v "python$PYTHON" || command -v python3)" -m venv .venv
  .venv/bin/python -m pip install "$TORCH_SPEC" torchvision --index-url "$index"
  .venv/bin/python -m pip install "${pkg[@]}" "$TORCH_SPEC" --extra-index-url "$index"
fi
.venv/bin/kohakumoss-doctor
