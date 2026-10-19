# Install

torch must come from the PyTorch index that matches the GPU: a CUDA build (cu126 /
cu128 / cu130) the NVIDIA driver can run, or the XPU build for Intel Arc. Every path
below installs torch from that index first, then KohakuMoss with the accelerator's
default extra (`cuda` or `xpu`), then runs `kohakumoss-doctor`.

| path | when |
| --- | --- |
| [From a clone](#from-a-clone) with the [install helper](#install-helper) | the usual install: Linux, Windows, Google Colab |
| [From PyPI](#from-pypi) | no clone; the wheel carries the configs and scripts |
| [Manual install](#manual-install) | your own venv or index choice |

## From a clone

```bash
git clone https://github.com/KohakuBlueleaf/KohakuMossTrainer && cd KohakuMossTrainer
```

| platform | command |
| --- | --- |
| Linux | `bash scripts/install/install.sh` |
| Windows | `powershell -ExecutionPolicy Bypass -File scripts\install\install.ps1` |
| Google Colab | `!git clone https://github.com/KohakuBlueleaf/KohakuMossTrainer /content/KohakuMossTrainer && python /content/KohakuMossTrainer/scripts/colab/setup.py` |

The helper installs the clone editable; the manual steps below with `-e ".[cuda]"` do the
same.

Colab: `scripts/colab/setup.py` makes a uv venv at `/content/km-venv` with the torch
build matching the VM driver, installs the package with its extras and runs the doctor;
`scripts/colab/kohakumoss_colab.ipynb` is the full notebook.

## Install helper

```bash
bash scripts/install/install.sh                                         # Linux
powershell -ExecutionPolicy Bypass -File scripts\install\install.ps1    # Windows
```

The helper:

1. detects the GPU and driver (`nvidia-smi`'s CUDA version; an Intel Arc adapter) and
   stops with a message when it cannot read the CUDA version from `nvidia-smi`;
2. picks the index: driver CUDA >= 13.0: cu130, >= 12.8: cu128, >= 12.6: cu126; Intel:
   xpu; neither: cpu;
3. makes a Python 3.12 venv `.venv`: with uv, `uv venv -p 3.12`; without it,
   `python -m venv` + pip, where Linux takes `python3.12` when installed (else
   `python3`) and Windows the `py` launcher of a python.org install (`python` alone may
   be the Microsoft Store alias);
4. installs `torch>=2.13,<2.15` from the index (`KM_TORCH` overrides the torch
   requirement);
5. installs KohakuMoss (editable when run from a clone, `kohakumoss` from PyPI
   otherwise) and runs the doctor.

Activate the venv:

| shell | command |
| --- | --- |
| bash | `source .venv/bin/activate` |
| PowerShell | `.venv\Scripts\Activate.ps1` (needs `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once per user) |
| `cmd` | `.venv\Scripts\activate.bat` |

## From PyPI

The helper run outside a clone, or the manual steps below, install `kohakumoss` from
PyPI. The wheel carries `configs/` and `scripts/` as package data (`kohakumoss/bundle/`),
so `kohakumoss-config new` and the recipes work without a clone:

- the commands it prints name the bundled scripts by absolute path (`kogine run
  <site-packages>/kohakumoss/bundle/scripts/train/moss.py --config my_run.py`);
- its configs go to the current directory.

## Manual install

torch first, from the index of your driver (`nvidia-smi` prints its CUDA version:
>= 13.0 cu130, >= 12.8 cu128, >= 12.6 cu126; Intel Arc: xpu), then the package with
that index as an extra index so the torch build stays:

```bash
uv venv -p 3.12
uv pip install "torch>=2.13,<2.15" torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install "kohakumoss[cuda]" "torch>=2.13,<2.15" --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
```

- From a clone, replace `"kohakumoss[cuda]"` with `-e ".[cuda]"`.
- On Intel Arc use the `xpu` index and `[xpu]`.
- Without uv, the same two steps with `pip install` (`--extra-index-url`, no
  `--index-strategy`).

LyCORIS (`lycoris-lora>=4.1.2`) installs from PyPI as a dependency. A LyCORIS checkout
replaces it with `uv pip install -e <path to LyCORIS>` after the package (the Colab
setup's `--lycoris <path>` does the same).

## The doctor

```bash
kohakumoss-doctor
```

It prints the OS, Python, torch build, every GPU with its arch, package versions, the
host compiler of each kernel backend, a smoke build of each backend on each device
(`status triton@cuda:0 ok` or the failure), which backend serves each op, and a
`missing:` section with the fix command for each entry:

- a requirement of the base package or the accelerator's default extra that is absent or
  at the wrong version;
- a CPU-only torch next to an NVIDIA GPU, or a CUDA torch the driver cannot run;
- Windows: MSVC `cl.exe` (VS Build Tools 2022 with the C++ workload; the toolchain
  imports `vcvars64` itself, from any console) and the Windows SDK;
- CUDA: the toolkit's `nvcc` (TileLang JIT);
- XPU: `pytorch-triton-xpu` absent, or `triton-windows` installed next to it;
- every kernel backend whose smoke build failed, with the likely fix.

| flag | effect |
| --- | --- |
| `--json report.json` | writes the same report (no paths, host or user names) |
| `--memory` | adds every memory profile resolved for the local device |
| `--clear-failures` | forgets the recorded build-time toolchain failures (a recorded failure makes later runs skip that backend for that op, with a WARNING) |

See [install-windows.md](install-windows.md) for the Windows toolchain and feature
matrix. Multi-GPU training is Linux-only.

## Triton and TileLang

| platform | Triton | TileLang |
| --- | --- | --- |
| Linux, CUDA | ships with the torch wheel | `tilelang` (extra `kernels-cuda`); JIT needs `nvcc` |
| Windows, CUDA | `triton-windows` (extra `kernels-cuda`) | `tilelang`; JIT needs MSVC `cl.exe`, the Windows SDK and `nvcc` |
| Intel Arc | `pytorch-triton-xpu`, from the XPU torch index with torch | - |

- `triton-windows` 3.7 pairs with torch 2.13 and 3.8 with torch 2.14; the extra pins
  `>=3.7,<3.9` and the package pins `torch>=2.13,<2.15` to keep the pair (the helpers
  and the commands here repeat that pin; `tests/rules/test_torch_pin.py` keeps them
  equal).
- The `xpu` extra uses `kernels-xpu` (empty: Intel's Triton comes with torch).

## Where caches go

The first runs on a machine build kernels and tune them; the results stay on disk, so
later runs start without that work. Each cache has its own location, and each location
can be moved with an environment variable (for a small home or system drive, or a
shared drive):

| cache | default | moved by |
| --- | --- | --- |
| KohakuMoss kernel cache: tuning tables (`tuning/`), device probes (`devices/`), recorded backend failures (`failures/`), the Windows toolchain (`toolchain/`) | `~/.cache/kohakumoss_kernels` | `KOHAKUMOSS_KERNEL_CACHE` |
| LyCORIS fused-kernel tuning | `~/.cache/lycoris_kernels` | `LYCORIS_KERNEL_CACHE_DIR` |
| Triton compiled kernels | `~/.triton/cache` | `TRITON_CACHE_DIR` (or `TRITON_HOME`) |
| `torch.compile` (TorchInductor) graphs and its Triton kernels | `<system temp>/torchinductor_<user>` | `TORCHINDUCTOR_CACHE_DIR` (its Triton kernels follow `TRITON_CACHE_DIR` when set) |
| TileLang compiled kernels (sm70 / sm75) | `~/.tilelang/cache` | `TILELANG_CACHE_DIR` |
| dataset caches: latents, text embeddings, catalog files | `<first dataset path>/cache` | the `CACHE_DIR` knob ([data.md](../concepts/data.md)) |

- Deleting a kernel cache is safe: the next run builds and tunes again.
- A tuning table is kept per card, CUDA and torch version, so one cache directory can be
  shared by several GPUs and installs.

## Extras

| extra | packages | needed for |
| --- | --- | --- |
| `cuda` | `kernels-cuda` + `lightning` + `quant` | the default set on NVIDIA GPUs |
| `xpu` | `kernels-xpu` + `lightning` + `quant` | the default set on Intel Arc |
| `kernels-cuda` (`kernels`) | `tilelang`, `triton-windows` (Windows) | TileLang kernels (sm70 / sm75), Triton on Windows |
| `kernels-xpu` | none | Intel's Triton comes with the XPU torch |
| `lightning` | `lightning` | `DRIVER="lightning"` |
| `quant` | `bitsandbytes` | `OPTIMIZER="adamw8bit"` (bitsandbytes 8-bit AdamW) |
| `optim` | `prodigyopt`, `schedulefree` | `OPTIMIZER="prodigy"` / `"schedule_free"` |
| `parquet` | `pyarrow` | the `parquet` dataset source |
| `wandb` | `wandb` | `LOGGER="wandb"` |
| `dev` | pytest, ruff, black, lpips + `bench` | tests and lint (the script tests import every script) |
| `bench` | matplotlib, pandas | benches, the fp16-fix charts |
| `baselines` | flash-linear-attention, causal-conv1d | comparison benches only |

The real-GPU bench matrix (`scripts/bench/matrix/`,
[bench-matrix.md](../performance/bench-matrix.md)) is its own dev package,
`kohakumoss-bench`: from a clone, `uv pip install -e scripts/bench/matrix` after the
package.

## Prebuilt wheels

- `kohakuvault` (a dependency) installs from wheels: a binary-only resolve
  (`uv pip compile --only-binary :all:`) for win_amd64 and manylinux_2_28 x86_64 on
  CPython 3.10 - 3.13 finds `kohakuvault 0.9.0` (with its dependency numpy), the newest
  release.
