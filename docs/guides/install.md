# Install

torch must come from the PyTorch index that matches the GPU: a CUDA build (cu126 /
cu128 / cu130) the NVIDIA driver can run, or the XPU build for Intel Arc. Every path
below installs torch from that index first, then KohakuMoss with the accelerator's
default extra (`cuda` or `xpu`), then runs `kohakumoss-doctor`.

## Install helper

The helper detects the GPU and driver (`nvidia-smi`'s CUDA version; an Intel Arc
adapter), picks the index (driver CUDA >= 13.0: cu130, >= 12.8: cu128, >= 12.6: cu126;
Intel: xpu; neither: cpu), makes `.venv` (uv when present, else `python -m venv` + pip),
installs `torch>=2.13,<2.15` from the index, then KohakuMoss (editable when run from a
clone, `kohakumoss` from PyPI otherwise) and runs the doctor. `KM_TORCH` overrides the
torch requirement.

```bash
bash scripts/install/install.sh                                         # Linux
powershell -ExecutionPolicy Bypass -File scripts\install\install.ps1    # Windows
```

Both make a Python 3.12 venv: with uv, `uv venv -p 3.12`; without it, Linux takes
`python3.12` when installed (else `python3`) and Windows the `py` launcher of a
python.org install (`python` alone may be the Microsoft Store alias). Either helper
stops with a message when it cannot read the CUDA version from `nvidia-smi`. Activating the venv in PowerShell (`.venv\Scripts\Activate.ps1`) needs
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once per user; `cmd` uses
`.venv\Scripts\activate.bat`.

## Manual install

torch first, from the index of your driver (`nvidia-smi` prints its CUDA version:
>= 13.0 cu130, >= 12.8 cu128, >= 12.6 cu126; Intel Arc: xpu), then the package with
that index as an extra index so the torch build stays:

```bash
uv venv -p 3.12
uv pip install "torch>=2.13,<2.15" torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install "kohakumoss[cuda]" "torch>=2.13,<2.15" --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
```

From a clone, replace `"kohakumoss[cuda]"` with `-e ".[cuda]"`; on Intel Arc use the
`xpu` index and `[xpu]`. Without uv, the same two steps with `pip install`
(`--extra-index-url`, no `--index-strategy`).

LyCORIS (`lycoris-lora>=4.1.2`) installs from PyPI as a dependency. A LyCORIS checkout
replaces it with `uv pip install -e <path to LyCORIS>` after the package (the Colab
setup's `--lycoris <path>` does the same).

## From PyPI

The helper run outside a clone, or the manual steps above, install `kohakumoss` from
PyPI. The wheel carries `configs/` and `scripts/` as package data (`kohakumoss/bundle/`), so
`kohakumoss-config new` and the recipes work without a clone; the commands it prints
name the bundled scripts by absolute path (`kogine run <site-packages>/kohakumoss/
bundle/scripts/train/moss.py --config my_run.py`) and its configs go to the current
directory.

## From a clone

```bash
git clone https://github.com/KohakuBlueleaf/KohakuMossTrainer && cd KohakuMossTrainer
```

| platform | command |
| --- | --- |
| Linux | `bash scripts/install/install.sh` |
| Windows | `powershell -ExecutionPolicy Bypass -File scripts\install\install.ps1` |
| Google Colab | `!git clone https://github.com/KohakuBlueleaf/KohakuMossTrainer /content/KohakuMossTrainer && python /content/KohakuMossTrainer/scripts/colab/setup.py` |

The helper installs the clone editable; the manual steps above with `-e ".[cuda]"` do the
same.

Colab: `scripts/colab/setup.py` makes a uv venv at `/content/km-venv` with the torch
build matching the VM driver, installs the package with its extras and runs the doctor;
`scripts/colab/kohakumoss_colab.ipynb` is the full notebook.

## Triton and TileLang

| platform | Triton | TileLang |
| --- | --- | --- |
| Linux, CUDA | ships with the torch wheel | `tilelang` (extra `kernels-cuda`); JIT needs `nvcc` |
| Windows, CUDA | `triton-windows` (extra `kernels-cuda`) | `tilelang`; JIT needs MSVC `cl.exe`, the Windows SDK and `nvcc` |
| Intel Arc | `pytorch-triton-xpu`, from the XPU torch index with torch | not used |

`triton-windows` 3.7 pairs with torch 2.13 and 3.8 with torch 2.14; the extra pins
`>=3.7,<3.9` and the package pins `torch>=2.13,<2.15` to keep the pair (the helpers and
the commands here repeat that pin; `tests/rules/test_torch_pin.py` keeps them equal). The `xpu` extra
uses `kernels-xpu` (empty: Intel's Triton comes with torch), so an XPU install never
pulls `triton-windows`.

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
- Windows: MSVC `cl.exe` (VS Build Tools 2022 with the C++ workload; no developer
  prompt is needed, the toolchain imports `vcvars64` itself) and the Windows SDK;
- CUDA: the toolkit's `nvcc` (TileLang JIT);
- XPU: `pytorch-triton-xpu` absent, or `triton-windows` installed next to it;
- every kernel backend whose smoke build failed, with the likely fix.

`--json report.json` writes the same report (no paths, host or user names); `--memory`
adds every memory profile resolved for the local device; `--clear-failures` forgets the
build-time toolchain failures earlier runs recorded (a recorded failure makes later
runs skip that backend for that op, with a WARNING). See
[install-windows.md](install-windows.md) for the Windows toolchain and feature matrix.
Multi-GPU training is Linux-only.

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
| `board` / `wandb` | `kohakuboard` / `wandb` | `LOGGER="board"` / `"wandb"` |
| `dev` | pytest, ruff, black, lpips | tests and lint |
| `bench` | matplotlib, pandas | benches |
| `baselines` | flash-linear-attention, causal-conv1d | comparison benches only |

The real-GPU bench matrix (`scripts/bench/matrix/`,
[bench-matrix.md](../performance/bench-matrix.md)) is its own dev package,
`kohakumoss-bench`, never part of the `kohakumoss` wheel: from a clone,
`uv pip install -e scripts/bench/matrix` after the package.

## Rust-built dependencies

KohakuMoss has no Rust code; `kohakuvault` (a dependency) and `kohakuboard` (extra
`board`) are the Rust-built packages. A binary-only resolve (`uv pip compile
--only-binary :all:`) for win_amd64 and manylinux_2_28 x86_64 on CPython 3.10 - 3.13
finds wheels for both: `kohakuvault 0.9.0` (with its dependency numpy) and
`kohakuboard 0.0.1`, the newest releases. No Rust toolchain is needed. One dependency
of `kohakuboard`, `click-completion`, is published as a source distribution only; it is
pure Python and builds without a compiler.
