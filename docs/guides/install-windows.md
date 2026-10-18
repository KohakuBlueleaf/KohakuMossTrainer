# Install on Windows

Every single-GPU feature runs on Windows ([Feature matrix](#feature-matrix)). Multi-GPU
training (DDP, FSDP2, pipeline, threaded) is Linux-only by design.

## Requirements

| requirement | for |
| --- | --- |
| Python 3.10+ (3.12 / 3.13 tested; the helper and the commands below make a 3.12 venv) | everything |
| an NVIDIA driver recent enough for the CUDA build of torch you install, or the Intel Arc driver | the GPU |
| Visual Studio Build Tools with the "Desktop development with C++" workload (MSVC x64 tools + Windows SDK) | kernel compilation, from any console ([Compiler setup](#compiler-setup)) |
| the CUDA toolkit, whose `nvcc` TileLang's JIT calls | NVIDIA sm70 / sm75 (V100, 2080 Ti, T4), whose kernels are TileLang; the doctor lists `nvcc` under `missing:` when it is absent |
| `triton-windows` (Triton backend, sm80+) and `tilelang` (TileLang backend, sm70 / sm75 and others) | kernels; both have Windows wheels |

## Install

The install helper does all of the following in one command, from a clone
([install.md](install.md#install-helper)):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install\install.ps1
```

The default execution policy of Windows PowerShell refuses scripts, so activating the
venv (`.venv\Scripts\Activate.ps1`) needs `Set-ExecutionPolicy -Scope CurrentUser
RemoteSigned` once per user; in `cmd`, `.venv\Scripts\activate.bat` runs as is.

By hand, NVIDIA GPUs (one environment; [install.md](install.md) lists the extras):

```bash
uv venv -p 3.12 .venv
uv pip install --python .venv/Scripts/python.exe "torch>=2.13,<2.15" torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/Scripts/python.exe -e ".[cuda]" "torch>=2.13,<2.15" --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
.venv/Scripts/kohakumoss-doctor
```

Intel Arc (XPU) on a machine that also has an NVIDIA GPU needs its own environment,
because the XPU build of torch replaces the CUDA build; the `xpu` extra leaves out the
CUDA-only `kernels` extra (the XPU torch wheel brings Intel's Triton):

```bash
uv venv -p 3.12 .venv-xpu
uv pip install --python .venv-xpu/Scripts/python.exe "torch>=2.13,<2.15" torchvision --index-url https://download.pytorch.org/whl/xpu
uv pip install --python .venv-xpu/Scripts/python.exe -e ".[xpu]" "torch>=2.13,<2.15" --extra-index-url https://download.pytorch.org/whl/xpu --index-strategy unsafe-best-match
.venv-xpu/Scripts/kohakumoss-doctor
```

- The text encoder's Gated DeltaNet, convolution and attention layers run on
  `kohakumoss_kernels` ops ([text-encoder.md](../internals/text-encoder.md)).
- The `baselines` extra (flash-linear-attention, causal-conv1d) is for comparison
  benches.

`kohakumoss-doctor` prints the GPU arch, the toolchain and package versions, the host
compiler each kernel backend uses, which kernel backend serves each op on this machine,
any backend failures recorded by earlier runs, and what is missing (packages, a CPU-only
torch, the C++ build tools) with the command that fixes it
([install.md](install.md#the-doctor)).

## Compiler setup

The Build Tools install is the whole user-side setup; the kernels configure the compiler
themselves.

The first kernel build of a process
(normally the first op resolution of a model) loads the backends once
(`registry.load_backends()`): it sets torch's compile cache-key tag and imports the
eager, compile, Triton and TileLang backends. The Triton and TileLang imports, and the
compile backend's first CPU probe, call
`kohakumoss_kernels.hardware.toolchain.ensure_toolchain()`:

- **MSVC environment.** Outside a Visual Studio developer shell it locates the newest
  Visual Studio with the x64 C++ tools through `vswhere`, runs its `vcvars64.bat` once
  and imports the environment it sets (`PATH`, `INCLUDE`, `LIB`, `LIBPATH`,
  `VCINSTALLDIR`, `VCToolsVersion`, Windows SDK variables) into the process. The result
  is cached in the kernel cache (`toolchain/vcvars.json`) and re-run when Visual Studio
  changes. The first process pays the `vcvars64.bat` run (0.8 s measured on VS 18 /
  MSVC 14.51); later ones read the cache.
- **`CXX`.** It is set to that `cl`, spelled `cl.EXE`, the name for which triton-xpu
  chooses MSVC flags. A `CXX` naming another compiler is kept.
- **nvcc's host compiler** for TileLang is that `cl` too: LyCORIS's kernel import sets
  `TILELANG_DISABLE_CLANG_CL=1`, so TileLang resolves MSVC `cl` even when its first
  compile runs before the toolchain setup, with a clang-cl on `PATH`
  (`tests/adapters/test_lycoris_host_compiler.py` runs both orders).
- **Console code page.** Intel Triton reads its UTF-8 C driver sources with the console
  code page; the toolchain makes those reads and writes UTF-8, so the XPU driver and
  inductor on XPU initialise on any console code page, cp950 / cp936 / cp932 included.

### When a backend fails

When a backend still cannot run an op (no compiler, a driver bug, any RuntimeError), the
op moves to the next backend of the arch priority (compile, then eager) with one
WARNING: `<op> on <arch>/<dtype>: backend <failed> failed, using <next> (<n> bound ops):
<error>`.

| failure | kept |
| --- | --- |
| a toolchain failure met at build time (the backend probe, the prewarm) | recorded on disk; later runs skip that backend for that op with a WARNING (`backend <b> skipped, failed here (<error>)`); the doctor lists the records and `--clear-failures` forgets them |
| a failure during a step or a sample | in that process only |

See [kernel-dev.md](../internals/kernel-dev.md#fallback).

## Feature matrix

| feature | Windows | notes |
| --- | --- | --- |
| model, loaders, savers, fp16 fix | works | |
| inference (MossPipeline, sample CLI) | works | 512^2, 20 steps, bf16 on an RTX 4090: 6.1 s, 15.7 GiB peak allocated |
| training: native and Lightning drivers, resume, sampling, validation, loggers | works | |
| LyCORIS adapters, every algorithm, fused kernels | works | |
| export to ComfyUI / LyCORIS / merged / diffusers | works | |
| dataset package, caches, loader workers | works | workers use spawn (Linux: forkserver); each worker imports its modules in turn before the first batch ([data.md](../concepts/data.md#loader)) |
| attention kernel | works, faster than SDPA | RTX 4090 1 MP x2 fwd+bwd 13.0 ms vs SDPA 26.9 ms |
| block-op kernels (Triton) | works | |
| quantized base (int8, fp8 on sm89+) | works | |
| memory profiles, auto planner | works | |
| activation offload, weight streaming | works | pinned H2D 19.7 / D2H 24.3 GB/s on an RTX 4090 under WDDM; pageable 8.3 / 13.5 |
| gradient offload, CPU AdamW, fused per-parameter step | works | |
| text encoder: kernels, batched / packed encode, resident / offload / ring / cpu placements, encode-ahead | works | the same Triton kernels as Linux on sm80+; TileLang on sm70 / sm75 |
| Intel Arc (XPU) kernels | works | Triton (Intel backend) builds with the toolchain above in a plain console; per op and shape the registry times the Triton kernel against `torch.compile` and binds the faster (see [performance](../performance/README.md)); the fp16 `kahan` optimizer writeback drifts on Arc (2e-3 after 200 steps of 1e-5, against under 1e-5 on CUDA); on Arc use bf16 parameters or the `nearest` writeback |
| multi-GPU (DDP / FSDP2 / pipeline / threaded) | not supported | Linux-only by design |

## Notes

- Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` on machines with several GPUs so device indices
  match `nvidia-smi`.
- A GPU that also drives the display has less free memory and noisier timings.
- The CUDA allocator on Windows has no expandable segments; training and sampling set
  `roundup_power2_divisions:4` instead, which keeps reserved memory flat across batches
  of different sizes (the run line shows `allocator roundup_power2_divisions:4`;
  [offload.md](../internals/offload.md#allocator)).
