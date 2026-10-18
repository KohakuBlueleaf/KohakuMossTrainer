# Install on Windows

Every single-GPU feature runs on Windows. Multi-GPU training (DDP, FSDP2, pipeline,
threaded) is Linux-only by design.

## Requirements

- Python 3.10+ (3.12 / 3.13 tested; the helper and the commands below make a 3.12 venv),
  an NVIDIA driver recent enough for the CUDA build of torch you install (or the Intel
  Arc driver), and the Visual Studio Build Tools with the "Desktop development with C++"
  workload (MSVC x64 tools + Windows SDK) for kernel compilation. No developer shell is
  needed; see [Compiler setup](#compiler-setup).
- NVIDIA sm70 / sm75 (V100, 2080 Ti, T4): the CUDA toolkit, whose `nvcc` TileLang's JIT
  calls (the kernels of those archs are TileLang). The doctor lists `nvcc` under
  `missing:` when it is absent.
- Kernel packages: `triton-windows` (Triton backend, sm80+) and `tilelang` (TileLang
  backend, sm70 / sm75 and others). Both have Windows wheels.

The install helper does all of the following in one command, from a clone:
`powershell -ExecutionPolicy Bypass -File scripts\install\install.ps1`
([install.md](install.md#install-helper)). The default execution policy of Windows
PowerShell refuses scripts, so activating the venv (`.venv\Scripts\Activate.ps1`) needs
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once per user; in `cmd`,
`.venv\Scripts\activate.bat` needs nothing. By hand, NVIDIA GPUs (one environment;
[install.md](install.md) lists the extras):

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

The text encoder's Gated DeltaNet, convolution and attention layers run on
`kohakumoss_kernels` ops (see docs/internals/text-encoder.md); no extra package is needed.
The `baselines` extra (flash-linear-attention, causal-conv1d) is only for comparison benches.

`kohakumoss-doctor` prints the GPU arch, the toolchain and package versions, the host
compiler each kernel backend uses, which kernel backend serves each op on this machine,
any backend failures recorded by earlier runs, and what is missing (packages, a CPU-only
torch, the C++ build tools) with the command that fixes it.

## Compiler setup

`import kohakumoss_kernels` has no side effects. The first kernel build of a process
(normally the first op resolution of a model) loads the backends once
(`registry.load_backends()`): it sets torch's compile cache-key tag and imports the
eager, compile, Triton and TileLang backends. The Triton and TileLang imports, and the
compile backend's first CPU probe, call
`kohakumoss_kernels.hardware.toolchain.ensure_toolchain()`:

- Outside a Visual Studio developer shell it locates the newest Visual Studio with the x64
  C++ tools through `vswhere`, runs its `vcvars64.bat` once and imports the environment it
  sets (`PATH`, `INCLUDE`, `LIB`, `LIBPATH`, `VCINSTALLDIR`, `VCToolsVersion`, Windows SDK
  variables) into the process. The result is cached in the kernel cache
  (`toolchain/vcvars.json`) and re-run when Visual Studio changes. The first process pays
  the `vcvars64.bat` run (0.8 s measured on VS 18 / MSVC 14.51); later ones read the cache.
- `CXX` is set to that `cl`, spelled `cl.EXE`: triton-xpu chooses MSVC flags only for that
  exact name (with `cl.exe` it passes GCC flags to MSVC and every launcher build fails).
  A `CXX` naming another compiler is kept. nvcc's host compiler for TileLang is that
  `cl` too: LyCORIS's kernel import sets `TILELANG_DISABLE_CLANG_CL=1`, so TileLang
  resolves MSVC `cl` even when its first compile runs before the toolchain setup, with a
  clang-cl on `PATH` (`tests/adapters/test_lycoris_host_compiler.py` runs both orders).
- Intel Triton reads its C driver sources with the console code page, and those sources
  are UTF-8, so on a cp950 / cp936 / cp932 console the XPU driver and inductor on XPU
  would not initialise. The toolchain makes Intel Triton's source reads and writes
  UTF-8, so neither `PYTHONUTF8=1` nor a developer shell is needed.

Nothing is needed from the user beyond the Build Tools install. When a backend still
cannot run an op (no compiler, a driver bug, any RuntimeError), the op moves to the next
backend of the arch priority (compile, then eager) with one WARNING:
`<op> on <arch>/<dtype>: backend <failed> failed, using <next> (<n> bound ops): <error>`.
Only a toolchain failure met at build time (the backend probe, the prewarm) is recorded
on disk; later runs skip that backend for that op with a WARNING (`backend <b> skipped,
failed here (<error>)`), and the doctor lists the records (`--clear-failures` forgets
them). A failure during a step or a sample stays in that process
([kernel-dev.md](../internals/kernel-dev.md#fallback)).

## Feature matrix

| feature | Windows | notes |
| --- | --- | --- |
| model, loaders, savers, fp16 fix | works | |
| inference (MossPipeline, sample CLI) | works | 512^2, 20 steps, bf16 on an RTX 4090: 6.1 s, 15.7 GiB peak allocated |
| training: native and Lightning drivers, resume, sampling, validation, loggers | works | |
| LyCORIS adapters, every algorithm, fused kernels | works | |
| export to ComfyUI / LyCORIS / merged / diffusers | works | |
| dataset package, caches, loader workers | works | workers use spawn (Linux: forkserver); each worker imports its modules in turn before the first batch ([data.md](../concepts/data.md#loader)) |
| attention kernel | works, faster than SDPA | torch's flash SDPA and varlen attention are absent from Windows torch wheels; RTX 4090 1 MP x2 fwd+bwd 13.0 ms vs SDPA 26.9 ms |
| block-op kernels (Triton) | works | |
| quantized base (int8, fp8 on sm89+) | works | |
| memory profiles, auto planner | works | |
| activation offload, weight streaming | works | pinned H2D 19.7 / D2H 24.3 GB/s on an RTX 4090 under WDDM; pageable 8.3 / 13.5 |
| gradient offload, CPU AdamW, fused per-parameter step | works | |
| text encoder: kernels, batched / packed encode, resident / offload / stream / cpu placements, encode-ahead | works | the same Triton kernels as Linux on sm80+; TileLang on sm70 / sm75 |
| Intel Arc (XPU) kernels | works | Triton (Intel backend) builds with the toolchain above in a plain console; per op and shape the registry times the Triton kernel against `torch.compile` and binds the faster (see [performance](../performance/README.md)); the fp16 `kahan` optimizer writeback drifts on Arc (2e-3 after 200 steps of 1e-5, against under 1e-5 on CUDA); on Arc use bf16 parameters or the `nearest` writeback |
| multi-GPU (DDP / FSDP2 / pipeline / threaded) | not supported | Linux-only by design |

## Notes

- Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` on machines with several GPUs so device indices
  match `nvidia-smi`.
- A GPU that also drives the display has less free memory and noisier timings.
