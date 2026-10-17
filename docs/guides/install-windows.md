# Install on Windows

Every single-GPU feature runs on Windows. Multi-GPU training (DDP, FSDP2, pipeline,
threaded) is Linux-only by design.

## Requirements

- Python 3.10+ (3.12 / 3.13 tested), an NVIDIA driver recent enough for the CUDA build of
  torch you install (or the Intel Arc driver), and the Visual Studio Build Tools with the
  "Desktop development with C++" workload (MSVC x64 tools + Windows SDK) for kernel
  compilation. No developer shell is needed; see [Compiler setup](#compiler-setup).
- Kernel packages: `triton-windows` (Triton backend, sm80+) and `tilelang` (TileLang
  backend, sm70 / sm75 and others). Both have Windows wheels.

NVIDIA GPUs (one environment):

```bash
uv venv -p 3.13 .venv
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/Scripts/python.exe -e ".[kernels,lightning,quant,parquet,board,dev,bench]"
.venv/Scripts/kohakumoss-doctor
```

Intel Arc (XPU) needs its own environment, because the XPU build of torch replaces the CUDA
build; the `kernels` extra is CUDA-only and is left out (the XPU torch wheel brings Intel's
Triton):

```bash
uv venv -p 3.13 .venv-xpu
uv pip install --python .venv-xpu/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/xpu
uv pip install --python .venv-xpu/Scripts/python.exe -e ".[lightning,quant,parquet,board,dev,bench]"
.venv-xpu/Scripts/kohakumoss-doctor
```

The text encoder's Gated DeltaNet and convolution layers run on KohakuMoss's own ops; no
extra package is needed. The `baselines` extra (flash-linear-attention, causal-conv1d) is only
for comparison benches.

`kohakumoss-doctor` prints the GPU arch, the toolchain and package versions, the host
compiler each kernel backend uses, which kernel backend serves each op on this machine,
and any backend failures recorded by earlier runs.

## Compiler setup

Every backend that JIT-compiles (Triton, TileLang, `torch.compile`) calls
`kohakumoss_kernels.hardware.toolchain.ensure_toolchain()` on import:

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
  are UTF-8; on a cp950 / cp936 / cp932 console the XPU driver and inductor on XPU could
  not initialise. The toolchain makes Intel Triton's source reads and writes UTF-8, so
  neither `PYTHONUTF8=1` nor a developer shell is needed.

Nothing is needed from the user beyond the Build Tools install. When a backend still
cannot build an op (no compiler, a driver bug), the registry records the error, warns
once, and runs the op on the next backend of the arch priority (compile, then eager); the
doctor lists those failures.

## Feature matrix

| feature | Windows | notes |
| --- | --- | --- |
| model, loaders, savers, fp16 fix | works | |
| inference (MossPipeline, sample CLI) | works | 512^2, 20 steps, bf16 on an RTX 4090: 6.1 s, 15.7 GiB peak |
| training: native and Lightning drivers, resume, sampling, validation, loggers | works | |
| LyCORIS adapters, every algorithm, fused kernels | works | |
| export to ComfyUI / LyCORIS / merged / diffusers | works | |
| dataset package, caches, loader workers | works | workers use spawn |
| attention kernel | works, faster than SDPA | torch's flash SDPA and varlen attention are absent from Windows torch wheels; RTX 4090 1 MP x2 fwd+bwd 13.0 ms vs SDPA 26.9 ms |
| block-op kernels (Triton) | works | |
| quantized base (int8, fp8 on sm89+) | works | |
| memory profiles, auto planner | works | |
| activation offload, weight streaming | works | pinned H2D 19.7 / D2H 24.3 GB/s on an RTX 4090 under WDDM; pageable 8.3 / 13.5 |
| gradient offload, CPU AdamW, fused per-parameter step | works | |
| text-encoder fast kernels | slower path | `causal-conv1d` has no Windows build; the text encoder runs the reference path for those layers |
| Intel Arc (XPU) kernels | works | Triton (Intel backend) builds with the toolchain above in a plain console; per op and shape the registry races the Triton kernel against `torch.compile` and binds the faster (see [performance](../performance/README.md)); the fp16 `kahan` optimizer writeback drifts on Arc (2e-3 after 200 steps of 1e-5, against under 1e-5 on CUDA; fp16 subnormals survive a store and load there, so the cause is elsewhere); on Arc use bf16 parameters or the `nearest` writeback |
| multi-GPU (DDP / FSDP2 / pipeline / threaded) | not supported | Linux-only by design |

## Notes

- Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` on machines with several GPUs so device indices
  match `nvidia-smi`.
- A GPU that also drives the display has less free memory and noisier timings.
