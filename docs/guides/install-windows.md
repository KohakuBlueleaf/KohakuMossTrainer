# Install on Windows

Every single-GPU feature runs on Windows. Multi-GPU training (DDP, FSDP2, pipeline,
threaded) is Linux-only by design.

## Requirements

- Python 3.10+ (3.12 / 3.13 tested), an NVIDIA driver recent enough for the CUDA build of
  torch you install, and the Visual Studio Build Tools (MSVC) for kernel compilation. The
  MSVC compiler does not need to be on PATH: `kohakumoss_kernels.hardware.toolchain` finds
  it through `vswhere` and sets `CXX`.
- Kernel packages: `triton-windows` (Triton backend, sm80+) and `tilelang` (TileLang
  backend, sm70 / sm75 and others). Both have Windows wheels.

```bash
uv venv -p 3.13 .venv
uv pip install torch --index-url https://download.pytorch.org/whl/cu130
uv pip install -e ".[kernels,lightning,dev]"
kohakumoss-doctor
```

`kohakumoss-doctor` prints the GPU arch, the toolchain and package versions, and which
kernel backend serves each op on this machine.

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
| Intel Arc (XPU) kernels | slower path | triton-xpu cannot build launchers on Windows (it passes GCC flags to MSVC), so XPU uses the eager ops |
| multi-GPU (DDP / FSDP2 / pipeline / threaded) | not supported | Linux-only by design |

## Notes

- Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` on machines with several GPUs so device indices
  match `nvidia-smi`.
- A GPU that also drives the display has less free memory and noisier timings.
