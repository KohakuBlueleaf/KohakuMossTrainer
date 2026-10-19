# Performance

| page | content |
| --- | --- |
| [attention.md](attention.md) | varlen attention vs PyTorch SDPA per GPU |
| [block-ops.md](block-ops.md) | norm / residual / qk-rope / SiTU / loss / fused linears vs PyTorch eager and compile per GPU |
| [offload-synthetic.md](offload-synthetic.md) | offload knobs and hardware-limit check on MOSS-sized synthetic blocks, V100 and RTX 4090 |
| [gemm-efficiency.md](gemm-efficiency.md) | achieved TFLOP/s of a training step's GEMM / flash kernels against the clock-derived peak and the measured cuBLAS ceiling, RTX 5090 / 3090 / 2080 Ti and V100 |
| [quant-rotate.md](quant-rotate.md) | the ConvRot rotation and the rotated int8 linear vs the plain int8 linear at the DiT shapes, separate vs fused, per GPU |
| [training-5090.md](training-5090.md) | LoRA and full-finetune tokens/s on one and four RTX 5090, the settings of the 5090 recipes and their neighbours |
| [vae.md](vae.md) | VAE encode / decode ms per image and peak VRAM per dtype, compile, tiling, batch, per GPU |
| [bench-matrix.md](bench-matrix.md) | the real-GPU bench matrix: every card's speed / VRAM / quality / inference runs, feature coverage, the report |
