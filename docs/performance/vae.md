# VAE: speed and VRAM per dtype

Encode (pixels to latents) and decode (latents to pixels) time and peak memory of the
FLUX.2 VAE per GPU, dtype and variant. The chosen defaults and the VAE's precision are
in [internals/offload.md#vae](../internals/offload.md#vae).

How to read the tables:

| term | meaning |
| --- | --- |
| ms | per image: the median of 3 timed repeats over 4 images |
| GiB (peak alloc) | the peak allocated memory during one repeat |
| `eager` | the default layout |
| `cl_compile` | channels_last plus `torch.compile(dynamic=True)` (`FluxVae.accelerate`) |
| `tiled` | the diffusers tiled / sliced encode and decode |
| `cl` | channels_last alone; it loses everywhere and is in the raw JSON only |
| bold | the fastest row of a GPU and area (batch 1) |

Inputs are real dataset images with the real weights, centre-cropped to 1 MP (1024²) or
2 MP (1440²). Bench: `scripts/bench/vae/vae.py`, which writes
`runs/bench/vae/<gpu>.json`; a configuration that runs out of memory is recorded as an
`oom` row and the bench continues.

## Batch 1

The defaults (`VAE_ENCODE` batch 1, `compile='auto'`):

| GPU | dtype | variant | area | encode ms | encode GiB (peak alloc) | decode ms | decode GiB (peak alloc) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5090 | fp32 | eager | 1MP | 102.0 | 2.80 | 131.5 | 3.10 |
| RTX 5090 | fp32 | cl_compile | 1MP | 87.1 | 2.82 | 117.8 | 2.91 |
| RTX 5090 | bf16 | eager | 1MP | 59.3 | 1.67 | 78.7 | 1.94 |
| RTX 5090 | bf16 | tiled | 1MP | 59.5 | 1.67 | 79.0 | 1.94 |
| RTX 5090 | bf16 | cl_compile | 1MP | **39.3** | 1.18 | **53.4** | 1.28 |
| RTX 5090 | fp16 | cl_compile | 1MP | 39.7 | 1.18 | 53.8 | 1.28 |
| RTX 5090 | fp32 | eager | 2MP | 218.7 | 5.26 | 275.0 | 5.86 |
| RTX 5090 | bf16 | eager | 2MP | 126.9 | 3.15 | 166.3 | 3.69 |
| RTX 5090 | bf16 | tiled | 2MP | 151.6 | 1.69 | 241.8 | 1.95 |
| RTX 5090 | bf16 | cl_compile | 2MP | **84.9** | 2.18 | **115.1** | 2.39 |
| RTX 3090 | fp32 | eager | 1MP | 258.4 | 2.77 | 328.5 | 3.07 |
| RTX 3090 | fp32 | cl_compile | 1MP | 243.7 | 2.80 | 330.1 | 2.88 |
| RTX 3090 | bf16 | eager | 1MP | 133.7 | 1.65 | 176.8 | 1.92 |
| RTX 3090 | bf16 | tiled | 1MP | 134.7 | 1.65 | 178.1 | 1.92 |
| RTX 3090 | bf16 | cl_compile | 1MP | **109.0** | 1.16 | **153.4** | 1.26 |
| RTX 3090 | fp16 | cl_compile | 1MP | 108.8 | 1.16 | 154.3 | 1.26 |
| RTX 3090 | fp32 | eager | 2MP | 626.9 | 5.24 | 742.5 | 5.83 |
| RTX 3090 | bf16 | eager | 2MP | 296.6 | 3.13 | 372.6 | 3.67 |
| RTX 3090 | bf16 | tiled | 2MP | 365.1 | 1.66 | 527.3 | 1.93 |
| RTX 3090 | bf16 | cl_compile | 2MP | **243.7** | 2.15 | **340.3** | 2.37 |
| V100 32G | fp32 | eager | 1MP | 327.4 | 2.77 | 387.1 | 2.51 |
| V100 32G | fp32 | cl_compile | 1MP | 354.8 | 3.30 | 424.1 | 3.26 |
| V100 32G | fp16 | eager | 1MP | 143.3 | 1.40 | 192.8 | 1.55 |
| V100 32G | fp16 | tiled | 1MP | 146.6 | 1.40 | 200.1 | 1.55 |
| V100 32G | fp16 | cl_compile | 1MP | **136.9** | 1.16 | **172.2** | 1.26 |
| V100 32G | fp32 | eager | 2MP | 824.8 | 5.24 | 923.8 | 4.72 |
| V100 32G | fp16 | eager | 2MP | 334.3 | 2.65 | 435.5 | 2.94 |
| V100 32G | fp16 | tiled | 2MP | 408.2 | 1.42 | 581.8 | 1.56 |
| V100 32G | fp16 | cl_compile | 2MP | **314.1** | 2.15 | **377.6** | 2.37 |

## Batch 4

4 same-size images per call; ms per image:

| GPU | dtype | variant | area | encode ms | encode GiB (peak alloc) | decode ms | decode GiB (peak alloc) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5090 | bf16 | eager | 1MP | 46.4 | 6.18 | 76.6 | 6.17 |
| RTX 5090 | bf16 | cl_compile | 1MP | 43.4 | 4.21 | 51.9 | 4.66 |
| RTX 5090 | bf16 | eager | 2MP | 110.8 | 6.13 | 164.1 | 12.05 |
| RTX 5090 | bf16 | cl_compile | 2MP | 98.8 | 4.18 | 114.4 | 9.07 |
| RTX 3090 | bf16 | eager | 1MP | 118.2 | 6.16 | 232.4 | 6.14 |
| RTX 3090 | bf16 | cl_compile | 1MP | 128.2 | 4.18 | 162.7 | 4.64 |
| RTX 3090 | bf16 | eager | 2MP | 278.2 | 6.10 | 495.2 | 12.02 |
| RTX 3090 | bf16 | cl_compile | 2MP | 287.0 | 4.15 | 357.5 | 9.05 |
| RTX 3090 | fp32 | eager | 2MP | 601.3 | 10.20 | OOM | - |
| V100 32G | fp16 | eager | 1MP | 127.4 | 5.16 | 283.4 | 5.40 |
| V100 32G | fp16 | cl_compile | 1MP | 161.6 | 4.18 | 182.4 | 4.64 |
| V100 32G | fp16 | eager | 2MP | 335.2 | 5.13 | 662.2 | 10.55 |
| V100 32G | fp16 | cl_compile | 2MP | 366.0 | 4.15 | 433.7 | 9.05 |

The RTX 3090's fp32 eager decode at 2 MP and batch 4 ran out of its 24 GB. The 3090
fp32 tiled rows are missing: that run stopped at its first out-of-memory.
