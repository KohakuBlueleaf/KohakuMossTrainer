# Training speed on the B300

Tokens per second of LoKr and full-finetune training on one NVIDIA B300 SXM6 (sm_103,
268 GiB), the kernels each step runs, and the neighbouring settings. Every arm runs the
same optimizer step as the RTX 5090 pages: 294,912 tokens, split as micro-batch
`MAX_TOKENS` x `GRAD_ACC`.

| setup | s/step | joint tok/s | vs one RTX 5090 |
| --- | --- | --- | --- |
| LoKr factor 8, `ADAPTER_MODE="merge"`, 36864 x 8, no checkpointing (steps 41-60) | **10.71** | **26,830** | 5.2x |
| full finetune, AdamW, 18432 x 16, no checkpointing | 12.05 | 23,500 | 4.6x |

The RTX 5090 reference is 5,140 joint tok/s per GPU (LoKr factor 8, the kernels path,
20,570 on four; [training-5090.md](training-5090.md)). Joint tokens are image + text;
both datasets cost the same per joint token within a few percent (attention is about a
tenth of a 4000-token sample's FLOPs).

## Setup

- **Hardware and tree.** One B300 SXM6, torch 2.13.0+cu130, Triton 3.7.1, cuDNN with
  the sm100 fused SDPA kernels.
- **Data.** One shard of a danbooru GBC dataset (30,432 images, native aspect around
  1 MP, `UPSCALE="never"`, so many images are smaller), the
  `examples/gbc-danbooru/caption_danbooru_mix.py` captions, offline latent and text
  caches. The shard and both caches sit in node RAM (`/dev/shm`): read from a network
  file system, the per-sample SQLite reads of the GBC records and the caches starve 4-16
  loader workers and the GPU idles.
- **Run.** `scripts/train/moss.py`, bf16 (`BASE_PRECISION="compute"`: the frozen base
  in 16 bits), `BATCHING="tokens_ffd"`, `TOKEN_BUDGET="exact"`, `MEMORY_PROFILE="manual"`,
  `TUNE="table"`, `COMPILE="off"`, `NUM_WORKERS=16`; LoKr factor 8 with
  `full_matrix=True`, DIM 10000, alpha 1; full finetune with fp32 masters, AdamW, no EMA.
- **Numbers.** s/step and joint tok/s over steps 2 to the end leaving out the traced
  step, unless a row names a window; the samples per step vary (85-100), so single steps
  spread by about 10 %.

## LoKr factor 8

| adapter mode | attention | micro-batch x GRAD_ACC | ckpt | s/step | joint tok/s | model TF/s | peak GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bypass | Triton | 9216 x 32 | none | 17.5 | ~16,300 | 360 | 53.6 |
| bypass | Triton | 18432 x 16 | none | 13.8 | ~21,000 | 470 | 96.3 |
| bypass | Triton | 36864 x 8 | none | 12.4 | ~22,400 | 507 | 198.8 |
| bypass | Triton | 73728 x 4 | none | out of memory | | | |
| bypass, `COMPILE="block"` | Triton | 36864 x 8 | none | 13.3 | ~21,000 | 474 | 251.6 |
| merge | Triton | 36864 x 8 | none | 10.9 | ~25,500 | 578 | 207.2 |
| merge, `COMPILE="block"` | Triton | 36864 x 8 | none | 10.8 | ~25,600 | 584 | 207.2 |
| merge | Triton | 18432 x 16 | none | 11.3 | ~25,300 | 566 | 113.1 |
| merge | Triton | 49152 x 6 | `partial:0.15` | 11.5 | ~24,800 | 836 | 235.0 |
| bypass, `BASE_PRECISION="mxfp8"` | SDPA loop | 36864 x 8 | none | out of memory | | | |
| merge | SDPA loop | 36864 x 8 | none | 13.9 (steps 2-20), 11.7 (21-40), **10.7 (41-60)** | 20,490 / 24,360 / **26,830** | | 207.2 |

- **Merge beats bypass.** The bypass path runs LyCORIS's LoKr bypass kernels next to the
  base GEMMs (forward 1.3 s, backward 1.8 s of a 12.4 s step, as much as the base GEMMs
  themselves); merge builds dW once per layer and micro-batch and runs one GEMM on
  W + dW, whose weight gradient is one more GEMM. Peak memory rises by 8 GiB.
- **Larger micro-batches are faster** up to 36864 tokens: 9216 tokens leave the GEMMs
  short of the card, and without checkpointing 36864 x 8 fits in 207 GiB.
- **`COMPILE="block"` buys nothing** at these sizes: the Triton block kernels already
  run the norm / residual / modulation work, and compiling the bypass path raises the
  peak by 53 GiB.

## Attention

Varlen attention fwd + bwd, bf16, 24 heads x 128, ms (`scripts/bench/kernel/attention.py`
and per-shape probes):

| pack | SDPA loop (cuDNN) | Triton kernels | torch `varlen_attn` (flash) | KohakuFA |
| --- | --- | --- | --- | --- |
| 2 x 1 MP (4396) | **1.57** | 3.21 | 5.05 | 3.46 |
| 1 x 2 MP (8692) | **2.52** | 6.00 | 9.72 | 4.97 |
| 12 x ~3k | **4.83** | 8.80 | | |
| 24 x 1500 | **3.10** | 4.82 | | |
| 3 mixed (4396, 2424, 1624) | **1.26** | 2.37 | 3.71 | 3.13 |
| 8 x 512 px (1104) | **0.99** | 1.07 | 1.49 | 2.71 |
| text refiner (6 short) | 1.32 | 0.40 | **0.38** | 1.36 |

- cuDNN's sm100 kernel runs one 3-8k segment at 1,000-1,400 TF/s; the Triton kernels
  reach about 500.
- The choice times the three PyTorch-side candidates per key, the mean segment length
  included ([kernel-dev.md](../internals/kernel-dev.md)), so short-segment packs (the
  text refiner, small images) keep Triton or flash.
- In a 36864 x 8 merge step attention drops from 3.3 s (Triton) to 1.9 s (cuDNN) of GPU
  time, 9.2 s of compute in all. cuDNN builds an execution plan per new segment length
  (about 75 ms forward + backward, kept for the process), so the GPU waits on those
  builds until the run has seen its lengths: 4.3 s of idle in step 13, 1.9 s in step 51,
  and the steps converge on the 9.2 s of compute.
- KohakuFA pays about 1.6 ms per call in host-side planning at these shapes and its
  bf16 dQ / dK errors at ordinary logits are above PyTorch's (3.7e-4 against 3.0e-4);
  it is benchmarked, not a candidate.

## Full finetune

| micro-batch x GRAD_ACC | ckpt | s/step | joint tok/s | model TF/s | peak GiB |
| --- | --- | --- | --- | --- | --- |
| **18432 x 16** | none | **12.05** | **~23,500** | 800 | 181.8 |
| 36864 x 8 | `partial:0.5` | 13.6 | ~20,100 | 928 (with the recompute) | 191.9 |

- A resident full finetune keeps the fp32 masters, the AdamW moments, fp32 gradients
  and the bf16 compute copy on the card (about 126 GiB for the DiT) and needs no
  checkpointing at 18432 tokens.
- Its step runs the same three GEMMs per layer as a merged LoKr step (forward, input
  gradient, weight gradient) plus one AdamW pass over the weights per optimizer step.
- These rows ran on the Triton attention; the cuDNN loop takes the same 1.4 s off.

## GEMMs: mxfp8 vs bf16

cuBLAS mxfp8 (activation quantizer included) vs bf16 `F.linear`, best path, bf16 inputs
(`scripts/bench/kernel/fp8.py --recipes mx`):

| shape (N x K) | tokens | forward | input gradient |
| --- | --- | --- | --- |
| DiT q (3072 x 3072) | 16k / 32k | 1.34x / 1.42x | 1.27x / 1.29x |
| DiT up + gate (16384 x 3072) | 16k / 32k | 1.44x / 1.63x | 1.38x / 1.42x |
| DiT down (3072 x 8192) | 16k / 32k | 1.18x / 1.35x | 1.56x / 1.60x |
| text encoder q (8192 x 4096) | 16k / 32k | 1.64x / 2.00x | 1.44x / 1.46x |

The DiT's GEMMs are about half of a merged step, so an mxfp8 base would take 10-20 % off
it at the cost of a quantized base; merge needs a 16-bit base, and the mxfp8 bypass
does not fit 36864 tokens.

## Encoders

- **Text encoder** (Qwen3.5 9B, resident, 512 captions of 25-520 tokens per call,
  `scripts/bench/textenc/textenc.py`): 49,200 tok/s bf16 packed at 512 prompts per call,
  36,400 at 8; mxfp8 46,700 at best. The offline text cache pass runs at a tenth of that
  when it writes its SQLite store to a network file system.
- **VAE** latent cache pass: 18-23 images/s at about 1 MP.
