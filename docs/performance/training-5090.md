# Training speed on the RTX 5090

Tokens per second of LoRA and full-finetune training on one and four RTX 5090 32 GB, the
settings behind the 5090 recipes, and the neighbouring settings each was chosen against.

| recipe | GPUs | setup |
| --- | --- | --- |
| `recipes_lora_5090.py` | 1 | LoRA r32 bf16, `MAX_TOKENS=4608`, `CHECKPOINT="partial:0.2"` |
| `recipes_lora_ddp_4x5090.py` | 4 | the same per rank, `PARALLEL="ddp"` |
| `recipes_full_5090.py` | 1 | full finetune, `ckpt-ring8-actoff-optstream`, `MAX_TOKENS=36864` |
| `recipes_full_ddp_4x5090.py` | 4 | the same per rank, `PARALLEL="ddp"` (one shared host copy of the masters) |

## Method

**Hardware and tree.** Four RTX 5090 32 GB (575 W cap) on PCIe without peer-to-peer,
Linux, tree 947155d, torch 2.14.1+cu130, Triton 3.8.0.

**Run.** `scripts/train/moss.py` on the scaling layer
(`configs/moss/scale/scale_lora_bf16.py`) with the HF-layout bf16 weights:

- data: 1 MP images (4096 image tokens plus the caption, about 4000 joint tokens per
  sample), static first-fit token batches (`BATCHING="tokens_ffd"`,
  `TOKEN_BUDGET="exact"`), offline latent and text caches;
- LoRA: dim 32, `attn-mlp`, bypass, AdamW;
- full finetune: AdamW with fp32 masters, no EMA;
- kernels: `TUNE="table"` with warm tables, `COMPILE="off"` unless a row says otherwise;
- 80 steps for LoRA, 25 to 40 for the full finetune; no profiler window.

The recipes keep the base layer's data knobs, except that the LoRA recipes set
`TOKEN_BUDGET="render"` (each caption's length from the offline text cache): their
full-batch estimate leaves 1.7 GiB free, so `MAX_TOKENS` has to bound every batch's real
token count. They also set `BASE_PRECISION="compute"`,
so the frozen base is 16-bit as measured here also on a ComfyUI folder, whose default
DiT file is the int8 one.

**Numbers.**

| column | meaning |
| --- | --- |
| tokens/s | tokens of steps 11 to the end over their time, leaving out steps longer than 3x the median step (one-time recompiles and retunes on a new batch shape); summed over the ranks |
| window | the same with every step of the window |
| s/step | median step time of the window |
| samples/s | samples of the window over its time, per rank |
| TF/s | the trainer's `tflops` metric, per GPU |
| peak | torch peak allocated / the nvidia-smi peak, GiB |
| full batch | the allocated peak extrapolated to a `MAX_TOKENS` batch (per-run fit of the step peak against the batch tokens) plus the allocator's reserved gap and the CUDA context, GiB; the card holds 31.36 GiB |
| scaling | four-GPU tokens/s over 4x the one-GPU tokens/s of the same per-rank setup |

## LoRA, one GPU

The speed follows the share of blocks that recompute in backward: the smallest batch that
holds one sample (4608 tokens) leaves the most memory for uncheckpointed blocks.
`partial:<f>` checkpoints the first `f` of the blocks.

| setup (`MAX_TOKENS=4608`, manual profile) | tokens/s | window | s/step | samples/s | TF/s | peak | full batch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `CHECKPOINT="none"` | out of memory | | | | | | |
| `CHECKPOINT="none"`, `COMPILE="block"` | 6781 (23 steps) | | 0.593 | 1.70 | 162 | 29.2 / 31.3 | out of memory at a 4.1k-token batch |
| `partial:0.15` | 6479 | 5890 | 0.615 | 1.59 | 211 | 28.1 / 29.7 | 30.6 |
| **`partial:0.2`** (recipe) | **6385** | 6189 | 0.628 | 1.67 | 221 | 27.2 / 28.7 | 29.6 |
| `partial:0.25` | 6287 | 5761 | 0.638 | 1.55 | 206 | 26.3 / 27.8 | 28.6 |
| `partial:0.3`, `BASE_PRECISION="fp8_compute"` | out of memory | | | | | | |
| `partial:0.4`, `fp8_compute`, `FP8_DGRAD="16bit"` | 5713 | 5103 | 0.671 | 1.38 | 182 | 24.3 / 25.7 | 25.7 |

- `partial:0.2` is the fastest setting whose full-batch estimate keeps 1.7 GiB free;
  `partial:0.25` keeps 2.7 GiB for a card that also drives a display.
- The fp8 base needs more memory than the bf16 base at the same checkpoint share
  (`partial:0.3` runs out of memory), and its fp8 GEMMs do not make up for the extra
  recompute.

## LoRA, four GPUs

| setup (per rank: `MAX_TOKENS=4608`, manual profile) | tokens/s | window | s/step | samples/s | TF/s | peak | full batch | scaling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **DDP, `partial:0.2`** (recipe) | **23509** | 22827 | 0.646 | 1.47 | 205 | 27.3 / 29.2 | 30.0 | 0.92 |
| DDP, `partial:0.25` | 23435 | 20239 | 0.655 | 1.31 | 181 | 26.4 / 28.3 | 29.0 | 0.93 |
| stock torch DDP (`DDP_BACKEND="torch"`), `partial:0.25` | 23790 | 22361 | 0.656 | 1.44 | 200 | 26.4 / 28.4 | 29.0 | 0.95 |
| pipeline `zbv`, 16 micro-batches, `CHECKPOINT="none"` | out of memory | | | | | | | |

- The gradient all-reduce (284 MB per step) waits 11-12 ms per step (median).
- The median DDP step is 3 % longer than the one-GPU step; the rest of the scaling loss is
  the one-time recompiles of new batch shapes, during which the other ranks wait. The
  three DDP rows differ by less than this step-to-step spread.

## Full finetune, one GPU

The weights stream (`swap`) with block checkpointing and activation offload; the fp32
masters and AdamW moments live on the host and stream through the optimiser step inside
backward. Rows marked "lanes" ran next to three other full-finetune jobs on the same host.

| setup | MAX_TOKENS | tokens/s | s/step | samples/s | TF/s | peak | full batch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **`ckpt-ring8-actoff-optstream`** (recipe, alone) | 36864 | **4088** | 8.76 | 1.08 | 195 | 18.3 / 22.1 | 22.3 |
| `ckpt-ring8-actoff-optstream` (lanes) | 36864 | 4052 | 8.81 | 1.07 | 193 | 18.3 / 22.1 | 22.3 |
| `ckpt-ring4-actoff-optstream` (lanes) | 36864 | 3964 | 8.99 | 1.05 | 189 | 17.4 / 21.3 | 21.4 |
| `ckpt-ring16-actoff-optstream` (lanes) | 36864 | 3937 | 8.98 | 1.04 | 188 | 19.9 / 23.8 | 23.9 |
| `ckpt-ring4-actoff-optstream` (lanes) | 55296 | 4082 | 13.06 | 1.07 | 195 | 21.1 / 25.9 | 26.0 |
| `ckpt-ring4-actoff-optcpu` (lanes) | 36864 | 2461 | 12.85 | 0.65 | 117 | 9.1 / 12.7 | 12.8 |
| `ckpt-resident-optstream` (lanes) | 9216 | 1896 | 3.88 | 0.49 | 91 | 27.5 / 31.3 | 31.5 |
| `ckpt-ring8-optstream` (lanes) | 36864 | out of memory | | | | | |
| `ckpt-resident-optstream`, `ckpt-resident-optcpu` (lanes) | 18432, 36864 | out of memory | | | | | |
| manual `partial:0.3`, `OPTIM_STATE="stream"`, `GRAD_OFFLOAD=True`, `GRAD_ACC=8` (lanes) | 4608 | out of memory | | | | | |

- The swap rows are compute-bound: the ring size and the batch size move the speed by
  less than 4 %.
- A resident full finetune keeps the fp32 masters on the GPU (18.5 GiB after the build),
  which leaves room only for small batches.
- The recipe run used 116 GiB of host RAM.

## Full finetune, four GPUs

| setup | tokens/s | s/step | samples/s (all ranks) | TF/s | peak | scaling |
| --- | --- | --- | --- | --- | --- | --- |
| **DDP, `ckpt-ring8-actoff-optstream`, 36864 per rank** (recipe) | **15271** | 9.25 | 3.92 | 184 | 18.6 / 24.0 | 0.93 |
| pipeline `1f1b`, `ckpt-resident`, 16 x 9216 | 10.3k | 12.54 | 2.70 | 123 | 28.3 / 30.4 | |
| FSDP2, `ckpt-resident`, 18432 per rank | 8306 | 7.83 | 2.23 | 99 | 27.6 / 31.4 | |
| pipeline `interleaved-1f1b`, `ckpt-resident`, 32 x 4608 | out of memory in step 2 | | | | | |

- DDP waits 6 ms per step for its reduction (median); the four ranks used 71 GiB of host
  RAM in all.
- The pipeline and FSDP2 rows trained every step and then ran out of memory in the
  end-of-run checkpoint, which gathers the full-finetune state on the GPU.
