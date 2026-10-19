# Full finetune

`ALGO="full"` trains the backbone's own parameters (the text refiner stays frozen unless
`TRAIN_TEXT_REFINER=True`). The parameters train in place through `FullTrainable`.

`configs/moss/_base/base_full.py` follows the MOSS tech report:

- Muon + AdamW at 5e-5;
- weight decay 0.01, off for biases / norms / embedders;
- decay-to-one on the last block's q/k norm weights;
- gradient clip 1.0, EMA 0.9999.

## Choosing a setup

State per parameter is about 16 bytes (fp32 master, two Adam moments, gradient), so the
5.1 B-parameter backbone needs one of:

| setup | recipe |
| --- | --- |
| 4x 32 GB with FSDP2 (Linux) | `recipes_full_fsdp_4xv100.py` |
| one 24 GB GPU, host masters + weight streaming | `recipes_full_24g_offload.py` |
| one RTX 5090 32 GB, host masters + weight streaming at 36864 tokens per step | `recipes_full_5090.py` |
| several 24 GB GPUs (Linux), DDP over the same streaming profile | `kohakumoss-config new train --algo full --gpus N` |
| 4x RTX 5090 32 GB on PCIe (Linux), DDP over the 5090 streaming profile | `recipes_full_ddp_4x5090.py` |

A full finetune trains the 16-bit DiT: in a ComfyUI folder it takes
`diffusion_models/moss_image_<VARIANT>_bf16.safetensors` (the default for `ALGO="full"`),
in the HF layout the `transformer/` folder.

```bash
# a config from the presets for this GPU, or one that extends a recipe
kohakumoss-config new train --algo full --weights ComfyUI/models --data <dataset> --name my-ft
kohakumoss-config new --from full_24g_offload my_ft.py    # then set WEIGHTS / DATASETS in it

# the data passes printed by the first command, then
kogine run scripts/train/moss.py --config configs/user/my-ft.py
```

## One GPU

`kohakumoss-config new train --algo full` picks the VRAM layer
([configs.md](configs.md#layers)):

| card | layer |
| --- | --- |
| holds 24 GiB within `--vram-tolerance` (default 0.05: from 22.8 GiB, so a 24 GB card that reports 23.99 GiB gets it) | `24g_full` |
| above 24 GiB | `auto` |
| less | refused |

The single-GPU streaming recipe:

- trains with AdamW (Muon steps whole matrices after backward, which a swapped block
  does not allow), with EMA off;
- needs about 70 GiB of host RAM ([Host RAM](#host-ram)).

## Several GPUs (Linux)

With `--gpus N` the starter picks:

| condition | engine |
| --- | --- |
| every pair of the N cards has peer access (NVLink / P2P) on the machine that runs `kohakumoss-config`, and the per-rank FSDP2 footprint (the `fsdp` layer's resident profile, every parameter, gradient and optimizer state sharded over N ranks; the memory estimate) fits the card within `--vram-tolerance` | FSDP2 |
| otherwise | DDP: the full VRAM layer's swap profile (`ckpt-ring4-actoff-optstream-gradoff`) runs unchanged on every rank, the ranks sharing one host copy of the masters ([multi-gpu.md](multi-gpu.md#shared-host-masters)) |
| `--vram auto` | DDP |

So 4x 32 GB V100 on NVLink gets FSDP2, and cards on plain PCIe get DDP: on 4x RTX 5090
(PCIe, no P2P) a full finetune runs 15.3k tok/s under DDP against 8.3k under FSDP2
([training-5090.md](../performance/training-5090.md)). Two RTX 3090 under DDP with that profile
measured 0.356 samples/s (scaling efficiency 0.79), an nvidia-smi peak of 17.3 GiB per
card and 63 GiB of host RAM in all.

## fp8 / MXFP8 compute (sm89+)

`BASE_PRECISION="mxfp8"` (or `"fp8_compute"` with `FP8_RECIPE`) runs every block
linear's forward and input gradient in fp8 from copies of the weights requantized after
each optimizer step.

- The weight gradient and the optimizer stay 16-bit / fp32.
- The weights stay resident (`stream="off"`, e.g. `MEMORY_PROFILE="ckpt-resident"`) and
  each carries 1 (`FP8_DGRAD="16bit"`) or 2 extra bytes.

See [quant-kernels.md](../internals/quant-kernels.md#full-finetune).

## Exports and EMA

| `SAVE_FORMAT` | output |
| --- | --- |
| `"comfy"` (default) | `<RUN_NAME>-merged.safetensors`: one bare-key safetensors (the ComfyUI single-file layout of [export-to-comfy.md](export-to-comfy.md); q / k / v fused per `SAVE_FUSED_QKV`) |
| `"hf"` | `<RUN_NAME>-diffusers/`: the sharded transformer folder |
| `"both"` / `"none"` | both / neither |

- On a ComfyUI folder the full finetune defaults to the bf16 DiT file; the int8 file
  trains only under `BASE_PRECISION="compute"` (its dequantized weights,
  [weights.md](weights.md#dit)).

- With `EMA` enabled the averaged weights are in every checkpoint (`ema.safetensors`);
  with a power EMA, `kohakumoss.training.ema.posthoc` reconstructs any EMA width after
  training ([training.md](../concepts/training.md#ema)).
- Sample the merged file with `scripts/sample/sample.py --set
  TRANSFORMER=<run>/export/<RUN_NAME>-merged.safetensors` (the text encoder and VAE come
  from `WEIGHTS`; [inference.md](inference.md#cli)).

## Host RAM

The single-GPU recipe (`ckpt-ring4-actoff-optstream-gradoff`, swapped blocks) keeps on
the host:

| part | size |
| --- | --- |
| per block parameter: the fp32 master, the two fp32 Adam moments and the 16-bit streaming copy (14 bytes for each of the 4.76 B block parameters) | about 62 GiB |
| with `GRAD_ACC > 1`: an fp32 host gradient sum per parameter (4 bytes), allocated on the first micro-batch that sends one | about 18 GiB more |
| the trainer process | 8 GiB budgeted (the bench matrix's budget, not a measurement) |
| the offloaded activations at 8192 tokens | about 2 GiB |
| at build: the text encoder for the sample prompts ([configs.md](configs.md#host-ram)) | 15.63 GiB |

Plan for about 70 GiB during the steps and 88 GiB at build. The two-GPU DDP run above,
whose ranks share one copy, measured 63 GiB of host RAM in all. Under FSDP2 the masters
and states are sharded over the GPUs.
