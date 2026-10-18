# Full finetune

`ALGO="full"` trains the backbone's own parameters (the text refiner stays frozen unless
`TRAIN_TEXT_REFINER=True`). `configs/moss/_base/base_full.py` follows the MOSS tech
report: Muon + AdamW at 5e-5, weight decay 0.01 off for biases / norms / embedders,
decay-to-one on the last block's q/k norm weights, gradient clip 1.0, EMA 0.9999.

State per parameter is about 16 bytes (fp32 master, two Adam moments, gradient), so the
5.1 B-parameter backbone needs one of:

| setup | recipe |
| --- | --- |
| 4x 32 GB with FSDP2 (Linux) | `recipes_full_fsdp_4xv100.py` |
| one 24 GB GPU, host masters + weight streaming | `recipes_full_24g_offload.py` |
| several 24 GB GPUs (Linux), DDP over the same streaming profile | `kohakumoss-config new train --algo full --gpus N` |

```bash
# a config from the presets for this GPU, or one that extends a recipe
kohakumoss-config new train --algo full --weights models/MOSS-Image-Base --data <dataset> --name my-ft
kohakumoss-config new --from full_24g_offload my_ft.py    # then set WEIGHTS / DATASETS in it

# the data passes printed by the first command, then
kogine run scripts/train/moss.py --config configs/user/my-ft.py
```

`kohakumoss-config new train --algo full` picks `24g_full` on a card that holds 24 GiB
within `--vram-tolerance` (default 0.05: from 22.8 GiB, so a 24 GB card that reports
23.99 GiB gets it), `auto` above 24 GiB, and refuses less
([configs.md](configs.md#layers)). The single-GPU streaming recipe trains with AdamW
(Muon steps whole matrices after backward, which a swapped block does not allow) and
without EMA. The single-GPU recipe needs about 70 GiB of host RAM
([Host RAM](#host-ram)).

With `--gpus N` (Linux) the starter picks FSDP2 only when the per-rank FSDP2 footprint
(the `fsdp` layer's resident profile, every parameter, gradient and optimizer state
sharded over N ranks; the memory estimate) fits the card within `--vram-tolerance`;
otherwise it picks DDP, and the full VRAM layer's swap profile
(`ckpt-stream4-actoff-optstream-gradoff`) runs unchanged on every rank, the ranks sharing
one host copy of the masters ([multi-gpu.md](multi-gpu.md#shared-host-masters)).
`--vram auto` always gives DDP. So 4x 32 GB gets FSDP2 and 2x 24 GB gets DDP; two RTX
3090 under DDP with that profile measured 0.356 samples/s (scaling efficiency 0.79), an
nvidia-smi peak of 17.3 GiB per card and 63 GiB of host RAM in all.

The parameters train in place through `FullTrainable`; no LyCORIS module is built.

fp8 / MXFP8 compute (sm89+): `BASE_PRECISION="mxfp8"` (or `"fp8_compute"` with
`FP8_RECIPE`) runs every block linear's forward and input gradient in fp8 from copies of
the weights requantized after each optimizer step; the weight gradient and the optimizer
stay 16-bit / fp32. The weights stay resident (`stream="off"`, e.g.
`MEMORY_PROFILE="ckpt-resident"`) and each carries 1 (`FP8_DGRAD="16bit"`) or 2 extra
bytes. See [quant-kernels.md](../internals/quant-kernels.md#full-finetune).

Exports: `EXPORT_FORMATS=("merged",)` writes one bare-key safetensors (the single-file
transformer layout of [export-to-comfy.md](export-to-comfy.md)); `"diffusers"` writes
the sharded transformer folder. With
`EMA` enabled the averaged weights are in every checkpoint (`ema.safetensors`); with a
power EMA, `kohakumoss.training.ema.posthoc` reconstructs any EMA width after training
([training.md](../concepts/training.md#ema)). Sample the merged file with
`scripts/sample/sample.py --set TRANSFORMER=<run>/export/<RUN_NAME>-merged.safetensors`
(the text encoder and VAE come from `WEIGHTS`; [inference.md](inference.md#cli)).

## Host RAM

The single-GPU recipe (`ckpt-stream4-actoff-optstream-gradoff`, swapped blocks) keeps
on the host, for every block parameter, the fp32 master, the two fp32 Adam moments and
the 16-bit streaming copy: 14 bytes for each of the 4.76 B block parameters, about
62 GiB. With `GRAD_ACC > 1` each parameter also gets an fp32 host gradient sum (4 bytes,
about 18 GiB more), allocated on the first micro-batch that sends one. With the trainer
process (the bench matrix budgets 8 GiB), the offloaded activations (about 2 GiB at 8192
tokens) and, at build, the 15.63 GiB text encoder for the sample prompts
([configs.md](configs.md#host-ram)), plan for about 70 GiB during the steps and 88 GiB
at build. The two-GPU DDP run above, whose ranks share one copy, measured 63 GiB of
host RAM in all. Under FSDP2 the masters and states are sharded over the GPUs.
