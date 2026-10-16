# Full finetune

`ALGO="full"` trains the backbone's own parameters (the text refiner stays frozen unless
`TRAIN_TEXT_REFINER=True`). `configs/moss/_base/base_full.py` follows the MOSS tech
report: Muon + AdamW at 5e-5, weight decay 0.01 off for biases / norms / embedders,
decay-to-one on the last block's q/k norm weights, gradient clip 1.0, EMA 0.9999.

State per parameter is about 16 bytes (fp32 master, two Adam moments, gradient), so the
5.1 B-parameter backbone needs one of:

| setup | recipe |
| --- | --- |
| 4x 32 GB with FSDP2 | `recipes_full_fsdp_4xv100.py` |
| one 24 GB GPU, host masters + weight streaming | `recipes_full_24g_offload.py` |

Exports: `EXPORT_FORMATS=("merged",)` writes one bare-key safetensors loadable as a
ComfyUI diffusion model; `"diffusers"` writes the sharded transformer folder. With
`EMA` enabled the averaged weights are in every checkpoint (`ema.safetensors`); with a
power EMA, `kohakumoss.training.ema.posthoc` reconstructs any EMA width after training.
