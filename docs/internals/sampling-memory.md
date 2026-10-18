# Sampling memory

A `MossPipeline.generate` call has three phases, and the device holds a different set
of tensors in each. `kohakumoss.inference.peaks.estimated` estimates every phase and
the pipeline logs them before sampling; `scripts/bench/infer/memory.py` measures them
(one row per phase).

| phase | on the device |
| --- | --- |
| text encode | the DiT's placed weights (every resident block, the ring slots of a streamed DiT, the non-block layers) + the text encoder's pass bytes (all of it when resident, the layer ring when streamed, the whole model when offloaded) |
| sampling | the DiT's placed weights + one forward of the largest packed batch (no activation is saved) + the text encoder's and the VAE's resident bytes |
| decode | the DiT's placed weights + the VAE's weights + one VAE pass over the image (`ACT_BYTES_PER_PIXEL` per pixel, one 512 x 512 tile when tiled) |

The peak of a call is the largest phase, not the sampling forward. The measured
`balanced` peak of 7.24 GiB per call (fp16, 1024^2, RTX 3090, text encoder streamed;
the 4.14 GiB of placed DiT weights are 16 resident blocks of 42 plus the slots and the
non-block layers) is those weights plus the largest phase on top; by the estimate it is
the text-encode or the decode phase, the sampling forward's working set being the
smaller one at this size (the per-phase measurement on the real model is
`scripts/bench/infer/memory.py`). `balanced` keeps 16 blocks resident, so it is not the
placement to compare with a small-card training profile: training's 8 GB / 6 GB layers
keep two ring slots of blocks.

What keeps sampling at or below training on the same card:

- the DiT takes the card's training placement: `kohakumoss-config new sample` writes
  the `MEMORY_PROFILE` of the card's training VRAM layer (the smallest layer's below
  6 GiB; resident above the largest), so the DiT weights are the ones training holds;
- the text encoder and the VAE default to `"auto"`, sized against the DiT forward's
  working set (`peaks.working_set` at one 1024^2 CFG sample) on the free memory: the
  text encoder takes the fastest of resident / offload / stream that fits, the VAE the
  fastest of resident / offload, then the same with tiled passes, then a streamed
  tiled pass (`offload.encoders.vae.choose_vae_plan`); training's small-card layers stream the
  text encoder and run the VAE offloaded and tiled in the same way;
- any placement set in `TEXT_ENCODE` / `VAE_ENCODE` is used as given.
