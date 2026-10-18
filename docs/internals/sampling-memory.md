# Sampling memory

What a `MossPipeline.generate` call holds on the device, phase by phase, and the
placement that keeps sampling at or below training on the same card.

- `kohakumoss.inference.peaks.estimated` estimates every phase, and the pipeline logs
  the estimates before sampling.
- `scripts/bench/infer/memory.py` measures them on the real model (one row per phase).

## Phases

| phase | on the device |
| --- | --- |
| text encode | the DiT's placed weights (every resident block, the ring slots of a streamed DiT, the non-block layers) + the text encoder's pass bytes (all of it when resident, the layer ring when streamed, the whole model when offloaded) |
| sampling | the DiT's placed weights + one forward of the largest packed batch (no activation is saved) + the text encoder's and the VAE's resident bytes |
| decode | the DiT's placed weights + the VAE's weights + one VAE pass over the image (`ACT_BYTES_PER_PIXEL` per pixel, one 512 x 512 tile when tiled) |

The peak of a call is the largest phase, not the sampling forward.

Measured example, `balanced`, fp16, 1024^2, RTX 3090, text encoder streamed:

- peak 7.24 GiB per call;
- of it, 4.14 GiB of placed DiT weights: 16 resident blocks of 42, plus the ring slots
  and the non-block layers;
- the rest is the largest phase on top. By the estimate it is the text-encode or the
  decode phase; the sampling forward's working set is the smaller one at this size.

`balanced` keeps 16 blocks resident, so it is not the placement to compare with a
small-card training profile: training's 8 GB / 6 GB layers keep two ring slots of
blocks.

## Keeping sampling at or below training

- **The DiT** takes the card's training placement. `kohakumoss-config new sample`
  writes the `MEMORY_PROFILE` of the card's training VRAM layer (the smallest layer's
  below 6 GiB; resident above the largest), so the DiT weights are the ones training
  holds.
- **The text encoder and the VAE** default to `"auto"`, sized against the DiT forward's
  working set (`peaks.working_set` at one 1024^2 CFG sample) on the free memory
  (`offload.encoders.vae.choose_vae_plan` for the VAE):

  | encoder | `auto` takes the first that fits |
  | --- | --- |
  | text encoder | `resident`, else `ring` |
  | VAE | the fastest of resident / offload; then the same with tiled passes; then a streamed tiled pass |

  Training's small-card layers stream the text encoder and run the VAE offloaded and
  tiled in the same way.
- **Any placement set** in `TEXT_ENCODE` / `VAE_ENCODE` is used as given.
