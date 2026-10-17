# Preparing a dataset

## 1. Point a config at it

Copy `configs/data/data_kohya_style.py` and set `DATA_ROOT`. A kohya-style tree looks
like:

```
dataset/
  5_some_artist/      image files + same-stem .txt tag files
  1_another/
```

Other layouts: `"source": "folder"` (any tree with sidecars), `"jsonl"`, `"parquet"`,
`"webdataset"`, or an already imported `"moss_shard"` directory. Several entries in
`DATASETS` are mixed by their `weight`.

## 2. Check what the planner will do

```bash
kogine run scripts/data/inspect_data.py --config configs/data/my_data.py
```

It prints the catalog (rows, epoch size, size and aspect quantiles, alpha share), the
plan of epoch 0 (samples and tokens per batch, budget fill, crop loss, upscaled share,
common sizes, over-long fits and drops) and sample captions. Tune `RESOLUTION_MODE`,
`BATCHING`, `OVERLONG`, `TARGET_AREA`, `MAX_TOKENS`,
`ASPECT_RANGE` and `CAPTION` here, before caching.

## 3. Cache (offline mode)

```bash
kogine run scripts/data/catalog_build.py --config configs/data/my_data.py
kogine run scripts/data/cache_text.py --config configs/data/my_data.py \
    --set TEXT_ENCODER=<factory returning an EncodeText callable>
kogine run scripts/data/cache_latents.py --config configs/data/my_data.py \
    --set IMAGE_ENCODER=kohakumoss.encoders.vae_encoder
```

`kohakumoss.encoders.vae_encoder(path, device, dtype)` loads the FLUX.2 VAE (`path` is
the `vae/` folder or a pipeline root) and returns its bound `encode_images`; set
`IMAGE_ENCODER_KWARGS = {"path": ..., "device": "cuda", "dtype": "fp32"}` in the data
config.

Run the text pass first: with text lengths cached, token batching and over-long fits
use exact text lengths.
Both passes skip keys already present, so they resume after an interruption. Any change
to the target sizes, crops, flips, caption template, variants or seed changes the keys:
re-run the passes after changing those knobs.

## 4. Optional: import into shards

Add `"import_to": "/path/out"` to a dataset entry and run `catalog_build.py`; then use
`{"source": "moss_shard", "path": "/path/out"}`. Shards are large sequential files,
which suit network storage better than many small images.

## Throughput

```bash
kogine run scripts/bench/data/loader.py --config configs/data/my_data.py
```

writes `runs/bench/data/loader_<mode>.json` (batches/s, images/s, joint tokens/s per
worker count).
