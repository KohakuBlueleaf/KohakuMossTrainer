# Preparing a dataset

Four steps: point a data config at the dataset, check the plan, cache text embeddings and
latents, and (optionally) import into shards. The commands below take the data config
`configs/data/my_data.py`; a starter from `kohakumoss-config new train` writes
`configs/user/<name>_data.py` to use in its place.

## 1. Point a config at it

`kohakumoss-config new train --data <dataset>` writes the data-preparation config
`<name>_data.py` next to the train config, with the same data knobs and the cache
encoders filled in ([configs.md](configs.md)).

By hand: a data config that reads the train config's data knobs and adds the two encoder
lines (section 3) caches exactly what that run reads:

```python
"""My data config. Drives scripts/data/inspect_data.py."""
from kohakuengine import use_config

use_config("../user/my-lora.py")   # the train config
TEXT_ENCODER_KWARGS = {"path": "ComfyUI/models", "device": "cuda"}
IMAGE_ENCODER_KWARGS = {"path": "ComfyUI/models", "device": "cuda"}
```

`configs/data/kohya/kohya_style.py` is a standalone example (set `data_root` and
`WEIGHTS`); its caches land in `<data_root>/cache` like a train config's, and a run
reads them when its key knobs ([data.md](../concepts/data.md#cache-keys)) equal the
example's.

### Kohya-style folders

```
dataset/
  5_some_artist/      image files + same-stem .txt tag files
  1_another/
```

Images directly in `dataset/` are read only with `"include_root": True` in the entry
(repeat 1). A kohya scan that finds no image stops with an error naming the layout, so a
flat folder of images takes `include_root` or `--source folder`.

### Other layouts

Each layout has a data layer (`kohakumoss-config new train --source <name>`,
`configs/moss/data/data_<name>.py`, every source argument spelled out), and one
`DATASETS` entry per source:

```python
{"source": "folder", "path": "dataset", "txt_as": "tags"}            # any tree, .txt / .caption / .json sidecars
{"source": "jsonl", "path": "dataset/train.jsonl", "root": None}    # {"image": "a/1.png", "caption": "...", "tags": [...]} per line
{"source": "parquet", "path": "dataset/train.parquet", "image_col": "image", "bytes_col": None, "root": None}
{"source": "webdataset", "path": "shards/*.tar", "txt_as": "tags"}  # <key>.png + <key>.txt / <key>.json members
{"source": "moss_shard", "path": "dataset_shards"}                  # an imported directory (section 4)
```

The manifest sources (`jsonl`, `parquet`) read:

| field | keys |
| --- | --- |
| image path | `image` / `file` / `path` / `file_name` (`parquet`: the `image_col` column, or encoded images in `bytes_col`), relative to `root` (default: the manifest's folder) |
| captions | `caption` / `text` / `caption_nl` (natural language) or `tags` (a list or a comma-separated string) |
| metadata | `artist`, `rating`, `weight` and `group` |
| anything else | stays in the row's json for the `json` caption template ([data.md](../concepts/data.md#sources)) |

Several entries in `DATASETS` are mixed by their `weight`.

## 2. Check what the planner will do

```bash
kogine run scripts/data/inspect_data.py --config configs/data/my_data.py
```

It prints:

- the catalog: rows, epoch size, size and aspect quantiles, alpha share;
- the plan of epoch 0: samples and tokens per batch, budget fill, crop loss, upscaled
  share, common sizes, over-long fits and drops;
- sample captions.

Tune `RESOLUTION_MODE`, `BATCHING`, `OVERLONG`, `TARGET_AREA`, `MAX_TOKENS`,
`ASPECT_RANGE` and `CAPTION` here, before caching.

## 3. Cache (offline mode)

```bash
kogine run scripts/data/catalog_build.py --config configs/data/my_data.py
kogine run scripts/data/cache_text.py --config configs/data/my_data.py
kogine run scripts/data/cache_latents.py --config configs/data/my_data.py
```

- Run the text pass first: with text lengths cached, `TOKEN_BUDGET="render"` and
  `"exact"` plan token batches and over-long fits on the encoder's real lengths
  (`"estimate"`, the default, uses the catalog's character counts and reads no cache).
- Both passes skip keys already present, so they resume after an interruption.
- On an Intel Arc GPU set `"device": "xpu"` in both encoder kwargs.

### The encoders

The encoders come from the data config: the two `*_ENCODER_KWARGS` lines of section 1
(an empty `path` is an error naming the knob). `TEXT_ENCODER` / `IMAGE_ENCODER` name
the encoder factories (dotted paths; the defaults are the real models):

| factory | loads | `path` |
| --- | --- | --- |
| `kohakumoss.encoders.vae_encoder(path, device, dtype, ...)` | the FLUX.2 VAE; returns its `encode_images` with the VAE's compute spec | the `vae/` folder, a pipeline root or one VAE file ([weights.md](weights.md#vae)) |
| `kohakumoss.encoders.text_encoder(path, device, dtype, plan)` | the Qwen3.5 text encoder (`plan`: TextEncoderPlan fields, checked for the device before the weights load) | a pipeline root or the `text_encoder/` folder |

Placement:

- The text encoder is placed by `TEXT_ENCODER_KWARGS["plan"]`: `memory` unset is `auto`,
  the fastest placement that fits the card (`resident`, else `ring`);
  `{"memory": "ring"}` or `{"memory": "cpu"}` (encode on the host) are used as given.
- The latent pass places the VAE by `VAE_ENCODE` the same way (`auto` when unset).
- Both scripts encode `ENCODE_BATCH` items per pass: captions in `cache_text.py`
  (default 512), images in `cache_latents.py` (default 8).

What the encoders compute comes from the train config the data config `use_config`s:
the latent pass takes `VAE_DTYPE` and `VAE_ENCODE`, the text pass takes `TEXT_ENCODE`'s
`dtype`, `precision`, `quant_group` and `fp8_recipe`; a field set in the kwargs wins.
The encoder's id must equal `TEXT_ENCODER_ID`.

### Cache keys

Both encoder specs enter the cache keys ([data.md](../concepts/data.md#cache-keys)), so
the caches hold what the run keys on, and a run reads the entries only when it computes
the same way.

- Any change to the target sizes, crops, flips, caption template, variants, seed or
  encoder spec changes the keys: re-run the passes after changing those knobs. The
  offline-miss error names the differing key knobs and prints the rebuild command.
- The latent keys include each image's content identity, which for an unimported source
  is a file fingerprint of its size, mtime and absolute path. A cache built from one
  mount of the dataset (`D:/data/my_set` on Windows) therefore misses for the same files
  under another mount (`/mnt/data/my_set` on Linux): build the caches once per mount
  path, or import the dataset into shards (section 4), whose content identity is a hash
  of each image's content.

### Online caches

With `LATENT_CACHE` / `TEXT_CACHE` set to `"online"` in the train config instead, the
trainer encodes the misses during training and writes them back to the same cache files;
`scripts/data/cache_merge.py` folds the per-rank shards of an online cache into the main
files afterwards ([data.md](../concepts/data.md#caches)).

## 4. Optional: import into shards

Add `"import_to": "/path/out"` to a dataset entry and run `catalog_build.py`; then use
`{"source": "moss_shard", "path": "/path/out"}`. Shards are large sequential files,
which suit network storage better than many small images.

A catalog file written in another layout than the current one:

- a scanning source's cached catalog is rebuilt by a rescan (one WARNING naming the
  file);
- a `moss_shard` directory, whose `catalog.db` is its only row store, is an error:
  import its source again with `catalog_build.py` and `import_to`.

The catalog of every scanning source is cached on disk and updated incrementally at
each start; `scripts/data/catalog_cache.py` shows, drops or compacts those files
([data.md](../concepts/data.md#the-catalog-cache)).

## Throughput

```bash
kogine run scripts/bench/data/loader.py --config configs/data/my_data.py
```

writes `runs/bench/data/loader_<mode>.json` (batches/s, images/s, joint tokens/s per
worker count).
