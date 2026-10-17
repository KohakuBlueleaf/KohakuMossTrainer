# Data pipeline

`kohakumoss_data` turns datasets into packed batches. It never imports `kohakumoss`
or `torch.distributed`: rank and world are plain ints, encoders are plain callables,
and the model-side layout (`SeqInfo`) is built by the trainer from `hw` and `text_len`.

```
Source.scan() ──▶ CatalogRow ──▶ Catalog ──▶ Planner(seed, rank, world) ──▶ BatchPlan
                                                         │
                     MossDataset.materialize: captions, text dropout, cache lookups,
                     decode ──▶ HostBatch (pinned) ──▶ trainer
```

## Sources

| source | layout | notes |
| --- | --- | --- |
| `kohya` | `<repeats>_<name>/img.ext` + `.txt` | `repeat_prefix`: `first` (leading int, kohya), `product` (product of all leading ints), `ignore`; `artist_from_folder`: `None`, `"name"` (folder name without numeric prefixes) or a regex |
| `folder` | any tree, sidecars `.txt` / `.caption` / `.json` | `txt_as`: `tags` or `nl`; json keys `tags`, `caption`, `artist`, `rating`, the rest stays in `row.json` |
| `jsonl` | one record per line, image key `image` / `file` / `path` / `file_name` | `weight`, `group` columns honoured |
| `parquet` | path column or bytes column | needs `pyarrow` |
| `webdataset` | uncompressed `.tar`, members `<key>.<ext>` | read by byte offset, no tar parsing at load time |
| `synthetic` | seeded random images + tag captions | smoke runs and tests |
| `moss_shard` | the native format | see below |

Rows are keyed on image files; a sidecar without an image is ignored. Scanning reads
image headers only (size, EXIF orientation, alpha); the content identity used by caches
is a file fingerprint (size + mtime) unless `hash_content=True` or the data has been
imported into shards (then it is a blake2b content hash).

## The `moss_shard` format

```
dataset/
  catalog.db          KohakuVault ColumnVault, one column per CatalogRow field
  shards/00000.bin    concatenated encoded images (rolled at shard_bytes, 1 GiB default)
  shards/00000.idx    32-byte records: offset u64, size u32, w u16, h u16, codec u8,
                      flags u8 (bit 0 = alpha), 6 pad, content-hash prefix u64
```

`scripts/data/catalog_build.py` imports any source (`import_to`) or just writes its
catalog (`catalog`) so later runs skip the scan.

## Captions

Templates (`CAPTION = {"template": ...}`):

- `tags`: keep fields first (`keep`, e.g. `artist`), then `keep_first` fixed tags, then
  the rest. A `|||` tag ends the fixed part; `%%` tags split groups. Each group drops
  with `group_dropout`, each tag with `tag_dropout`; groups and tags shuffle when
  `shuffle`. `underscores` turns `_` into spaces, `unescape` turns `\(` into `(`.
- `nl`: `caption_nl`, keep fields prepended as `a, b. `, falling back to tags.
- `mix`: natural language with probability `p_nl`, tags otherwise.
- `json`: `row.json` plus listed row fields as `key: value` lines or compact JSON, with
  `field_dropout` (fields in `keep` never drop).

Caption variant `k` of row `r` is rendered from a generator seeded by
`(seed, r, k)`, so there are exactly `CAPTION_VARIANTS` captions per row: what the text
cache stores is what training draws.

## Resolution, batching, over-long samples

Three independent choices. All sizes are multiples of `SIZE_STEP` (16 = VAE /8 x
patchify 2); `hw` in a batch is the latent grid (pixels / 16).

`RESOLUTION_MODE` decides each sample's target size:

| mode | target size |
| --- | --- |
| `resize` (default) | native aspect, area ~= `TARGET_AREA`: every sample has nearly the same image-token count |
| `bucket` | nearest log-aspect bucket of ~`TARGET_AREA`; with `UPSCALE="never"` a smaller image keeps its native size as its own bucket; batches never mix buckets |
| `square_crop` | side sqrt(`TARGET_AREA`), centre crop |
| `shifted_square` | same, seeded crop position |

`ASPECT_RANGE` bounds w/h (images outside are cropped to the bound). `UPSCALE="never"`
keeps images smaller than the target at their native size. `DataConfig.crop` is
`center` (default) or `random`; an offline latent cache needs `center`.

`BATCHING` decides which samples share a batch:

| mode | rule |
| --- | --- |
| `tokens` (default) | walk the seeded sample order, adding samples (image + text tokens) while the batch stays <= `MAX_TOKENS`; the sample that would exceed it starts the next batch |
| `tokens_ffd` | first-fit-decreasing per window of 256 samples into `MAX_TOKENS` bins (denser batches, order shuffled) |
| `count` | `BATCH_SIZE` consecutive samples (trailing partial batch dropped) |

`MAX_SAMPLES` additionally caps the samples per token batch (0 = no cap).
`TOKEN_BUDGET="exact"` takes every caption length from the offline text cache, so
`MAX_TOKENS` is a hard bound; a static pipeline micro-batch shape (`MAX_TOKENS`,
`MAX_SAMPLES`) needs it (`kohakumoss_data.pipeline_problems` lists what is missing).

Every `DataConfig` field except `latent_factor` is a train-script knob of the same
name in upper case (`DataConfig.from_knobs`), including `CROP`, `PLANNER_KWARGS`
(`window`, `drop_last`, ...; these override the planner arguments derived from the
other knobs), `LATENT_ENCODER_ID`, `TEXT_ENCODER_ID`, `LATENT_DRAW` (also the VAE draw
used for online encoding), `BACKGROUND`, `RESAMPLE`, `TEXT_CHARS_PER_TOKEN`,
`TEXT_OVERHEAD_TOKENS`, `MAX_TEXT_TOKENS`, `PREFETCH` and `PIN_MEMORY`.
`kohakumoss_data.data_problems(cfg)` lists every value that would make
`MossDataset(cfg)` fail.

`OVERLONG` decides what happens to a single sample longer than `MAX_TOKENS`:

| mode | effect |
| --- | --- |
| `drop` (default) | skipped |
| `truncate_text` | only the first `MAX_TOKENS - image tokens` text tokens are kept. The text encoder is causal, so a prefix of the cached embedding is exactly the embedding of the caption's prefix; the end of the chat template (`<|im_end|>` / assistant turn) is cut with it |
| `drop_tokens` | `MAX_TOKENS - text tokens` image tokens are kept, chosen by a seeded draw; `img_keep[i]` holds their sorted indices into the row-major grid, `latents` holds only them, and `build_seqinfo(hw, text_len, img_keep=...)` gives each kept token its original grid position |

A fitted sample keeps at least one text token and one image token; otherwise it is
dropped. With online encoding the loader returns the full image (`pixels`) and full
prompts plus `img_keep` / `text_limit`; the trainer encodes, then gathers / cuts.
`MAX_TOKENS` applies per sample under `count` batching too.

Each epoch is a seeded permutation of the weighted row multiset (row i appears
floor(w) times, plus once more with probability frac(w)); each occurrence gets a
seeded crop / flip / caption variant / keep draw. Text tokens for planning come from
the offline text cache when it has the caption, otherwise (and always with
`TEXT_CACHE="online"` or `"off"`) from `ceil(chars / text_chars_per_token) +
text_overhead_tokens`. An online cache grows during the run, so planning never reads
it: the plan is a pure function of the data, the seed and the knobs, and a restart or
resume maps every index to the same samples. Global batch `g = index * world + rank`,
so ranks never share a batch.

## Loader

`Loader` wraps a `DataLoader(batch_size=None)` over a `BatchStream`: worker `w` of
`W` materialises indices `start + w, start + w + W, ...`, and the DataLoader returns
them in order, so the stream is the same for any worker count. Resume state is
`{seed, start}`. Workers use `spawn` on every OS, so the dataset (sources, catalog,
templates, caches) is pickled by value; file handles reopen lazily per process.

`Loader(transform=...)` / `build_loader(cfg, rank, world, transform=...)` (a picklable
callable or a dotted path) runs on every batch inside the worker, before the
DataLoader pins it. The trainer passes `kohakumoss.training.step.LayoutTransform`,
which builds the batch's SeqInfo there (when the text lengths are final) and attaches
it as `hb.layout`, so the layout is not built on the training thread.

Text dropout (`TEXT_DROP`) is decided per (seed, index, rank, slot) before lengths are
known: a dropped sample's caption is `""`, and its embedding is the cached empty
prompt.

## Caches

`LATENT_CACHE` and `TEXT_CACHE` are `off | offline | online`; both live in `CACHE_DIR`
(`latents.db`, `text.db`, KohakuVault). An empty `CACHE_DIR` resolves to `<path>/cache`
of the first dataset with a `path` (its folder when the path is a file or a glob)
(`kohakumoss_data.resolved_cache_dir`); the trainer and the cache scripts use the same
rule. Keys:

- latent: content identity x encoder id x storage dtype x target size x crop box x
  flip x background (alpha images only) x resample filter x VAE draw
  (`latent_draw`, `mode` or `sample`)
- text: rendered caption x encoder id x storage dtype

`offline` raises on a miss (run the cache scripts first); `online` returns pixels /
prompts plus `latent_keys` / `text_keys` so the trainer can encode and write the
results back; `off` always returns pixels / prompts. Storage dtype `bf16`, `fp16`,
`fp32` or `fp8` (e4m3 with a per-tensor fp32 scale `absmax / 448`, about 7% max
relative error, half the bytes of bf16).

A cache hit returns the stored (rounded) value. `LatentCache.as_cached` /
`TextCache.as_cached` round a fresh encoding exactly the same way, on its device, so
an online run trains on identical values whether a sample hits or misses.

Online write-back is `CacheWriter`, the trainer's `on_encoded` hook: it queues each
batch's fresh encodings and a background thread writes them. Only full values are
written: full-grid latents (a kept-token subset is not the value of its key) and text
that was not cut by `text_limit`. Every rank of a multi-process run writes the same
vault; a busy vault is retried with backoff (up to `busy_timeout`, 600 s) instead of
failing the run.

The text cache's lengths are read through a packed length table (`idx:lengths`, 20
bytes per entry, binary-searched) that the text cache pass writes at its end; a vault
without the table is scanned once at startup.

Sizes at MOSS shapes: a 1 MP latent is 64 x 64 x 128 bf16 = 1 MiB; a text embedding is
L x 4096 bf16 = 8 KiB per token, so a 300-token caption is ~2.4 MiB per variant.
