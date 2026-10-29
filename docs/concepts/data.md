# Data pipeline

`kohakumoss_data` turns datasets into packed batches. Its one first-party dependency
is `kohakumoss_common`: rank and world are plain ints, encoders are plain callables,
and the model-side layout (`SeqInfo`) is built by the trainer from `hw` and `text_len`.

```
Source.walk() ──▶ catalog file (cache) ──▶ Catalog (columns) ──▶ Planner(seed, rank,
                                                                  world) ──▶ BatchPlan
                                                         │
                     MossDataset.materialize: rows fetched, captions, text dropout,
                     cache lookups, decode ──▶ HostBatch (pinned) ──▶ trainer
```

The how-to (layouts, cache scripts, import into shards) is
[datasets.md](../guides/datasets.md). This page covers, in order: dataset entries and
sources, captions, resolution and batching, caches, then the catalog, the batch, the
loader and the file formats.

## Dataset entries

Each `DATASETS` item is `{"source": name, **source arguments}` plus the dataset-level
keys, which the source never sees:

| key | effect |
| --- | --- |
| `weight` | multiplies every row weight of the entry (mixing between datasets) |
| `caption` | the entry's caption template (default: `CAPTION`) |
| `loss_weight` | per-sample loss weight of the entry's rows (`HostBatch.weight`) |
| `catalog` | a prebuilt catalog file, read instead of scanning when it exists |
| `import_to` | `catalog_build.py` only: copy the images into a `moss_shard` directory |

## Sources

| source | layout | arguments and fields |
| --- | --- | --- |
| `kohya` | `<repeats>_<name>/img.ext` + `.txt` | `repeat_prefix`: `first` (leading int, kohya), `product` (product of all leading ints), `ignore`; `artist_from_folder`: `None`, `"name"` (folder name without numeric prefixes) or a regex; `include_root` (default False): also the images directly in `path`, repeat 1, no artist; no image found is an error naming the layout and `include_root` |
| `folder` | any tree, sidecars `.txt` / `.caption` / `.json` | `txt_as`: `tags` or `nl`; `recursive` (default True); json keys `tags`, `caption`, `artist`, `rating`, the rest stays in `row.json` |
| `jsonl` | one record per line, image key `image` / `file` / `path` / `file_name` | `root` (image paths relative to it; default the file's folder); record keys `tags` (list or string), `caption` / `text` / `caption_nl`, `artist`, `rating`, `weight`, `group`, the rest in `row.json` |
| `parquet` | path column (`image_col`, default `image`) or bytes column (`bytes_col`) | `root` as for `jsonl`; the same columns as the jsonl keys; needs `pyarrow` (extra `parquet`) |
| `webdataset` | uncompressed `.tar` (`path` is a glob), members `<key>.<ext>` | `txt_as` for `<key>.txt`; `<key>.json` with the folder json keys; members read by byte offset at load time |
| `moss_shard` | the native format | see [The `moss_shard` format](#the-moss_shard-format) |
| `gbc` | a GBC dataset (KohakuGBC): `dataset.json` + `shards/shard_XXXXX/{images.zip, graphs.db, meta.db, manifest.parquet}` | `shards` (default all); needs `kohakugbc-data` (extra `gbc`, Python 3.11+); captions through the `gbc` template, see [GBC datasets](#gbc-datasets) |

- Rows are keyed on image files; a sidecar without an image is ignored.
- Scanning reads image headers only (size, EXIF orientation, alpha).
- **Content identity** (what the caches key on):

  | identity | when |
  | --- | --- |
  | file fingerprint: absolute path + size + mtime | default for `kohya`, `folder`, `jsonl`, `parquet` (path column) |
  | blake2b of the image bytes | `hash_content=True` on those sources; always for `webdataset` and a parquet bytes column; data imported into shards |

  The path makes two files with equal size and mtime (archives extracted with
  second-resolution timestamps, `cp -p` copies) two identities; it also ties a
  fingerprint-keyed cache entry to the file's location, so moving the dataset re-encodes
  it, while content hashes are location-free.

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
- `gbc`: a `gbc` source's samples from their graph and metadata, see
  [GBC captions](#gbc-captions).

A template is bound to its dataset entry once, at build (`bind_template`): a template
with a `bind(source, geometry_of, max_chars)` reads the source's records, the row's
geometry (its target size and centre crop) and the caption length the text budget
allows (`MAX_TEXT_TOKENS` under the planner's chars-per-token estimate); the others
render from the row alone. A template that follows the row's crop (`geometric`) refuses
`CROP="random"`, `RESOLUTION_MODE="shifted_square"` and `FLIP`.

Caption variant `k` of row `r` is rendered from a generator seeded by `(seed, r, k)`, so
there are exactly `max(1, CAPTION_VARIANTS)` captions per row (0 draws variant 0): what
the text cache stores is what training draws, since the cache pass takes the count from
the planner.

Text dropout (`TEXT_DROP`) is decided per (seed, index, rank, slot) before lengths are
known: a dropped sample's caption is `""`, and its embedding is the cached empty
prompt.

## Resolution, batching, over-long samples

Four independent choices: target size, batching, text-token budget and over-long
samples. All sizes are multiples of `SIZE_STEP` (16 = VAE /8 x patchify 2); `hw` in a
batch is the latent grid (pixels / 16).

### Target size: `RESOLUTION_MODE`

| mode | target size |
| --- | --- |
| `resize` (default) | native aspect, area ~= `TARGET_AREA`: every sample has nearly the same image-token count |
| `bucket` | nearest log-aspect bucket of ~`TARGET_AREA`; with `UPSCALE="never"` a smaller image keeps its native size as its own bucket; batches never mix buckets |
| `square_crop` | side sqrt(`TARGET_AREA`), centre crop |
| `shifted_square` | same, seeded crop position |

- Sizes are (h, w) pixels; a crop box is (left, top, right, bottom) in source pixels
  with the target's aspect, cropped and then resized to the target.
- `ASPECT_RANGE` bounds w/h (images outside are cropped to the bound).
- `UPSCALE="never"` keeps images smaller than the target at their native size.
- `CROP` (`DataConfig.crop`) is `center` (default) or `random`; an offline latent cache
  needs `center`.

### Batching: `BATCHING`

| mode | rule |
| --- | --- |
| `tokens` (default) | walk the seeded sample order, adding samples (image + text tokens) while the batch stays <= `MAX_TOKENS`; the sample that would exceed it starts the next batch |
| `tokens_ffd` | first-fit-decreasing per window of 256 samples into `MAX_TOKENS` bins (denser batches, order shuffled) |
| `count` | `BATCH_SIZE` consecutive samples (trailing partial batch dropped) |

- `MAX_SAMPLES` additionally caps the samples per token batch (0 = no cap; the cap is
  then set above any batch size, so every batching function applies it
  unconditionally).
- `bucket` runs the batching function once per bucket; `bucket` and `tokens_ffd`
  batches are shuffled after batching.

### Text tokens in planning: `TOKEN_BUDGET`

| value | text tokens |
| --- | --- |
| `estimate` (default) | from the catalog's char columns: the dataset's caption template estimates each row's caption length without dropout (`estimate_chars`), then `ceil(chars / TEXT_CHARS_PER_TOKEN) + TEXT_OVERHEAD_TOKENS`, capped at `MAX_TEXT_TOKENS`; no row is read |
| `render` | each sampled caption is rendered (rows fetched): its offline text-cache length when cached, else the same estimate of the rendered string |
| `exact` | every caption's length from the offline text cache, so `MAX_TOKENS` is a hard bound; a static pipeline micro-batch shape (`MAX_TOKENS`, `MAX_SAMPLES`) needs it (`kohakumoss_data.pipeline_problems` lists what is missing) |

### Over-long samples: `OVERLONG`

What happens to a single sample longer than `MAX_TOKENS`:

| mode | effect |
| --- | --- |
| `drop` (default) | skipped |
| `truncate_text` | only the first `MAX_TOKENS - image tokens` text tokens are kept. The text encoder is causal, so a prefix of the cached embedding is exactly the embedding of the caption's prefix; the end of the chat template (`<|im_end|>` / assistant turn) is cut with it |
| `drop_tokens` | `MAX_TOKENS - text tokens` image tokens are kept, chosen by a seeded draw; `img_keep[i]` holds their sorted indices into the row-major grid, `latents` holds only them, and `build_seqinfo(hw, text_len, img_keep=...)` gives each kept token its original grid position |

- A fitted sample keeps at least one text token and one image token; otherwise it is
  dropped.
- With online encoding the loader returns the full image (`pixels`) and full prompts
  plus `img_keep` / `text_limit`; the trainer encodes, then gathers / cuts.
- `MAX_TOKENS` applies per sample under `count` batching too.

### Data knobs

Every `DataConfig` field except `latent_factor` is a train-script knob of the same name
in upper case (`DataConfig.from_knobs`), including `CROP`, `PLANNER_KWARGS` (`window`,
`drop_last`, ...; these override the planner arguments derived from the other knobs),
`LATENT_ENCODER_ID`, `TEXT_ENCODER_ID`, `LATENT_ENCODER_SPEC` / `TEXT_ENCODER_SPEC` (set
by the trainer and the cache passes, see [Cache keys](#cache-keys)), `LATENT_DRAW` (also
the VAE draw used for online encoding), `BACKGROUND`, `RESAMPLE`,
`TEXT_CHARS_PER_TOKEN`, `TEXT_OVERHEAD_TOKENS`, `MAX_TEXT_TOKENS`, `PREFETCH` and
`PIN_MEMORY`. `kohakumoss_data.data_problems(cfg)` lists every value that would make
`MossDataset(cfg)` fail.

### The plan

- Each epoch is a seeded permutation of the weighted row multiset (row i appears
  floor(w) times, plus once more with probability frac(w)); each occurrence gets a
  seeded crop / flip / caption variant / keep draw.
- Text tokens for planning follow `TOKEN_BUDGET` (above); an online cache grows during
  the run, so planning never reads it: the plan is a pure function of the data, the
  seed and the knobs, and a restart or resume maps every index to the same samples.
- Tombstoned rows have weight 0 and are never planned.
- Global batch `g = index * world + rank`, so ranks never share a batch; `g` is located
  by walking the epoch lengths, and the planner keeps its last five epochs.

## Caches

`LATENT_CACHE` and `TEXT_CACHE` are `off | offline | online`; both live in `CACHE_DIR`
(`latents.db`, `text.db`, KohakuVault). An empty `CACHE_DIR` resolves to `<path>/cache`
of the first dataset with a `path` (its folder when the path is a file or a glob)
(`kohakumoss_data.resolved_cache_dir`); the trainer and the cache scripts use the same
rule.

| mode | on a miss |
| --- | --- |
| `offline` | raises `CacheMiss` (run the cache scripts first) |
| `online` | returns pixels / prompts plus `latent_keys` / `text_keys` so the trainer can encode and write the results back |
| `off` | always returns pixels / prompts |

Storage dtype: `bf16`, `fp16`, `fp32` or `fp8` (e4m3 with a per-tensor
fp32 scale `absmax / 448`, about 7% max relative error, half the bytes of bf16).

A cache hit returns the stored (rounded) value. `LatentCache.as_cached` /
`TextCache.as_cached` round a fresh encoding exactly the same way, on its device, so
an online run trains on identical values whether a sample hits or misses.

### Cache keys

- latent: content identity x encoder x storage dtype x target size x crop box x
  flip x background (alpha images only) x resample filter x VAE draw
  (`latent_draw`, `mode` or `sample`)
- text: rendered caption x encoder x storage dtype

The encoder part is the encoder id plus its compute spec (`LATENT_ENCODER_SPEC`,
`TEXT_ENCODER_SPEC`; `kohakumoss_data.encoder_key_id`):

- the VAE's compute dtype (`fp32` / `bf16` / `fp16`);
- the text encoder's `<dtype>/<precision>` plus the quantization group of a weight-only
  int8 / fp8 base or the fp8 kernel recipe (`kohakumoss.encoders.specs`).

A VAE in fp16 and one in bf16, or an int8 text encoder and a 16-bit one, give different
encodings, so a cache built one way never serves a run that computes the other way: the
run misses and the miss message names the differing spec. The placement (memory mode,
batching, layout, kernels) is not in the key. The cache passes take the spec from the
encoder they build (`encoder_spec`) and refuse a dataset whose spec differs; the trainer
derives it from its own encoder knobs (`VAE_DTYPE`, `TEXT_ENCODE` resolved on its
device; `training.build.model.data_knobs`).

`MAX_TOKENS` is not a key knob: the caches hold full-grid latents and whole captions,
whatever the batch budget.

### Offline precompute

`scripts/data/cache_latents.py` / `cache_text.py` fill `latents.db` / `text.db` for
`offline` (tests/data/cache/test_cache.py). The latent pass reads each chunk's image
bytes on the main thread and decodes them on `NUM_WORKERS` threads, one chunk ahead of
the encoder.

The text cache's lengths are read through a packed length table (`idx:lengths`, 20
bytes per entry, binary-searched) that the text cache pass writes at its end; a vault
without the table is scanned once at startup.

### Online write-back

An `online` run writes its misses back through `CacheWriter`
(tests/data/cache/test_writer.py), so a later epoch or run hits. Online latents come
from the VAE under its `VAE_ENCODE` placement, `ahead` host batches per VAE pass
([offload.md](../internals/offload.md#vae)).

- `CacheWriter` is the trainer's `on_encoded` hook: it copies each batch's fresh
  encodings into pinned host memory on a side stream (after the compute stream's work,
  one event per batch), queues the host tensors, and a background thread waits for the
  event and writes them; the queue holds host tensors and every copy runs on the side
  stream.
- Only full values are written: full-grid latents (a kept-token subset is not the value
  of its key) and text that was not cut by `text_limit`.
- Each rank writes only its own shards, `latents.r<rank>.kv` / `text.r<rank>.kv` next to
  the main files, so no file has two writers. Every lookup (`get`, `in`, the lengths)
  reads the union of the main file and the shards present (re-listed at most every
  5 s).
- `scripts/data/cache_merge.py` (`kohakumoss_data.cache.store.merge_shards`, one
  process, no run writing) folds the shards into the main file and deletes them.

### Cache misses

Each cache pass, and each `online` run, records the knobs that shape its keys in
`<cache_dir>/latents.settings.json` / `text.settings.json` (one entry per distinct build;
`kohakumoss_data.cache.settings.FIELDS`). An `offline` miss raises `CacheMiss`, which:

- names the first missing sample and the key knobs that differ from the nearest
  recorded build (all of the run's key knobs when the cache recorded none);
- prints the key knobs as data-config lines and the cache pass with every key knob
  passed as `--set` (PowerShell quoting on Windows, POSIX elsewhere), so the rebuild
  matches the run (tests/data/cache/test_cache_miss.py).

A cache whose recorded builds lack one of the key knobs was written under other keys,
so every entry misses; the message says so and gives the same rebuild command.

## The catalog

`Catalog` holds the planning columns in numpy (`w`, `h` int64, `weight` float32,
`has_alpha` and `live` bool, `group_id` int64 into `groups`, the caption-source char
counts `chars[...]`, `source` int64).

- `catalog[i]` builds a `CatalogRow` from the columns and the row's variable fields
  (locator, tags, caption_nl, artist, rating, json, extra), fetched on demand through a
  per-process LRU (4096 rows).
- Planners, the latent cache pass, the statistics and `Catalog.concat` read only the
  columns; rows are fetched when a batch is materialised or a caption rendered.
- `Catalog(rows)` builds the same object from in-memory rows.

### The catalog cache

Scanning sources (`kohya`, `folder`, `jsonl`, `parquet`, `webdataset`) list walk
items cheaply and probe them expensively. A walk item has a stable key, an 8-int stamp
and a probe:

| source | item | stamp |
| --- | --- | --- |
| `kohya`, `folder` | an image | (size, mtime_ns) of the image and of its `.txt`, `.caption`, `.json` (-1 when absent), from one `os.scandir` per directory |
| `jsonl`, `parquet` (path column) | a record, key `locator#n` | the image's (size, mtime_ns) plus a 63-bit hash of the record |
| `parquet` (bytes column) | the file | its (size, mtime_ns) |
| `webdataset` | a tar | its index, size, mtime_ns |
| `gbc` | a shard | (size, mtime_ns) of its `manifest.parquet`, `graphs.db`, `meta.db`, `images.zip` |

Each dataset entry has a catalog file `<cache>/catalog/<source>-<digest>.db` (the digest
covers the source kind, its absolute path and its options); `<cache>` is the latent /
text cache dir, or the dir it would be with those caches off.

At startup the writer walks the source and compares stamps:

| item | action |
| --- | --- |
| new | probed and appended |
| changed | re-probed and rewritten in place (same ids) |
| not found by the walk | tombstoned (`live` 0, weight 0; its id is never reused for another item) |

A cold scan appends in walk order, so its ids equal an uncached scan's. Probing runs in
`DATA_SCAN_WORKERS` spawn processes that return rows; only the writer touches the file,
committing every 4096 rows. One line reports `catalog: N images (M from cache, K
probed) in X s`.

| `DATA_RESCAN` | scan |
| --- | --- |
| `False` (default) | incremental |
| `True` | re-probe every item in place |
| `"never"` | open the file without walking: for a large dataset known not to change |

`scripts/data/catalog_cache.py` shows, drops or compacts the files (`ACTION`,
`kohakumoss_data.scan.manage`); a compaction rewrites the live rows in walk order with
new ids, copying unchanged rows, which is exactly what a cold scan writes.

**Writer coordination.**

- The writer takes `<file>.lock` (O_CREAT | O_EXCL, holding its pid and host; a lock
  whose pid is gone on this host is stale), clears `<file>.ready`, updates the file,
  writes `<file>.ready` (generation, row count) and releases the lock.
- A reader (`MossDataset(cfg, writer=False)`, or a writer that finds a live lock) waits
  up to `DATA_WAIT` seconds for no live lock and a marker, then opens the file.
- A lock file that cannot be read (its writer has not written it yet) counts as stale
  after 60 s; a lock from another host is never stale.
- The trainer builds the dataset on rank 0 first, then the other ranks build it as
  readers after a barrier.

### The catalog file

One KohakuVault (>= 1.1) file, a ColumnVault and a KVault on the same path; readers
open both read-only:

| part | contents |
| --- | --- |
| fixed i64 columns | `w`, `h`, `has_alpha`, `live`, `group_id`, `tags_chars`, `nl_chars`, `artist_chars`, `rating_chars`, `json_chars`, `prompt_chars` (a source's bound on its longest caption, `extra["prompt_chars"]`; 0 for the others); the stamps `st_size`, `st_mtime`, `s1_size` ... `s3_mtime` |
| `weight32` | 4 little-endian float32 bytes per row |
| `item` (str) | the scan item each row came from |
| KVault `r<id, 8-byte BE>` | msgpack of the row's variable fields |
| KVault `meta:info`, `meta:groups` | schema, generation, row count, source kind; group names |

Opening reads each planning column with one range read (`Column.read_buffer`) straight
into numpy; the stamps and items are read only by a rescan. `catalog.db` of a
`moss_shard` dataset and `catalog_build.py` files use the same layout, the only one
the runtime reads: a file whose fixed columns are not all there (another layout, or one
written by a KohakuVault before 1.0, whose columns 1.x does not read) is rebuilt by a
rescan when it is a scan cache, and is an error naming the re-import for a prebuilt one
([datasets.md](../guides/datasets.md#4-optional-import-into-shards)). A `moss_shard`
dataset without its `shards/` is an error saying so.

## The `moss_shard` format

```
dataset/
  catalog.db          a catalog file (layout above); each row's extra holds the
                      blake2b content hash and the source locator (`origin`)
  shards/00000.bin    concatenated encoded images (rolled at shard_bytes, 1 GiB default)
  shards/00000.idx    32-byte records: offset u64, size u32, w u16, h u16, codec u8,
                      flags u8 (bit 0 = alpha), 6 pad, content-hash prefix u64
```

`scripts/data/catalog_build.py` imports any source (`import_to`) or just writes its
catalog (`catalog`) so later runs skip the scan.

## GBC datasets

The `gbc` source reads a dataset in KohakuGBC's format through `kohakugbc-data`
(`pip install "kohakumoss[gbc]"`, CPython 3.11+):

```
<root>/dataset.json                format, version, the finished shards, a header
<root>/shards/shard_XXXXX/
  images.zip                       the images as captioned (KohakuClip image shard)
  graphs.db                        key -> GBC graph (typed msgpack, zstd)
  meta.db                          key -> metadata dict (any content)
  manifest.parquet                 key, index, w, h, codec, sizes, extra columns
```

- One walk item per shard (the catalog cache keeps a shard's rows until one of its four
  files changes); a row's locator is `<shard>/<key>`, its size the manifest's.
- The probe reads each sample's graph and metadata once to store `prompt_chars`: the
  graph's full-detail prompt text plus every top-level metadata string, a bound on any
  `gbc` caption of the sample (the planner's text estimate reads this column).
- `open(locator)` is the stored image bytes; `record(locator)` the graph and metadata.
  The reader opens its files on first use in each process.
- GBC boxes are fractions of the stored image. Captions follow the row's centre crop and
  target size (see below), so a `gbc` caption refuses `CROP="random"`,
  `RESOLUTION_MODE="shifted_square"` and `FLIP`.

## GBC captions

The `gbc` template turns a sample's record into a caption in three steps: the sections,
a format, the draw.

**Sections.**

| section | from |
| --- | --- |
| metadata sections (any name) | `meta`: `{name: rule}` per section, or `meta_formatter`: a dotted path to `f(meta) -> {name: str or [str]}` |
| `tags` | one of `tags["sources"]` per sample, by weight: `"meta:<field>"` (a metadata list) or `"graph:root"` (the root's tags in the graph) |
| `nl` | the root's long caption; its short one (its `short` caption, else the long one's first sentence) with probability `nl["short"]` |
| `graph` | the GBC prompt text (below) |

A rule takes the value at `field` (a dotted path) and writes it as items:

| rule key | does |
| --- | --- |
| `bins` | `[[lower, tag], ...]`, highest first: the first `lower <= value`; `""` writes nothing |
| `ranges` | `[[first, last, tag], ...]`, both ends inclusive |
| `map` | `{value: text or [texts]}` |
| (none) | the value itself (a string, a number or a list) |
| `drop` | the whole section's drop probability (a rate: a number or `[lo, hi]`) |

A missing field, or a value outside every bin / range / map key, writes nothing.
`underscores` (default True) turns `_` into spaces in list items; nothing is escaped.

Every tag list (both `tags` sources and the graph's tags) drops the items matching a
`tags["exclude"]` regex and, case-insensitively, the items of the `tags["dedupe"]`
metadata sections (e.g. a tagger's `rating:*` tags, or names an artist section already
holds).

**Format.** `formats` is a weighted list of section orders. A sample draws among the
formats whose content sections (`tags`, `nl`, `graph`) it has, by weight; a metadata
section with no items is left out. Items join with `list_sep` (`", "`); a section is
followed by `text_sep` (`"\n"`) after a text section (`nl`, `graph`, which end as
written) and by `section_sep` (`",\n"`) otherwise; `prefix` / `suffix` wrap the
caption. A format may also hold `requires` (metadata sections the sample must have
items in, else the format is not drawn for it, e.g. an artist-only caption for posts
with an artist) and `drop` (`{section: rate}` overriding a metadata section's or the
tags' (`"tags"`) drop rate in that format, e.g. an artist that never drops there).

**Rates.** Every drop rate (`meta` sections' `drop`, `tags["drop"]`,
`graph["tag_drop"]`) and `graph["keep"]` is a number or `[lo, hi]`; a range is drawn
uniformly once per caption, so `[0.0, 0.4]` gives some captions with no dropout at all.
A fixed rate draws nothing.

**Draw.**

- `tags`: each tag of the drawn source drops with probability `tags["drop"]`, the rest
  shuffled (`tags["shuffle"]`).
- `graph`: the graph's DAG with its boxes mapped into the row's crop (a box falling
  outside is dropped, its vertex kept), at detail levels from
  `kohakugbc_data.prompt.detail.sample_levels` (`graph["keep"]`; the root always at full
  detail), every shown tag list (the root's included) dropped per tag with
  `graph["tag_drop"]`, rendered on the row's training canvas with the renderer's default
  header and coordinates (`<gbc bbox=yxyx space=abs_centralized canvas=WxH>`).
- **Fit.** A caption longer than the text budget (`MAX_TEXT_TOKENS` in characters under
  the planner's estimate, times `fit["margin"]`) is drawn again with the dropout raised
  by a step (tag drop `+ fit["tag_drop_step"]` up to `fit["tag_drop_max"]`, keep
  `- fit["keep_step"]` down to `fit["keep_min"]`), same format and tag source, until it
  fits; then `fit["fallback"]` (a section order) is drawn. Metadata sections are never
  dropped by the fit.

Every draw comes from the caption generator of (seed, row, variant), so a caption is a
fixed function of the row and the variant, as for the other templates.

## HostBatch

`MossDataset.materialize(index, rank, world)` returns a `HostBatch`; image tokens are
sample-major and row-major over each sample's latent grid.

| field | contents |
| --- | --- |
| `ids` | int64 [B] catalog row ids |
| `hw` | int32 [B, 2] latent grid (pixels / 16) |
| `text_len` | int32 [B] text tokens; final when `text` is set, filled by the consumer after encoding `prompts` |
| `text_drop` | bool [B] text dropout draws |
| `weight` | float32 [B] the entries' `loss_weight` |
| `img_keep` | per sample, sorted kept grid indices after `drop_tokens`, or None (all kept) |
| `latents` | cached packed latents [N_img, C], kept tokens only |
| `text` | cached packed text embeddings, already cut to `text_limit` |
| `pixels` | uint8 [H, W, 3] full images when latents are not cached; the consumer encodes, then gathers `img_keep` |
| `prompts` | captions when text is not cached; the consumer keeps the first `text_limit` rows after encoding |
| `text_limit` | per sample, > 0 caps the text tokens (`truncate_text`) |
| `latent_keys` / `text_keys` | each sample's cache keys (empty when that cache is off), used by online write-back |
| `index` / `rank` | the plan coordinates |
| `layout` | the consumer's packed-sequence layout, built in the loader worker by a transform when the text lengths are final (None otherwise) |

`HostBatch.pin()` (the DataLoader pin hook) pins every tensor field, the tensors of the
list fields (`img_keep`) and the `layout` through its own `pin_memory()` (the trainer's
SeqInfo has one). A worker hands every tensor over in shared memory, and the main
process holds one open file per shared storage for as long as that storage lives. A
field left unpinned keeps its file open while the batch is held, and the trainer holds a
whole GRAD_ACC group of batches before its first forward. With the SeqInfo unpinned,
that is ~16 open files per batch, and GRAD_ACC=64 passed Linux's default limit of 1024
open files. Pinned, a held batch keeps no file open.

## Loader

`Loader` wraps a `DataLoader(batch_size=None)` over a `BatchStream`: worker `w` of `W`
materialises indices `start + w, start + w + W, ...`, and the DataLoader round-robins
its workers in order, so the stream is the same for any worker count. Resume state is
`{seed, start}`.

- **Fresh worker processes.** Workers start from a fresh interpreter (`kohakumoss_data.loader.stateless.worker_context`: `forkserver` on
  Linux, `spawn` on Windows), so the dataset (sources, catalog, templates, caches) is
  pickled by value; file handles reopen lazily per process. Batches do not depend on
  the start method. Each worker takes the tensor sharing strategy of the loader's
  process (`share_like`): a process that holds many batches (the encode feed's main
  process, `file_system`) keeps no file descriptor per buffered tensor.
- **Transform.** `Loader(transform=...)` / `build_loader(cfg, rank, world,
  transform=...)` (a picklable callable or a dotted path) runs on every batch inside the
  worker, before the DataLoader pins it (with `PIN_MEMORY` and an accelerator with
  pinned host memory: CUDA or XPU; `kohakumoss_common.devices.streamed_type` is the one
  predicate the loader, the SeqInfo transfer and the offload engines share). The
  trainer passes `kohakumoss.training.step.LayoutTransform`, which builds the batch's
  SeqInfo there (when the text lengths are final) and attaches it as `hb.layout`, so the
  layout is built off the training thread.

### Worker start-up time

The DataLoader starts its workers one after another, and each start returns only once
the new worker has read most of its pickled arguments, which it reads after importing
the launching process's main module and while importing the modules the arguments name
(torch, `kohakumoss_data`, the transform's module; the trainer's `LayoutTransform`
brings in `kohakumoss_kernels` with every backend).

| start method | wait before step 1 |
| --- | --- |
| `spawn` | every worker pays those imports in turn: with a launcher that imports the trainer at its top, step 1 waits 46 s for 4 workers on an RTX 5090 host (Linux), whatever the cache modes |
| `forkserver` (Linux) | the server preloads the main module, the DataLoader's worker loop and the modules of the dataset and the transform once, and every worker forks from it with them imported, so the wait is one set of imports |

Windows has only `spawn`, where the wait grows with `NUM_WORKERS`.

## Single-writer files

Every KohakuVault file has exactly one writer process; any number of processes read
it.

- The catalog files are written by the process that holds their lock (the trainer's
  rank 0; the others read after it).
- The online caches write one shard file per rank.
- The offline cache scripts and `cache_merge.py` / `catalog_cache.py` are one process
  each.

### Tensor store files

Each cache file is a KVault (`TensorStore`) with one writer: the main file for the
offline passes and `cache_merge.py`, `<stem>.r<rank>.kv` shards for online writers.

| key | value |
| --- | --- |
| `t:<key>` | magic `KMT1`, dtype code u8, ndim u8, pad u16, scale f32, shape u32 x ndim, then the raw little-endian payload |
| `m:<key>` | the first dim (token count) as ASCII digits, written after `t:<key>` |
| `idx:lengths` | u32 count, the sorted 16-byte keys, their i32 lengths |

fp8 values are e4m3 with a per-tensor fp32 scale `absmax / 448` computed in fp32 (1 for
an all-zero tensor). A merge takes each copied entry's length from its `t:` header, so a
shard whose writer stopped between the two writes still merges.

Sizes at MOSS shapes: a 1 MP latent is 64 x 64 x 128 bf16 = 1 MiB; a text embedding is
L x 4096 bf16 = 8 KiB per token, so a 300-token caption is ~2.4 MiB per variant.
