# GBC danbooru run

LoKr factor 8 trained on a danbooru dataset in KohakuGBC's GBC format (images, one GBC
graph per image, the danbooru post metadata with normalized quality scores).

| file | holds |
| --- | --- |
| `caption_danbooru.py` | the caption layer: `artist, copyright, character, <middle>, quality, era, rating`, the middle `tags` / `nl` / `tags, nl` / `graph` at 2 : 2 : 3 : 3, every dropout |
| `caption_danbooru_mix.py` | the same rules with an artist-heavy share: 10 % empty, 40 % `artist, quality, era, rating`, 30 % `tags` / `tags, nl` / `nl`, 20 % `graph`; every drop rate a per-caption range, so some captions keep every item |
| `train_4x3090_2x2080ti.py` | the run: four RTX 3090 train, two RTX 2080 Ti encode, 16 samples x 7,000 steps |

## Captions

- `artist`, `copyright`, `character`: the post's tags of that category, bare names,
  underscores as spaces; each section drops with probability 0.2.
- `tags`: the post's general tags or the graph root's tags (one per sample, 1 : 1), each
  tag dropped with probability 0.2, shuffled; the tagger's `rating:*` tags and the names
  the artist / copyright / character sections hold are removed (from the graph's tags
  too).
- Sections join with `,\n`; after the NL caption or the graph text only a newline.
- `nl`: the root's long caption, its short one with probability 0.3.
- `graph`: the GBC prompt text, the root at full detail, the other vertices at sampled
  detail levels (`keep` 0.55), tags dropped with probability 0.2.
- `quality` from `selection.quality_pct` (masterpiece >= 0.97, best quality >= 0.90,
  great quality >= 0.75, none >= 0.45, low quality >= 0.25, bad quality >= 0.10, worst
  quality below), `era` from the upload year (newest 2025-2026, recent 2023-2024, mid
  2020-2022, early 2015-2019, old 2005-2014), `rating` as safe / sensitive / nsfw /
  nsfw, explicit; these three never drop.
- A caption over the text budget is drawn again with stronger dropout until it fits, then
  falls back to `artist, copyright, character, nl, quality, era, rating`.

## Run

```bash
pip install "kohakumoss[cuda,gbc]"
CUDA_DEVICE_ORDER=PCI_BUS_ID kogine run scripts/train/moss.py \
    --config examples/gbc-danbooru/train_4x3090_2x2080ti.py
```

Set `WEIGHTS` (a ComfyUI MOSS-Image folder with the int8 ConvRot DiT and text encoder)
and the dataset `path` first; CUDA devices 0-3 train and 4-5 encode
(`DEVICES` / `ENCODE_DEVICES`, [encode-feed.md](../../docs/internals/encode-feed.md)).
