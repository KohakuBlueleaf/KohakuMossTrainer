# SeqInfo and packing

Every batch is packed: each tensor in the model is a flat token axis, and `SeqInfo`
carries the sample boundaries.

- Image tokens of all samples form one axis `[N_img, C]`, sample-major, row-major over each
  sample's latent grid `(h, w)`.
- Text tokens form a second axis `[N_txt, D]`.
- The joint sequence of sample i is `[img_i ; txt_i]`; joint tensors are sample-major in
  that order. `img_in_joint` / `txt_in_joint` scatter the two axes into it.

## Building it

- `build_seqinfo(hw, text_len, img_keep=None)` runs on the host once per batch.
- `img_keep[i]` (sorted indices into sample i's row-major grid, or None for all) keeps
  only those image tokens (random token drop): `n_img`, `cu_img` and every image axis
  count the kept tokens, and each kept token keeps the RoPE position of its original
  grid cell.
- `SeqInfo.to(device)` moves the host tensors as one pinned buffer per dtype (one copy
  each on CUDA); the fields on the device are views of it.

## Fields

| field | meaning |
| --- | --- |
| `cu_img`, `cu_txt`, `attn_joint.cu_seqlens`, `attn_text.cu_seqlens` | segment boundaries |
| `seg_img`, `seg_txt`, `seg_joint` | sample index of every token (selects per-sample modulation) |
| `pos_joint`, `pos_txt` | RoPE coordinates `(r, f, h, w)` in fp32 |

## Positions

Positions follow the official layout: image tokens get `f = text_len` and centred `h, w`;
text token j gets `f = j`. Image positions therefore depend on each sample's true text
length, which is why text is never padded and attention never crosses a sample boundary.

A batch packs mixed resolutions and text lengths at the cost of their tokens alone:
attention runs per segment, every other op is per token.
