# SeqInfo and packing

Every batch is packed: there is no padding anywhere in the model.

- Image tokens of all samples form one axis `[N_img, C]`, sample-major, row-major over each
  sample's latent grid `(h, w)`.
- Text tokens form a second axis `[N_txt, D]`.
- The joint sequence of sample i is `[img_i ; txt_i]`; joint tensors are sample-major in
  that order. `img_in_joint` / `txt_in_joint` scatter the two axes into it.

`build_seqinfo(hw, text_len)` runs on the host once per batch and carries:

| field | meaning |
| --- | --- |
| `cu_img`, `cu_txt`, `attn_joint.cu_seqlens`, `attn_text.cu_seqlens` | segment boundaries |
| `seg_img`, `seg_txt`, `seg_joint` | sample index of every token (selects per-sample modulation) |
| `pos_joint`, `pos_txt` | RoPE coordinates `(r, f, h, w)` in fp32 |

Positions follow the official layout: image tokens get `f = text_len` and centred `h, w`;
text token j gets `f = j`. Image positions therefore depend on each sample's true text
length, which is why text is never padded and attention never crosses a sample boundary.

Mixed resolutions and text lengths in one batch cost nothing extra: attention runs per
segment, every other op is per token.
