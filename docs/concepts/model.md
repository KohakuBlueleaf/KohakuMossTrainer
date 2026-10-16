# Model

`kohakumoss.model.MossBackbone` is a re-implementation of the MOSS-Image transformer with
module names equal to the official state-dict keys, so HF and ComfyUI checkpoints load
without renaming.

## Shape (base / sft)

| part | structure |
| --- | --- |
| `video_embedder` | Linear 128 -> 3072 (one token per latent pixel of the /16 grid) |
| `text_embedder` | `proj` Linear 4096 -> 3072, then 2 `TextBlock`s (biased fused `qkv`, qk-norm, RoPE, SiTU FFN, no modulation) |
| `time_embedding` | sinusoid(1000 sigma, 256, float64) -> Linear -> SiLU -> Linear |
| `shared_block_modulation` | one Linear 3072 -> 6 x 3072 for all 42 blocks: shift_a, scale_a, gate_a, shift_f, scale_f, gate_f |
| `layers.{0..41}` | joint blocks over `[image ; text]`: RMSNorm + modulation, q/k/v, per-head qk RMSNorm + 4-axis RoPE, attention, gated residual; RMSNorm + modulation, `ffn_up_gate` (first half up, second half gate), SiTU, `ffn_down`, gated residual |
| `final_norm_video`, `video_head_modulation`, `final_linear_video` | modulated RMSNorm and Linear 3072 -> 128 on the image tokens |

SiTU is `(4 tanh(g/4) sigmoid(g)) * (25 tanh(u/25))`. The model predicts the flow
velocity `noise - x0` at sigma in [0, 1] (sigma = 1 is pure noise).

## Calling convention

```python
v = backbone(img, text, sigma, seq, skip_layers=frozenset(), segment_skip=None)
```

- `img` [N_img, 128] and `text` [N_txt, 4096] are packed (see `seqinfo.md`); `sigma` is
  fp32 [B].
- `skip_layers` skips blocks for every sample; `segment_skip` maps a block index to a bool
  [B] of samples that skip it. Packed CFG uses the latter for the negative-branch skip of
  block 22: the block's gates are zeroed for those samples, which leaves their stream
  exactly as skipping the block would.
- Time embedding, shared modulation and head modulation are computed per sample in fp32;
  those modules stay fp32 (`KEEP_FP32`).

## Ops and runners

Blocks call ops through the `OpSet` resolved once per (device, dtype). `bind_ops()` binds
each layer's fused linear+epilogue callables; it runs after any structural change
(adapter attach, base storage swap). The block loop calls `runner.run(i, block, x, ctx)`;
`PlainRunner` calls the block directly, the memory engine supplies the others.

## Checkpoints

| input | handled |
| --- | --- |
| HF `transformer/` dir (3 shards + index) | as is |
| ComfyUI single file, bare or `model.diffusion_model.` keys | prefix stripped, fused `proj.qkv` split into q / k / v |
| our single-file saves | as is |

Loading is strict: every model key must be filled and every source key consumed
(`rope.bands` is derived and optional). Savers write a single bare-key file (split or
fused qkv) or diffusers shards with `config.json`; the residual fold is undone first.
