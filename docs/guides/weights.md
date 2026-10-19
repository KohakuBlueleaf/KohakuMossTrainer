# Weights: where each file goes

MOSS-Image ships in two layouts, and KohakuMoss reads both for training, sampling,
the cache passes and the export scripts:

- the ComfyUI layout: one file per component under a ComfyUI `models/` folder, the
  Comfy-Org/MOSS-Image repack (the default in the shipped configs);
- the HF layout: OpenMOSS-Team/MOSS-Image-Base / -SFT, a pipeline folder with
  `transformer/`, `text_encoder/`, `tokenizer/`, `vae/`.

## ComfyUI layout

```
models/
  diffusion_models/  moss_image_{base,sft}_int8_convrot.safetensors   (5.4 GB)
                     moss_image_{base,sft}_bf16.safetensors           (10.2 GB)
  text_encoders/     qwen3.5_9b_int8_convrot.safetensors              (9.7 GB)
                     qwen3.5_9b_bf16.safetensors                      (19.3 GB)
                     qwen3.5_9b_w4a8_gptq.safetensors                 (7.3 GB)
  vae/               flux2-vae_small.safetensors                      (0.25 GB)
                     flux2-vae.safetensors                            (0.34 GB)
```

```bash
hf download Comfy-Org/MOSS-Image diffusion_models/moss_image_base_int8_convrot.safetensors \
    text_encoders/qwen3.5_9b_int8_convrot.safetensors vae/flux2-vae_small.safetensors \
    --local-dir ComfyUI/models
```

`WEIGHTS` is the `models/` folder. Each component has a knob that names its file (a
file name under the component's subfolder, with or without `.safetensors`, or any path);
left empty it takes the first default that exists:

| knob | subfolder | defaults, in order |
| --- | --- | --- |
| `TRANSFORMER` | `diffusion_models/` | `moss_image_<VARIANT>_int8_convrot`, `moss_image_<VARIANT>_bf16` (a full finetune: the bf16 file first) |
| `TEXT_ENCODER` | `text_encoders/` | `qwen3.5_9b_int8_convrot`, `qwen3.5_9b_bf16`, `qwen3.5_9b_w4a8_gptq` |
| `VAE` | `vae/` | `flux2-vae_small`, `flux2-vae` |

- The train script (`scripts/train/moss.py`), the sample and grid scripts
  (`scripts/sample/`) and the export scripts (`scripts/export/export.py`, `merge.py`:
  `TRANSFORMER`, `VARIANT`) take the same knobs.
- The cache passes take the folder as `TEXT_ENCODER_KWARGS["path"]` /
  `IMAGE_ENCODER_KWARGS["path"]` (the default text-encoder and VAE files), or a file.
- `configs/moss/weights/weights_comfy.py` names every file; `weights_hf.py` is the HF
  layout.
- A missing file is a knob problem naming every path that was tried.

### DiT

The two DiT files hold the MOSS-Image transformer under bare keys with the joint-block
q / k / v fused (`layers.N.proj.qkv`, rows q;k;v). KohakuMoss splits them into its own
`q` / `k` / `v` layers on load.

The int8 file stores the four projections of every block (`proj.qkv`, `proj.o`,
`proj.ffn_up_gate`, `proj.ffn_down`) as [quantized files](#quantized-files) do: int8
codes of the rotated weight with one fp32 scale per row, everything else bf16.

- `BASE_PRECISION="auto"` (the default) keeps those int8 bases: the frozen base runs the
  rotated int8 linear (`docs/internals/quant-kernels.md#rotated-int8-linear`), adapters
  attach in bypass, the 16-bit weight is never built. A bf16 file under `auto` is 16-bit
  (`compute`).
- Any other `BASE_PRECISION` dequantizes the int8 layers first (W = s q R, one INFO
  line): `compute` runs the dequantized weights in 16 bits, `int8` / `fp8` /
  `fp8_compute` / `mxfp8` quantize them (and the other linears) as for a 16-bit file.
- A full finetune trains dense weights: on the int8 file it needs
  `BASE_PRECISION="compute"` (the dequantized weights train); `auto` is refused by the
  knob checks. The full-finetune default picks the bf16 file.
- Each block's `comfy_attention.config` tensor is a ComfyUI runtime hint (its attention
  kernel); KohakuMoss does not read it (one INFO line counts them).
- The memory planner counts the int8 bytes under `auto`.

### Text encoder

A ComfyUI text-encoder file is the full Qwen3.5-9B checkpoint: the language model under
`model.language_model.*`, the vision tower (`model.visual.*`), the MTP head (`mtp.*`) and
the LM head (`lm_head.*`). MOSS-Image reads the language model's last hidden state, so
KohakuMoss loads `model.language_model.*` onto the text model's own keys and leaves the
other three parts unread.

| file | language-model linears | embedding |
| --- | --- | --- |
| `qwen3.5_9b_bf16` | bf16 | bf16 |
| `qwen3.5_9b_int8_convrot` | int8 ConvRot (group 256) | int8 ConvRot |
| `qwen3.5_9b_w4a8_gptq` | W4A8: 4-bit codes, group 16, ConvRot 256 | int8 ConvRot |

- `TEXT_ENCODE["precision"]="auto"` (the default) keeps what the file stores: the int8
  linears and embedding run as stored, the W4A8 linears decode their codes to the same
  int8 grid. On the host (memory `cpu`) `auto` is `compute`.
- `TEXT_ENCODE["w4a8"]` picks the W4A8 decode: `int8` decodes once at load (1 byte per
  weight, the int8 linear per call), `packed` keeps the 4-bit codes (0.56 bytes per
  weight) and decodes them per call, `auto` (default) takes `int8` when the decoded
  encoder fits resident next to the DiT step and `packed` otherwise. Both give the same
  int8 values; the costs are in `docs/performance/quant-rotate.md#w4a8-decode`.
- Any other precision dequantizes the stored layers into the 16-bit model first, then
  quantizes as for an HF checkpoint.
- The text cache key records the stored layout (`src:int8-convrot256`, ...), so a cache
  is reused only with the same file format.

A ComfyUI file carries no config and no tokenizer. KohakuMoss ships both in
`kohakumoss/encoders/text/assets/qwen3.5-9b/` (from MOSS-Image's `text_encoder/` and
`tokenizer/` folders): `config.json`, `tokenizer.json.gz` (3.8 MB, decompressed in memory
on load), `tokenizer_config.json`, `chat_template.jinja`. The packaged tokenizer gives the
same token ids and chat template as the HF `tokenizer/` folder.

### VAE

`VAE` (and `IMAGE_ENCODER_KWARGS["path"]` of the latent cache) takes either file:

| file | keys | decoder | latents |
| --- | --- | --- | --- |
| `flux2-vae_small.safetensors` | diffusers (`encoder.down_blocks.N.resnets.N...`) | widths 96 / 192 / 384 / 384, the HF `vae/` decoder exactly | the fp32 encoder whose bf16 rounding the HF `vae/` folder holds |
| `flux2-vae.safetensors` | BFL / ldm (`encoder.down.N.block.N...`) | widths 128 / 256 / 512 / 512 (the full FLUX.2 decoder) | the small file's encoder exactly: the same latents |

- The small file is MOSS-Image's VAE in fp32. Its decoder is the HF `vae/` decoder
  tensor for tensor. Its encoder, `quant_conv` and latent statistics (`bn.*`) are fp32;
  the HF folder stores the same tensors rounded to bf16. The two give latents that
  differ at the level of that rounding (relative L2 difference 4.6e-3 on a 256 x 256
  dataset image in fp32; decode PSNR 37.08 dB from the file's latents, 37.09 dB from
  the folder's), so a latent cache is encoded with one VAE and used with it.
- The full file has the small file's encoder (the same latents, the same latent cache)
  and FLUX.2's original, wider decoder (decode PSNR 37.07 dB on the same image).
- A single file is loaded in fp32 (the normalisation statistics stay fp32) and cast to
  `VAE_DTYPE`, as the folder is.
- The BFL keys are mapped onto diffusers' `AutoencoderKLFlux2` names: encoder down blocks
  in order, decoder up blocks in reverse order (BFL level 0 is the finest, diffusers
  block 0 the coarsest), `nin_shortcut` -> `conv_shortcut`, the mid attention's 1x1
  convolutions -> linears, `encoder.quant_conv` / `decoder.post_quant_conv` -> the
  top-level `quant_conv` / `post_quant_conv`, `bn.*` unchanged.
- The `AutoencoderKLFlux2` config of a file is read from its tensor shapes (block widths,
  layers per block, channels, quant convs, mid attention); the remaining fields are the
  FLUX.2 defaults.

## HF layout

```
models/MOSS-Image-Base/   transformer/  text_encoder/  tokenizer/  vae/
```

```bash
hf download OpenMOSS-Team/MOSS-Image-Base --local-dir models/MOSS-Image-Base
```

`WEIGHTS` is the pipeline folder (the train script) or a weights root holding `base/`
and `sft/` (the sample script, `VARIANT` picks one). `TRANSFORMER` / `TEXT_ENCODER` /
`VAE` empty read `transformer/`, `text_encoder/` (+ `tokenizer/`), `vae/`; set, they are
paths (a single DiT file, an fp16-fix checkpoint, another text-encoder folder or a
ComfyUI file). Every weight is bf16: `BASE_PRECISION` and `TEXT_ENCODE["precision"]`
`auto` are `compute`.

## Quantized files

A ComfyUI quantized layer `<n>` stores its codes as `<n>.weight`, its scales, and
`<n>.comfy_quant`, a uint8 JSON record of its format:

| format | tensors | weight |
| --- | --- | --- |
| `int8_tensorwise` | `weight` I8 [N, K], `weight_scale` F32 [N, 1] | q x s per row |
| `float8_e4m3fn` | `weight` F8 [N, K], `weight_scale` / `scale_weight` | q x s |
| `asym_w4a8_int8` | `weight` I8 [N, K/2] (two 4-bit codes per byte), `weight_s_rel` F8 [N, K/16], `weight_s_channel` F32 [N], `weight_codebook` F32 [16] | round(codebook[code] x s_rel) (the int8 grid) x s_channel |

With `"convrot": true` the stored weight is the rotated one, W R, R = blockdiag(H_g)
over K with H_g the normalised regular Hadamard of `convrot_groupsize` g (ConvRot); the
loader keeps the rotation with the base and every consumer reads W (the linear rotates
its input, a dequantized weight is rotated back). A record-less `<n>.weight` in int8 /
fp8 next to `weight_scale` / `scale_weight` is read the same way without a rotation.
The formulas follow comfy-kitchen's eager backend; `tests/quant/test_comfy_formats.py`
checks every one against an independent numpy port.

## Saving

Exports are written in the ComfyUI layout by default (`SAVE_FORMAT`, `SAVE_MERGED`,
`SAVE_FUSED_QKV`): [export-to-comfy.md](export-to-comfy.md).

## Third-party files

The packaged Qwen3.5-9B config and tokenizer come from OpenMOSS-Team/MOSS-Image-Base
(the Qwen team's Qwen3.5-9B tokenizer, as also shipped by Qwen/Qwen3.5-9B and
Comfy-Org/MOSS-Image), all under the Apache License, Version 2.0. The license text and a
NOTICE naming the sources are next to the files
(`kohakumoss/encoders/text/assets/qwen3.5-9b/LICENSE`, `NOTICE`); the wheel ships them.
