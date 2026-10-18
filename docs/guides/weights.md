# Weights: where each file goes

MOSS-Image ships in two layouts, and KohakuMoss reads both:

- the ComfyUI layout: one file per component under a ComfyUI `models/` folder, the
  Comfy-Org/MOSS-Image repack;
- the HF layout: OpenMOSS-Team/MOSS-Image-Base / -SFT, a pipeline folder with
  `transformer/`, `text_encoder/`, `tokenizer/`, `vae/`.

## ComfyUI layout

```
models/
  diffusion_models/  moss_image_{base,sft}_bf16.safetensors
                     moss_image_{base,sft}_int8_convrot.safetensors
  text_encoders/     qwen3.5_9b_bf16.safetensors
                     qwen3.5_9b_int8_convrot.safetensors
                     qwen3.5_9b_w4a8_gptq.safetensors
  vae/               flux2-vae_small.safetensors
                     flux2-vae.safetensors
```

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

### Text encoder

A ComfyUI text-encoder file is the full Qwen3.5-9B checkpoint: the language model under
`model.language_model.*`, the vision tower (`model.visual.*`), the MTP head (`mtp.*`) and
the LM head (`lm_head.*`). MOSS-Image reads the language model's last hidden state, so
KohakuMoss loads `model.language_model.*` onto the text model's own keys and leaves the
other three parts unread.

A ComfyUI file carries no config and no tokenizer. KohakuMoss ships both in
`kohakumoss/encoders/text/assets/qwen3.5-9b/` (from MOSS-Image's `text_encoder/` and
`tokenizer/` folders): `config.json`, `tokenizer.json.gz` (3.8 MB, decompressed in memory
on load), `tokenizer_config.json`, `chat_template.jinja`. The packaged tokenizer gives the
same token ids and chat template as the HF `tokenizer/` folder.

## Third-party files

The packaged Qwen3.5-9B config and tokenizer come from OpenMOSS-Team/MOSS-Image-Base
(the Qwen team's Qwen3.5-9B tokenizer, as also shipped by Qwen/Qwen3.5-9B and
Comfy-Org/MOSS-Image), all under the Apache License, Version 2.0. The license text and a
NOTICE naming the sources are next to the files
(`kohakumoss/encoders/text/assets/qwen3.5-9b/LICENSE`, `NOTICE`); the wheel ships them.
