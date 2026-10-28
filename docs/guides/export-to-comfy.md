# Export to ComfyUI

A training run writes its exports to `runs/moss/<RUN_NAME>/export/` at the end and,
with `CHECKPOINT_EXPORT=True` (the default), into every checkpoint's `export/` folder;
`scripts/export/export.py` exports any checkpoint again (an interrupted run's too).

## What a run writes

| knob | values |
| --- | --- |
| `SAVE_FORMAT` | `comfy` (default), `hf`, `both`, `none` |
| `SAVE_MERGED` | `False` (default); `True` adds the merged DiT (adapters only) |
| `SAVE_FUSED_QKV` | `False` (default: separate q / k / v keys), `True` (one fused `proj.qkv`), `auto` (as the DiT file the run trained on stores q / k / v; the Comfy-Org files are fused) |
| `CHECKPOINT_EXPORT` | `True` (default): every checkpoint also holds the exports |

| run | `comfy` | `hf` |
| --- | --- | --- |
| adapters | `<run>-comfy.safetensors`: the ComfyUI LyCORIS file (`lora_unet_...` keys, separate q/k/v unless `SAVE_FUSED_QKV`) | `<run>-lycoris.safetensors`: the LyCORIS file with diffusers module keys |
| adapters, `SAVE_MERGED` | + `<run>-merged.safetensors`: one ComfyUI DiT file (bare keys, bf16) | + `<run>-diffusers/`: the HF transformer folder |
| full finetune | `<run>-merged.safetensors`: one ComfyUI DiT file | `<run>-diffusers/` |

- `both` writes both columns; `none` writes no export (the checkpoints still hold the
  training state, which `export.py` reads).
- The build log names the files of one export and their size; with
  `CHECKPOINT_EXPORT` every checkpoint adds that much (`KEEP_LAST` prunes them with the
  checkpoint).
- A merged file from a run on the int8 DiT is the dequantized base plus dW in bf16.
- A pipeline run's checkpoint export first gives rank 0 every stage's adapters (the
  same step as the final export).
- The older `EXPORT_FORMATS` list maps onto these knobs at validation (one INFO line):
  `comfy` / `lycoris` are the adapter columns, `merged` / `diffusers` the DiT columns
  (`SAVE_MERGED` on an adapter run), an empty list is `none`.

## Exporting

```bash
# ComfyUI adapter file from a checkpoint (WEIGHTS: the ComfyUI folder or the HF dir)
kogine run scripts/export/export.py --set WEIGHTS=ComfyUI/models \
    --set ADAPTER=runs/moss/<RUN_NAME>/checkpoints/step_00001000/trainable.safetensors \
    --set OUT=moss_lora.safetensors
# merged transformer (one ComfyUI DiT file)
kogine run scripts/export/export.py --set FORMAT=merged ...
# the LyCORIS file
kogine run scripts/export/export.py --set FORMAT=hf ...
# merge any adapter files (LyCORIS, ComfyUI, training states) into the base
kogine run scripts/export/merge.py --config <file with WEIGHTS / ADAPTERS / OUT>
```

- `FORMAT`: `comfy`, `hf` (= `lycoris`), `lycoris`, `merged`, `diffusers`; a
  full-finetune checkpoint (`ALGO="full"`) exports as `merged` (`comfy`) or `diffusers`
  (`hf`).
- `FUSED_QKV` (`False` default: separate q / k / v; `True`; `auto`: as `WEIGHTS` stores
  q / k / v) writes the fused layout of the Comfy-Org files for `comfy` and `merged`.
- With a ComfyUI folder as `WEIGHTS`, `TRANSFORMER` / `VARIANT` pick the DiT file as in
  training ([weights.md](weights.md)).
- `SAVE_DTYPE` (`bf16` default, `fp16`, `fp32`) is the dtype of the written tensors in
  `export.py` and `merge.py`, as in the other export scripts.

## Loading the file

- With a ComfyUI MOSS-Image loader, load the adapter with a LoRA loader node. On the
  fused Comfy-Org DiT files, ComfyUI maps the separate q / k / v keys onto their slices
  of `proj.qkv`; the merging mode handles them, and the bypass mode needs
  [Comfy-Org/ComfyUI#17408](https://github.com/Comfy-Org/ComfyUI/pull/17408).
- Without one, `scripts/sample/sample.py` samples the `comfy` file directly (`ADAPTERS`,
  [inference.md](inference.md#cli)).

The `comfy` format targets a ComfyUI MOSS-Image loader that maps `lora_unet_<module path
with _>.<tensor>` or `diffusion_model.<module path>.<tensor>` keys onto the
transformer's own state-dict names, and full transformers as one safetensors with bare
keys. The key layout follows the module paths of ComfyUI's native MOSS-Image model; see
[Fused q/k/v](#fused-qkv) for the fused repack. The weight math is checked against
ComfyUI's `comfy/weight_adapter/*`: LoRA / LoHa / LoKr alpha-over-rank, DoRA normalised
by `||W||` on the output axis, OFT merges ignoring `rescale`, strength applied inside and
on top of the rotation.

## Folds and fp16-fix weights

A checkpoint's `trainable.safetensors` records what export needs in its metadata:

| key | content |
| --- | --- |
| `kohakumoss_spec` | the adapter configuration |
| `kohakumoss_fold` | the weight fold the layers trained on: the fold of an fp16-fix WEIGHTS file, or none |

- `export.py` puts WEIGHTS into that fold before it reads the state, so any WEIGHTS file
  of the model works and the fold is undone in the output; a state that records no fold
  trained on WEIGHTS as loaded.
- `merge.py` on an fp16-fix WEIGHTS file merges each dW with the fold's row scales, so
  the merged (unfolded) file is exact; `scripts/export/fp16_fix.py` on it writes the
  fp16-fix checkpoint of the merged model.

## What the `comfy` format contains

Each layer is written so that ComfyUI's weight adapters compute the trained dW at
strength 1:

| algorithm | tensors | conversion |
| --- | --- | --- |
| LoRA / LoCon | `lora_up.weight`, `lora_down.weight`, `alpha` | none (scalar folded into up) |
| LoHa | `hada_w{1,2}_{a,b}`, `alpha` | none |
| LoKr | `lokr_w1` / `lokr_w1_{a,b}`, `lokr_w2` / `lokr_w2_{a,b}`, `alpha` | none |
| GLoRA | `{a1,a2,b1,b2}.weight`, `alpha` | none |
| DoRA (output axis) | + `dora_scale` [out, 1] | `dora_scale * \|\|W\|\| / \|\|W + dW\|\|`: ComfyUI normalises by the un-merged weight on this axis |
| DoRA (input axis) | + `dora_scale` [1, in] | none |
| OFT | `oft_blocks`, `alpha` | `alpha` is the effective constraint (LyCORIS stores constraint / out_dim) |
| OFT with `rescaled` | `diff` | ComfyUI's merge ignores `rescale`, so the full dW is written |
| BOFT | `oft_blocks`, `rescale`, `alpha` | `alpha` as for OFT |
| IA3 | `diff` | the full dW, loaded by ComfyUI's `diff` adapter |
| DyLoRA | `lora_up.weight`, `lora_down.weight`, `alpha = rank` | every rank, eval scale folded into up |
| T-LoRA | rank-2r LoRA | `[P*lam*s, -P0*lam0*s] @ [Q ; Q0]`, no rank mask |
| full | `diff`, `diff_b` | none |
| norms | `w_norm`, `b_norm` | none |

## Fused q/k/v

Joint-block q/k/v are separate layers in KohakuMoss and are written as separate
`..._proj_q` / `_k` / `_v` keys. In ComfyUI:

- the MOSS-Image model (`comfy/ldm/moss_image`) names them `layers.N.proj.{q,k,v}` on an
  unfused checkpoint and `layers.N.proj.qkv` (rows q;k;v, `fused_qkv=True`) on the
  Comfy-Org repack;
- the generic LoRA key map (`comfy/lora.py`, `model_lora_keys_unet`) maps
  `lora_unet_<path with _>` and `diffusion_model.<path>` to the model's own
  `<path>.weight` keys, and on a fused checkpoint maps each `layers.N.proj.{q,k,v}` key
  to its row slice of `layers.N.proj.qkv.weight`.

So a separate-q/k/v file loads on both checkpoints, and is the default
(`SAVE_FUSED_QKV` / `FUSED_QKV` `False`): its adapter tensors stay in their trained form
(a LoKr stays a LoKr). Bypass loading of it on the fused repack needs
[Comfy-Org/ComfyUI#17408](https://github.com/Comfy-Org/ComfyUI/pull/17408). `True`
writes the fused layout below; `auto` follows the DiT file the run trained on (the
Comfy-Org files: fused; the HF transformer folder: unfused).

A fused `comfy` file (`True`) writes each block's q/k/v as one `layers.N.proj.qkv`
layer:

| q/k/v layers | written as |
| --- | --- |
| three plain LoRA sets | one LoRA of rank r_q + r_k + r_v: down = the three downs stacked, up block-diagonal with each layer's alpha / rank folded in, alpha = rank |
| anything else (DoRA, LoHa, LoKr, OFT and the rest) | the exact `diff`: the three dW stacked by rows; a `diff` scales linearly with strength |

KohakuMoss's own loader splits a fused LoRA or `diff` layer back into q / k / v (also a
fused LoRA from another trainer); other fused kinds are an error.

## Strength

- ComfyUI scales dW linearly with strength for the low-rank, DoRA, `diff` and norm
  formats, which equals KohakuMoss's multiplier except for `diff`-written
  OFT-with-rescale.
- ComfyUI's OFT and BOFT merges apply strength inside the rotation and again on dW, so
  those (and OFT-with-rescale) are exact at strength 1.
- KohakuMoss's own loader (`kohakumoss.adapters.load_adapters`) reproduces the
  training-time multiplier for all.
