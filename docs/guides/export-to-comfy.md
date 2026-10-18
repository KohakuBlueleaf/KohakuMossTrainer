# Export to ComfyUI

A training run writes its `EXPORT_FORMATS` (default `comfy`; the shipped configs add
`lycoris`) to `runs/moss/<RUN_NAME>/export/` at the end. `scripts/export/export.py`
exports any checkpoint of a run (an interrupted one too).

## Exporting

```bash
# adapter file from a checkpoint
kogine run scripts/export/export.py --set WEIGHTS=<model dir> \
    --set ADAPTER=runs/moss/<RUN_NAME>/checkpoints/step_00001000/trainable.safetensors \
    --set OUT=moss_lora.safetensors
# merged transformer (fused q/k/v like the Comfy-Org repack)
kogine run scripts/export/export.py --set FORMAT=merged --set FUSED_QKV=True ...
# adapter file for the fused-q/k/v Comfy-Org repack (see "Fused q/k/v")
kogine run scripts/export/export.py --set FUSED_QKV=True ...
# merge any adapter files (LyCORIS, ComfyUI, training states) into the base
kogine run scripts/export/merge.py --config <file with WEIGHTS / ADAPTERS / OUT>
```

- A full-finetune checkpoint (`ALGO="full"`) exports as `merged` or `diffusers`.
- `SAVE_DTYPE` (`bf16` default, `fp16`, `fp32`) is the dtype of the written tensors in
  `export.py` and `merge.py`, as in the other export scripts.

## Loading the file

- With a ComfyUI MOSS-Image loader, load the adapter with a LoRA loader node in its
  default (merging) mode.
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
  `<path>.weight` keys.

So the default file targets an unfused checkpoint; the fused repack takes the
`FUSED_QKV=True` file below.

`FUSED_QKV=True` with `FORMAT="comfy"` writes each block's q/k/v as one
`layers.N.proj.qkv` layer for the fused repack:

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
