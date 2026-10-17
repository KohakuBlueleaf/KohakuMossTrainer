# Export to ComfyUI

The `comfy` format targets a ComfyUI MOSS-Image loader that maps `lora_unet_<module
path with _>.<tensor>` or `diffusion_model.<module path>.<tensor>` keys onto the
transformer's own state-dict names, and full transformers as one safetensors with bare
keys. Neither the per-algorithm weight math nor the key layout has been checked against
a released ComfyUI MOSS model yet: no ComfyUI checkout carries one. The weight math is
checked against ComfyUI's `comfy/weight_adapter/*` (LoRA / LoHa / LoKr alpha-over-rank,
DoRA normalised by `||W||` on the output axis, OFT merges ignoring `rescale`, strength
applied inside and on top of the rotation); the key layout is an assumption.

```bash
# adapter file
kogine run scripts/export/export.py --set WEIGHTS=<model dir> \
    --set ADAPTER=<run>/trainable.safetensors --set OUT=moss_lora.safetensors
# merged transformer (fused q/k/v like the Comfy-Org repack)
kogine run scripts/export/export.py --set FORMAT=merged --set FUSED_QKV=True ...
```

Load the adapter with a LoRA loader node in its default (merging) mode.

## What the `comfy` format contains

Each layer is written so that ComfyUI's weight adapters compute the trained dW at
strength 1:

algorithm | tensors | conversion
--- | --- | ---
LoRA / LoCon | `lora_up.weight`, `lora_down.weight`, `alpha` | none (scalar folded into up)
LoHa | `hada_w{1,2}_{a,b}`, `alpha` | none
LoKr | `lokr_w1` / `lokr_w1_{a,b}`, `lokr_w2` / `lokr_w2_{a,b}`, `alpha` | none
GLoRA | `{a1,a2,b1,b2}.weight`, `alpha` | none
DoRA (output axis) | + `dora_scale` [out, 1] | `dora_scale * ||W|| / ||W + dW||`: ComfyUI normalises by the un-merged weight on this axis
DoRA (input axis) | + `dora_scale` [1, in] | none
OFT | `oft_blocks`, `alpha` | `alpha` is the effective constraint (LyCORIS stores constraint / out_dim)
OFT with `rescaled` | `diff` | ComfyUI's merge ignores `rescale`, so the full dW is written
BOFT | `oft_blocks`, `rescale`, `alpha` | `alpha` as for OFT
IA3 | `diff` | ComfyUI has no IA3 loader
DyLoRA | `lora_up.weight`, `lora_down.weight`, `alpha = rank` | every rank, eval scale folded into up
T-LoRA | rank-2r LoRA | `[P*lam*s, -P0*lam0*s] @ [Q ; Q0]`, no rank mask
full | `diff`, `diff_b` | none
norms | `w_norm`, `b_norm` | none

Joint-block q/k/v are separate layers in KohakuMoss and are written as separate
`..._proj_q` / `_k` / `_v` keys. A loader for the fused `proj.qkv` repack must map them
onto row slices of the fused weight; one that does not skips them.

## Strength

ComfyUI scales dW linearly with strength for the low-rank, DoRA, `diff` and norm
formats, which equals KohakuMoss's multiplier except for `diff`-written OFT-with-rescale.
ComfyUI's OFT and BOFT merges apply strength inside the rotation and again on dW, so
those (and OFT-with-rescale) are exact at strength 1; KohakuMoss's own loader
(`kohakumoss.adapters.load_adapters`) reproduces the training-time multiplier for all.
