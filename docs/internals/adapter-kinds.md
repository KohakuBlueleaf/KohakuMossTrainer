# Adapter kind records

`kohakumoss.adapters.kinds.Kind` holds everything the applier does differently per
LyCORIS algorithm. A Target looks its record up once, from the module class, when it is
built; the binder, the exporters and the inference stack read the record's fields and
never branch on a kind name. `adapters.register(kind)` adds or replaces one.

field | meaning
--- | ---
`name`, `module`, `algos` | kind name, LyCORIS module class, `ALGO` names -> LyCORIS `network_module`
`diff_weight(t)` | dW at multiplier 1 (fp32, before DoRA); `Target.unit_fn` binds it
`diff_weight_at(t, m)` | the (dW, db) builder at `m` when it is not `diff_weight * m` (OFT / BOFT rotations), or None
`kernel_diff_weight(t, ops)` | W -> dW at multiplier 1 from the layer's kernel ops, or None
`kernel_merged_weight(t, ops)` | W -> W + dW at multiplier 1 with the add inside the op (`base=`), or None
`kernel_bypass(t, ctx)` / `kernel_merge(t, ctx)` | the bound forward from the kernel ops for the context, or None for the module path
`module_forward(t, ctx)` | the bypass forward over module methods (default: the module's `bypass_forward_diff`)
`own_forward(t, ctx)` | replaces the selection entirely (DyLoRA's train / eval bypass forwards)
`comfy_tensors(t)` | the ComfyUI tensor set (default: the native set with DoRA's magnitude folded)
`undo_comfy(lyco, parts)` | undoes our own ComfyUI conversion on load
`uniform_scale_tensors(tensors)` | tensors that a uniform row scale multiplies (one factor per term)
`row_scale_tensors(tensors)` | tensors whose output rows a per-row scale multiplies
`stacking` | inference stacking: `additive` (per-row scale chains in bypass), `commuting` (block-uniform scale commutes), `chained` (bypass chain, dense under a fold scale), `dense`
`bypass_exact_at(m)` | the module's bypass is exact at `m` (OFT: only at 1)
`loraplus_tensors` | the zero-initialised factors that LoRA+ boosts
`per_token_mask` | per-token rank masks (T-LoRA): bypass only, a mask group per axis
`draw_step(t, gen)` | the per-step random draw (DyLoRA's rank prefix)
`supports_dora`, `rankless`, `training_dependent` | DoRA allowed, needs no `DIM`, forward re-binds on train / eval

A uniform row scale c of dW is written by multiplying one factor of each additive term:
LoRA's `lora_up`, LoHa's `hada_w1_a`, LoKr's `lokr_w1` (or `lokr_w1_a`), T-LoRA's
`lambda_layer` and `base_lambda`, and GLoRA's `b1` alone (its W*a1*a2 term scales with
the base). Rotations and IA3 commute with a uniform scale of their base. A per-row
scale, or a set with no such factor, is written as an exact `diff`.

LyCORIS keeps presets as class state on `LycorisNetwork`, so `resolve_preset` sets every
preset key on each call.
