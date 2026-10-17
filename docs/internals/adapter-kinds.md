# Adapter kind records

`kohakumoss.adapters.kinds.Kind` holds everything the applier does differently per
LyCORIS algorithm. A Target looks its record up once, from the module class, when it is
built; the binder, the exporters and the inference stack read the record's fields and
never branch on a kind name. `adapters.register(kind)` adds or replaces one.

field | meaning
--- | ---
`name`, `module`, `algos` | kind name, LyCORIS module class, `ALGO` names -> LyCORIS `network_module`
`unit(t)` | a zero-argument builder of dW at multiplier 1 (fp32, before DoRA)
`at(t, m)` | the (dW, db) builder at `m` when it is not `unit * m` (OFT / BOFT rotations, `full`), or None
`fused_unit(t, kit)` | the fused merge-kernel dW builder of the kit, or None
`bypass(t, ctx)` / `merge(t, ctx)` | the fused bound forward for the context, or None for the eager body
`eager(t, ctx)` | the eager bypass forward (default: the module's `bypass_forward`)
`special(t, ctx)` | replaces the selection entirely (`full`, DyLoRA's train / eval forwards)
`comfy(t)` | the ComfyUI tensor set (default: the native set with DoRA's magnitude folded)
`from_comfy(lyco, parts)` | undoes our own ComfyUI conversion on load
`row_names(tensors)` | tensors that a uniform row scale multiplies (one factor per term)
`stacking` | inference stacking: `additive` (per-row scale chains in bypass), `commuting` (block-uniform scale commutes), `chained` (bypass chain, dense under a fold scale), `dense`
`exact_at(m)` | the module's bypass is exact at `m` (OFT: only at 1)
`up_side` | the zero-initialised factors that LoRA+ boosts
`token_axis` | per-token rank masks (T-LoRA): bypass only, a mask group per axis
`draw(t, gen)` | the per-step random draw (DyLoRA's rank prefix)
`dora`, `owns_weight`, `dimless`, `training_dependent` | DoRA allowed, trains the weight itself, needs no `DIM`, forward re-binds on train / eval

A uniform row scale c of dW is written by multiplying one factor of each additive term:
LoRA's `lora_up`, LoHa's `hada_w1_a`, LoKr's `lokr_w1` (or `lokr_w1_a`), T-LoRA's
`lambda_layer` and `base_lambda`, and GLoRA's `b1` alone (its W*a1*a2 term scales with
the base). Rotations and IA3 commute with a uniform scale of their base. A per-row
scale, or a set with no such factor, is written as an exact `diff`.

LyCORIS keeps presets as class state on `LycorisNetwork`, so `resolve_preset` sets every
preset key on each call.
