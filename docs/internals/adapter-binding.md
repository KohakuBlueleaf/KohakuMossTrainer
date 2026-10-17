# Adapter binding

## Bound forwards

`bind.select.bind_target` resolves each layer's forward once from its kind record and a
`BindCtx` (mode, kit, multiplier, DoRA-detach, module dropout, training state, draws).
Every function in `bind.fused` and `bind.eager` takes its bound state first and the
runtime tensor last, so `functools.partial` freezes the state at bind time; each
variant (scalar parameter or not, rescale or not, multiplier 1 or not, bias or not) is
its own function, and no bound forward branches on bound state.

## Kits

`bind.kits.Pinned(backend)` calls one backend's fused kernels
(`lycoris.kernels.autograd.*`) directly. `bind.kits.Raced(backend)` binds every op to a
`bind.race.TierSlot`. Both expose the same factories and return None where they have no
fused path.

## Tier ops

`bind.ops` mirrors the `tier.run` call of each `lycoris.functional` entry as a
`race.Op`: the same op name, operand tuple, scope test (`supported`), fused kernel,
eager body and exact extras, so the race and its recorded decision are the ones LyCORIS
makes for that call. `operands(t)` builds the operand objects once per call and every
arm reads those same objects: the race's fp64 gate maps gradients by operand identity.
Bypass ops take the (tokens, channels) activation (the input for the low-rank kinds,
the base output for the rotations), with factors cast to its dtype as the modules do;
merge ops take the fp32 base weight.

## Tier slots

A `TierSlot` asks `tier.best` on the first call of a bucket and keeps the arm once
`tier.is_settled` holds; a later call of the bucket is one dict lookup and the arm.
LyCORIS fixes a key's tier at its first call and races only outside checkpointed
forwards and backwards, so a checkpoint recompute reruns the arm its forward ran. The
bucket covers what varies per call (token bucket, activation dtype, grad pattern);
factor shapes, dtypes and grad flags are fixed between `AdapterSet.bind()` calls, each
of which builds fresh slots. Inside an enclosing `torch.compile` a miss stores nothing
and the compile arm runs the eager body for the enclosing compiler. `TierSlot.warm`
resolves every bucket on scratch operands (`AdapterSet.prewarm`).
