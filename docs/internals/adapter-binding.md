# Adapter binding

## Bound forwards

`bind.select.bind_target` resolves each layer's forward once from its kind record and a
`BindCtx` (mode, backend, the layer's kernel ops, multiplier, DoRA-detach, module
dropout, training state, whether the kernel ops apply, draws). Every forward is a
closure built at bind time (`bind.forwards`, `bind.eager`, `bind.compose`): each variant
(scalar parameter or not, rescale or not, multiplier 1 or not, bias or not, base add
inside the op or not) is its own closure, and no bound forward branches on bound
state. The kernel ops apply unless the module trains with input or rank dropout, which
only the LyCORIS module methods carry (`eager.module_bypass`).

## Kernel ops

`bind.ops` (activation-side ops: bypass, rotations, IA3 scale) and `bind.weight_ops`
(weight ops and the DoRA decompose) build each kind's ops as an `autotuned.Op` over one
layer's module:

- `operands(t)`: the operand tuple for the runtime input `t`, built once per call; every
  candidate reads those same objects, so gradients taken with respect to them reach the
  module's parameters;
- `kernel(backend, ops)`: the LyCORIS fused op of `lycoris.kernels.autograd`, by
  `backend=`;
- `body(ops, *consts)`: the eager formula, the public `lycoris.functional` entry on
  backend `"torch"` (`bind.bodies`; IA3 and GLoRA, which have no functional entry, as
  their formula);
- `in_scope(backend)`: whether that backend's fused kernel covers the layer: a static
  scale, LoHa rank within `lycoris.kernels.limits.MAX_HADA_RANK`, a plain LoKr second
  half with every kron factor within the backend's `lokr.bypass.MAX_KRON_FACTOR`, a
  diag-OFT block within the backend's `oft.MAX_BLOCK` (default
  `lycoris.functional.diag_oft.MAX_BLOCK`).

Bypass ops take the (tokens, channels) activation (the input for the low-rank kinds and
input-axis IA3, the base output for the rotations and output-axis IA3), with factors
cast to its dtype as the modules do; weight ops take the fp32 base weight.

`bind.kernel_ops` turns an Op into the callable a forward wrapper takes:

- `FixedBackendOps(backend)` (the adapter configuration names a backend): the fused kernel of a fused
  backend (None outside its scope, so the layer takes the module path), the compiled
  body for `compile`, the eager body for `torch`;
- `AutotunedOps(backend)` (the configured backend is `auto` and the device has a fused
  backend): an `AutotunedOp` per op; an op outside the kernel's scope runs the compiled
  body.

## Autotuned ops

An `AutotunedOp` has two candidates: `kernel` (the fused op of the device's most-prior
fused backend, `lycoris.kernels.dispatch` order) and `compile` (`torch.compile` of the
eager body through `kohakumoss_kernels.backends.compile.ops.CompiledOp`; see
[Checkpoint recompute](#checkpoint-recompute) for how its graphs are cached;
`AdapterConfig.compile_dynamic`: `tokens` (default) marks the token axes dynamic,
`static`, `dynamic`). The first call of a bucket (token bucket, a power of two from
`TOKEN_FLOOR`, activation dtype, grad pattern) asks
`kohakumoss_kernels.tuning.tuner.select` with the problem key (every operand's shape,
token axes bucketed, dtype and grad flag, and the grad mode): a stored winner of this
card binds at once; otherwise both candidates are timed, forward plus backward, on
detached copies of the operands, and the winner is stored and bound. A candidate
raising a build or scope error loses. Where the tuner may not time (inference mode, a
backward, a trace) the bucket runs the kernel unsettled and is asked again on its next
call. Operands no fused kernel takes (mixed fp16 / bf16, bf16 below sm_80) bind the
compiled body. Inside an enclosing `torch.compile` the eager body is traced.

### Checkpoint recompute

Whatever an adapter call binds inside a checkpointed forward is what its recompute
binds, whatever runs in between: other layers, other token counts, other micro-batches
(the threaded and process-group pipelines under `gpipe` run every micro-batch's forward
before any recompute), another stage's thread, the autograd thread. Three things hold it:

- **The global state.** Dynamo guards a graph on the intra-op thread count, and
  `CompiledOp` keys its call pattern on it. A recompute runs on whatever thread runs
  the backward: on Linux the autograd device threads run at 1 thread, the trainer's
  thread at the host's count (32 on a 64-core host). At another count the recompute
  misses the forward's graph and traces a new one from its own inputs, whose partition
  can save a different set of tensors (`CheckpointError`; the threaded pipeline under
  `gpipe` gives it with LoRA, where stage 1's graphs are first traced in training).
  `kohakumoss.memory.runners.CheckpointRunner` therefore runs every recompute under its
  forward's thread count (`forward_state`, a checkpoint `context_fn`; the selective
  contexts inside it), restoring the backward thread's count after it. Grad mode and
  autocast are restored by `torch.utils.checkpoint` itself.

- **The candidate.** A settled bucket stays bound. A bucket first called under
  saved-tensor hooks (a checkpointed forward, `autotuned._recomputed`) where the tuner
  may not time binds the kernel for the process: the problem is shared by every layer
  with the same signature, so another layer outside a checkpoint could settle it to the
  compiled body between the forward and its recompute. Such a bucket is not timed later
  in the process; `AdapterSet.prewarm` settles the planned buckets before step 0.
- **The compiled graph.** `autotuned._graph` holds one `CompiledOp` per body and operand
  shapes apart from the token axes (`autotuned.fixed_shapes`), each with one Dynamo cache
  per call pattern. With the token axes marked dynamic every such cache holds one graph
  per device, traced from the same input spec whatever ran before it. One cache shared
  across layer shapes collects a static graph per shape and then automatic-dynamic
  graphs as new shapes arrive; Dynamo tries the newest graph first, so a recompute that
  comes after other shapes can run a graph that saves different tensors than its
  forward did (`CheckpointError`: a different number or metadata of saved tensors). An
  autotuned op binds a bucket's compile candidate to that `CompiledOp` once
  (`CompiledBody.at`; the token-axis indices are computed once per operand count), and
  resolves that `CompiledOp`'s runner once per intra-op thread count and autocast state
  (the bucket fixes the rest of the call pattern), so a bound call marks its token axes
  dynamic and calls the runner, with no shape or pattern lookup
  (`tests/adapters/test_autotuned_pattern.py`, negative: the pattern per call).
  A pattern past `GRAPH_LIMIT` graphs raises instead of running eager.

### Body candidate

Precision is a pass/fail threshold. The speed choice only ever sees candidates that
pass it. A body that runs through a custom `torch.autograd.Function` (LoHa: LyCORIS's
`HadaWeight` behind `bodies.loha`, `loha_with_base`, `loha_weight`, `loha_merged`, the
set `bodies.AUTOGRAD_FUNCTION_BODIES`) runs the Function's backward outside autocast in
eager, which makes it an fp32 backward. torch.compile traces that same backward under
autocast with 16-bit matmuls. The compiled candidate's gradients then fall below
PyTorch's (fp16 loha.weight grad: abs_mean 8.1e-4 compiled against 2.0e-7 eager), so
it fails the threshold. `autotuned.body_candidate` resolves this at bind time: such an
op's body candidate is `eager` (the eager body), not `compile`. An `AutotunedOp` over
it times kernel against eager. A layer outside the kernel's scope, and the fixed
`compile` backend, run the eager body.

`AdapterSet.prewarm(rows, dtype, devices)` settles every distinct autotuned op
(`AutotunedOp.signature`: name and the layouts of the operands without a token axis)
over the planned token buckets and grad patterns on scratch inputs, before step 0; a
problem already stored is only bound, so a restart times nothing.

## Forward wrappers

With `AdapterConfig.fuse_base_add` (default on) the bind selects the
ops that take the base into the LyCORIS kernel (`base=` / `delta=` / `diff=False` of
`lycoris.kernels.autograd`), so the add happens in the kernel's store and the gradient
passes through to the base; the compile candidate adds it inside the compiled graph:

- bypass (LoRA, LoHa, LoKr): `forwards.with_base(linear, op)` computes
  `op((x, base(x)))`;
- merge at multiplier 1: `forwards.merged_with_base` computes `lin(x, merge(W))`, the
  fp32 base weight as the base (the record's `kernel_merged_weight`);
- DoRA: `forwards.dora_with_delta` decomposes `(W, dW)` with W + dW read inside the
  decompose op (`apply_dora(..., delta=)`);
- OFT / BOFT bypass: `forwards.rotated(linear, rotate)` computes the rotated base output
  itself (`diff=False`), and OFT merge at multiplier 1 the rotated weight. BOFT merge
  keeps the separate add: `boft_diff_weight` has no `diff=False`.

A multiplier other than 1 and a trainable scalar keep the separate add (the scale
applies to dW only).

## Quantized and streamed bases

A base that is not a plain 16/32-bit dense parameter (`Target.stored`: int8 / fp8
weight-only storage, fp8-compute copies, ring / swap streamed slots) keeps the empty
`weight` placeholder its storage left. LyCORIS sizes every module from
`in_features` / `out_features` and, seeing a non-native linear, runs it in bypass; the
construction-time state LyCORIS would read from the base weight (the DoRA magnitude,
the T-LoRA SVD, the OFT / BOFT rescale) is built by `network.Deferred` from one
transient copy of each layer's base. No dense base weight and no dense dW is held:

- training binds bypass only (`ADAPTER_MODE='merge'` on such a base is a validation
  error naming the base); the bypass kernel ops take the 16-bit activation and add to
  the base's own output (fp8 GEMM, quantized kernel or streamed slot);
- OFT at a multiplier other than 1, BOFT on the module path, DyLoRA and GLoRA run
  LyCORIS's bypass forwards (`eager.module_rotation`, one base forward through the
  module's `org_forward`; `eager.dylora_train`; `eager.glora_bypass`, taking back the
  base bias LyCORIS's GLoRA delta adds a second time);
- DoRA (`bind.dora.bypass_out` / `bypass_out_biased` / `bypass_in`) reads one transient
  copy of W for the
  norm of W + dW, then computes `y + m ((s - 1) W x + s dW x)` (output axis) or
  `y + m (W' (c x) - y)` (input axis) from the base forward and the bypass delta;
- inference (`adapters.infer.stored`) attaches every kind as a bypass chain on such a base,
  whatever the mode (a requested merge falls back with one log line); a rotation or
  channel kind reads the unfolded base output through `org_forward` under an fp16 fold,
  a DoRA ratio is computed once at attach, and a dense-diff file is refused.

## Tests

`tests/adapters/test_kernel_ops.py` checks every kernel op on one linear's LyCORIS
module: each candidate the bind can run (the fused kernel where it applies, the body
candidate, the eager body) against the eager body under autocast to the test dtype (as
training runs it) and in fp64, forward and backward, fp16 and bf16. A candidate's
output dtype is eager's or the promotion of the operand dtypes (a kernel stores in the
operands' natural dtype and never forces 16-bit). `tests/adapters/test_body_candidate.py`
checks that the LoHa ops bind kernel against eager, with no compile candidate. `tests/adapters/test_stored_bypass.py` checks one
linear on an int8 / fp8 weight-only base per kind (LoRA, LoHa, LoKr full matrix factor
16, OFT, BOFT, IA3, GLoRA, DoRA, T-LoRA): the bound forward and the inference attach
(bypass and merge mode) against an independent fp64 formula through the error
statistics, with the same adapter on a dense base as the baseline; no stored-weight
read in a bypass forward (DoRA: exactly one); the merge error; the dense-diff refusal.
`tests/training/prewarm/test_adapter_prewarm*.py` check the prewarm's buckets, that it leaves
no gradient on the factors, and that a restart times nothing.
`tests/integration/test_adapters_real.py` runs LoRA and LoKr on the real backbone over
16-bit, int8 and fp8-compute bases: a training step, the adapted layer against fp64,
and the LyCORIS export re-imported.
