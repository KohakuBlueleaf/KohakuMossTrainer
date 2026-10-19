# Reference cycles

Device memory is released by reference counting: a model, an optimiser or a trainer
state frees its tensors when its last name goes away. An object that sits in a reference
cycle is freed only by Python's cyclic collector, which runs on object-allocation counts,
not on device memory: until it runs, every tensor reachable from the cycle stays
allocated. A backbone in a cycle holds its whole weight set after `del`; a second model
loaded in the same process then runs the device out of memory. The source tree therefore
holds no cycle through an object that holds tensors, modules, streams or buffers.

## The rule

An object never stores a callable that refers back to itself, and a hook or callback an
object installs on something it holds reaches it only through a weak reference.

| pattern | instead |
| --- | --- |
| `self._impl = self._dense` (a bound method on self) | the class function, `self._impl = MossLinear._dense`, called as `self._impl(self, x)` |
| `partial(self._run_block, i)` kept on self | a partial over a module (or class) function without self, `partial(_run_block, i)` in `model.backbone`, called with the model |
| a closure or lambda over self kept on self | a partial over a module function that takes the object as an argument |
| `p.register_post_accumulate_grad_hook(partial(self._hook, b))` with `p` held by self | `weak_call(Owner._hook, self, b)` (`kohakumoss_common.weak`) |
| `child.on_grad = self._on_grad` with `child` held by self | `weak_method(self._on_grad)` |
| an attribute called from outside the class (`base.linear(x, bias)`) | a method that calls the stored class function: `def linear(self, x, bias): return self._linear(self, x, bias)` |

`nothing` (`kohakumoss_common.calls`) takes any arguments, so it fills a class-function
slot unchanged. A class-function slot is the first choice; a weak reference is used only
where the object is truly referred back to (a hook on a tensor or module it holds, a
callback its child calls). A weak callback called after its object is gone raises
ReferenceError: a hook that silently did nothing would drop gradients or optimiser
steps. A FusedStep and a LayerStreamer keep no reference to the parameters / layers
they hook, so their hooks hold them strongly and need no weak reference. Every weakly
held object has a strong owner for its working life (the table below).

| weak edge | weakly held | strong owner |
| --- | --- | --- |
| GradOffload post-accumulate hook on each parameter | the GradOffload | the memory engines (`engines.grads`) and the bundle's `hooks` |
| GradReducer post-accumulate hook on each parameter | the GradReducer | `AsyncReducer.inner` (the session's `hooks.ddp`), or its creator |
| `AsyncReducer.reduced` hook body (FusedStep / GradOffload `final`) | the AsyncReducer | the session's `hooks.ddp` |
| per-parameter step (`step_param`) of FusedStep, GradOffload `final`, AsyncReducer slabs | the bundle | the trainer state (`bundle`) |
| StreamedAdamW block transform | the bundle | the trainer state (`bundle`) |
| weight stream `on_grad` / `on_block_backward`, `WeightStream.optimizer` | the StreamedAdamW | the bundle's `blocks` |
| pipeline schedule `loss_fn`, stage `on_chunk` | the PipelineEngine | the pipeline driver (`build_engine`'s caller) |
| `Target.layer`, the bound adapter forwards' layer, the inference chain's base forward, LyCORIS `org_module` / `org_forward` | the adapted linear | the model |
| LyCORIS state-dict hooks | the LyCORIS module | its layer (`_modules["lyco"]`) or the AdapterStack's `live` list |
| prewarm twin grad-carrying forward | the twin's linear | the twin model |

A cycle outside the source tree that holds a frame holds every caller frame above it,
and a frame that outlives its call keeps the locals it returned with. Importing
`torch._dynamo` leaves such a cycle (`torch.fx.wrap` keeps its own frame); the first
kernel build imports it under `load_backbone`, so `registry.load_backends` ends with one
cyclic gc pass ([kernel-dev.md](kernel-dev.md#lazy-backends)).

The dataset is pickled into the data-loader workers, so it holds no weak reference: its
resolution planner keeps the dataset's `text_len` method (a cycle through host-side
catalog columns only, no device tensor). A thread started on an object's method holds
the object until the thread returns (the online-cache writer's thread returns at
`close`).

Selection stays at bind time: the stored class function is the selected implementation,
called directly. The call costs the same as a bound method (one attribute load, one
call with self passed explicitly). A weak callback adds one weak-reference
dereference and one partial call per call; these run once per parameter per step.

## Optimiser wiring

The bundle (`FusedStepBundle` / `Bundle`), the memory engines and the DDP reducer are the
roots the trainer keeps; every edge from what they own back to them is weak:

- the per-parameter step a FusedStep, a GradOffload (`final`) or the AsyncReducer
  delivers to is `weak_method(bundle.step_param)`;
- `AsyncReducer.reduced` returns its hook body over a weak reference to the reducer;
  `block_reduce` returns a strong one (nothing the reducer holds keeps it, and a caller
  may hold only that callable);
- StreamedAdamW installs its gradient and block-backward callbacks on the weight stream
  weakly, and `WeightStream.optimizer` is a weak reference to it; the bundle's block
  transform on StreamedAdamW is weak;
- a GradReducer hook carries its bucket's index, not the bucket: a bucket holds its
  parameters, so a hook on one of them holding the bucket would close a cycle through the
  parameter;
- the stream's BlockReducer reads block ownership from the owners list, not from the
  stream;
- a CPUAdamW lane's copy and step functions take the lane (and the optimiser) as call
  arguments instead of holding them.

## Adapters

A bound adapter forward is installed in its layer's instance dictionary and the LyCORIS
module is registered as the layer's child, so everything the forward and the module refer
to must not refer back to the layer:

- `Target` holds its layer weakly (`Target.linear` is a property over a weak reference);
  `Target.layer` is the weak reference the bound forwards capture, read once per call.
- LyCORIS's `org_module` / `org_forward` are re-pointed at the layer weakly
  (`adapters.build.target.weak_org`) after a module is built, for training and for
  inference loading; the module's own state-dict hooks, which LyCORIS registers as
  bound methods of the module, are held weakly too (`weak_self_hooks`).
- `LycorisNetwork.__init__` leaves the network in a cycle of its own (a recursive
  nested function over the network), so `build_targets` takes the modules out of the
  network's `loras` list; the network itself holds no tensor after that.
- The train/eval re-bind hook sits on the adapted layer (its `train`) and holds the
  binding settings and the target, not the adapter set: the target refers to its
  LyCORIS module, so a hook on the module itself would close a cycle through it.
- The inference chain (`AdapterStack`) starts from the layer's class forward over a
  weak reference; the prewarm twin's grad-carrying forward is a `weak_call` over its
  linear.

## Verification

`tests/_util/release.py` (`check_released`) builds an object, drops it with the cyclic
collector disabled, and checks that weak references to the object and to the parts it
owns (a weight, a buffer) are dead. Each area's test runs it on a variant with one cycle
put back and requires the check to fail.
