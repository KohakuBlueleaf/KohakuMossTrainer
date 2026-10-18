# Memory profiles

Memory placement is set in one of two exclusive modes, chosen by `MEMORY_PROFILE`
(the same knob in `scripts/train/moss.py` and `scripts/sample/sample.py`):

- **A named profile** (`balanced`, `ckpt-stream4-actoff`, ...) sets every memory knob.
  Setting a memory knob next to it is a validation error naming both the profile and the
  knob.
- **`manual`** takes every memory knob from the config; a knob left `None` takes the
  default in the table below.

Either way the plan resolves once, at build time, into a concrete `MemoryPlan`, and knob
validation resolves it for the local device, so every invalid combination is reported in
the one startup error, before any weight is loaded. `MEMORY_OVERRIDES` (a dict of raw
plan fields) applies on top of either mode.

## Memory knobs

| knob | plan field | values | default | effect |
| --- | --- | --- | --- | --- |
| `CHECKPOINT` | `checkpoint` | `none`, `block`, `sub-block`, `selective`, `partial:<fraction>` | `block` | which activations are recomputed in backward |
| `STREAM` | `stream` | `off`, `ring` (frozen, host to device only), `swap` (trainable) | `off` | block weights live on the host and stream in |
| `STREAM_RESIDENT` | `stream_resident` | int >= 2 | `0` (unused with `STREAM='off'`) | blocks kept on the device while streaming |
| `PINNED` | `pinned` | bool | `True` | pinned vs staged host buffers (streams, offloads, host optimiser states) |
| `ACT_OFFLOAD` | `act_offload` | bool | `False` | saved activations go to pinned host memory, prefetched one block ahead |
| `OPTIM_STATE` | `optim_state` | `gpu`, `stream` (host states, update on the GPU), `cpu` (host update) | `gpu` | AdamW state placement only (`OPTIMIZER='adamw'`; other optimisers keep `gpu`) |
| `GRAD_OFFLOAD` | `grad_offload` | bool | `False` | gradients summed in host memory across GRAD_ACC |
| `FUSED_STEP` | `fused_step` | bool | `False` | per-parameter optimiser step inside backward |

Inference (`sample.py`) has no backward, so only `STREAM`, `STREAM_RESIDENT` and `PINNED`
exist there; its plan never checkpoints or offloads activations, and `swap` becomes
`ring`.

`compile` and `kernels` are plan fields owned by the `COMPILE` and `KERNELS` knobs.
`act_offload_inflight` (int >= 1, default 2; `MEMORY_OVERRIDES` only) is how many blocks
of activation-offload copies may be queued before the host waits for the oldest block's
(see [offload.md](../internals/offload.md#activation-offload)).

## Precision knobs

Precision is never part of a profile. These knobs apply under every profile and in
manual mode, and a profile never picks a quantized base:

| knob | plan field | values | default | effect |
| --- | --- | --- | --- | --- |
| `BASE_PRECISION` | `base_precision` | `compute`, `fp8`, `int8`, `quant` (fp8 where the arch has fp8 GEMMs, else int8), `fp8_compute`, `mxfp8` | `compute` | storage of the frozen base weights (`compute`: the PRECISION dtype; `fp8` / `int8` weight-only); `fp8_compute` / `mxfp8` run the GEMMs in fp8 on sm89+; see [quant-kernels.md](../internals/quant-kernels.md#fp8-compute) |
| `FP8_RECIPE` | `fp8_recipe` | `row`, `tensor`, `block` | `row` | scale granularity of `fp8_compute` |
| `FP8_DGRAD` | `fp8_dgrad` | `fp8`, `16bit` | `fp8` | input-gradient GEMM in fp8 (a W^T fp8 copy, 2 bytes per weight in all) or 16-bit (1 byte per weight) |

## Rules that constrain a plan

Weight streaming needs no checkpointing, for every base precision: a streamed linear
saves only its input (an fp8-compute linear over a frozen base saves nothing), and its
backward streams the block's weights in again
([offload.md](../internals/offload.md#weight-streaming)). The remaining rules are
physical:

| rule | reason |
| --- | --- |
| `stream_resident >= 2` | one slot computes while the next one loads |
| `stream='ring'` is frozen-base only; `swap` is trainable-only | a ring slot is overwritten without a write-back; `swap` writes trained blocks back to the host |
| `swap`: `fused_step` is `True` with a GPU `optim_state`, `False` with `optim_state='cpu'` | a swapped block leaves the device after its backward, so a GPU step runs inside backward; a host step runs after it |
| a quantized base (`fp8`, `int8`, `quant`) is not trainable | the stored weights are not the trained values |
| an `fp8_compute` / `mxfp8` full FT keeps the weights resident | the fp8 copies are requantized after each step |
| `fp8`, `fp8_compute`, `mxfp8` need sm89+ (CUDA) | fp8 GEMMs |
| `optim_state` other than `gpu` needs `OPTIMIZER='adamw'` | it places AdamW states |
| `swap` needs `OPTIMIZER='adamw'`; Muon needs `fused_step=False` | Muon steps whole matrices after backward (full FT with Muon uses a resident, unfused profile, e.g. `ckpt-resident`) |
| DDP runs every profile; FSDP / pipeline / threaded need resident compute-precision weights, GPU states, no activation offload, no `fused_step` and no `grad_offload` | DDP sums every gradient across ranks wherever the plan moves it; the other multi-GPU strategies own the parameters; see [multi-gpu.md](../guides/multi-gpu.md) |
| `checkpoint='sub-block'` not with `COMPILE='block'` or FSDP | sub-block checkpointing calls block parts directly |

With a full finetune (`ALGO='full'`), a catalog entry's `ring` resolves to `swap`, and
the fused step follows the `swap` rule above.

## The catalog

The catalog is the full product of five placement axes, every entry named by its
non-empty parts:

| axis | parts |
| --- | --- |
| checkpoint | `nockpt` (none), `ckpt` (block), `subckpt` (sub-block) |
| weights | `resident`, `stream16`, `stream8`, `stream4`, `stream2` (ring with that many resident blocks) |
| activations | (none), `actoff` |
| optimizer | (gpu), `optfused` (gpu, fused step), `optstream` (fused), `optcpu` |
| gradients | (none), `gradoff` |

3 x 5 x 2 x 4 x 2 = 240 entries (8-bit states are an `OPTIMIZER`, `adamw8bit` or
`adamw8bit_kernel`, with any GPU placement, not a placement axis); every combination of
these axes validates for a frozen base at compute precision, so none is excluded. The
short names are catalog entries; a run that uses one logs the entry it equals:

| short name | catalog entry |
| --- | --- |
| `speed` | `nockpt-resident` |
| `balanced` | `ckpt-stream16` |
| `low` | `ckpt-stream4-actoff-optstream` |
| `minimal` | `ckpt-stream2-actoff-optstream-gradoff` |

`auto` takes the first of `speed`, `balanced`, `low`, `minimal` that is runnable with the
other knobs and whose estimate fits the device's free memory at planning time (times
0.92; the context is already outside free memory, so the estimate's context part is not
counted twice); the planner's per-profile rows are kept in the run's memory reports.
Under a process group rank 0 plans and every rank resolves rank 0's profile (handed over
on the store, `kohakumoss.parallel.launch.rank0_value`), so every rank builds the same
plan whatever its own free memory.
With text encoded online on the training device, each profile's estimate includes the
text encoder in the fastest placement (`resident`, else `stream`) that fits the budget
next to that profile's DiT step, or `TEXT_ENCODE['memory']` when set: its sizes come from its `config.json` alone, a weightless meta-device twin in
the plan's dtype and precision (`kohakumoss.training.build.text_estimate`), with one
pass of `TEXT_ENCODE['max_tokens']` (default 16384) tokens; no weight is read before
the plan. An offload / stream pass peaks on top of the parts allocated for the whole run
(context, weights, adapter, optimiser state and slots, the state and gradient rings,
the EMA). Each row records `text_memory` and `estimate_with_text`.
When no profile's estimate fits, `auto` stops the run with an error listing each
profile's estimate and the free memory. `MEMORY_AUTO_FORCE=True` runs the last runnable
one (normally `minimal`) instead and logs `MEMORY_PROFILE='auto' -> <profile> (nothing
fits, forced): no profile's estimate fits the free memory; MEMORY_AUTO_FORCE runs it
anyway` at WARNING.
`STEP_RELEASE` returns free cached device memory to the driver between steps: `"none"`
(default, the allocator keeps its blocks), `"empty_cache"` (after every optimizer step)
or an integer N (every N steps); it trades a sync and re-allocation per release for a
lower nvidia-smi steady use on small cards.
The estimates are analytic: they model allocated bytes plus a fragmentation factor, not
measured device use. `kohakumoss-doctor --memory` prints those four
resolved for the local device.

Training's encoders follow `TEXT_ENCODE["memory"]` and `VAE_ENCODE["memory"]`, which
always win (the encoders can be offloaded alone next to a resident DiT). When the dict
leaves `memory` unset, the text encoder is `auto` under every plan: the fastest
placement (`resident`, else `stream`) that fits the memory left next to the DiT step,
so a large card keeps it resident whether or not the DiT streams. The VAE follows the
resolved memory plan, whatever spelling chose it (a short name, a catalog entry,
`auto` or `manual`): offloaded next to streamed DiT blocks, resident otherwise
(`kohakumoss.training.build.model.encoder_memory`). The build logs each placement and
its source, e.g. `text encoder placement: resident (TEXT_ENCODE['memory'] unset: the
fastest placement that fits)`. The VRAM layers set both dicts explicitly
([configs.md](../guides/configs.md#layers)).

## Estimated peak VRAM

The analytic memory model (`kohakumoss.memory.estimate`) for one micro-step of
the MOSS base DiT at 1 MP (4096 image + 300 text tokens, bf16, fused kernels): LoRA
rank 16 on every block linear, and full FT (where `stream N` is `swap`). Text encoder and
VAE are not included (cached or placed separately). `kohakumoss-config list` prints the
same table; `tests/memory/profiles/test_catalog_table.py` keeps the two equal. A full
finetune's device parameters are its masters: fp32 (the compute dtype under `adamw16`),
4 B per resident trainable parameter, with no separate master copy, and so are their
gradients; a `swap` block's resident slots and gradients are in the compute dtype. The
full FT column counts the device buffers the offload optimisers allocate:
`StreamedAdamW`'s slot ring for `swap` blocks (sized by a block), `HostStateAdamW`'s
state ring and gradient offload's staging ring (each sized by the largest device
parameter, the 6 D x D block-sharing modulation).

The total is a device-level number: the allocator's parts times the plan's
fragmentation (reserved over allocated: `FRAGMENTATION` 1.08 when every buffer lives on
the compute stream, `FRAGMENTATION_OFFLOAD` 1.12 when the plan keeps buffers on side
streams: streamed weights, host optimiser states, gradient or activation offload), plus
`context` (0.88 GiB: the CUDA context, loaded kernels and libraries, outside the
allocator). A measured `torch.cuda.max_memory_allocated` peak compares with the sum of
the parts without `context`. Full FT under `minimal` with 4 resident `swap` blocks
(`ckpt-stream4-actoff-optstream-gradoff` at 1 MP, bf16), each part against the buffer
the code allocates for it:

| part | GiB | what `minimal` keeps on the device |
| --- | --- | --- |
| `weights` | 2.06 | the non-block parameters (0.33 B, fp32 masters) + the 4 resident `swap` slots of the weight stream (bf16) |
| `optim_slots` | 5.70 | `StreamedAdamW`: 3 slots x per block parameter (fp32 master, m, v rows; fp32 gradient sum; bf16 updated weight) = 18 B x 113 M |
| `state_ring` | 2.53 | `HostStateAdamW`: 3 slots x 4 fp32 rows (master, m, v, gradient) of the largest device parameter (56.6 M) |
| `grad_ring` | 0.63 | gradient offload: 3 fp32 staging buffers of the largest device parameter |
| `grads` | 0.21 | one block's bf16 gradients before they land in its slot |
| `recompute` | 0.50 | one checkpointed block's recompute window |
| `activations` / `text_refiner` / `io` | 0.23 | 2 in-flight offloaded blocks' inputs, the text-refiner activations, step inputs / outputs |
| `workspace` | 0.25 | GEMM / kernel workspaces |
| `context` | 0.88 | the CUDA context, kernels and libraries (outside the allocator) |

Allocated: 12.11 GiB; reserved (x 1.12): 13.56 GiB; device total: 14.44 GiB. Measured on
an RTX 3090 (`scripts/bench/train/memory.py`): 11.93 GiB allocated, 13.26 GiB reserved
(1.11 x allocated), 14.14 GiB device use (0.88 GiB outside the allocator);
`tests/training/memory/test_estimate_swap_slots.py` keeps the estimate at or above each
of the three. Every part is a buffer that exists at the step's peak; none is counted
twice, and none is something `minimal` offloads (masters, moments and gradient sums of
the blocks stay on the host; only the rings above and the non-block masters are on the
device).

<!-- catalog:begin -->
| profile | checkpoint | weights | act_offload | optimizer | grad_offload | LoRA GiB | full FT GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `speed` | none | resident | off | gpu | off | 35.0 | 106.0 |
| `balanced` | block | stream 16 | off | gpu | off | 8.2 | 53.0 |
| `low` | block | stream 4 | on | stream fused | off | 3.9 | 13.7 |
| `minimal` | block | stream 2 | on | stream fused | on | 3.3 | 14.0 |
| `nockpt-resident` | none | resident | off | gpu | off | 35.0 | 106.0 |
| `nockpt-resident-gradoff` | none | resident | off | gpu | on | 36.1 | 89.8 |
| `nockpt-resident-optfused` | none | resident | off | gpu fused | off | 35.0 | 86.0 |
| `nockpt-resident-optfused-gradoff` | none | resident | off | gpu fused | on | 36.1 | 89.8 |
| `nockpt-resident-optstream` | none | resident | off | stream fused | off | 36.0 | 49.5 |
| `nockpt-resident-optstream-gradoff` | none | resident | off | stream fused | on | 35.8 | 50.2 |
| `nockpt-resident-optcpu` | none | resident | off | cpu | off | 36.0 | 67.4 |
| `nockpt-resident-optcpu-gradoff` | none | resident | off | cpu | on | 35.8 | 47.4 |
| `nockpt-resident-actoff` | none | resident | on | gpu | off | 13.7 | 87.3 |
| `nockpt-resident-actoff-gradoff` | none | resident | on | gpu | on | 13.6 | 67.3 |
| `nockpt-resident-actoff-optfused` | none | resident | on | gpu fused | off | 13.7 | 66.6 |
| `nockpt-resident-actoff-optfused-gradoff` | none | resident | on | gpu fused | on | 13.6 | 67.3 |
| `nockpt-resident-actoff-optstream` | none | resident | on | stream fused | off | 13.4 | 27.0 |
| `nockpt-resident-actoff-optstream-gradoff` | none | resident | on | stream fused | on | 13.3 | 27.7 |
| `nockpt-resident-actoff-optcpu` | none | resident | on | cpu | off | 13.4 | 44.9 |
| `nockpt-resident-actoff-optcpu-gradoff` | none | resident | on | cpu | on | 13.3 | 24.9 |
| `nockpt-stream16` | none | stream 16 | off | gpu | off | 30.1 | 74.9 |
| `nockpt-stream16-gradoff` | none | stream 16 | off | gpu | on | 30.0 | 75.6 |
| `nockpt-stream16-optfused` | none | stream 16 | off | gpu fused | off | 30.1 | 74.9 |
| `nockpt-stream16-optfused-gradoff` | none | stream 16 | off | gpu fused | on | 30.0 | 75.6 |
| `nockpt-stream16-optstream` | none | stream 16 | off | stream fused | off | 29.8 | 39.6 |
| `nockpt-stream16-optstream-gradoff` | none | stream 16 | off | stream fused | on | 29.6 | 40.3 |
| `nockpt-stream16-optcpu` | none | stream 16 | off | cpu | off | 29.8 | 30.4 |
| `nockpt-stream16-optcpu-gradoff` | none | stream 16 | off | cpu | on | 29.6 | 31.1 |
| `nockpt-stream16-actoff` | none | stream 16 | on | gpu | off | 7.6 | 52.4 |
| `nockpt-stream16-actoff-gradoff` | none | stream 16 | on | gpu | on | 7.4 | 53.1 |
| `nockpt-stream16-actoff-optfused` | none | stream 16 | on | gpu fused | off | 7.6 | 52.4 |
| `nockpt-stream16-actoff-optfused-gradoff` | none | stream 16 | on | gpu fused | on | 7.4 | 53.1 |
| `nockpt-stream16-actoff-optstream` | none | stream 16 | on | stream fused | off | 7.3 | 17.1 |
| `nockpt-stream16-actoff-optstream-gradoff` | none | stream 16 | on | stream fused | on | 7.1 | 17.8 |
| `nockpt-stream16-actoff-optcpu` | none | stream 16 | on | cpu | off | 7.3 | 7.9 |
| `nockpt-stream16-actoff-optcpu-gradoff` | none | stream 16 | on | cpu | on | 7.1 | 8.6 |
| `nockpt-stream8` | none | stream 8 | off | gpu | off | 28.2 | 73.0 |
| `nockpt-stream8-gradoff` | none | stream 8 | off | gpu | on | 28.1 | 73.8 |
| `nockpt-stream8-optfused` | none | stream 8 | off | gpu fused | off | 28.2 | 73.0 |
| `nockpt-stream8-optfused-gradoff` | none | stream 8 | off | gpu fused | on | 28.1 | 73.8 |
| `nockpt-stream8-optstream` | none | stream 8 | off | stream fused | off | 27.9 | 37.7 |
| `nockpt-stream8-optstream-gradoff` | none | stream 8 | off | stream fused | on | 27.8 | 38.4 |
| `nockpt-stream8-optcpu` | none | stream 8 | off | cpu | off | 27.9 | 28.5 |
| `nockpt-stream8-optcpu-gradoff` | none | stream 8 | off | cpu | on | 27.8 | 29.2 |
| `nockpt-stream8-actoff` | none | stream 8 | on | gpu | off | 5.7 | 50.5 |
| `nockpt-stream8-actoff-gradoff` | none | stream 8 | on | gpu | on | 5.5 | 51.2 |
| `nockpt-stream8-actoff-optfused` | none | stream 8 | on | gpu fused | off | 5.7 | 50.5 |
| `nockpt-stream8-actoff-optfused-gradoff` | none | stream 8 | on | gpu fused | on | 5.5 | 51.2 |
| `nockpt-stream8-actoff-optstream` | none | stream 8 | on | stream fused | off | 5.4 | 15.2 |
| `nockpt-stream8-actoff-optstream-gradoff` | none | stream 8 | on | stream fused | on | 5.2 | 15.9 |
| `nockpt-stream8-actoff-optcpu` | none | stream 8 | on | cpu | off | 5.4 | 6.0 |
| `nockpt-stream8-actoff-optcpu-gradoff` | none | stream 8 | on | cpu | on | 5.2 | 6.7 |
| `nockpt-stream4` | none | stream 4 | off | gpu | off | 27.3 | 72.1 |
| `nockpt-stream4-gradoff` | none | stream 4 | off | gpu | on | 27.1 | 72.8 |
| `nockpt-stream4-optfused` | none | stream 4 | off | gpu fused | off | 27.3 | 72.1 |
| `nockpt-stream4-optfused-gradoff` | none | stream 4 | off | gpu fused | on | 27.1 | 72.8 |
| `nockpt-stream4-optstream` | none | stream 4 | off | stream fused | off | 27.0 | 36.8 |
| `nockpt-stream4-optstream-gradoff` | none | stream 4 | off | stream fused | on | 26.8 | 37.5 |
| `nockpt-stream4-optcpu` | none | stream 4 | off | cpu | off | 27.0 | 27.6 |
| `nockpt-stream4-optcpu-gradoff` | none | stream 4 | off | cpu | on | 26.8 | 28.3 |
| `nockpt-stream4-actoff` | none | stream 4 | on | gpu | off | 4.8 | 49.6 |
| `nockpt-stream4-actoff-gradoff` | none | stream 4 | on | gpu | on | 4.6 | 50.3 |
| `nockpt-stream4-actoff-optfused` | none | stream 4 | on | gpu fused | off | 4.8 | 49.6 |
| `nockpt-stream4-actoff-optfused-gradoff` | none | stream 4 | on | gpu fused | on | 4.6 | 50.3 |
| `nockpt-stream4-actoff-optstream` | none | stream 4 | on | stream fused | off | 4.4 | 14.2 |
| `nockpt-stream4-actoff-optstream-gradoff` | none | stream 4 | on | stream fused | on | 4.3 | 14.9 |
| `nockpt-stream4-actoff-optcpu` | none | stream 4 | on | cpu | off | 4.4 | 5.0 |
| `nockpt-stream4-actoff-optcpu-gradoff` | none | stream 4 | on | cpu | on | 4.3 | 5.7 |
| `nockpt-stream2` | none | stream 2 | off | gpu | off | 26.8 | 71.6 |
| `nockpt-stream2-gradoff` | none | stream 2 | off | gpu | on | 26.7 | 72.3 |
| `nockpt-stream2-optfused` | none | stream 2 | off | gpu fused | off | 26.8 | 71.6 |
| `nockpt-stream2-optfused-gradoff` | none | stream 2 | off | gpu fused | on | 26.7 | 72.3 |
| `nockpt-stream2-optstream` | none | stream 2 | off | stream fused | off | 26.5 | 36.3 |
| `nockpt-stream2-optstream-gradoff` | none | stream 2 | off | stream fused | on | 26.3 | 37.0 |
| `nockpt-stream2-optcpu` | none | stream 2 | off | cpu | off | 26.5 | 27.1 |
| `nockpt-stream2-optcpu-gradoff` | none | stream 2 | off | cpu | on | 26.3 | 27.8 |
| `nockpt-stream2-actoff` | none | stream 2 | on | gpu | off | 4.3 | 49.1 |
| `nockpt-stream2-actoff-gradoff` | none | stream 2 | on | gpu | on | 4.1 | 49.8 |
| `nockpt-stream2-actoff-optfused` | none | stream 2 | on | gpu fused | off | 4.3 | 49.1 |
| `nockpt-stream2-actoff-optfused-gradoff` | none | stream 2 | on | gpu fused | on | 4.1 | 49.8 |
| `nockpt-stream2-actoff-optstream` | none | stream 2 | on | stream fused | off | 4.0 | 13.8 |
| `nockpt-stream2-actoff-optstream-gradoff` | none | stream 2 | on | stream fused | on | 3.8 | 14.5 |
| `nockpt-stream2-actoff-optcpu` | none | stream 2 | on | cpu | off | 4.0 | 4.6 |
| `nockpt-stream2-actoff-optcpu-gradoff` | none | stream 2 | on | cpu | on | 3.8 | 5.3 |
| `ckpt-resident` | block | resident | off | gpu | off | 13.9 | 84.8 |
| `ckpt-resident-gradoff` | block | resident | off | gpu | on | 14.2 | 67.9 |
| `ckpt-resident-optfused` | block | resident | off | gpu fused | off | 13.9 | 64.8 |
| `ckpt-resident-optfused-gradoff` | block | resident | off | gpu fused | on | 14.2 | 67.9 |
| `ckpt-resident-optstream` | block | resident | off | stream fused | off | 14.0 | 27.6 |
| `ckpt-resident-optstream-gradoff` | block | resident | off | stream fused | on | 13.9 | 28.3 |
| `ckpt-resident-optcpu` | block | resident | off | cpu | off | 14.0 | 45.5 |
| `ckpt-resident-optcpu-gradoff` | block | resident | off | cpu | on | 13.9 | 25.5 |
| `ckpt-resident-actoff` | block | resident | on | gpu | off | 13.2 | 86.8 |
| `ckpt-resident-actoff-gradoff` | block | resident | on | gpu | on | 13.1 | 66.8 |
| `ckpt-resident-actoff-optfused` | block | resident | on | gpu fused | off | 13.2 | 66.1 |
| `ckpt-resident-actoff-optfused-gradoff` | block | resident | on | gpu fused | on | 13.1 | 66.8 |
| `ckpt-resident-actoff-optstream` | block | resident | on | stream fused | off | 12.9 | 26.5 |
| `ckpt-resident-actoff-optstream-gradoff` | block | resident | on | stream fused | on | 12.7 | 27.2 |
| `ckpt-resident-actoff-optcpu` | block | resident | on | cpu | off | 12.9 | 44.4 |
| `ckpt-resident-actoff-optcpu-gradoff` | block | resident | on | cpu | on | 12.7 | 24.4 |
| `ckpt-stream16` | block | stream 16 | off | gpu | off | 8.2 | 53.0 |
| `ckpt-stream16-gradoff` | block | stream 16 | off | gpu | on | 8.1 | 53.7 |
| `ckpt-stream16-optfused` | block | stream 16 | off | gpu fused | off | 8.2 | 53.0 |
| `ckpt-stream16-optfused-gradoff` | block | stream 16 | off | gpu fused | on | 8.1 | 53.7 |
| `ckpt-stream16-optstream` | block | stream 16 | off | stream fused | off | 7.9 | 17.7 |
| `ckpt-stream16-optstream-gradoff` | block | stream 16 | off | stream fused | on | 7.7 | 18.4 |
| `ckpt-stream16-optcpu` | block | stream 16 | off | cpu | off | 7.9 | 8.5 |
| `ckpt-stream16-optcpu-gradoff` | block | stream 16 | off | cpu | on | 7.7 | 9.2 |
| `ckpt-stream16-actoff` | block | stream 16 | on | gpu | off | 7.1 | 51.9 |
| `ckpt-stream16-actoff-gradoff` | block | stream 16 | on | gpu | on | 6.9 | 52.6 |
| `ckpt-stream16-actoff-optfused` | block | stream 16 | on | gpu fused | off | 7.1 | 51.9 |
| `ckpt-stream16-actoff-optfused-gradoff` | block | stream 16 | on | gpu fused | on | 6.9 | 52.6 |
| `ckpt-stream16-actoff-optstream` | block | stream 16 | on | stream fused | off | 6.8 | 16.6 |
| `ckpt-stream16-actoff-optstream-gradoff` | block | stream 16 | on | stream fused | on | 6.6 | 17.3 |
| `ckpt-stream16-actoff-optcpu` | block | stream 16 | on | cpu | off | 6.8 | 7.4 |
| `ckpt-stream16-actoff-optcpu-gradoff` | block | stream 16 | on | cpu | on | 6.6 | 8.1 |
| `ckpt-stream8` | block | stream 8 | off | gpu | off | 6.3 | 51.1 |
| `ckpt-stream8-gradoff` | block | stream 8 | off | gpu | on | 6.2 | 51.8 |
| `ckpt-stream8-optfused` | block | stream 8 | off | gpu fused | off | 6.3 | 51.1 |
| `ckpt-stream8-optfused-gradoff` | block | stream 8 | off | gpu fused | on | 6.2 | 51.8 |
| `ckpt-stream8-optstream` | block | stream 8 | off | stream fused | off | 6.0 | 15.8 |
| `ckpt-stream8-optstream-gradoff` | block | stream 8 | off | stream fused | on | 5.8 | 16.5 |
| `ckpt-stream8-optcpu` | block | stream 8 | off | cpu | off | 6.0 | 6.6 |
| `ckpt-stream8-optcpu-gradoff` | block | stream 8 | off | cpu | on | 5.8 | 7.3 |
| `ckpt-stream8-actoff` | block | stream 8 | on | gpu | off | 5.2 | 50.0 |
| `ckpt-stream8-actoff-gradoff` | block | stream 8 | on | gpu | on | 5.0 | 50.7 |
| `ckpt-stream8-actoff-optfused` | block | stream 8 | on | gpu fused | off | 5.2 | 50.0 |
| `ckpt-stream8-actoff-optfused-gradoff` | block | stream 8 | on | gpu fused | on | 5.0 | 50.7 |
| `ckpt-stream8-actoff-optstream` | block | stream 8 | on | stream fused | off | 4.9 | 14.7 |
| `ckpt-stream8-actoff-optstream-gradoff` | block | stream 8 | on | stream fused | on | 4.7 | 15.4 |
| `ckpt-stream8-actoff-optcpu` | block | stream 8 | on | cpu | off | 4.9 | 5.5 |
| `ckpt-stream8-actoff-optcpu-gradoff` | block | stream 8 | on | cpu | on | 4.7 | 6.2 |
| `ckpt-stream4` | block | stream 4 | off | gpu | off | 5.4 | 50.2 |
| `ckpt-stream4-gradoff` | block | stream 4 | off | gpu | on | 5.2 | 50.9 |
| `ckpt-stream4-optfused` | block | stream 4 | off | gpu fused | off | 5.4 | 50.2 |
| `ckpt-stream4-optfused-gradoff` | block | stream 4 | off | gpu fused | on | 5.2 | 50.9 |
| `ckpt-stream4-optstream` | block | stream 4 | off | stream fused | off | 5.1 | 14.9 |
| `ckpt-stream4-optstream-gradoff` | block | stream 4 | off | stream fused | on | 4.9 | 15.6 |
| `ckpt-stream4-optcpu` | block | stream 4 | off | cpu | off | 5.1 | 5.6 |
| `ckpt-stream4-optcpu-gradoff` | block | stream 4 | off | cpu | on | 4.9 | 6.4 |
| `ckpt-stream4-actoff` | block | stream 4 | on | gpu | off | 4.3 | 49.1 |
| `ckpt-stream4-actoff-gradoff` | block | stream 4 | on | gpu | on | 4.1 | 49.8 |
| `ckpt-stream4-actoff-optfused` | block | stream 4 | on | gpu fused | off | 4.3 | 49.1 |
| `ckpt-stream4-actoff-optfused-gradoff` | block | stream 4 | on | gpu fused | on | 4.1 | 49.8 |
| `ckpt-stream4-actoff-optstream` | block | stream 4 | on | stream fused | off | 3.9 | 13.7 |
| `ckpt-stream4-actoff-optstream-gradoff` | block | stream 4 | on | stream fused | on | 3.8 | 14.4 |
| `ckpt-stream4-actoff-optcpu` | block | stream 4 | on | cpu | off | 3.9 | 4.5 |
| `ckpt-stream4-actoff-optcpu-gradoff` | block | stream 4 | on | cpu | on | 3.8 | 5.2 |
| `ckpt-stream2` | block | stream 2 | off | gpu | off | 4.9 | 49.7 |
| `ckpt-stream2-gradoff` | block | stream 2 | off | gpu | on | 4.7 | 50.4 |
| `ckpt-stream2-optfused` | block | stream 2 | off | gpu fused | off | 4.9 | 49.7 |
| `ckpt-stream2-optfused-gradoff` | block | stream 2 | off | gpu fused | on | 4.7 | 50.4 |
| `ckpt-stream2-optstream` | block | stream 2 | off | stream fused | off | 4.6 | 14.4 |
| `ckpt-stream2-optstream-gradoff` | block | stream 2 | off | stream fused | on | 4.4 | 15.1 |
| `ckpt-stream2-optcpu` | block | stream 2 | off | cpu | off | 4.6 | 5.2 |
| `ckpt-stream2-optcpu-gradoff` | block | stream 2 | off | cpu | on | 4.4 | 5.9 |
| `ckpt-stream2-actoff` | block | stream 2 | on | gpu | off | 3.8 | 48.6 |
| `ckpt-stream2-actoff-gradoff` | block | stream 2 | on | gpu | on | 3.6 | 49.3 |
| `ckpt-stream2-actoff-optfused` | block | stream 2 | on | gpu fused | off | 3.8 | 48.6 |
| `ckpt-stream2-actoff-optfused-gradoff` | block | stream 2 | on | gpu fused | on | 3.6 | 49.3 |
| `ckpt-stream2-actoff-optstream` | block | stream 2 | on | stream fused | off | 3.5 | 13.3 |
| `ckpt-stream2-actoff-optstream-gradoff` | block | stream 2 | on | stream fused | on | 3.3 | 14.0 |
| `ckpt-stream2-actoff-optcpu` | block | stream 2 | on | cpu | off | 3.5 | 4.0 |
| `ckpt-stream2-actoff-optcpu-gradoff` | block | stream 2 | on | cpu | on | 3.3 | 4.8 |
| `subckpt-resident` | sub-block | resident | off | gpu | off | 14.7 | 85.7 |
| `subckpt-resident-gradoff` | sub-block | resident | off | gpu | on | 15.1 | 68.8 |
| `subckpt-resident-optfused` | sub-block | resident | off | gpu fused | off | 14.7 | 65.7 |
| `subckpt-resident-optfused-gradoff` | sub-block | resident | off | gpu fused | on | 15.1 | 68.8 |
| `subckpt-resident-optstream` | sub-block | resident | off | stream fused | off | 14.9 | 28.5 |
| `subckpt-resident-optstream-gradoff` | sub-block | resident | off | stream fused | on | 14.8 | 29.2 |
| `subckpt-resident-optcpu` | sub-block | resident | off | cpu | off | 14.9 | 46.4 |
| `subckpt-resident-optcpu-gradoff` | sub-block | resident | off | cpu | on | 14.8 | 26.4 |
| `subckpt-resident-actoff` | sub-block | resident | on | gpu | off | 13.0 | 86.6 |
| `subckpt-resident-actoff-gradoff` | sub-block | resident | on | gpu | on | 12.8 | 66.5 |
| `subckpt-resident-actoff-optfused` | sub-block | resident | on | gpu fused | off | 13.0 | 65.8 |
| `subckpt-resident-actoff-optfused-gradoff` | sub-block | resident | on | gpu fused | on | 12.8 | 66.5 |
| `subckpt-resident-actoff-optstream` | sub-block | resident | on | stream fused | off | 12.7 | 26.3 |
| `subckpt-resident-actoff-optstream-gradoff` | sub-block | resident | on | stream fused | on | 12.5 | 27.0 |
| `subckpt-resident-actoff-optcpu` | sub-block | resident | on | cpu | off | 12.7 | 44.2 |
| `subckpt-resident-actoff-optcpu-gradoff` | sub-block | resident | on | cpu | on | 12.5 | 24.1 |
| `subckpt-stream16` | sub-block | stream 16 | off | gpu | off | 9.1 | 53.9 |
| `subckpt-stream16-gradoff` | sub-block | stream 16 | off | gpu | on | 9.0 | 54.6 |
| `subckpt-stream16-optfused` | sub-block | stream 16 | off | gpu fused | off | 9.1 | 53.9 |
| `subckpt-stream16-optfused-gradoff` | sub-block | stream 16 | off | gpu fused | on | 9.0 | 54.6 |
| `subckpt-stream16-optstream` | sub-block | stream 16 | off | stream fused | off | 8.8 | 18.6 |
| `subckpt-stream16-optstream-gradoff` | sub-block | stream 16 | off | stream fused | on | 8.6 | 19.3 |
| `subckpt-stream16-optcpu` | sub-block | stream 16 | off | cpu | off | 8.8 | 9.4 |
| `subckpt-stream16-optcpu-gradoff` | sub-block | stream 16 | off | cpu | on | 8.6 | 10.1 |
| `subckpt-stream16-actoff` | sub-block | stream 16 | on | gpu | off | 6.9 | 51.7 |
| `subckpt-stream16-actoff-gradoff` | sub-block | stream 16 | on | gpu | on | 6.7 | 52.4 |
| `subckpt-stream16-actoff-optfused` | sub-block | stream 16 | on | gpu fused | off | 6.9 | 51.7 |
| `subckpt-stream16-actoff-optfused-gradoff` | sub-block | stream 16 | on | gpu fused | on | 6.7 | 52.4 |
| `subckpt-stream16-actoff-optstream` | sub-block | stream 16 | on | stream fused | off | 6.5 | 16.3 |
| `subckpt-stream16-actoff-optstream-gradoff` | sub-block | stream 16 | on | stream fused | on | 6.4 | 17.1 |
| `subckpt-stream16-actoff-optcpu` | sub-block | stream 16 | on | cpu | off | 6.5 | 7.1 |
| `subckpt-stream16-actoff-optcpu-gradoff` | sub-block | stream 16 | on | cpu | on | 6.4 | 7.8 |
| `subckpt-stream8` | sub-block | stream 8 | off | gpu | off | 7.2 | 52.0 |
| `subckpt-stream8-gradoff` | sub-block | stream 8 | off | gpu | on | 7.1 | 52.7 |
| `subckpt-stream8-optfused` | sub-block | stream 8 | off | gpu fused | off | 7.2 | 52.0 |
| `subckpt-stream8-optfused-gradoff` | sub-block | stream 8 | off | gpu fused | on | 7.1 | 52.7 |
| `subckpt-stream8-optstream` | sub-block | stream 8 | off | stream fused | off | 6.9 | 16.7 |
| `subckpt-stream8-optstream-gradoff` | sub-block | stream 8 | off | stream fused | on | 6.7 | 17.4 |
| `subckpt-stream8-optcpu` | sub-block | stream 8 | off | cpu | off | 6.9 | 7.5 |
| `subckpt-stream8-optcpu-gradoff` | sub-block | stream 8 | off | cpu | on | 6.7 | 8.2 |
| `subckpt-stream8-actoff` | sub-block | stream 8 | on | gpu | off | 5.0 | 49.8 |
| `subckpt-stream8-actoff-gradoff` | sub-block | stream 8 | on | gpu | on | 4.8 | 50.5 |
| `subckpt-stream8-actoff-optfused` | sub-block | stream 8 | on | gpu fused | off | 5.0 | 49.8 |
| `subckpt-stream8-actoff-optfused-gradoff` | sub-block | stream 8 | on | gpu fused | on | 4.8 | 50.5 |
| `subckpt-stream8-actoff-optstream` | sub-block | stream 8 | on | stream fused | off | 4.6 | 14.5 |
| `subckpt-stream8-actoff-optstream-gradoff` | sub-block | stream 8 | on | stream fused | on | 4.5 | 15.2 |
| `subckpt-stream8-actoff-optcpu` | sub-block | stream 8 | on | cpu | off | 4.6 | 5.2 |
| `subckpt-stream8-actoff-optcpu-gradoff` | sub-block | stream 8 | on | cpu | on | 4.5 | 5.9 |
| `subckpt-stream4` | sub-block | stream 4 | off | gpu | off | 6.3 | 51.1 |
| `subckpt-stream4-gradoff` | sub-block | stream 4 | off | gpu | on | 6.1 | 51.8 |
| `subckpt-stream4-optfused` | sub-block | stream 4 | off | gpu fused | off | 6.3 | 51.1 |
| `subckpt-stream4-optfused-gradoff` | sub-block | stream 4 | off | gpu fused | on | 6.1 | 51.8 |
| `subckpt-stream4-optstream` | sub-block | stream 4 | off | stream fused | off | 6.0 | 15.8 |
| `subckpt-stream4-optstream-gradoff` | sub-block | stream 4 | off | stream fused | on | 5.8 | 16.5 |
| `subckpt-stream4-optcpu` | sub-block | stream 4 | off | cpu | off | 6.0 | 6.5 |
| `subckpt-stream4-optcpu-gradoff` | sub-block | stream 4 | off | cpu | on | 5.8 | 7.3 |
| `subckpt-stream4-actoff` | sub-block | stream 4 | on | gpu | off | 4.0 | 48.8 |
| `subckpt-stream4-actoff-gradoff` | sub-block | stream 4 | on | gpu | on | 3.9 | 49.5 |
| `subckpt-stream4-actoff-optfused` | sub-block | stream 4 | on | gpu fused | off | 4.0 | 48.8 |
| `subckpt-stream4-actoff-optfused-gradoff` | sub-block | stream 4 | on | gpu fused | on | 3.9 | 49.5 |
| `subckpt-stream4-actoff-optstream` | sub-block | stream 4 | on | stream fused | off | 3.7 | 13.5 |
| `subckpt-stream4-actoff-optstream-gradoff` | sub-block | stream 4 | on | stream fused | on | 3.5 | 14.2 |
| `subckpt-stream4-actoff-optcpu` | sub-block | stream 4 | on | cpu | off | 3.7 | 4.3 |
| `subckpt-stream4-actoff-optcpu-gradoff` | sub-block | stream 4 | on | cpu | on | 3.5 | 5.0 |
| `subckpt-stream2` | sub-block | stream 2 | off | gpu | off | 5.8 | 50.6 |
| `subckpt-stream2-gradoff` | sub-block | stream 2 | off | gpu | on | 5.6 | 51.3 |
| `subckpt-stream2-optfused` | sub-block | stream 2 | off | gpu fused | off | 5.8 | 50.6 |
| `subckpt-stream2-optfused-gradoff` | sub-block | stream 2 | off | gpu fused | on | 5.6 | 51.3 |
| `subckpt-stream2-optstream` | sub-block | stream 2 | off | stream fused | off | 5.5 | 15.3 |
| `subckpt-stream2-optstream-gradoff` | sub-block | stream 2 | off | stream fused | on | 5.3 | 16.0 |
| `subckpt-stream2-optcpu` | sub-block | stream 2 | off | cpu | off | 5.5 | 6.1 |
| `subckpt-stream2-optcpu-gradoff` | sub-block | stream 2 | off | cpu | on | 5.3 | 6.8 |
| `subckpt-stream2-actoff` | sub-block | stream 2 | on | gpu | off | 3.6 | 48.4 |
| `subckpt-stream2-actoff-gradoff` | sub-block | stream 2 | on | gpu | on | 3.4 | 49.1 |
| `subckpt-stream2-actoff-optfused` | sub-block | stream 2 | on | gpu fused | off | 3.6 | 48.4 |
| `subckpt-stream2-actoff-optfused-gradoff` | sub-block | stream 2 | on | gpu fused | on | 3.4 | 49.1 |
| `subckpt-stream2-actoff-optstream` | sub-block | stream 2 | on | stream fused | off | 3.2 | 13.0 |
| `subckpt-stream2-actoff-optstream-gradoff` | sub-block | stream 2 | on | stream fused | on | 3.1 | 13.7 |
| `subckpt-stream2-actoff-optcpu` | sub-block | stream 2 | on | cpu | off | 3.2 | 3.8 |
| `subckpt-stream2-actoff-optcpu-gradoff` | sub-block | stream 2 | on | cpu | on | 3.1 | 4.5 |
<!-- catalog:end -->
