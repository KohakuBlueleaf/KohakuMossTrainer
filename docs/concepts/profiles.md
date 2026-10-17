# Memory profiles

`MEMORY_PROFILE` names a point on the speed / VRAM trade-off; it resolves once, at
build time, into a concrete `MemoryPlan`. Every field of the plan can be overridden with
`MEMORY_OVERRIDES`, and the profile is only a starting point for them. Knob validation
resolves the plan for the local device, so every invalid combination is reported in the
one startup error, before any weight is loaded.

| field | values | effect |
| --- | --- | --- |
| `checkpoint` | `none`, `block`, `sub-block`, `selective`, `partial:<fraction>` | which activations are recomputed in backward |
| `act_offload` | bool | saved activations go to pinned host memory, prefetched one block ahead |
| `base_precision` | `compute`, `fp8`, `int8`, `quant` (fp8 where the arch has fp8 GEMMs, else int8) | storage of the frozen base weights (`compute`: the PRECISION dtype) |
| `stream` | `off`, `ring` (frozen, host to device only), `swap` (trainable) | block weights live on the host and stream in |
| `stream_resident` | int | blocks kept on the device while streaming |
| `pinned` | bool | pinned vs staged host buffers (every host-buffer owner: streams, offloads, host optimiser states) |
| `optim_state` | `gpu`, `stream` (host states, update on the GPU), `8bit`, `cpu` (host update) | AdamW state placement (`OPTIMIZER='adamw'`; other optimisers need `gpu`) |
| `grad_offload` | bool | gradients summed in host memory across GRAD_ACC |
| `fused_step` | bool | per-parameter optimiser step inside backward (`swap` with a GPU `optim_state` always) |
| `compile`, `kernels` | | set by the `COMPILE` and `KERNELS` knobs, not by profiles or overrides |

Profiles, from fastest to leanest: `speed`, `balanced`, `low`, `minimal`. `auto` walks
them in that order and takes the first runnable one (valid with the other knobs) whose
estimate fits the device's free memory; the planner's per-profile rows are kept in the
run's memory reports. `kohakumoss-doctor --memory` prints every profile resolved for the
local device.
