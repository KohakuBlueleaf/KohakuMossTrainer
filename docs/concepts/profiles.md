# Memory profiles

`MEMORY_PROFILE` names a point on the speed / VRAM trade-off; it resolves once, at
build time, into a concrete `MemoryPlan`. Every field of the plan can be overridden with
`MEMORY_OVERRIDES`, and the profile is only a starting point for them.

| field | values | effect |
| --- | --- | --- |
| `checkpoint` | `none`, `block`, `sub-block`, `selective`, `partial:<fraction>` | which activations are recomputed in backward |
| `act_offload` | bool | saved activations go to pinned host memory, prefetched one block ahead |
| `base_precision` | `compute`, `bf16`, `fp16`, `fp32`, `fp8`, `int8`, `quant` (fp8 where the arch has fp8 GEMMs, else int8) | storage of the frozen base weights |
| `stream` | `off`, `ring` (frozen, host → device only), `swap` (trainable) | block weights live on the host and stream in |
| `stream_resident` | int | blocks kept on the device while streaming |
| `pinned` | bool | pinned vs staged host buffers |
| `optim_state` | `gpu`, `8bit`, `cpu` | optimiser state placement |
| `fused_step` | bool | per-parameter optimiser step inside backward |
| `kernels`, `compile` | | kernel backend choice and compile granularity |

Profiles, from fastest to leanest: `speed`, `balanced`, `low`, `minimal`. `auto` walks
them in that order and takes the first whose estimate (and a one-step probe) fits the
device. `kohakumoss-doctor --memory` prints every profile resolved for the local device.
