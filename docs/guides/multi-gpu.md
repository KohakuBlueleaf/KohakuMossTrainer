# Multi-GPU training

Set `PARALLEL` in the train config; the launcher spawns one process per entry of
`DEVICES` (see [distributed launch](../internals/distributed-windows.md)).

Multi-GPU training is Linux-only by design: on Windows, knob validation rejects any
`PARALLEL` other than `"none"` and more than one entry in `DEVICES`.

| `PARALLEL` | engine | when |
| --- | --- | --- |
| `ddp` | `GradReducer`: bucketed SUM all-reduce of the trainable parameters, overlapped with backward | LoRA / LyCORIS: only adapter gradients cross the wire |
| `fsdp` | FSDP2 `fully_shard` per joint block, then the root; adapters are sharded with their block | full finetune that does not fit one card |
| `pipeline` | `torch.distributed.pipelining` stages over contiguous block ranges | cards without a fast interconnect |
| `threaded` | one process, stages on different devices | mixed GPUs (e.g. 3090 + 2080 Ti) |

## Loss normalisation

The loss of every rank is divided by the global token count of the step, so gradients
are summed across ranks (`GradReducer` sums; `fsdp.shard` sets the FSDP2 divide factor
to 1) and the logged loss is the sum of the ranks' shares. Under FSDP2 with GRAD_ACC,
gradients are reduce-scattered on the last micro-batch only
(`set_requires_gradient_sync`).

## Supported memory plans

Knob validation resolves the MemoryPlan and rejects every combination outside this
table with one message each:

| `PARALLEL` | stream | optim_state | base_precision | act_offload | fused_step / grad_offload |
| --- | --- | --- | --- | --- | --- |
| `none` | any | any | any | any | any |
| `ddp` | `off`, `ring` | any | any | any | no |
| `fsdp` | `off` | `gpu` | `compute` | no | no |
| `pipeline`, `threaded` | `off` | `gpu` | `compute` | no | no |

- `ddp` + `swap`: the streamed blocks' host masters receive no all-reduce.
- `fsdp` needs `DIST_BACKEND='nccl'`: FSDP2's gradient reduce-scatter uses
  `ReduceOp.PREMUL_SUM`, which gloo does not implement.
- `fsdp` runs no sampling during training (`SAMPLE_EVERY=0`): the sample forward runs
  on rank 0 only and cannot all-gather the sharded blocks. Validation runs on every
  rank. `checkpoint='sub-block'` is rejected under `fsdp` (the block parts bypass the
  FSDP2 unshard on the block's `__call__`).
- FSDP2 checkpoints hold full tensors: every rank gathers the trainable / EMA tensors
  and the optimiser state (`get_optimizer_state_dict(full_state_dict=True)`), rank 0
  writes; resume re-shards them. The final export runs with every unit unsharded.

## Pipeline memory

Each rank builds the whole model on its device, keeps its stages' layers, and moves the
other layers (and top-level modules with no local parameter) to host memory.

## Pipeline schedules

`PIPELINE_SCHEDULE`: `gpipe`, `1f1b`, `interleaved-1f1b`, `looped-bfs`,
`interleaved-zb`, `zbv`, `dualpipe-v` (the last two use a V placement: rank `r` holds
stages `r` and `2W-1-r`). Every rank draws the same step and builds the micro-batch
layouts itself; only `(x, modulation, head modulation, micro-batch index)` cross a stage
boundary. Stages are built with static input / output metadata, and each stage counts
its forwards (and the last stage its losses) on the host, so no micro-batch index is
read back from the device. Shapes are static: each micro-batch is padded to the pipe's token budget and
sample count with a zero-weight padding sample. `PipeShape.from_knobs` takes that shape
from the data knobs (`MAX_TOKENS` joint tokens, `MAX_SAMPLES` samples, `MICROBATCHES`)
and refuses knobs under which they are not hard bounds of the planner
(`BATCHING='tokens'` or `'tokens_ffd'`, `MAX_SAMPLES >= 1`, `TOKEN_BUDGET='exact'`,
`TEXT_CACHE='offline'`). Token drop (`OVERLONG='drop_tokens'`) passes each sample's kept
grid indices as `MicroBatch.img_keep`; the kept tokens keep their grid positions.

## Threaded pipeline

`ThreadedPipeline` places layer ranges on devices of one process. Kernel launches are
asynchronous per device, so stages overlap without threads; `schedule="async"` runs each
micro-batch's backward on a helper thread while the next forward is issued. Each stage
can bind its own op set (`opset_for(device)`), so a 3090 stage runs Triton kernels and a
2080 Ti stage runs TileLang kernels.

## Heterogeneous devices (library API)

The trainer splits stages evenly (`even_bounds`); the balancing functions below are a
library API for custom drivers, not wired to a knob.

`balance.measure_speed` times a dense matmul per device; `token_budgets` splits the
step's tokens across data-parallel ranks proportionally, and `split_layers` finds the
contiguous stage split minimising the slowest stage's cost / speed (optionally under
per-stage memory caps). A mixed 3090 + 2080 Ti job runs fp16, since the 2080 Ti has no
bf16.
