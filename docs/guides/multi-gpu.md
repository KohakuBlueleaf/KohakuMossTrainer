# Multi-GPU training

Set `PARALLEL` in the train config; the launcher spawns one process per entry of
`DEVICES` (see [distributed on Windows](../internals/distributed-windows.md)).

| `PARALLEL` | engine | when |
| --- | --- | --- |
| `ddp` | `GradReducer`: bucketed SUM all-reduce of the trainable parameters, overlapped with backward | LoRA / LyCORIS: only adapter gradients cross the wire |
| `fsdp` | FSDP2 `fully_shard` per joint block, then the root; adapters are sharded with their block | full finetune that does not fit one card |
| `pipeline` | `torch.distributed.pipelining` stages over contiguous block ranges | cards without a fast interconnect, Linux |
| `threaded` | one process, stages on different devices | Windows, mixed GPUs (e.g. 3090 + 2080 Ti) |

## Loss normalisation

The loss of every rank is divided by the global token count of the step, so gradients
are summed across ranks (`GradReducer` sums; `fsdp.shard` sets the FSDP2 divide factor
to 1).

## Pipeline schedules

`PIPELINE_SCHEDULE`: `gpipe`, `1f1b`, `interleaved-1f1b`, `looped-bfs`,
`interleaved-zb`, `zbv`, `dualpipe-v` (the last two use a V placement: rank `r` holds
stages `r` and `2W-1-r`). Every rank draws the same step and builds the micro-batch
layouts itself; only `(x, modulation, head modulation, micro-batch index)` cross a stage
boundary. Shapes are static: each micro-batch is padded to the pipe's token budget and
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

## Heterogeneous devices

`balance.measure_speed` times a dense matmul per device; `token_budgets` splits the
step's tokens across data-parallel ranks proportionally, and `split_layers` finds the
contiguous stage split minimising the slowest stage's cost / speed (optionally under
per-stage memory caps). A mixed 3090 + 2080 Ti job runs fp16, since the 2080 Ti has no
bf16.
