# Distributed launch

`kohakumoss.parallel.launch.spawn(fn, plan, *args)` starts one process per device with
`torch.multiprocessing.spawn` and initialises the process group itself.

Multi-GPU training (`PARALLEL` other than `"none"`) is Linux-only by design and knob
validation rejects it on Windows. On Windows the launcher serves the CPU gloo
[self-checks](#self-checks).

## Knobs

| knob | values (default first) | effect |
| --- | --- | --- |
| `DIST_BACKEND` | `auto`, `nccl`, `gloo` | `auto` is nccl on Linux CUDA, xccl on XPU when available (only `auto` selects it), gloo otherwise (Windows, CPU); gloo all-reduces CUDA tensors through host memory |
| `DIST_BIND` | `lazy`, `device` | process-group device binding ([below](#device-binding)) |
| `DIST_TIMEOUT` | 600 s | every collective's and the store's wait |
| `DIST_GATE_TIMEOUT` | 7200 s (`launch.RANK0_WAIT_S`) | a store gate's wait for rank-0-only or per-rank startup work ([below](#timeouts-and-gates)) |

## Process group

- **Store**: `init="tcp"` builds a `TCPStore` explicitly. The parent process owns the
  store master on an OS-assigned port and the workers connect as clients, so the port is
  bound before any worker uses it.
  - Windows wheels have no libuv, so the store uses `use_libuv=False` there and
    `USE_LIBUV=0` is set in every worker.
  - On Windows the store address is `localhost`: the `127.0.0.1` literal fails inside
    the non-libuv store with a formatting error from torch.
- **File init**: `init="file"` uses a `file://` rendezvous in a fresh temp directory,
  removed when the job ends (a normal exit, a failing rank, or SIGTERM to the
  launcher, which then terminates the rank processes). The tcp init creates no
  directory.
- **Workers**: `fn` must be a module-level function, because spawned processes import it.

### Device binding

`DIST_BIND` (`ParallelPlan.bind`):

| value | binding |
| --- | --- |
| `"lazy"` (default) | the process group initialises without `device_id`, so NCCL builds one 2-rank communicator per point-to-point pair on first use instead of routing every pipeline send / receive through one shared communicator |
| `"device"` | nccl / xccl bind to the rank's device at init (`device_id`): the communicator is created there and a `dist.barrier()` runs on that device |

The pipeline stays lazy under either value; gloo is never bound.

## Timeouts and gates

`DIST_TIMEOUT` (`ParallelPlan.timeout_s`, default 600 s): every collective and the store
wait that long for every rank, so a hung rank stops the job within it.

Startup work that is not the same length on every rank meets on the store instead of in
a collective, and waits up to `DIST_GATE_TIMEOUT` (default 2 h):

- **Rank-0 gates.** Rank 0's one-time work before the step loop is not waited for in a
  collective: the other ranks wait on the store (`launch.wait_for_rank0`) until rank 0
  sets its key (`release_ranks`), then meet in the collective that follows. This covers
  the catalog scan in `writer_first`, the kernel prewarm's measurement, and
  `MEMORY_PROFILE='auto'`, which is planned on rank 0 and handed to the other ranks
  (`launch.rank0_value`).
- **Store barriers.** Startup work whose length differs on every rank ends in a store
  barrier (`launch.ranks_meet` / `startup_barrier`): each rank adds one to
  `kohakumoss/<gate>/arrived`, the last one sets `kohakumoss/<gate>`, and every rank
  waits for that key. The gates:

  | gate | where |
  | --- | --- |
  | `data` | after each rank's loader build, before the model build |
  | `prewarmed` | after each rank's prewarm, before the stage split and the gradient-sync hooks |
  | `checkpoint` | around rank 0's checkpoint write |
  | `sampled` | after rank 0's sample grid |

Between the `data` and `prewarmed` gates the build's collectives (the shared host region
of DDP `swap` blocks) wait `DIST_TIMEOUT`, so a weight load whose length differs between
ranks by more than that needs a larger `DIST_TIMEOUT`.

## Failing rank

A rank whose `fn` raises prints its traceback, kills its own multiprocessing children and
leaves with `os._exit(1)`, without the process-group teardown; `spawn` then sees the
exit and terminates the other ranks.

- **No teardown.** On nccl the teardown blocks while the failing rank has a send or
  receive posted (a pipeline stage mid-schedule), so tearing down first would leave the
  failing rank alive and its peers waiting in a receive for `DIST_TIMEOUT`.
- **Children killed.** `os._exit` skips the DataLoader shutdown: forkserver-started
  loader workers watch the forkserver, not the rank, and each holds the forkserver's
  keep-alive pipe, so worker and forkserver would otherwise both outlive the rank.
- `tests/parallel/test_launch_failfast.py` checks the exit (gloo, and nccl on two GPUs,
  where the teardown variant hangs) and the children (gloo, Linux).

## Self-checks

`kogine run scripts/dist/check_comm.py --set DEVICES=0,1` runs, in spawned workers and
without a model:

- the bucketed gradient reducer against the sum of every rank's and micro-batch's
  gradient (small buckets, two micro-batches; it also checks that the gradients are
  views of the bucket buffers);
- sweeps of all-reduce, all-gather / reduce-scatter (FSDP2's collectives, bf16) and the
  point-to-point hop around the rank ring (the pipeline's link).

The engines themselves are checked on the real model in
`tests/integration/test_multi_gpu_real.py`.

Measured all-reduce bandwidth (algorithm GB/s, 64 MB):

| pair | link | GB/s |
| --- | --- | --- |
| 2x V100 16 GB | NVLink | 90.6 |
| V100 32 GB + V100 16 GB | PCIe, different root complexes | 1.07 |
