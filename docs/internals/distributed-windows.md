# Distributed launch on Windows and Linux

`kohakumoss.parallel.launch.spawn(fn, plan, *args)` starts one process per device with
`torch.multiprocessing.spawn` and initialises the process group itself; there is no
`torchrun` dependency.

- **Backend**: `auto` is nccl on Linux CUDA, xccl on XPU when available, gloo otherwise
  (Windows, CPU). gloo all-reduces CUDA tensors through host memory.
- **Store**: `init="tcp"` builds a `TCPStore` explicitly. The parent process owns the
  store master on an OS-assigned port and the workers connect as clients, so no port is
  picked and then bound later by someone else. Windows wheels have no libuv,
  so the store uses `use_libuv=False` there and `USE_LIBUV=0` is set in every worker.
  On Windows the store address is `localhost`: the `127.0.0.1` literal fails inside the
  non-libuv store with a formatting error from torch.
- **File init**: `init="file"` uses a `file://` rendezvous in a fresh temp directory.
- **Workers**: `fn` must be a module-level function, because spawned processes import it.

## Self-checks

`scripts/dist/check_comm.py --devices 0,1` runs, in spawned workers:
the bucketed gradient reducer against the sum of every rank's gradients, a 1F1B pipeline
against the single-process reference, FSDP2 against the unsharded reference (nccl), and
an all-reduce bandwidth sweep. `tests/parallel/test_distributed.py` runs the same checks
under pytest (gloo on CPU everywhere, nccl with two GPUs).

Measured all-reduce bandwidth (algorithm GB/s, 64 MB):

| pair | link | GB/s |
| --- | --- | --- |
| 2x V100 16 GB | NVLink | 90.6 |
| V100 32 GB + V100 16 GB | PCIe, different root complexes | 1.07 |
