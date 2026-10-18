# Internals

| page | content |
| --- | --- |
| [kernel-dev.md](kernel-dev.md) | writing and registering a kernel backend |
| [tuning.md](tuning.md) | planner, timed shortlist, tuning tables, prewarm |
| [fp16.md](fp16.md) | precision rules of the block ops |
| [quant-kernels.md](quant-kernels.md) | int8 / fp8 weight-only frozen-base GEMMs; fp8 / MXFP8 compute (formats, recipes, kernels, selection) |
| [offload.md](offload.md) | activation offload, weight streaming, gradient offload, CPU optimiser, text-encoder placement |
| [sampling-memory.md](sampling-memory.md) | what a generate call holds on the device per phase (text encode, sampling, decode), and the placement that keeps sampling at or below training |
| [text-encoder.md](text-encoder.md) | text-encoder ops and kernels, batched / packed encode, TextEncoderPlan, encode-ahead, quantization |
| [distributed-launch.md](distributed-launch.md) | process launch, store and backends (multi-GPU training is Linux-only; the CPU gloo self-checks also run on Windows) |
| [adapter-kinds.md](adapter-kinds.md) | the per-algorithm adapter records |
| [adapter-binding.md](adapter-binding.md) | bound adapter forwards, kernel ops, autotuned ops, forward wrappers |
| [checkpoints.md](checkpoints.md) | streaming checkpoint writes, memory and disk cost |
