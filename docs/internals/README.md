# Internals

How the parts work inside: for contributors, and for users who want to know what a knob
does on the device. Start from the [concepts](../concepts/README.md) and
[guides](../guides/README.md); come here for the design and its cost.

| area | page | content |
| --- | --- | --- |
| kernels | [kernel-dev.md](kernel-dev.md) | writing and registering a kernel backend |
| | [tuning.md](tuning.md) | planner, timed shortlist, tuning tables, prewarm, tuning windows |
| | [fp16.md](fp16.md) | precision rules of the block ops, error statistics, kernel or compile per problem |
| | [quant-kernels.md](quant-kernels.md) | int8 / fp8 weight-only frozen-base GEMMs; fp8 / MXFP8 compute (formats, recipes, kernels, selection) |
| memory | [offload.md](offload.md) | activation offload, weight streaming, gradient offload, CPU optimiser, text-encoder placement |
| | [sampling-memory.md](sampling-memory.md) | what a generate call holds on the device per phase (text encode, sampling, decode), and the placement that keeps sampling at or below training |
| | [checkpoints.md](checkpoints.md) | streaming checkpoint writes, memory and disk cost |
| | [reference-cycles.md](reference-cycles.md) | no reference cycle through a tensor holder: class-function slots, weak hooks and callbacks |
| text encoder | [text-encoder.md](text-encoder.md) | text-encoder ops and kernels, batched / packed encode, TextEncoderPlan, encode-ahead, quantization |
| encode feed | [encode-feed.md](encode-feed.md) | ENCODE_DEVICES: encoder processes on their own cards, fed by the main process, streaming ready batches to the training processes |
| adapters | [adapter-kinds.md](adapter-kinds.md) | the per-algorithm adapter records |
| | [adapter-binding.md](adapter-binding.md) | bound adapter forwards, kernel ops, autotuned ops, forward wrappers |
| multi-GPU | [distributed-launch.md](distributed-launch.md) | process launch, store and backends (multi-GPU training is Linux-only; the CPU gloo self-checks also run on Windows) |
