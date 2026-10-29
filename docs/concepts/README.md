# Concepts

How each part of KohakuMoss works. Read [model.md](model.md) and
[seqinfo.md](seqinfo.md) for the batch layout, then the page of the part you configure.

| page | content |
| --- | --- |
| [model.md](model.md) | the MOSS-Image transformer as implemented here: shape, calling convention, checkpoints, ops and runners |
| [seqinfo.md](seqinfo.md) | packed batches: sample boundaries, joint order, positions, token drop |
| [massive-channels.md](massive-channels.md) | the base model's massive residual channels: what they do, why shrinking them breaks the model, the annealed heal, the stream rescale and its effect on training |
| [numerics.md](numerics.md) | dtypes, conditioning precision, PyTorch's precision defaults, `FP16_FIX` modes, the fp16-fix checkpoint, the residual fold |
| [adapters.md](adapters.md) | the adapter knobs, LyCORIS algorithms, targeting presets, multiplier, export, inference loading, binding through the KohakuMoss applier, quantized bases |
| [training.md](training.md) | build order, kernel prewarm and tuning windows, objective, timestep samplers, optimisers, the optimiser step, precision, pipeline layouts, EMA, drivers, checkpoints, callbacks, shutdown |
| [profiles.md](profiles.md) | choosing a memory profile, `auto`, encoder placement, `STEP_RELEASE`, the memory and precision knobs, the catalog, estimated peak VRAM |
| [data.md](data.md) | dataset entries, sources, captions, resolution and batching, caches, the catalog cache, HostBatch, loader, file formats |
