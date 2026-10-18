# Concepts

| page | content |
| --- | --- |
| [model.md](model.md) | the MOSS-Image transformer as implemented here: blocks, modulation, RoPE, calling convention, ops and runners, loaders |
| [seqinfo.md](seqinfo.md) | packed batches: sample boundaries, joint order, positions, token drop |
| [numerics.md](numerics.md) | dtypes, conditioning precision, PyTorch's precision defaults, the residual fold, `FP16_FIX` modes, the fp16-fix checkpoint |
| [adapters.md](adapters.md) | LyCORIS algorithms through the KohakuMoss applier, presets, binding, quantized bases, export, inference loading |
| [training.md](training.md) | build order, kernel prewarm, objective, timestep samplers, optimisers, precision, EMA, drivers, checkpoints, callbacks |
| [profiles.md](profiles.md) | memory profiles, the memory and precision knobs, the catalog, estimated peak VRAM |
| [data.md](data.md) | dataset format, sources, catalog cache, captions, resolution, batching, loader, caches |
