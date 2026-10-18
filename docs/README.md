# KohakuMoss docs

| section | content |
| --- | --- |
| [guides/](guides/README.md) | install (Windows / Linux / Intel Arc / Colab), configs and presets, host RAM, datasets and caches, first LoRA, full finetune, inference and comparison grids, export to ComfyUI, logging, multi-GPU, the script catalogue |
| [concepts/](concepts/README.md) | the model, SeqInfo and packing, numerics, PyTorch precision defaults and the fp16 fix, adapters, training, memory profiles, the data pipeline |
| [internals/](internals/README.md) | kernel development, tuning, fp16 block ops, quantized and fp8 kernels, offload engine, sampling memory, text encoder, distributed launch, adapter records and binding, checkpoint writes |
| [performance/](performance/README.md) | measured tables per GPU: attention, block ops, VAE, the real-GPU bench matrix |

New users: [install](guides/install.md), then [your first LoRA](guides/first-lora.md);
for a run from a few options, [the example scripts](../scripts/examples/README.md). The
architecture charts (packages, train step, memory placement, kernel selection,
multi-GPU) are in the [README](../README.md#architecture).
