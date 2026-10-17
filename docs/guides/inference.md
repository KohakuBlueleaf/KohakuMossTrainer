# Inference

`kohakumoss.inference.MossPipeline` generates images with the official MOSS sampler:
Euler over the shift-5 sigma grid snapped to the 1000-step training grid, CFG 3, and the
negative branch skipping block 22.

## Python

```python
import torch
from kohakumoss.inference import MossPipeline

pipe = MossPipeline.from_pretrained(
    "path/to/MOSS-Image",       # root with base/ (and sft/), or one pipeline dir
    variant="base",
    dtype=torch.bfloat16,
    adapters=[("my_lora.safetensors", 0.8)],
    adapter_mode="merge",        # or "bypass"
    residency="gpu",             # "offload" | "te_once"
)
out = pipe.generate(
    ["a cat on a windowsill", "a lighthouse at dusk"],
    sizes=[(1024, 1024), (832, 1216)],
    seeds=[0, 1],
    steps=50, cfg=3.0, shift=5.0,
)
out.images[0].save("cat.png")
```

- **Packed CFG** (`cfg_mode="packed"`, default): the conditional and unconditional samples
  of a batch run in one packed forward; `two_pass` runs them as two forwards with half
  the tokens each.
- **Per-sample determinism**: sample i's noise and math depend only on its prompt, size
  and seed, not on which other samples share its batch — up to kernel rounding. On CPU
  in fp64 batch composition changes nothing measurable; on CUDA in bf16 the packed token
  count changes GEMM / attention problem shapes and therefore rounding, about 1e-3
  relative after 4 steps on the tiny model (`tests/inference/test_determinism_cuda.py`).
  Bitwise-identical samples need the same batch composition.
- **Weights root**: `variant` names a subfolder of `path` that must exist. Each
  component comes from `path/variant` when it holds it, else from `path`, else from
  `path/base` (the SFT folder carries only the transformer); the folders used are in
  `out.settings["sources"]`.
- **Embeddings in**: `prompt_embeds` skips encoding the prompts; `negative_prompts`
  strings are still encoded unless `negative_embeds` is given.
- **Linear algebra**: an owned CUDA pipeline selects cuSOLVER (OFT's batched Cayley
  inverse) for the duration of each `generate` call and restores the previous backend
  after it.
- **Residency**: `gpu` keeps the text encoder and VAE on the device; `offload` parks both
  on CPU between uses; `te_once` encodes every prompt in the first call and frees the
  text encoder.
- **Text device**: `text_device="cpu"` (`--text-device cpu` in `scripts/sample`) runs the
  text encoder on the CPU and moves only the embeddings to the compute device, as ComfyUI
  does on small cards. The 9B encoder then needs ~18 GB of host RAM in bf16 (`--text-dtype
  fp32`: ~36 GB) and no VRAM; encoding speed is the CPU's. Training takes the same
  choice as `TEXT_DEVICE = "cpu"`.
- **fp16** (GPUs without bf16): pass `dtype=torch.float16` and an `fp16_fix`
  (`"weights"`, `"online"` or `"fp32_residual"`, with `fp16_fix_kwargs={"c": 0.0625}`),
  or load an fp16-fix checkpoint as `transformer=...`; see `docs/concepts/numerics.md`.
- **Training-time samples**: `MossPipeline.from_components(backbone, text_encoder, vae)`
  shares the training model's modules (no reload, runs under `no_grad`).

## CLI

```bash
python scripts/sample/sample.py --weights path/to/MOSS-Image --variant base \
    --prompt "a cat" --prompt "a dog" --size 1024x1024 --seed 0 --seed 1 \
    --adapter my_lora.safetensors:0.8 --out out/samples
```

`--fp16-fix online --fp16-fix-kwargs '{"c": 0.0625}'` selects an fp16 fix.
`--prompts-file` takes a `.txt` (one prompt per line) or a `.json` list of strings or of
`{"prompt", "negative", "size": [H, W], "seed"}` objects. The output directory gets one
PNG per prompt and `metadata.json` with every setting.
