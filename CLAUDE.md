# KohakuMoss

A finetuning framework for MOSS-Image (OpenMOSS-Team/MOSS-Image-Base / -SFT): LyCORIS
algorithms and full finetune on text-image data at native resolutions, with per-arch
kernels, fp16 and bf16, speed/VRAM profiles, offload and multi-GPU, on Windows and Linux.
Sibling to KohakUwULLM (LLM training); same house conventions.

## Project Overview

The framework provides slots; you select or drop in what you need. An adapter algorithm,
a timestep sampler, a dataset source, a resolution planner, a kernel backend or a memory
profile is a registry entry or a dotted import path -- no core edits.

Four packages under `src/`:

| package | role | imports |
| --- | --- | --- |
| `kohakumoss_common` | logger, registry / build, dotted-path imports, shared no-op callables | neither torch nor a sibling |
| `kohakumoss_kernels` | MOSS-block ops, registry, constraints, backends (eager / triton / tilelang), tuning | `kohakumoss_common` only |
| `kohakumoss_data` | dataset format, sources, catalog, captions, resolution planning, loader, caches | `kohakumoss_common` only, never torch.distributed |
| `kohakumoss` | model, encoders, adapters, diffusion, training, memory, offload, parallel, quant, logging | every sibling |

Adapters come from the `lycoris-lora` package: its algorithms, save format and fused
kernels. KohakuMoss carries a custom applier (`kohakumoss.adapters`) that binds each
layer to the LyCORIS entry point once; adapter kernel work belongs in LyCORIS itself.

Targets: sm70 / sm75 (V100, 2080 Ti) through TileLang kernels; sm86 / sm89 / sm120 (3090,
4090, RTX 50xx) through Triton kernels with per-arch designs; Intel Arc (XPU) through
Triton-XPU and eager. Windows support is mandatory for sm75 / sm86 / sm89 / sm120.

## Code Conventions

### File Organization
- Source: `src/{kohakumoss,kohakumoss_kernels,kohakumoss_data,kohakumoss_common}/`
- Scripts: `scripts/{train,data,sample,export,probe,dist,bench}/`
- KohakuEngine configs: `configs/<domain>/<group>/<group>_<name>.py`
- Docs: `docs/{concepts,guides,internals,performance}/`, each with a `README.md` index
- Tests: `tests/<area>/`
- Max lines per file: 600 (hard max 1000). Split modules before they grow.
- Highly modularized -- one responsibility per module; nest, do not flatten.

### Coding style rules

**Code files carry only *what*. Every *why* and *how* belongs in `docs/`.**

1. **Docstring -- what a module / class / function is for**, and the caller-facing
   contract (arguments, units, dtype/layout, return). One to three lines.
2. **Comment -- what a piece of code is doing.** One line; two at the outside.
3. **NO rationale, derivation, measurement, history or memo comments in source.** That
   goes to `docs/`; the code gets at most `See docs/<file>.md`.
4. No editorial comments, none addressed to a future reader.

### Import Rules
1. No imports inside functions (except optional deps such as triton / tilelang /
   lightning, guarded at module level where possible).
2. Grouping order: built-in, third-party, first-party.
3. Within a group: `import` before `from`, shorter dotted paths first, then alphabetical.

### Python Style
- Target Python 3.10+. Modern hints: `list`, `dict`, `tuple`, `X | None`.
- Prefer `match-case` over nested `if-elif-else`.
- Python source and printed messages are ASCII-only (Windows consoles with a non-UTF-8
  code page fail on non-ASCII text).
- `black` and `ruff check` over every directory before commit.

### The architectural rules

**Select, don't dispatch.** Configuration resolves a concrete class or callable once, at
build time, via `kohakumoss_common.registry.build(choice, REGISTRY)` or
`kohakumoss_kernels.registry.resolve(...)`. The
result is held as a plain attribute and called directly. No per-step `if mode == ...`.

Selection is a cost judgement, not a ban on `if`. Weigh a branch by how often it runs
and how much it is duplicated: a branch inside a module that is instantiated hundreds of
times in the model (every adapted linear, every block, every kernel wrapper) runs
layers x calls times per step and is resolved at bind time into a distinct callable
(`functools.partial` over a specialised function, a table keyed at bind time) — a kind
`match`, "multiplier == 1", "has scalar / rescale / bias", an optional tensor that is
always or never present. A branch the train loop takes once per step, or a one-off in
build / export code, is negligible and stays a plain `if` when that reads better. A branch
the same decision repeats in many places is also a selection candidate, for clarity.
Train/eval differences in high-multiplicity modules re-bind on the mode change.

**The backbone is a pure function.** `MossBackbone(img, text, sigma, seq) -> v`. It knows
nothing about the loss; objectives live in `kohakumoss.diffusion.objective`.

**Packed is the layout.** Every tensor that flows through the model is a flat token axis;
`SeqInfo` carries the sample boundaries. Only attention reads the boundaries; every other
op is a last-dim op. Attention must never cross a sample boundary.

**The eager ops are the reference.** `kohakumoss_kernels.backends.eager` defines what every
op computes. Every other backend equals it within the precision bar, forward and backward.

### Numerics rules
- Reduce in fp32; token counts are int64; per-sample math (time embedding, modulation,
  head modulation) is fp32.
- **No fp64 anywhere in the trainer, the model, the kernels or inference.** fp64 exists only
  as the precision reference (tests, `kohakumoss_kernels.error_stats`, benches, the eager
  reference run on fp64 inputs), never an fp64 tensor or an fp64 register.
- **fp32 is for vector operations only.** Every matmul (linear layers incl. the KEEP_FP32
  conditioning modules, attention QK^T / PV, adapter products, the text encoder) runs on
  fp16 / bf16 inputs with fp32 accumulation, the GPU default: no fp32 / tf32 matmuls and no
  hi/lo-split multi-pass MMAs to buy extra precision. fp32 storage (masters, optimizer
  state) and fp32 element-wise / reduction math are fine. The precision bar compares against
  PyTorch at this same matmul precision.
- Precision: match PyTorch or beat it, with fp64 as the reference (the ruler). Every kernel
  is checked in fp16 and bf16, forward and backward: kernel and PyTorch are both measured
  against fp64 (cosine; abs and rel error, read together, as mean / median / min / max), and
  the kernel's stats must be no worse than PyTorch's. No fixed numeric threshold: a worse
  stat gets a written justification or a fix; tests fail only on explosion
  (`kohakumoss_kernels.error_stats` measures, `tests/_util/error_stats.py` decides). Precision is a threshold, never a ranking: among kernels
  that match PyTorch, the fastest is selected. Attention adds the huge-logit, padded-key and
  empty-row cases.
- A kernel is selected only where it is faster than PyTorch on that arch; where it loses
  to `torch.compile` of the eager composition (typical for FFN / matmul-heavy layers), the
  op uses the compiled path. The choice is measured and recorded, never guessed per call.
- Varlen attention must be faster than PyTorch SDPA (flash / mem-efficient). FA4-like
  designs follow KohakuFA.
- Precision is at least that of PyTorch / `torch.compile`, or better: kernels use what the
  hardware provides, 16-bit matmul inputs with fp32 accumulation; the vector work after a
  matmul runs on the fp32 accumulator (epilogues, LSE), and the result is stored in the
  output tensor's own dtype. PyTorch / eager code stays plain PyTorch: no wrapper forces a
  matmul's input or output dtype.
- fp16 training uses the residual-stream fold (`kohakumoss.model.numerics`), fp32
  accumulators, and the device-flag scaler.

### Testing
- **No fake model, ever**: no tiny, random, synthetic or stand-in model, text encoder or
  VAE. A test is one of two kinds:
  - **model-free**: a kernel's inputs -> outputs against the eager reference and fp64
    (`kohakumoss_kernels.error_stats`), or a pure function (planners, captions, knob and
    plan validation, catalog, schedules, report formatting, rule checks);
  - **whole integration on the real model** (`tests/integration/`, real weights from
    `KOHAKUMOSS_WEIGHTS`, the user's dataset from `KOHAKUMOSS_DATASET`): training runs,
    memory placement and offload, inference, grids, VAE / text-encoder placement.
- **Fast**: one fixed kernel config per op, tuning off (`tests/conftest.py` sets the tuner
  to "off"), a few representative shapes, no exhaustive knob matrices or tile sweeps. Every
  model-free test has a time budget (`tests/conftest.py`, `KOHAKUMOSS_TEST_BUDGET`); the
  model-free suite runs in well under a minute per area, apart from a first kernel
  compile.
- Only two kinds of run may take over a minute: first-time kernel tuning, and a real
  training / inference run that tests an integration. The modules a real run uses are
  bug-free before it starts; a real run that hits an internal module bug stops and the
  module goes back to review and fix.
- A check is shown to fail on a constructed broken variant (`tests/_util/negative.py`).
- Markers: `cpu, cuda, gpu, sm70, sm75, sm86, sm89, sm120, xpu, windows, multi_gpu,
  real_weights, slow, budget(seconds)`. The default run is the model-free suite;
  `-m real_weights` runs the integration.

### Audit loop (REQUIRED)
1. Implement every item of the goal in full; no command, test or job runs before that.
2. Review and audit the whole diff for clear bugs, integrity bugs and behaviour bugs, and
   its cost (host syncs, extra copies, saved tensors, peak memory).
3. Lint, then the model-free suite once.
4. Then the real-model integration once, on every GPU, everything to disk.
5. If a bug slipped past a check, fix the check first, then the bug.
6. Audit benchmarks the same way; an unaudited measurement is not evidence.

### Post-impl tasks
1. Verify the rules above, especially in-function imports and file length.
2. `black` and `ruff`.
3. Logically separated commits: sentence-case imperative subject, body with the measured
   problem then the change.

### Windows parity
Most users run Windows, so every single-GPU feature works on Windows. A slower Windows
implementation is acceptable (offload and other driver/OS-bound paths may use CPU staging
instead of pinned transfers); "does not work on Windows" is not, unless a written
justification in `docs/guides/install-windows.md` says why the feature does not matter
there. Multi-GPU training (DDP, FSDP2, pipeline, threaded) is Linux-only by design.
Every test that can run on Windows runs there.

### Public repository
Everything tracked by git is public. No machine names, hosts, users, local paths,
credentials or private dataset details in tracked files, commit messages or test data.
Local machine notes live in `CLAUDE.local.md`, plans in `plans/`, live state in `STATE.md`
(all gitignored).

## Development Setup

```bash
uv pip install -e ".[cuda,dev]"    # or ".[xpu,dev]"; extras: kernels, lightning, quant, parquet, wandb, bench, baselines
uv pip install -e scripts/bench/matrix    # the bench matrix dev package (kohakumoss-bench), not in the wheel
kogine run scripts/train/moss.py --config configs/moss/recipes/recipes_lora_24g_bf16.py
```

Never use `sys.path` hacks in scripts; always import from the installed package.
