# KohakuMoss

A finetuning framework for MOSS-Image (OpenMOSS-Team/MOSS-Image-Base / -SFT): LyCORIS
algorithms and full finetune on text-image data at native resolutions, with per-arch
kernels, fp16 and bf16, speed/VRAM profiles, offload and multi-GPU, on Windows and Linux.
Sibling to KohakUwULLM (LLM training); same house conventions.

## Project Overview

The framework provides slots; you select or drop in what you need. An adapter algorithm,
a timestep sampler, a dataset source, a resolution planner, a kernel backend or a memory
profile is a registry entry or a dotted import path -- no core edits.

Three packages under `src/`:

| package | role | imports |
| --- | --- | --- |
| `kohakumoss_kernels` | MOSS-block ops, registry, constraints, backends (eager / triton / tilelang), tuning | never `kohakumoss*` |
| `kohakumoss_data` | dataset format, sources, catalog, captions, resolution planning, loader, caches | never `kohakumoss`, never torch.distributed |
| `kohakumoss` | model, encoders, adapters, diffusion, training, memory, offload, parallel, quant, logging | both siblings |

Adapters come from the `lycoris-lora` package: its algorithms, save format and fused
kernels. KohakuMoss carries a custom applier (`kohakumoss.adapters`) that binds each
layer to the LyCORIS entry point once; adapter kernel work belongs in LyCORIS itself.

Targets: sm70 / sm75 (V100, 2080 Ti) through TileLang kernels; sm86 / sm89 / sm120 (3090,
4090, RTX 50xx) through Triton kernels with per-arch designs; Intel Arc (XPU) through
Triton-XPU and eager. Windows support is mandatory for sm75 / sm86 / sm89 / sm120.

## Code Conventions

### File Organization
- Source: `src/{kohakumoss,kohakumoss_kernels,kohakumoss_data}/`
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
- `black` and `ruff check` over every directory before commit.

### The architectural rules

**Select, don't dispatch.** Configuration resolves a concrete class or callable once, at
build time, via `build(spec, REGISTRY)` or `kohakumoss_kernels.registry.resolve(...)`. The
result is held as a plain attribute and called directly. No per-step `if mode == ...`.

**The backbone is a pure function.** `MossBackbone(img, text, sigma, seq) -> v`. It knows
nothing about the loss; objectives live in `kohakumoss.diffusion.objective`.

**Packed is the layout.** Every tensor that flows through the model is a flat token axis;
`SeqInfo` carries the sample boundaries. Only attention reads the boundaries; every other
op is a last-dim op. Attention must never cross a sample boundary.

**The eager ops are the spec.** `kohakumoss_kernels.backends.eager` defines what every op
computes. Every other backend equals it within the precision bar, forward and backward.

### Numerics rules
- Reduce in fp32; token counts are int64; per-sample math (time embedding, modulation,
  head modulation) is fp32.
- Every kernel needs a precision check against an fp64 reference in both fp16 and bf16,
  forward and backward, judged in ULP (`tests/_util/precision.py`). Attention adds the
  huge-logit, padded-key and empty-row cases.
- Every kernel must beat PyTorch (eager composition and `torch.compile` of it) on the arch
  it targets, measured per op and inside the full step.
- fp16 training uses the residual-stream fold (`kohakumoss.model.numerics`), fp32
  accumulators, and the device-flag scaler.

### Testing
- Tests are specifications: each one is shown to fail on a constructed broken variant
  (`tests/_util/negative.py`) before it counts. A check that passes before and after a bug
  proves nothing.
- Markers: `cpu, cuda, sm70, sm75, sm86, sm89, sm120, xpu, windows, multi_gpu,
  real_weights, slow`. Real weights come from `KOHAKUMOSS_WEIGHTS`.

### Audit loop (REQUIRED)
1. Implement the slice in full.
2. Construct the negative case and confirm the check catches it.
3. Lint.
4. Audit the diff for clear bugs, integrity bugs and behaviour bugs, and its cost
   (host syncs, extra copies, saved tensors, peak memory).
5. If a bug slipped past a check, fix the check first, then the bug.
6. Audit benchmarks the same way; an unaudited measurement is not evidence.

### Post-impl tasks
1. Verify the rules above, especially in-function imports and file length.
2. `black` and `ruff`.
3. Logically separated commits: sentence-case imperative subject, body with the measured
   problem then the change.

### Public repository
Everything tracked by git is public. No machine names, hosts, users, local paths,
credentials or private dataset details in tracked files, commit messages or test data.
Local machine notes live in `CLAUDE.local.md`, plans in `plans/`, live state in `STATE.md`
(all gitignored).

## Development Setup

```bash
uv pip install -e ".[kernels,lightning,reference,dev]"
kogine run scripts/train/moss.py --config configs/moss/smoke/smoke_tiny.py
```

Never use `sys.path` hacks in scripts; always import from the installed package.
