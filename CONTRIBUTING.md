# Contributing to KohakuMoss

Thanks for your interest in the project. This document is short on purpose; please read it
end-to-end before opening an issue or PR.

**English only.** Issues, PRs, commit messages and code comments in this repository are in
English. Python source and printed messages are ASCII-only (Windows consoles with a
non-UTF-8 code page fail on non-ASCII text).

Issues: https://github.com/KohakuBlueleaf/KohakuMossTrainer/issues. The templates live in
[`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE/). For a bug, include the run's full log,
its generated `config.py` and the output of `kohakumoss-doctor`.

---

## Before You Open a PR

### Feature PRs need an approved issue first

A feature PR adds functionality, changes a public API or a knob, alters the architecture
(a package, a registry slot, a backend, a memory engine), or changes user-visible behavior.
Open an issue first and wait for a maintainer's explicit approval ("go ahead", "approved", or
an `approved` label). An issue and a PR opened at the same time does not count. PRs without
that trail are closed with a pointer here; reopen once the issue is approved.

### Bug fixes and small changes can go straight to a PR

- bug fixes with a clear cause and no architectural change;
- documentation, typo and dead-link fixes;
- tests for existing behavior;
- small refactors inside one module that keep every public API.

A short issue first is still recommended: the fix may collide with in-flight work.

### What the PR description holds

Use the PR template and keep its checklist. At minimum: a summary, the PR type, the measured
problem (log line, number, traceback), the exact validation commands and the GPU they ran
on, the linked issue, and any skipped check with its reason.

---

## Setup

```bash
git clone https://github.com/KohakuBlueleaf/KohakuMossTrainer && cd KohakuMossTrainer
uv pip install -e ".[cuda,dev]"          # or ".[xpu,dev]"
uv pip install -e scripts/bench/matrix   # the bench matrix dev package, if you touch it
```

Windows first: see [docs/guides/install-windows.md](docs/guides/install-windows.md). Never use
`sys.path` / `PYTHONPATH` hacks; scripts and tests import the installed packages.

---

## Tests

- **Model-free suite** (the default `pytest`): kernel inputs -> outputs against the eager
  reference and fp64 (`kohakumoss_kernels.error_stats`), and pure functions (planners,
  captions, knob checks, catalog, schedules, reports). Fast: one fixed kernel config per op,
  tuning off, a few representative shapes. Every test has a time budget
  (`KOHAKUMOSS_TEST_BUDGET`, `@pytest.mark.budget(seconds)` for first-compile tests).
- **Real-weights integration** (`pytest -m real_weights`, with `KOHAKUMOSS_WEIGHTS` and
  `KOHAKUMOSS_DATASET`): training runs, memory placement, offload, inference, export.
- **No fake model, ever**: no tiny, random or stand-in model, text encoder or VAE.
- **Every check has a negative**: a check counts only if it fails on a constructed broken
  variant ([`tests/_util/negative.py`](tests/_util/negative.py) `must_fail`).
- Markers: `cpu, cuda, gpu, sm70, sm75, sm86, sm89, sm120, xpu, windows, multi_gpu,
  real_weights, slow, budget(seconds)`.

Local pre-flight before pushing:

```bash
ruff check src scripts tests configs
black --check src scripts tests configs
pytest -q
```

---

## Code Conventions

The full conventions are in **[CLAUDE.md](CLAUDE.md)**; read it before submitting code. The
non-negotiables:

- **Code carries only *what*; every *why* goes to `docs/`.** Docstrings say what a module,
  class or function is for and its contract (1-3 lines); comments say what a piece of code
  does (one line). No rationale, derivation, measurement or history comments in source.
- **Python 3.10+**, modern hints (`list`, `dict`, `X | None`), `match-case` over nested
  `if-elif-else`.
- **Imports**: built-in, third-party, first-party; `import` before `from`. No imports inside
  functions (optional dependencies such as triton / tilelang are guarded at module level).
- **Files**: one responsibility per module; nest, do not flatten.
- **Select, don't dispatch**: configuration resolves a concrete class or callable once, at
  build time (`kohakumoss_common.registry.build`, `kohakumoss_kernels.registry.resolve`);
  no per-step `if mode == ...` in high-multiplicity code.
- **The backbone is a pure function** `MossBackbone(img, text, sigma, seq) -> v`; objectives
  live in `kohakumoss.diffusion.objective`. Packed is the layout: attention never crosses a
  sample boundary.
- **Numerics**: reduce in fp32; matmuls on fp16 / bf16 inputs with fp32 accumulation. A
  kernel must match PyTorch or beat it against the
  fp64 reference (fp16 and bf16, forward and backward), and is selected only where it is
  measured faster.
- **Windows parity**: every single-GPU feature works on Windows; multi-GPU training is
  Linux-only by design.
- **Public repository**: no machine names, hosts, users, local paths, credentials or private
  dataset details in tracked files or commit messages.

### Before opening the PR

1. Re-read your diff against CLAUDE.md, especially in-function imports.
2. Run `black` and `ruff` over every directory.
3. Add tests in the right kind (model-free or real-weights), each with its negative.
4. Split commits along logical boundaries: a sentence-case imperative subject, and a body
   with the measured problem, then the change.
5. Fill out the PR template completely.

---

## Adding Things

The framework provides slots; most contributions are a registry entry or a dotted import
path, not a core edit:

- **an adapter algorithm**: through `lycoris-lora` (kernel work belongs in LyCORIS itself);
  KohakuMoss binds it in `kohakumoss.adapters`;
- **a kernel or backend**: [docs/internals/kernel-dev.md](docs/internals/kernel-dev.md);
- **a memory profile**: [docs/concepts/profiles.md](docs/concepts/profiles.md);
- **a dataset source**: [docs/concepts/data.md](docs/concepts/data.md);
- **a config preset**: [docs/guides/configs.md](docs/guides/configs.md).

Any of these is still a feature contribution and needs the approved issue first.

---

## License

KohakuMoss is licensed under the [KohakuMoss License 1.0](LICENSE): the Apache License 2.0
with one additional term (derivative works must include "Kohaku" in their name). By
contributing, you agree that your contributions are licensed under the same terms.
