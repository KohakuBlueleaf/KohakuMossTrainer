# Kernel development

`kohakumoss_kernels` holds the MOSS-block ops. Each op has one eager reference
(`backends/eager/`, the spec) and per-arch implementations:

| arch | implementation | why |
| --- | --- | --- |
| sm70 / sm75 (V100, 2080 Ti) | TileLang | Triton has no tensor-core dots below sm80 (they lower to FMA) |
| sm80 / sm86 / sm89 / sm120 | Triton (TileLang also registered) | Inductor sees `triton_op` launches |
| XPU | Triton (Intel backend), per op where measured faster (see performance docs) | TileLang has no XPU target |
| CPU | eager | |

## Adding an op to a backend

1. Write `backends/<backend>/<op>.py`; the backend package imports every public
   submodule on import (files starting with `_` are helpers and are skipped), so no
   `__init__` edit is needed. A submodule that fails to import is listed in
   `backends.<backend>.FAILED` and warned about; the rest still register.
2. At module bottom call `registry.register(op, backend, fn, supports)`, with
   `supports` built from `constraints.requires(...)`. `fn` has the eager signature,
   supports autograd and `torch.compile(fullgraph=True)`, and handles every shape it
   accepts (resolution happens once per build, never per call).
3. torch.library wiring:
   - Triton: `torch.library.triton_op` + `register_autograd`; launch through
     `backends.triton._common.Launcher` (direct on real tensors, `wrap_triton` while
     traced) so Inductor records the launches.
   - TileLang: `torch.library.custom_op` + `register_fake` + `register_autograd`.
     Allocate outputs with torch and pass them in: tensors TileLang allocates lose
     autograd when a custom op returns them.
4. Tiles come from `tuning.select` (see [tuning.md](tuning.md)).
5. Tests in `tests/kernels/`: the fp64 bar of `tests/_util/precision.py` in fp16 and
   bf16, forward and backward, plus a broken variant the check must reject
   (`tests/_util/negative.py`). Benchmark in `scripts/bench/kernel/` against every
   PyTorch alternative, eager and compiled, accuracy next to speed.

## Per-sample vectors

The segment-indexed ops (`rms_norm_mod` shift / scale, `gated_residual` and
`linear_gated_residual` gate) receive `mod[:, k]` slices of the backbone's fp32
`[B, 6, D]` block modulation and `[B, 2, D]` head modulation: row stride `6 D` (or
`2 D`), not `D`. The kernels take the row and column strides of each per-sample vector
as arguments and address `seg * row_stride + col * col_stride`; no copy is made. With a
single sample every row reads row 0, so only packed batches of two or more samples
expose a layout error.
`tests/kernels/test_modulation_layout.py` runs every backend on the strided slices
against its own result on contiguous copies; `tests/model/test_packed_kernels.py` runs a
packed batch against per-sample runs under the device's default kernel selection.

## TileLang notes (0.1.15)

- Token and segment counts are `T.dynamic`; one compile per (heads, head dim, dtype,
  tile config, arch) serves every batch. A grid extent can be a scalar argument.
- `T.copy` from a symbolic-shaped tensor bounds-checks rows past the end (zero fill);
  element stores to symbolic tensors are bounds-checked too.
- Use slice regions (`X[a : a + bm, head, :]`) in `T.copy`, not start indices, on
  tensors of rank > 2.
- Moving a GEMM accumulator into the A operand of the next GEMM goes through shared
  memory. A fragment-to-fragment copy between the two layouts fails layout inference
  ("Layout infer conflict") on every arch, and Volta's MMA has no transposed-A form.
- Warp partition: each warp owns a multiple of 16 rows and 16 columns of every
  `T.gemm` tile (checked before compiling).
- Loops whose trip count depends on the program use `T.serial` (no pipelining).
- On Windows nvcc accepts only MSVC; `hardware.toolchain.ensure_toolchain` (run by the
  triton, tilelang and compile backends on import) imports the `vcvars64.bat`
  environment and sets `CXX` to that `cl.EXE` (see
  [install-windows.md](../guides/install-windows.md#compiler-setup)).

## When a backend cannot run an op

`registry.resolve` returns a `Bound` op that calls its implementation directly. An
implementation that cannot run here raises `errors.BackendUnavailable`:

- the tuner raises `ConfigsFailed` (a subclass) when every shortlisted config fails to
  compile or run;
- a raced kernel (`backends/compile/race.py`) raises it when its once-per-device probe
  fails;
- the compile backend raises it when an op's first compile on a device fails.

`Bound` then records the failure for (op, backend, arch) in `registry.failures()` and in
`<kernel cache>/failures/backends.json` (the doctor prints both), warns with the
collected errors, and rebinds to the next backend of the arch priority; `resolve` skips
the failed backend for that op from then on. A pinned backend (`resolve(...,
backend=...)`) re-raises instead. The fallback covers failures raised while the forward
runs, so a backend surfaces its backward failures there:

- the attention kernels build and launch their backward once (zero `dO`, no grad) on the
  first grad-enabled forward of a problem key, inside the same `BackendUnavailable`
  translation as the forward, so a backward that cannot build or launch hands the op on
  before any activation was saved on that backend; under `TUNE=off` the untimed first
  pick is the one launched;
- `registry.exercise(op, run)` runs a forward + backward (prewarm does this for every
  block op and every attention length bucket) and falls the op back on a failure
  raised anywhere in it.

A backward failure that still reaches the backward pass (a new key first met inside a
checkpoint recompute) raises there.

What counts as "cannot run here" is `errors.BUILD_ERRORS` (RuntimeError, OSError,
UnicodeError, Triton's `TritonError`), never out-of-memory: an OOM or a bug propagates and
disables nothing. Before any op binds a backend on a device, `resolve` runs that
backend's health check once per process and device (`health.register_check`): Triton
builds and launches one minimal kernel (this also initialises Intel's driver utilities
on XPU), the compile backend inherits Triton's result on a GPU and needs the host C++
compiler on CPU, and TileLang needs the host compiler for nvcc. A failed check warns once
and the backend is skipped for every op on that device, so a missing compiler costs one
build, not one per op. A recorded failure is stored with the environment fingerprint
(package versions, toolchain, Intel source mode) and honoured by later processes with
the same fingerprint; a changed environment ignores it. `kohakumoss-doctor` lists the
health results and the current-fingerprint failures; `kohakumoss-doctor
--clear-failures` forgets them.

Kernels launch on the current device's queue, so every Triton launch (`Launcher`) and
TileLang launch (`_blockops.compiled`) runs under `hardware.current.on_device(...)` for
its tensors' device; without it a tensor on `cuda:1` / `xpu:1` launched while device 0
is current reads the wrong device (DEVICE_LOST on Intel Arc).

## Varlen attention

`attention_varlen(q, k, v, cu_seqlens, max_seqlen, scale)` on packed `[N, H, D]`
tensors (any token and head strides, contiguous head dim for Triton). The grid's first
axis covers the longest segment; programs past their segment's end exit, so no host
work is needed per batch.

Numerics (the three attention fixes, all applied in both backends):

1. **Two-plane row statistics.** The forward stores the raw score max `m` and
   `log2(l)` as separate fp32 planes `[H, N]`, not one log-sum-exp. A single
   `c*m + log2(l)` loses `log2(l)` once `c*m` is large.
2. **Shift before scale.** Probabilities are `exp2((s - m) * c - log2 l)`; `s - m` is
   exact near the row max, so the dominant probability is exactly 1 and
   `delta = rowsum(dO * O)` cancels.
3. **Scale on the accumulator.** dS is rounded to 16 bits unscaled; the softmax scale
   multiplies the fp32 dQ / dK accumulators at the end.

Design space (the tuner picks per arch and problem key; each variant is a separate
compiled kernel):

| kernel | knobs |
| --- | --- |
| Triton fwd | BM, BN, warps, stages |
| Triton bwd | strategy `fused` (dK/dV per key block, dQ through fp32 atomics into an `[N, H, D]` accumulator, then one scale-and-cast pass) or `split` (dK/dV kernel + dQ kernel, no atomics; forced by `KOHAKUMOSS_DETERMINISTIC=1`); each kernel tuned separately |
| TileLang fwd | BM, BN, threads, `qreg` (query tile in registers), `share` (K and V through one shared buffer) |
| TileLang dK/dV | BM, BN, threads, `kvreg` (K / V tile in registers), `pds` (P and dS through one shared buffer) |
| TileLang dQ | BM, BN, threads, `qreg` (Q / dO in registers), `share` (V then K through one buffer) |

The register / shared-buffer variants trade shared memory for registers so more
programs fit on cards with small shared memory per SM (Turing: 64 KiB).

The fp16 / bf16 MMAs use fp32 accumulation everywhere. GeForce tensor cores run fp16
accumulation about twice as fast, but rounding each partial sum to fp16 adds an error
the size of the output rounding to O and to the gradients, which the precision bar
rejects; bf16 MMAs have no fp16-accumulating form at all.

### Measured design per arch

Every kernel's candidates start with the arch's own measured seeds
(`ARCH_SEEDS[sm]` in `backends/triton/attention.py`, `SEEDS[(kind, sm)]` in
`backends/tilelang/attention.py`), then the shared seeds, then the planner's ranking;
the tuner times the first `SHORTLIST`. Under `TUNE=off` or a trace the first legal seed
runs untimed.

| arch | source | fwd (BM, BN, warps, stages) | fused bwd | dK/dV | dQ |
| --- | --- | --- | --- | --- | --- |
| sm120 (RTX 5090, 99 KiB shared per block) | full sweep of 512 tiles per kernel at 2 MP x1, 1 MP x2, 512 px x8 | 128, 64, 8, 3 (235 TFLOP/s) | 16, 128, 8, 4 (200) | 32, 32, 4, 3 (209) | 128, 32, 8, 3 (241) |
| sm80 (A100, 163 KiB) | FA2's / Triton's A100 tiles (not measured here) | 128, 64, 4, 3 | 64, 128, 8, 2 | 64, 128, 8, 2 | 128, 64, 8, 3 |
| sm86 / sm89, XPU | the shared seeds, tuned per key | | | | |
| sm70 / sm75 (TileLang) | measured per-arch seeds | see `SEEDS` | | | |

On sm120 the fused backward (dQ through fp32 atomics) beats the split one at every
measured shape (11.6 vs 14.7 ms at 2 MP x1) and reaches 86 % of FA2's backward; FA2 and
our forward both run at the card's measured MMA rate (233 TFLOP/s probe), so a single
long segment ties with SDPA ([performance](../performance/attention.md#the-single-segment-ceiling)).
The fused kernel holds BN = 128 keys with dK and dV in fp32 registers; larger query
tiles than 16 rows lose to register pressure. TileLang runs on sm120 too (race
candidate) and wins only the text-refiner shapes. sm80 seeds are a design point for
A100-class cards; the tuner replaces them with measured choices on first use.

Registry entry on sm80+ / XPU (`backends/triton/attention_race.py`): a per-key race of
our Triton kernels, our TileLang kernels and torch's `varlen_attn` (when the torch build
ships flash and determinism is not requested), timed forward + backward on the race's
own leaf copies. The race times only outside inference mode, checkpointed forwards,
recomputes and traces; trainers call `tuning.prewarm_attention` before step 0 so every
length bucket is decided ahead of the checkpointed training forward.

Call path: the first call of a problem key tunes (or reads the table) and binds the
chosen kernels into one launch callable; later calls look the binding up by key and
launch. Provisional picks made while tracing are not bound.

Guards: masked key columns are `-inf` before the max (fully valid rows always have a
finite max), masked tiles zero `p` explicitly so out-of-range rows never produce
`inf * 0`, empty segments launch nothing. The backward kernels read row statistics
(`m`, `log2 l`, delta) only for rows inside the segment; rows past the segment end read
0 (explicit `if_then_else`, independent of TileLang's automatic bounds guards).

### Precision modes and the precision gate

The Triton kernels compile in one of two precision modes (`SPLIT` constexpr):

| mode | P, dS into the MMAs | delta | MMAs fwd / bwd | extra memory |
| --- | --- | --- | --- | --- |
| `p16` | rounded to 16 bit once | from the 16-bit O | 2 / 5 | none |
| `split` | hi + lo 16-bit pair, two MMAs into the fp32 accumulator (lo scaled by 2^11 so fp16 keeps its bits) | from O plus its stored rounding residual (`olo`, an O-sized 16-bit tensor saved for backward) | 3 / 8 | one O-sized tensor per call |

`split` leaves the fp32 accumulations and the single rounding at the store as the only
error sources. PyTorch's reference differs per device, and the registry gate
(`precision_ok` in the Triton `SUPPORTS`) requires the least mode that reaches it:

| device | PyTorch SDPA, fp16 / bf16, vs fp64 (max / RMS ULP) | p16 kernels | required |
| --- | --- | --- | --- |
| CUDA flash (sm86 / sm89) | o 34 - 65 / 0.27; dQ, dK 70 - 270 / 0.37 - 0.38 | o 1.0x; dQ, dK RMS 0.84x, 99.9 % 0.75x | `p16` |
| CUDA flash (sm120) | o 42 - 44 / 0.26; dQ 124 - 134 / 0.31; dK 91 - 99 / 0.31 | RMS 1.00 - 1.02x, 99.9 % <= 1.06x | `p16` |
| CUDA memory-efficient (sm70 / sm75, Windows) | dQ, dK RMS 0.37 | dQ, dK RMS 0.83x, 99.9 % 0.71x | `p16` |
| XPU (B50) | 0.50 / 0.20 on o, dQ, dK, dV (correctly rounded) | max 44 - 184, RMS 0.26 - 0.32 | `split` |

`split` measures max 0.50 - 0.56 / RMS 0.20 - 0.21 ULP on o, dQ, dK, dV (sm120, both
dtypes): correctly rounded up to the fp32 accumulation, at +50 % forward and +15 %
backward time on the 5090. No CUDA SDPA backend is correctly rounded, so CUDA runs
`p16`; XPU's SDPA is, so XPU runs `split`.

Where `p16` falls short of a correctly rounded reference: with clustered queries and
keys the softmax is nearly one-hot and `sum_j dS_ij = 0` holds only before rounding.
dQ = `scale * sum_j dS_ij K_j`, and K carries a large component common to all keys, so
the rounding of dS to 16 bits (bf16: 8 bits) leaves a residue `(sum_j round(dS_ij)) * K`
that the true dQ cancels. Measured dQ cosine to fp64 (logits ~1e5, bf16, the test's
first draw, sm120): flash SDPA 0.974, p16 0.988, `split` 0.999997; on XPU the SDPA reference is
correctly rounded, which is why the gate requires `split` there.

`KOHAKUMOSS_ATTN_PRECISION=auto|p16|split` forces a mode; a forced mode below the
device's requirement makes the gate reject the Triton backend there, and the race key
carries `-split` so the two modes keep separate tuning tables. On devices without
torch's flash `varlen_attn` (XPU) the race adds the per-segment SDPA loop as a
candidate, so the kernel is bound only where it is faster than PyTorch at the required
precision.

### Precision bar

The attention tests judge o, dQ, dK, dV against the fp64 reference and against PyTorch's
SDPA in the same dtype (`tests/kernels/_attention_case.py`), with no ULP floor:

| statistic | bar | measured, sm89, 16 seeds, fp16 / bf16 |
| --- | --- | --- |
| RMS error | <= 1.1 x PyTorch | dQ, dK <= 0.85 x; o, dV 1.00 x |
| 99.9 % quantile of the elementwise error | <= 1.1 x PyTorch | dQ, dK <= 0.75 x; o, dV <= 1.01 x |
| elementwise max | <= 2.5 x PyTorch | dQ, dK 0.49 - 2.09 x (mean 0.93 - 0.97 x); o, dV 0.94 - 1.00 x |

Both implementations round the same quantities once each (P and dS to 16 bit for the
MMAs, delta from the 16-bit O); the elementwise max over one draw lands on a near-zero
element of the shortest segment (largest magnitudes, fewest terms), where which of two
equally rounded results is closer is a coin flip. Its ratio spreads 0.49 - 2.09 x over
seeds while the distribution (RMS, 99.9 % quantile) is uniformly better than PyTorch's,
so the max carries the 2.5 x margin and catches localized bugs (masking, tile edges:
errors orders of magnitude above rounding), and precision is judged on the RMS and the
quantile at 1.1 x. A result 1.3 x less precise than PyTorch everywhere fails the bar
(the test's negative case).

The huge-logit test pools 16 draws before applying the bar: with a nearly one-hot
softmax a handful of elements carry each draw's dK error, and the per-draw RMS ratio of
the same kernel spreads 0.69 - 2.01 x (sm70 TileLang vs memory-efficient SDPA, fp16)
while the pooled ratio is 0.92 x (sm120 p16 vs flash: 0.36 - 0.66 x pooled).

Determinism per backend and strategy (repeat calls on the same inputs and card):

| path | o | dK, dV | dQ |
| --- | --- | --- | --- |
| Triton fwd + `split` bwd | bitwise | bitwise | bitwise |
| Triton `fused` bwd | bitwise | bitwise | not bitwise (fp32 atomic adds in arrival order) |
| TileLang fwd + bwd (delta, dK/dV, dQ; no atomics) | bitwise | bitwise | bitwise |
| torch `varlen_attn` (race candidate) | as torch | as torch | as torch |

The Triton strategy is a tuned choice, so dQ is bitwise only when `split` wins or
`KOHAKUMOSS_DETERMINISTIC=1` forces it (which also drops `varlen_attn` from the race).
A different tuned tile config gives different (equally accurate) bits.

The eager spec runs one SDPA per segment and needs the segment bounds on the host. It
takes them from a cache keyed by the `cu_seqlens` tensor: `SeqInfo.to` registers the
bounds it already holds (`remember_bounds`) when it moves the plan to the device, so the
eager path never syncs; an unregistered tensor is read back once. An entry is used only
while the tensor lives and its version counter is unchanged, so an in-place rewrite or
a new tensor at a reused address reads back again. Inference tensors carry no version
counter and are read back on every call.
