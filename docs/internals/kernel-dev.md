# Kernel development

`kohakumoss_kernels` holds the MOSS-block ops. Each op has one eager reference
(`backends/eager/`, the spec) and per-arch implementations:

| arch | implementation | why |
| --- | --- | --- |
| sm70 / sm75 (V100, 2080 Ti) | TileLang | Triton has no tensor-core dots below sm80 (they lower to FMA) |
| sm80 / sm86 / sm89 / sm120 | Triton (TileLang also registered) | Inductor sees `triton_op` launches |
| XPU | Triton (Intel backend) | TileLang has no XPU target |
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
- On Windows nvcc accepts only MSVC; `hardware.toolchain.ensure_host_compiler` sets
  `CXX` to the newest `cl.exe` found through vswhere or the standard install roots.

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
| Triton bwd | strategy `fused` (dK/dV per key block, dQ through fp32 atomics into an `[N, H, D]` accumulator, then one scale-and-cast pass) or `split` (dK/dV kernel + dQ kernel, no atomics, bitwise deterministic; forced by `KOHAKUMOSS_DETERMINISTIC=1`); each kernel tuned separately |
| TileLang fwd | BM, BN, threads, `qreg` (query tile in registers), `share` (K and V through one shared buffer) |
| TileLang dK/dV | BM, BN, threads, `kvreg` (K / V tile in registers), `pds` (P and dS through one shared buffer) |
| TileLang dQ | BM, BN, threads, `qreg` (Q / dO in registers), `share` (V then K through one buffer) |

The register / shared-buffer variants trade shared memory for registers so more
programs fit on cards with small shared memory per SM (Turing: 64 KiB).

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
`inf * 0`, empty segments launch nothing. The backward is three kernels (delta, dK/dV
per key block, dQ per query block) with no atomics: results are bitwise
deterministic.
