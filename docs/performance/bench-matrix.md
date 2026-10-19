# The real-GPU bench matrix

One frozen copy of the working tree (plus the extra repos, such as LyCORIS) runs on
every card of every host:

- real weights, real data, every speed / VRAM setting;
- the training quality runs with their comparison grids, and inference checks;
- the ncu efficiency pass and the feature-coverage runs.

Every run writes its config, log, metric rows, nvidia-smi trace and result record to
disk, and one report answers speed, VRAM, whether a step is optimized, quality and
coverage per card model.

```
scripts/bench/matrix/launch.py                   local driver: plan, freeze, ship, install, start, status, join, collect, report
scripts/bench/matrix/host.py                     host runner (on the host, from the frozen tree)
scripts/bench/matrix/src/kohakumoss_bench/       the dev package both use: run/, plan/, measure/, report/
```

The package installs separately from the kohakumoss wheel, next to the editable
kohakumoss install, with `uv pip install -e scripts/bench/matrix` (the hosts' default
`install` command installs both from the frozen tree). Module paths below are relative
to `kohakumoss_bench/`.

## Launch

`launch.py --phase <phase>` runs one phase per call against a [hosts file](#hosts-file):

| phase | what it does |
| --- | --- |
| `plan` | the dry run, nothing trains: per host the cards nvidia-smi lists (over ssh; which ones the host would use and the memory in use now), read-only preflight checks (venv interpreter, weights, fix file, datasets, `/dev/shm` space, `ncu --version`), the card pools and every run of each pool's queue with an expected duration ([Plan](#plan)), written to `runs/bench/<bench>/plan.md` and `plan.json` |
| `start` | freezes, ships and installs the trees, then starts the [host runner](#host-runner) (below) |
| `status` | prints each host's `status.json` |
| `join --gpus 2,3` | writes `join.json` into a running bench: those cards join the pool of their model once idle |
| `collect` | copies `runs/<bench>/<host>/` to `runs/bench/<bench>/<host>/` |
| `report` | writes `report.md` and `report.json` there ([Report](#report)) |

`start` lists the shipped files of each repo (tracked and untracked, not ignored,
without the local-only directories in `run/remote.py` `SKIP`) and hashes them into one
identity per repo (`<HEAD>-<sha1 of paths and bytes>`). It ships each as one tar
(skipped when the identity is there), runs the install command for each, uploads the
host runner config (`configs/<bench>-<host>.json`) and starts the host runner (`nohup`
over ssh). Later edits never reach a running bench: every host runs the trees it got.

## Hosts file

A JSON file `{"bench": {...}, "hosts": [...]}`. The bench block fills `Options`
(`plan/config.py`; `--option key=value` overrides one on the command line); each
host fills `Host`:

| field | meaning |
| --- | --- |
| `name`, `transport`, `ssh` | "ssh" with a command prefix (`["ssh", "user@host"]`), "local" (this machine, foreground) or "manual" (print the commands) |
| `root` | bench root on the host: `trees/<identity>/`, `trees/<repo>-<identity>/`, `configs/`, `runs/<bench>/<host>/` |
| `python`, `install` | the venv interpreter and the command that installs a frozen tree into it (`{python}`, `{tree}`; "" skips: the environment's `PYTHONPATH` can point at the trees instead) |
| `device_type`, `gpus`, `xpu_cards` | CUDA cards come from nvidia-smi (`gpus` limits the indices); XPU cards are listed |
| `ram_gib` | gates the host-offloaded full finetune |
| `weights`, `fix_file`, `copies` | `{"source", "stage"}`: copied once to fast local storage; `hf:<repo>` downloads; `copies` stages more (datasets) |
| `datasets`, `cache_dir`, `cache_from` | the DATASETS entries on the host, the offline cache directory, an existing cache to copy instead of building |
| `env` | extra environment of every run; `{tree}` and `{tree:<repo>}` expand to the frozen trees |
| `quality`, `kinds`, `profiles`, `bases`, `catalog`, `runs` | quality run names, run kinds, named profiles (no catalog points), base precisions to keep, the host's catalog points ("all" / "frontier" / "none"; "" = the bench's), run-name patterns to keep (fnmatch; a kept run keeps what it depends on) |
| `ncu` | the Nsight Compute command (`["sudo", "-n", "/usr/local/cuda/bin/ncu"]` where counters are admin-only); empty: no ncu pass |

The bench block's `repos` (`{"lycoris": "<local path>"}`) lists the extra repos frozen,
shipped and installed next to the main tree.

## Host runner

Prepare once, then one pool per card model:

1. **Staging.** The weights, the fp16-fix file and the copies are copied to their stage
   paths (NVMe or `/dev/shm`); files already there with the same size are kept, so the
   copy happens once per host. Every run loads from the stage.
2. **Probe.** Per card: the dense GEMM rate (8192^3, fp32 accumulation; bf16 where the
   card has it, fp16, and fp8 `_scaled_mm` on sm89+), the tensor-core MMA peak (a
   Triton loop of register-resident 128 x 128 x 64 `tl.dot`s into fp32, 8 CTAs per SM,
   same dtypes; the report's "MFU (measured MMA)" divides by it) and the pinned / pageable
   host <-> device copy bandwidth (1 GiB). These are the ceilings the report holds runs
   against.
3. **Caches.** The offline latent and text caches every run reads are built once on the
   host's largest card (`cache_latents.py`, `cache_text.py` with the same data block as
   the runs) and marked with that block; an unchanged block skips the build.
4. **Grid prompts.** The most frequent artists of the catalog, a seeded draw of their
   rows, each caption rendered by the training caption template (variant 0).
5. **Pools.** The idle cards (no other compute process in nvidia-smi) of one model share
   one queue, split into units (`matrix.units`: a run with every run that depends on it,
   in order, on one card). The first unit (the tuning run) runs alone; the pool's other
   cards start once it has finished. The final unit (the restart run) starts when every
   other unit has finished. Each card takes the next unit when it is free; a card that
   turns busy is waited for, then the run is skipped. Identical cards never run the same
   configuration twice.

One process per run: the weights load from the stage in seconds, and the kernel tuning
tables, the torch.compile cache and the Triton / TileLang caches are on disk, so only a
pool's first run tunes; every later run (and the restart run at the end) binds from the
tables. Tables are per card model; each write merges the file first
([tuning.md](../internals/tuning.md)), so cards of one pool tuning at once keep each other's rows.

## Matrix

Per pool (`plan/matrix.py`):

| kind | runs |
| --- | --- |
| speed | precision (bf16 where the card has it, fp16) x base (compute and int8 weight-only everywhere, plus fp8_compute and MXFP8 on sm89+; `bases` keeps a subset) x algo (LoRA r32, LoKr factor 16 full matrix) x the named profiles (speed, balanced, low, minimal, auto) that fit; one `auto` row each for DoRA r16, LoHa r16 and OFT block 64; the reference combination (first precision, compute base, LoRA r32) over the catalog points that fit |
| cache | the flagship setups under three cache modes: no cache (`LATENT_CACHE` and `TEXT_CACHE` "off": the VAE and text encoder run every step), the offline latent cache, both offline caches. Setups: LoRA r32 `balanced` in the first precision and in fp16, LoRA r32 `minimal`, LoKr 16 `balanced`, fp8_compute LoRA r32 `balanced` (sm89+), and LoKr 16 fp16 on an int8 base under `ckpt-ring2-actoff-optstream-gradoff` with the text encoder streamed 8 batches ahead; on a card without bf16 the fp16 equivalents once. The encoders take the profile's placement (text streamed and VAE offloaded under `low` / `minimal`). They run right after the pool's first run, ahead of every other rank-1 run |
| placement | no cache, per text-encoder / VAE placement: LoRA r32 `balanced` (text `auto`, `resident`, `offload`, and `ring` with the VAE `resident` and `offload`), LoRA r32 `minimal` and the low-VRAM LoKr setup (text `auto`, `resident`, `offload`, `ring`; the profile's VAE). A placement past the card's memory ends `oom`. The report's "auto picked" column is what the `auto` row of the same setup logged (`text encoder placement:` / `VAE placement:`). They run right after the cache rows |
| batch | LoRA r32 under `balanced` and `ckpt-resident` in the first precision, plus fp16 `balanced` on a bf16 card, at `batch_scales` (2, 4, 8) x `max_tokens` tokens per step (MAX_TOKENS, the BatchLimits budget); past the card's memory a row ends `oom`. They run right after the pool's first run |
| online | the reference combination under `auto` with online latent and text caches into an empty directory |
| cap | `minimal`, LoRA r8 and LoKr 16, every base, under each VRAM target below the card's memory (8 GB, 6 GB) |
| quality | LoRA r8 and LoKr 16 for 400 steps in bf16 and fp16 (`fp16_fix`), plus LoRA r8 fp16 with the other fix where the host has the fix file; checkpoints every 100 steps, samples during training, LyCORIS + ComfyUI exports |
| export, grid | each checkpoint exported to LyCORIS, then `compare_grid` (base model, then step 100 .. 400 as rows, the grid prompts as columns) |
| infer | `sample.py`: base model, adapter merged, adapter in bypass, fp8 DiT + TE (sm89+), low-VRAM placement |
| coverage | the host's largest model only: `plan/coverage_runs.py` |
| ncu | the reference combination under `auto` under Nsight Compute (hosts with `ncu`; see below) |
| restart | the pool's first speed run again at the end: it must tune nothing |

fp16 runs use `fp16_fix`: "online" (runtime factors, c = 1/16) or "file" (the host's
fp16-fix checkpoint; online on a host without one).

A profile fits when its estimated peak (`kohakumoss/memory/estimate.py`) plus
`overhead_gib` (default 2.5: the context, loaded kernels and the allocator's cache,
which the estimate leaves out) is at most the card's memory. The catalog points are the
fitting entries (`catalog="all"`) or their frontier (`catalog="frontier"`, the default):
an entry is dropped when an entry that streams, offloads and recomputes no more on every
axis needs at most `frontier_slack` (3%) more estimated memory.

Speed runs:

- 40 steps, one metric row per step (one host sync per step, the same in every run).
- Steps 38 .. 39 (`trace_steps`, 2) run under `TRACE_STEPS` (torch.profiler) for the
  step breakdown (`trace_compute_ms`, `trace_copy_ms`, `trace_idle_ms` per step,
  [multi-gpu.md](../guides/multi-gpu.md)).
- Medians are over the steps after step 10 outside the profiled steps.

## Plan

The plan's durations are estimates:

| part | estimate |
| --- | --- |
| a training step | its model FLOPs (`kohakumoss/model/flops.py`, recompute included) at `plan_mfu` (0.4) of the nominal peak (`plan_tflops` for a card missing from the table), or the streamed weight and activation bytes at `plan_link_gbps` (20 GB/s) when that is longer |
| every process | + `plan_overhead_s` (120 s) |
| a pool's first run | + `plan_tuning_s` (1800 s) |
| each ncu step | + `plan_ncu_step_s` (300 s) |
| images | sampler steps x 2 (CFG) forwards at 1 MP |

A pool's wall time places its units longest-first on the least-loaded card after the
first unit.

## ncu

The ncu pass records steps `warmup + 1 .. warmup + ncu_steps` (step 11) of one training
run per pool: `PROFILE_STEPS` brackets them with `cudaProfilerStart` / `Stop`, ncu runs
with `--clock-control none --cache-control none` (clocks and caches untouched: other
users' cards stay as they are, each kernel sees the cache state the step leaves) and the
replay mode `ncu_replay` picks (`measure/ncu.py` `REPLAYS`):

| `ncu_replay` | results | cost of one profiled step |
| --- | --- | --- |
| `app-range` (default) | one per profiler range: the whole step as one unit | the whole run once per metric pass (about 1 min per pass) |
| `application` | one per kernel (`--profile-from-start off`, relaxed matching) | every kernel serialized per pass: ~0.12 s x ~9-11k kernels, about 20 min per pass |
| `kernel` | one per kernel (`--profile-from-start off`) | as `application`, plus a device-memory save / restore around every kernel |

The four metrics need 6 (ga100) to 9 (gb202) passes, so a per-kernel pass of one step
takes 2-3.5 h. `app-range` measures the same step in minutes and gives the whole-step
tensor / SM / DRAM % the bound result uses, without the per-kernel split (the speed
runs' torch traces carry the kernel times). ncu rejects `--profile-from-start` under
range replay: the profiler calls themselves delimit the range.

Collected per result: `gpu__time_duration.sum`, the tensor-pipe metric
`.avg.pct_of_peak_sustained_elapsed`, `sm__throughput` and `dram__throughput` (% of
peak, elapsed). The tensor metric is the first of
`sm__pipe_tensor_subpipe_hmma_cycles_active`, `sm__pipe_tensor_op_hmma_cycles_active`,
`sm__pipe_tensor_cycles_active` that `ncu --query-metrics --chip <chip>` lists for the
card (else the subpipe name on sm90+ and the op name below).

Under `sudo` the environment is reset, so the run's environment goes through an env file
(`run/child.py --env-file`), and the run's caches (HOME, Triton, inductor) and a copy
of the kernel tables live inside the run directory: nothing the root process writes
lands in the runner account's own caches.

The report is `ncu.ncu-rep`, imported as `ncu.csv` and summarized into `ncu.json`:

- per kernel (per range under `app-range`; grouped by name): calls, ms per step, time share, duration-weighted
  tensor / SM / DRAM %, and the class: compute (SM % leads and reaches `ncu_bound_pct`,
  60), memory (DRAM % leads and reaches it) or latency (neither);
- totals: duration-weighted tensor / SM / DRAM %, kernel ms per step, the time share of
  each class;
- against the matching speed run (same precision, base, algo, profile): kernel busy share
  (kernel ms per step / the unprofiled step time), tensor active over the step (tensor %
  x busy share, the whole-step figure DCGM's tensor-active metric reports), and the
  result: PCIe-bound (the speed run's transfer share >= `--transfer-bound`), host-bound
  (kernels fill under half of the step), else the class with the largest time share.

A failed ncu process (sudo, an unsupported metric) is `profile-failed`: queued for
review, the queue goes on.

The achieved TFLOP/s of each GEMM / flash kernel against its clock-derived peak, and
how the tensor-pipe % relates to it per arch, are in
[gemm-efficiency.md](gemm-efficiency.md) (`measure/tflops.py`, `measure/ceiling.py`).

## Coverage

The coverage runs exercise what the matrix does not: the other LyCORIS algorithms
(LoCon, decomposed LoKr, BOFT, GLoRA, IA3, DyLoRA, T-LoRA, dropouts), the optimisers
(8-bit AdamW, Prodigy, schedule-free, Muon), EMA, timestep samplers, huber loss, x0
weighting, schedules, the Lightning driver, block compile, the LyCORIS torch backend,
exports, samples and validation during training, the resolution planners and
batchings, flip / random crop, manual memory knobs, streaming without checkpointing,
pageable host memory, the fp8 weight-only base, online caches with offloaded / streamed
encoders, the fp8 variants (16-bit input gradient, fp8 text encoder) on sm89+,
checkpoint + resume, and a direct full finetune where the host RAM holds its masters and
states. The report maps each feature of `plan/coverage.py` `FEATURES` to its runs or
tests.

## Host RAM

Concurrent runs on one host share what the host has free (`plan/hostmem.py`).

- **Budget:** `MemAvailable` of /proc/meminfo, after every process's resident, pinned
  and shared use (other jobs on the host included), minus `host_ram_reserve_gib` (16)
  and the shares of runs started in the last 3 minutes (not yet resident). Without
  /proc: `host_ram_share` (0.6) of `ram_gib`.
- **A run's share:** a process baseline (8 GiB), plus the streamed base weights when its
  profile streams, the offloaded activations when it offloads them (tokens x saved
  elements per token x depth x 2 bytes: about 42 GiB at 8192 tokens without
  checkpointing, 2 GiB with block checkpointing) and, for a full finetune, host
  optimizer state and gradients.
- **Admission:** a run waits until its share fits; a run larger than the whole budget
  waits until nothing else holds any.

## Release

A host with `priorities` holding 0 runs the release stage instead of its pool queue
(`run/release.py`): LoRA r32 `balanced` and `minimal`, LoKr 16 `balanced`, the
low-VRAM LoKr setup, fp8-compute LoRA r32 `balanced` (sm89+), LoRA r32 under the
`vram_8g` / `vram_6g` layers (configs/moss/vram, their memory profile, token budget and
encoder placement) with the allocator capped at 8 / 6 GB, all on the offline caches,
plus LoRA r32 `balanced` without caches.

The report's "Release" table (`report/build.py` `release_rows`) holds throughput, the
encoder seconds per step, the steady step peak, and the prewarm phase (process start
to the first step) against the step phase (`report/results.py` `phase_peaks`, GiB):

- nvidia-smi peak of each phase (the process; under WDDM the card above its idle
  baseline);
- torch max allocated / reserved: the prewarm's (`prewarm_peak_alloc` /
  `prewarm_peak_reserved`, logged at step 1) and the steps' (the larger of step 1's own
  peak, parsed from its `VRAM peak: prewarm ...; step 1 ...` log line, and every later
  step's metric-row peak: `vram_step_peak`, `vram_peak_alloc`, `vram_peak_reserved`;
  step 1's row is left out, its window can hold the prewarm);
- "prewarm > step": yes when any prewarm peak exceeds the step phase's. The tuning
  prewarm is bounded by the kernel prewarm's planned-maximum replay; on the release
  rows it peaks at or within 0.8 GiB of the step peak (the minimal LoRA rows: 3.1 vs
  2.8 GiB on the RTX 5090). Peaks are compared with peaks: a median of the step peaks
  understates a run whose packs vary in length.

The offline caches are built once per encoder dtype the host's cards resolve to
(`knobs.encoder_dtype`: bf16 where the card has it, else fp16; the VAE dtype and the
text-encoder dtype are part of every cache key), on a card of that dtype with both
encoders pinned to it. A marker per dtype (`bench-cache-<dtype>.json`) records the
data block, the dtype and the cache-key layout (`CACHE_LAYOUT`), so a change of any of
them rebuilds.

## Disk

`run/disk.py`, per pool:

- **Cleanup** (`cleanup`, on): once a run and every queued run that reads its directory
  (`{run:<name>}` in its knobs: exports read a quality run's checkpoints, grids and
  inference checks read exports, a resume run reads the checkpoint run) have finished,
  the run's `train/checkpoints/` and its exported `.safetensors` are deleted. Results,
  metric rows, logs, traces, nvidia-smi samples and ncu reports stay. Runs a previous
  start finished are handed to the cleanup when the pools are built, before any card
  starts. Nothing outside the host's runs directory is touched.
- **Floor**: a row starts only with at least `max(disk_floor_gib, 2 x its checkpoint
  estimate)` free on the runs disk (50 GiB; the estimate is `full_checkpoint_gib`, 60,
  for a full finetune, `adapter_checkpoint_gib`, 2, otherwise). Below it the card
  holds, reports the hold once (`held` in `status.json`, one log line) and rechecks every
  `disk_poll_s` (60 s).
- **Reruns**: a run that executes again in a directory a previous attempt used first
  moves that attempt's `train/` to `train.attempt-<n>/` (its checkpoints deleted), so
  the new metric rows start empty.

## Resume

`--option resume=true` on `start` skips every run of the bench whose `result.json`
(on any card of the host) ended in a final status (ok, oom, config, nonfinite, retuned,
profile-failed); a run that crashed, timed out or was cut off runs again and its old
record becomes `result.superseded.json`. A runner failure (a full disk, a lost file) is
a `runner-error` record and stops the host; a crash or a timeout only goes to the review
queue ([Failures](#failures)).

## VRAM

A 500 ms nvidia-smi sampler runs for the whole host (two `--loop-ms` processes). Per
run, it writes the child process's used memory and its card's used memory to
`vram_smi.csv`, with:

- the process peak and steady value (the median of the second half of the samples);
- the card peak, and the card peak and steady value above the idle baseline taken before
  launch. Under Windows WDDM the per-process column is [N/A] and the card above its
  baseline stands in;
- for a training run, the same split at the start of its first step (the first metric
  row's `time` less its `step_time`) into the startup + prewarm phase and the step loop,
  each with its own peak and steady values (`vram_smi_phases`), so the prewarm spike
  does not hide the loop's own use.

The metric rows add torch's allocated / reserved peaks, the per-step peak
(`vram_step_peak`) and the device-use breakdown
([logging.md](../guides/logging.md#vram)).

A cap run caps the caching allocator at the target minus `cap_allowance_gib` (1.5) and
passes when the nvidia-smi peak stays at or below the target.

## Failures

Each run's process classifies an exception before it exits (`run/child.py`):

| class | meaning | effect |
| --- | --- | --- |
| `oom` | out of memory | recorded: the setting does not fit the card |
| `config` | the knobs were rejected (validation, a missing optional package) | recorded and queued for review; the queue goes on |
| `crash` | any other exception or failed assertion inside a module | that row stops and is queued for review; the queue goes on |
| `timeout` | the run exceeded its timeout (the first run of a pool gets the tuning one) | as `crash` (SIGTERM to the run's process group, SIGKILL a minute later; through sudo for the ncu pass) |
| `terminated` | the run got SIGTERM from outside the runner (a drain) | not final: a resumed bench runs it again |
| `runner-error` | the host runner itself failed around a run (a full disk, a lost file) | the host stops: its other cards finish the run they are on and start nothing new |

Each run starts as the leader of its own process group (a new session on Linux, a new
process group on Windows: `proc.session`). A timeout signals the whole group, and once
the run has exited whatever is left of the group (a loader worker or forkserver that
outlived its trainer) is killed (`proc.reap`). On Windows the timeout kills the process
tree.

A training run that ends with non-finite steps is `nonfinite`, a restart run whose
kernel or adapter prewarm timed anything is `retuned`, a failed ncu process is
`profile-failed`: queued for review, the queue goes on. Nothing is retried.
`review-queue.jsonl` holds every entry with its traceback, frames and run directory.

## Results

`<root>/runs/<bench>/<host>/`: `cards.json`, `staging.json`, `probes.json`,
`prompts.json`, `queues.json` (per pool), `status.json`, `kernels.json` (per model),
`review-queue.jsonl`, the tuning tables under `tuning/<model>/`, and per card
`<card>/<run>/` with `config.py`, `log.txt`, `train/metrics.jsonl` (training runs),
`train/ranks/` (the per-rank rows, summary and profiled trace), `vram_smi.csv`,
`failure.json`, `ncu.*` (the ncu pass) and `result.json` (the summary: medians of every
metric, maxima, the step breakdown, kernel and adapter prewarm lines, losses, exports,
the nvidia-smi peaks and steady values).

## Report

Per card model, in order:

| section | content |
| --- | --- |
| probes | each card's probe ceilings |
| cache modes | samples/s, tokens/s, TFLOP/s, text-encoder and VAE seconds per step and share of the wall time, the median per-step torch peak, nvidia-smi peak / steady of the prewarm phase and of the step loop, the fp16 loss scale, non-finite steps |
| speed | card, samples/s, s/step, tokens/s, nominal MFU, share of the measured GEMM rate, nvidia-smi peak / steady, allocated / reserved, compute / copy / idle ms per step, GB moved per step, transfer share, the bound class, prewarm tuned / from the table, non-finite steps, loss first -> last |
| VRAM targets | pass / fail |
| ncu | the efficiency table and top kernels |
| quality | quality, grids and inference |
| kernels | per tuned kernel: problems, chosen configs, and where the table timed a PyTorch / compile path the chosen config's speedup over it |

Then the feature coverage (`plan/coverage.py` `FEATURES`: working / broken / does not
fit / not run, or test passed / failed / present from a `--junit` XML) and every
failure.

The bound class of a speed run:

- transfer share: (h2d GB / pinned h2d GB/s + d2h GB / pinned d2h GB/s) / step time, over
  the DiT engines (`xfer_*_gb`, [logging.md](../guides/logging.md#transfer)) and the
  text encoder / VAE weight traffic;
- compute share: model TFLOP/s (kohakumoss/model/flops.py) / the probe's GEMM rate (fp8 for an
  fp8 base);
- transfer-bound at a transfer share of `--transfer-bound` (0.7) or more, else
  compute-bound at a compute share of `--compute-bound` (0.5) or more, else
  software-bound: neither the link nor the math units are the limit.

### Nominal peaks

Nominal MFU (the trainer's `mfu`, the report's MFU column) divides the model TFLOP/s by
`kohakumoss/model/flops.py` `PEAK_TFLOPS`, matched as a substring of the device name (first
match wins). Every value is the dense peak (no 2:4 sparsity) of FP16 / BF16 tensor-core
inputs with FP32 accumulation, the precision the trainer's matmuls run at:

| key | TFLOP/s | source |
| --- | --- | --- |
| `V100`, `PG503` | 125 | Tesla V100 datasheet, SXM2 tensor performance |
| `2080 Ti` | 53.8 | Turing architecture whitepaper, FP16 tensor with FP32 accumulate (107.6 with FP16 accumulate) |
| `Tesla T4` | 65 | T4 datasheet, mixed precision FP16 / FP32 |
| `Tesla T10` | 68.8 | no NVIDIA tensor figure is published; 8x the listed FP32 peak of 8.6 TFLOPS (3072 CUDA cores, 384 tensor cores, about 1.40 GHz), the FP16-tensor to FP32 ratio of the T4 datasheet (65 / 8.1) |
| `3060` | 25.6 | 112 tensor cores at the 1.78 GHz boost clock, half-rate FP32 accumulate as on the GA102 GeForce cards |
| `3090` | 71 | Ampere GA102 whitepaper, FP16 tensor with FP32 accumulate (142 with sparsity) |
| `4090` | 165.2 | Ada Lovelace architecture whitepaper, FP16 tensor with FP32 accumulate (330.4 with sparsity) |
| `5090` | 209.5 | RTX Blackwell architecture whitepaper, FP16 tensor with FP32 accumulate (419 with sparsity) |
| `RTX PRO 6000 Blackwell` | 500 | RTX PRO 6000 Blackwell Server Edition (Colab G4) datasheet: FP4 4 PFLOPS with sparsity, so FP16 1000 with sparsity and 500 dense (FP32 accumulate at full rate on the professional card); the datasheet lists no FP16 row. The Workstation Edition clocks higher (FP32 125 against 120 TFLOPS), about 4% above this value |
| `A100` | 312 | A100 datasheet, BF16 / FP16 tensor (624 with sparsity) |
| `H100 PCIe` | 756 | H100 datasheet, PCIe, BF16 / FP16 tensor (1513 with sparsity) |
| `H100` | 989 | H100 datasheet, SXM, BF16 / FP16 tensor (1979 with sparsity) |
| `NVIDIA L4` | 121 | L4 datasheet, BF16 / FP16 tensor (242 with sparsity) |

For an fp8 base (fp8_compute, MXFP8) the report adds "MFU (fp8 nominal)": the same
TFLOP/s against twice the 16-bit peak, the FP8-to-FP16 tensor ratio at FP32 accumulate
of the Ada Lovelace (4090: 330.3 against 165.2) and RTX Blackwell (5090: 419 against
209.5) whitepapers. "of GEMM peak" holds an fp8 run against the probe's measured fp8
`_scaled_mm` rate.

A device whose name matches no key reports no MFU. The bench's compute share uses
the probe's measured GEMM rate instead, so it does not depend on this table.
