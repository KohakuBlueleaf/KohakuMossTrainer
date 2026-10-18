# Logging and progress

KohakuMoss logs through stdlib `logging`: one logger per module, one line format,
configured once by the entry point.

```
[12:00:00] [inference.pipeline] [INFO] estimated peak 9.60 GiB while sampling [dit_gib=6.2, tokens=8704]
[12:00:03] [rank1] [training.entry] [INFO] training from step 0 to 1000
```

- **Line.** `[HH:MM:SS] [module.name] [LEVEL] message [key=value, ...]`. The module
  column drops the package root (`kohakumoss.inference.pipeline` shows as
  `inference.pipeline`, `kohakumoss_kernels.tuning.tuner` as `tuning.tuner`) and is
  trimmed to 26 characters with a leading `...` only when still longer; `[rankN]`
  follows the time in a distributed run. Floats in fields print with four significant digits.
- **Colour.** Every token has its own colour, never one colour over a line: time grey
  (`90`), rank blue (`34`), module magenta (`35`), the level token alone in its level's
  colour (DEBUG cyan `36`, INFO green `32`, WARNING yellow `33`, ERROR red `31`,
  CRITICAL bold white on red `1;37;41`). The message keeps the terminal's colour; at
  WARNING and above it is tinted to the level colour (CRITICAL: red). Field keys are dim
  cyan (`2;36`), `=` plain, values bold (`1`), numbers and quantities with a unit
  (GiB, MiB, s, ms, it/s, s/step, tokens, x, %) bright yellow (`93`), paths underlined
  blue (`4;34`). Every span ends with a reset (`0`).
- **Highlights.** `hl(text, kind="value")` wraps a span of a message: `value` picks the
  field rule (number yellow, path blue, else bold), `path`, `ok` (bright green), `bad`
  (bright red). It returns plain text unless the console handler of the last `setup()`
  writes colour; the file handler strips every span, so log files stay plain:
  `log.info(f"loaded {hl(path, 'path')} in {hl(f'{secs:.1f} s')}")`. The load, plan,
  peak, tuning-progress, generate and step lines use it and carry their numbers as
  spans or fields.
- **Library code** only calls `get_logger(__name__)` (from `kohakumoss_common.log`, or
  `kohakumoss.logging.log`, which adds the bars); importing a package configures
  no handler. Until an entry point calls `setup()`, records propagate to Python's root
  logger (so an embedding application's own logging configuration receives them).
- **Entry points** (`kogine` scripts, `kohakumoss-config`, `kohakumoss-tune`, the
  trainer and its DDP workers) call `setup(level=None, stream=None, path=None,
  rank=None, world=1)` once. It is idempotent: it replaces the handlers it attached
  before. With `world` > 1 lines carry `[rankN]` and only rank 0 writes. It attaches a stderr handler that flushes
  every record (and writes replacement characters on a console whose code page cannot
  carry a character) to the three package roots `kohakumoss`, `kohakumoss_kernels`,
  `kohakumoss_data`, and stops their propagation. `path` adds an uncoloured file
  handler; `add_log_file(path)` adds one later, at the level `setup` set (the trainer's
  rank 0 writes `<output>/log.txt` from the moment its run directory exists).
  `set_level(level)` changes the level later, handlers included.
- **Structured fields.** `log.info("transformer loaded", gib=9.6, secs=27.1)`: keyword
  arguments become record attributes and are appended to the line; a name that clashes
  with a `LogRecord` attribute (`name`, `msg`, `module`, ...) gets a `_` suffix. The
  record's file and line are the caller's, also through `timed(log, what)`.
- **Records and tables, never a repr.** `log_object(log, title, obj, only=None,
  suffix="")` logs a dataclass or mapping (a MemoryPlan, a TextEncoderPlan, a knob set)
  as one header line, `title: <its profile / memory / name / kind>` plus the suffix
  (`(applied in 17.9s)`), then one aligned `key  value` line per field, nested records
  and mappings indented. Fields equal to their default are hidden at INFO behind an
  `(n defaults hidden)` line and shown at DEBUG. `log_table(log, title, headers, rows)`
  logs an ASCII table (`|` columns, a `-+-` rule, numeric columns right-aligned): the
  placement summary, the estimated-peak parts, the per-op prewarm times. `render(obj)`
  and `table(headers, rows)` return the lines. Keys are dim cyan, values in the field
  colours, `True` green and `False` dim; without a colour console (and in the file
  handler) the lines are plain.

  ```
  [12:00:00] [inference.loading] [INFO] DiT memory plan: balanced on cuda:0 (applied in 17.9s)
    checkpoint      none
    base_precision  fp8_compute
    (12 defaults hidden)
  ```
- **One implementation.** `kohakumoss_common/log.py`, a torch-free package every other
  package imports. `get_logger` returns a wrapper around the stdlib logger of the name
  (the stdlib logger keeps its own class).

| setting | effect |
| --- | --- |
| `KOHAKUMOSS_LOG_LEVEL` | read by `setup()`: `DEBUG` / `INFO` (default) / `WARNING` / `ERROR` or a number |
| `FORCE_COLOR=1` | ANSI colours on a pipe or a file (a wrapper script that tees the output); wins over `NO_COLOR`, so an environment that exports `NO_COLOR` globally can still be overridden per command |
| `NO_COLOR` (any value) | no ANSI colours |
| stderr not a terminal | no ANSI colours |
| Windows console | `setup()` switches on VT processing for the stderr console |

Third-party floors (`NOISY_LIBRARY_LEVELS`), applied by `setup()` unless the level is
DEBUG: diffusers and transformers at ERROR, huggingface_hub, urllib3 and httpx at
WARNING.

## What a run logs

Inference (`MossPipeline`, `scripts/sample/sample.py`, `compare_grid.py`):

- each component as it reads (the transformer's tensors, GiB and seconds; the text
  encoder and VAE weights read into host RAM, with seconds; transformers' own loading
  bar is off), then one placement line per component naming where its weights live
  and where they compute, with sizes:

  ```
  transformer: <path> -> resident on cuda:0 (9.63 GiB, fp16, fp16 fold c=1/16 from metadata)
  transformer: <path> -> streamed from host, 16 of 42 blocks resident (4.10 GiB on cuda:0, 5.53 GiB host; bf16)
  text encoder: <path> -> stream (weights in host RAM, pinned, 14.80 GiB; ring of 2 layers on cuda:0, ~0.70 GiB, 0.20 GiB resident; precision fp8, bf16)
  VAE: <path> -> offload (host RAM, moved to cuda:0 per pass, 0.23 GiB)
  placement: DiT resident 9.63 GiB | TE stream 0.90 GiB on GPU / 14.80 GiB host | VAE offload; estimated GPU peak 12.10 GiB (one 1024x1024 image, CFG)
  ```

  (`kohakumoss.inference.placement_report`). Backend selection for host-side work
  (quantizing the text encoder's linears on the CPU) logs at DEBUG;
- backend selection, once per op, arch and dtype (`attention_varlen on sm89/bfloat16:
  backend triton`); a backend that fails logs one WARNING per rebind (`<op> on
  <arch>/<dtype>: backend <failed> failed, using <next> (<n> bound ops): <error>`), a
  backend skipped for a toolchain failure recorded by an earlier run logs `backend <b>
  skipped, failed here (<error>)`, and a text-encoder op whose probe fails logs `text
  encoder op <name> on <device>: backend <b> failed its probe, using eager: <error>`
  ([kernel-dev.md](../internals/kernel-dev.md#fallback));
- kernel decisions (`kohakumoss_kernels.tuning.decisions`): every tuned kernel and
  kernel-vs-compile problem is recorded once, when its config settles, from the card's
  tuning table, measured, or the untimed default (`TUNE="off"` with no stored row, or a
  single candidate).
  Per problem a DEBUG line names the key, the config and the timings
  (`choice_triton_rms_norm_mod 4096x3072bfloat16|...: compile (kernel 0.21 ms, compile
  0.17 ms), measured`). A provisional binding (the problem is queued for a retune, the
  own kernel runs meanwhile) is no decision and logs only `queued <kernel> <key>` at
  DEBUG. At INFO each batch of decisions is one table, logged after a prewarm, after a
  retune that measured something (training: the first `RETUNE_STEPS` steps; then
  `kernel tuning closed after step N`), at the end of training step 1 and after a
  generate. A key first met after the tuning windows binds untimed with one WARNING
  line (`met after the tuning window`):

  ```
  after a retune (0.41s): 12 kernel decision(s)
    kernel                     | from table | measured | default | chosen
    ---------------------------+------------+----------+---------+---------------------
    choice_triton_rms_norm_mod |          2 |        1 |       0 | compile x2, kernel x1
  ```

  After step 1 a newly measured problem (one the prewarm missed) logs one INFO line
  with its key, choice and timings; other decisions after step 1 stay at DEBUG. A
  generate binds stored decisions without a prewarm; under `TUNE="table"` its
  table lists the problems it measured on first use and after the call;
- prompt encoding (prompts, tokens, seconds), the sampling plan (images, batches, steps,
  cfg, sizes), a progress bar per batch over the sampler steps with the current sigma,
  the VAE decode (a bar when more than one image), the total time, and the generate's
  VRAM: peak allocated, peak reserved and device used (`GenerateResult.settings["vram"]`,
  also written to the sample script's metadata as `run.vram`);
- every file the sample script writes.

Training shows one tqdm bar over `STEPS` (rank 0) whose postfix updates every step:
the loss smoothed by an EMA (beta 0.9), lr, grad_norm, s/it, samples/s, tokens/s and
`vram` = the step's peak allocated / peak reserved / the device's use in GiB
(see [VRAM](#vram)); tqdm's own field shows the ETA. The bar is on when
`PROGRESS=True` (the default), stderr is a terminal or `FORCE_COLOR=1` is set, and the
level is INFO or lower. Log lines (the `LOG_EVERY` line, sampling, validation, saves)
print above it through `logging_redirect_tqdm`.

While the bar is shown the console prints no per-step line: the bar's postfix carries
those numbers, and event lines (sampling, validation, saves, exports, warnings, tuner
lines) still print above the bar. The console step line `step N [loss=, grad_norm=,
lr=]` prints only when the bar is off (no terminal and no `FORCE_COLOR`,
`PROGRESS=False`, or a level above INFO). The choice is made once at build: the run
gets either the bar or the StepLine callback.

| knob | default | what it sets |
| --- | --- | --- |
| `LOG_EVERY` | 10 | the console step line (bar off only); the default `METRICS_EVERY` |
| `METRICS_EVERY` | None (= `LOG_EVERY`) | metric rows to the sinks (`metrics.jsonl`, KohakuBoard, wandb) and the throughput, VRAM and encoder-share monitors that fill them; 1 = every step |
| `PROGRESS` | True | the bar |

`LOGGER` picks the sink next to `metrics.jsonl`, which every run writes: `console`
(no other sink; the console shows the bar or the step line), `board` (KohakuBoard under
`<output>/board`) or `wandb`. The wandb run starts in wandb's own default mode
(`WANDB_MODE` when set, else online), writing under the run directory; set
`WANDB_MODE=offline` to keep it local (and `wandb sync` it afterwards). Under every
logger the console shows exactly what it shows with `LOGGER="console"`: wandb runs
silent and captures no console output, and KohakuBoard writes its own log lines to
`<output>/board/kohakuboard.log` and captures no stdout / stderr.

A run that does not resume (`RESUME=""`, or `"latest"` with no checkpoint yet) into an
existing `<output>` first moves an earlier attempt's `metrics.jsonl`, `log.txt` and
`board/` aside to the first free `metrics.<n>.jsonl`, `log.<n>.txt`, `board.<n>`, with a
WARNING line each (`fresh run: an earlier metrics.jsonl in <output> moved to
<output>/metrics.1.jsonl`). A resumed run appends to them; rows of `metrics.jsonl`
past the resumed step (an attempt that went on after that checkpoint) move to
`metrics.<n>.jsonl` first (`resumed at step <s>: <n> later rows of <path> moved to
<path>`), so the file holds one row per step.

Both cadences also fire at the run's first step (after a resume: the first resumed
step) and at its last (a callback with `edges = True`). The cadence is resolved once at
build into each callback's `every`; the per-step cost is the runner's one integer
compare. Each emitted metric row costs one host sync (the step's device metrics are
read to floats) plus the monitors' host work, so `METRICS_EVERY=1` adds one sync per
step; the bar itself reads host values only.

Training also logs: the run summary, the model / adapters / encoders build with its memory plan,
the text encoder's and the VAE's placement lines after it (the inference lines above; a
VAE also as `stream (... 2-layer ring on cuda:0 ...)` or `cpu (... runs on the host)`;
`text encoder: not kept, TEXT_CACHE='offline'` / `VAE: not loaded, ...` when the run
holds none), the kernel prewarm (per-op progress lines, the wall time per op and in total) and the
adapter prewarm, the data loader build, resume, one line per `LOG_EVERY` steps,
step 1's s/step against the median of steps 2-10 (the prewarm's benefit), and each
export.

## Kernel tuning

| `TUNE` | effect |
| --- | --- |
| `table` (training default) | bind the card's stored decisions; a problem not in the table is tuned once and stored (training: in the prewarm before step 0, forward + backward on the planned shapes and dtype, workspaces released after; then only the text-encoder prewarm, the kernel-vs-compile retune of the first `RETUNE_STEPS` steps and the first sample grid's first sampler step; inference: on its first call, no prewarm) |
| `off` (inference default) | bind the card's stored decisions, else the default config; nothing is timed or stored |

`kohakumoss-tune --device cuda:0 --dtype bf16 --sizes 1024x1024 832x1216` (or
`kogine run scripts/bench/tune/tune.py`) fills the table once for a card, dtype and
size list (`--backward` for training shapes); the table lives under the kernel cache
(`KOHAKUMOSS_KERNEL_CACHE`, default `~/.cache/kohakumoss_kernels/tuning/<card>/`).

A pipeline that owns its model applies `MossPipeline.from_pretrained(tune=...)` as the
process's TUNE; a pipeline sharing the training model (`from_components`) uses the
trainer's. See [../internals/tuning.md](../internals/tuning.md#tune).

Bars go to stderr, like the log lines, and are hidden above INFO or with
`generate(..., progress=False)`; log lines printed while a bar is open go through
`tqdm.write` (`logging_redirect_tqdm` over the package roots), so neither breaks the
other.

## VRAM

The bar's `vram=9.6/10.7/12.7G` is, per step:

| number | meaning | source |
| --- | --- | --- |
| peak allocated | the step's highest live-tensor bytes | the VRAM timeline's step window on the peak accumulator |
| peak reserved | the step's highest bytes the torch caching allocator held from the driver | the same window |
| device | the device's use, as nvidia-smi shows it (every process, this one's CUDA / XPU context included) | the driver: `mem_get_info` (total - free), re-read at most once a second |

All three are host-side reads (one allocator-statistics read per window restart, one
`cudaMemGetInfo` per second); none synchronises the device.

Every peak in the project (the bar, the metric rows, the prewarm and step-1 report,
`generate`'s VRAM record, the bench rows) is a window on one process-wide peak
accumulator (`kohakumoss_kernels.hardware.memory`): `with window(device) as w` (or
`open_window` / `close_window`) gives the peak allocated / reserved bytes while the
window was open. Each open, read and close folds the allocator's peak counter into every
open window of the device before resetting it, so a nested measurement (a sampling
`generate` inside a training step, a prewarm replay inside a generate, the encoder's
phase marks) never cuts an enclosing one. Nothing else calls `reset_peak_memory_stats`
or reads `max_memory_*`.

Reports, benches and metadata carry the three labelled numbers: `peak_alloc` (live
tensors), `peak_reserved` (the torch allocator's hold) and `device_used` (the whole
device, every process: what nvidia-smi shows). A docs table whose column says only
"peak GiB" was measured with the allocated counter. On Windows (WDDM) nvidia-smi shows the whole process, which is
the third number plus other processes' background.

The gap between the device use and torch's numbers is logged as a table after the
build, after the prewarm and after step 1, and goes into the metric rows
(`vram_device`, `vram_other`, `vram_context`, `vram_non_torch`, `vram_fragmented`,
`vram_cached`, `vram_allocated`, GiB):

| part | what it holds |
| --- | --- |
| `other` | other processes (and the driver's own reservation) on the device, measured through NVML before this process's first device call; 0 with no NVML (pynvml), where `context` holds it |
| `context` | the CUDA / XPU context (driver and runtime), measured as what the first device call added; with no NVML it also holds `other` (the device's use at the first device call) |
| `non_torch` | device use minus other, context and torch reserved: loaded kernel modules (torch's own lazily loaded CUDA kernels, cuBLAS / cuBLASLt / cuDNN library state, Triton / TileLang cubins), NCCL buffers, anything allocated outside the torch caching allocator |
| `fragmented` | torch reserved, free inside split blocks (`inactive_split_bytes`) |
| `cached` | torch reserved, free whole blocks (returned by `empty_cache`) |
| `allocated` | live tensors |

The cuBLAS / cuBLASLt workspaces are not in `non_torch`: torch allocates them from its
caching allocator, so they count in `allocated` (torch 2.14, V100: the first fp16
1024x1024 matmul adds 8.1 MiB allocated, which `torch._C._cuda_clearCublasWorkspaces()`
frees, and 10 MiB of device use outside the allocator for the library's kernels and
handle). The parts sum to `device` by construction, and `device` is the same driver
number as the bar's third value and `device_used`. If another process frees or takes
memory after `other` was measured, the difference lands in `non_torch` (which can then
go negative). The trainer returns no `cached` memory to the driver between steps
([../internals/offload.md](../internals/offload.md#between-step-release)): the
allocator reuses it in the next step, so the tables after step 1 show the steady
reserved memory of the run.

## Transfer

Runs with gradient offload or a host-stepped weight stream also add
`grad_offload_wait_s`: host seconds per step backward waited for a busy slot of the
shared gradient staging ring (`HostAccumulator`, 4 slots), measured only when a slot is
busy; a value near the step time means the host adds bound backward.

Runs with DiT-side offload add `xfer_h2d_gb` / `xfer_d2h_gb` to the metric rows: GB moved
per step between host and device, summed over the engines that count their bytes
(weight streaming, gradient offload, activation offload, the host-state and CPU AdamW
optimisers). The counters are host integers; reading them costs no device sync. The
text encoder and VAE report their own traffic (`te_h2d_gb`, `vae_h2d_gb`). Divided by
the step time and set against the link's measured bandwidth they say how much of the
step the transfers need (the bench matrix report's transfer share,
`scripts/bench/matrix/`).

## Sample output files

`scripts/sample/sample.py` numbers its images after the highest `NNNN_` index already
in `OUT` and writes one `metadata-<first index>.json` per run, so reusing `OUT` keeps
every earlier image and its settings.
