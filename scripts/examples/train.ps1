# Example training run: a short LoRA / LoKr run on your dataset, from a generated config.
#
#   .\scripts\examples\train.ps1 -Weights ComfyUI\models -Dataset D:\data\my_set
#   .\scripts\examples\train.ps1 -Algo lokr -Precision fp16 -BasePrecision int8 -Memory minimal
#   .\scripts\examples\train.ps1 -Steps 30 -SampleEvery 10        # quick run with sample images
#   .\scripts\examples\train.ps1 -Device xpu -Index 1              # second Intel Arc GPU
#   .\scripts\examples\train.ps1 -TuneOnly                         # tune the training shapes, no weights loaded
#   .\scripts\examples\train.ps1 -DryRun                           # write and check the config, do not train
#
# Run it from the activated venv (kogine, kohakumoss-tune and python on PATH). -Weights and
# -Dataset default to $env:KOHAKUMOSS_WEIGHTS / $env:KOHAKUMOSS_DATASET. The run writes its
# generated config, logs, checkpoints, exported adapters and samples under
# <Out>\<yyyyMMdd-HHmmss>-<device>-<algo>-<precision>\. Environment variables set for the
# device are restored when the script exits. See scripts/examples/README.md.

[CmdletBinding()]
param(
    # Device kind: cuda (NVIDIA) | xpu (Intel Arc).
    [ValidateSet("cuda", "xpu")] [string] $Device = "cuda",
    # GPU index within that kind: cuda -> CUDA_VISIBLE_DEVICES (nvidia-smi order), xpu -> ZE_AFFINITY_MASK.
    [int] $Index = 0,
    # Weights: a ComfyUI models folder (diffusion_models\, text_encoders\, vae\) or an HF folder (transformer\, text_encoder\, vae\).
    [string] $Weights = $env:KOHAKUMOSS_WEIGHTS,
    # The fp16-fix checkpoint, used with -FixFile (single-file transformer with the fold in its metadata).
    [string] $FixCheckpoint = "",
    # Dataset folder.
    [string] $Dataset = $env:KOHAKUMOSS_DATASET,
    # Dataset layout: kohya (<repeats>_<name>\ folders + .txt tag sidecars) | folder (image + .txt pairs).
    [ValidateSet("kohya", "folder")] [string] $Source = "kohya",
    # Adapter: lora = LoRA of rank -Dim; lokr = LoKr full matrix, factor -Factor (needs a higher LR).
    [ValidateSet("lora", "lokr")] [string] $Algo = "lora",
    # LoRA rank (ignored for lokr, which uses a full-matrix factorization).
    [int] $Dim = 8,
    # LoKr Kronecker factor.
    [int] $Factor = 16,
    # Learning rate; 0 = the algo's short-run default (lora 5e-4, lokr 2e-3).
    [double] $Lr = 0,
    # Training steps.
    [int] $Steps = 100,
    # Precision: bf16 | fp16 (fp16 uses the online residual fix, or the fix checkpoint with -FixFile).
    [ValidateSet("bf16", "fp16")] [string] $Precision = "bf16",
    # fp16 from the fp16-fix checkpoint instead of the online fix (text encoder / VAE still from -Weights).
    [switch] $FixFile,
    # DiT base: auto = what the DiT file stores (the ComfyUI int8 file: int8); compute = bf16 / fp16; int8 / fp8 = weight-only; fp8_compute / mxfp8 = fp8 GEMMs (sm89+).
    [ValidateSet("auto", "compute", "int8", "fp8", "fp8_compute", "mxfp8")] [string] $BasePrecision = "auto",
    # Memory profile (weights stay at -BasePrecision). Any catalog name, built
    # from its parts in this order (omit a part for its default):
    #   checkpoint: nockpt | ckpt | subckpt
    #   weights:    resident | ring16 | ring8 | ring4 | ring2   (blocks kept on the GPU)
    #   activations: actoff                                             (offload saved activations)
    #   optimizer:  optfused | optstream | optcpu                        (default: GPU AdamW)
    #   gradients:  gradoff                                             (offload gradients)
    #   e.g. ckpt-ring8, nockpt-resident-optcpu, ckpt-ring2-actoff-optstream-gradoff
    #   (the older spellings stream16 .. stream2 are accepted too)
    # Short names: speed, balanced, low, minimal; auto = the fastest whose estimate fits the GPU.
    # Full list: kohakumoss-config list. "manual" = no profile: the plan comes from -Checkpoint ...
    # -FusedStep below (unset ones take their default).
    [string] $Memory = "auto",
    # The manual knobs (only with -Memory manual; with a profile they must stay "" / 0).
    # Gradient checkpointing: none | block | sub-block | selective | partial:<fraction>   [block]
    [string] $Checkpoint = "",
    # Block weights: off (all on the GPU) | ring (streamed from host RAM through GPU slots)   [off]
    [ValidateSet("", "off", "ring")] [string] $Stream = "",
    # GPU block slots when streaming (>= 2); 0 = default.
    [int] $StreamResident = 0,
    # Pinned host memory for every offload: on | off (pageable, staged copies)   [on]
    [ValidateSet("", "on", "off")] [string] $Pinned = "",
    # Saved activations offloaded to host RAM between forward and backward: on | off   [off]
    [ValidateSet("", "on", "off")] [string] $ActOffload = "",
    # AdamW state: gpu | stream (host, streamed per parameter) | cpu (host update)   [gpu]
    [ValidateSet("", "gpu", "stream", "cpu")] [string] $OptimState = "",
    # Gradients summed in host fp32: on | off   [off]
    [ValidateSet("", "on", "off")] [string] $GradOffload = "",
    # Optimizer step per parameter inside backward: on | off   [off]
    [ValidateSet("", "on", "off")] [string] $FusedStep = "",
    # fp8 base scaling (with -BasePrecision fp8_compute): row | tensor | block.
    [ValidateSet("row", "tensor", "block")] [string] $Fp8Recipe = "row",
    # fp8 input-gradient GEMM: fp8 (keeps a second fp8 weight copy, fastest) | 16bit (no copy, less VRAM).
    [ValidateSet("fp8", "16bit")] [string] $Fp8Dgrad = "16bit",
    # Target image area in pixels (1048576 = 1 MP = 4096 image tokens; 589824 = 768^2).
    [int] $TargetArea = 1048576,
    # Joint (image + text) tokens per packed batch.
    [int] $MaxTokens = 8192,
    # Text encoder during training: auto (fastest that fits) | ring (layers from host RAM; "stream" is accepted too) | offload | resident.
    [ValidateSet("auto", "ring", "stream", "offload", "resident")] [string] $TextMemory = "auto",
    # Caches: off = no cache, the VAE and text encoder encode every batch in the loop and nothing is
    # written | online = encode on a miss and write latents / text embeddings to the cache dir on disk
    # (later runs read them back) | offline = read a prebuilt cache only.
    [ValidateSet("off", "online", "offline")] [string] $Cache = "off",
    # Cache directory ("" = <dataset>\cache).
    [string] $CacheDir = "",
    # Sample images every N steps (0 = off).
    [int] $SampleEvery = 0,
    # Sample prompt (used when -SampleEvery > 0).
    [string] $SamplePrompt = "a lighthouse on a rocky coast at sunset",
    # Save a checkpoint every N steps (0 = only at the end).
    [int] $SaveEvery = 0,
    # Kernel tuning: table = stored per-card decisions, missing ones tuned in the prewarm before step 0
    # and between steps, then stored | off = stored decisions, else fixed defaults; nothing timed.
    [ValidateSet("table", "off")] [string] $Tune = "table",
    # Only tune: kohakumoss-tune on the training shapes (no weights loaded; forward + backward, -Precision,
    # a square of -TargetArea packed to -MaxTokens, frozen base) and exit; later runs bind the results.
    [switch] $TuneOnly,
    # Kernel backend per op, e.g. -Kernel gated_residual=eager (default: the measured choice).
    [string[]] $Kernel = @(),
    # Seed.
    [int] $Seed = 0,
    # Full log line (and logger sinks) every N steps; between them the progress bar shows loss / lr / speed / VRAM.
    [int] $LogEvery = 10,
    # Metrics sink: console | wandb (needs `wandb login` once).
    [ValidateSet("console", "wandb")] [string] $Logger = "console",
    # Project name for wandb.
    [string] $Project = "kohakumoss",
    # Output root (relative paths are under the current directory).
    [string] $Out = "runs\examples\train",
    # Write and check the generated config and print the command, without training.
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$script = Join-Path $repo "scripts\train\moss.py"
$cfgRoot = Join-Path $repo "configs\moss"

# device selection is process environment; save it here and restore it in the finally below
$envNames = @("CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "ZE_AFFINITY_MASK", "PYTHONUTF8")
$envSaved = @{}
foreach ($n in $envNames) { $envSaved[$n] = [Environment]::GetEnvironmentVariable($n) }

# Python literals for the generated config (forward slashes keep paths free of escapes)
function PyStr([string] $s) {
    $e = $s -replace '\\', '\\' -replace "'", "\'" -replace "`r", '\r' -replace "`n", '\n'
    return "'" + $e + "'"
}
function PyPath([string] $p) { return PyStr ($p -replace "\\", "/") }
function PyBool([string] $s) { if ($s -eq "on") { "True" } else { "False" } }
function FullPath([string] $p) { return $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($p) }
function Need([string] $tool) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "$tool not on PATH: activate the KohakuMoss venv first"
    }
}

try {
    switch ($Device) {
        "cuda" {
            $env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"
            $env:CUDA_VISIBLE_DEVICES = "$Index"
        }
        "xpu" {
            $env:ZE_AFFINITY_MASK = "$Index"
            $env:PYTHONUTF8 = "1"
        }
    }

    if ($TuneOnly) {
        Need "kohakumoss-tune"
        # one square image of -TargetArea per sample (16 px per token), as many samples as -MaxTokens packs
        $side = [int]([math]::Floor([math]::Sqrt($TargetArea) / 16) * 16)
        $sample = ($side / 16) * ($side / 16) + 512
        $batch = [math]::Max(1, [int][math]::Floor($MaxTokens / $sample))
        Write-Host "== tune $Device $Precision ${side}x${side}, $batch per batch (forward + backward)"
        & kohakumoss-tune --device $Device --dtype $Precision --sizes "${side}x${side}" --batch $batch --backward --trainable adapter
        if ($LASTEXITCODE -ne 0) { throw "tuning failed, exit $LASTEXITCODE" }
        return
    }

    Need "kogine"
    Need "python"
    if (-not $Weights) { throw "no weights: pass -Weights <dir> or set KOHAKUMOSS_WEIGHTS" }
    if (-not $Dataset) { throw "no dataset: pass -Dataset <dir> or set KOHAKUMOSS_DATASET" }
    if (-not (Test-Path $Weights)) { throw "weights root not found: $Weights" }
    if (-not (Test-Path $Dataset)) { throw "dataset not found: $Dataset" }
    if ($FixFile -and $Precision -ne "fp16") { throw "-FixFile needs -Precision fp16" }
    if ($FixFile -and -not $FixCheckpoint) { throw "-FixFile needs -FixCheckpoint <fp16-fix .safetensors>" }
    if ($FixFile -and -not (Test-Path $FixCheckpoint)) { throw "fp16-fix checkpoint not found: $FixCheckpoint" }
    $Weights = FullPath $Weights
    $Dataset = FullPath $Dataset
    $Out = FullPath $Out

    $manual = @()
    if ($Checkpoint) { $manual += "CHECKPOINT = $(PyStr $Checkpoint)" }
    if ($Stream) { $manual += "STREAM = '$Stream'" }
    if ($StreamResident -gt 0) { $manual += "STREAM_RESIDENT = $StreamResident" }
    if ($Pinned) { $manual += "PINNED = $(PyBool $Pinned)" }
    if ($ActOffload) { $manual += "ACT_OFFLOAD = $(PyBool $ActOffload)" }
    if ($OptimState) { $manual += "OPTIM_STATE = '$OptimState'" }
    if ($GradOffload) { $manual += "GRAD_OFFLOAD = $(PyBool $GradOffload)" }
    if ($FusedStep) { $manual += "FUSED_STEP = $(PyBool $FusedStep)" }
    if ($manual -and $Memory -ne "manual") {
        throw "-Checkpoint / -Stream / ... set the plan only with -Memory manual (got -Memory $Memory)"
    }

    $runId = Get-Date -Format "yyyyMMdd-HHmmss"
    $tag = "$runId-$Device-$Algo-$Precision" + $(if ($FixFile) { "-fixfile" } else { "" })
    $runDir = Join-Path $Out $tag
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null

    $lrValue = if ($Lr -gt 0) { $Lr } elseif ($Algo -eq "lokr") { 2e-3 } else { 5e-4 }
    $cacheLine = if ($CacheDir) { "CACHE_DIR = $(PyPath (FullPath $CacheDir))" } else { "" }
    $dataset = if ($Source -eq "kohya") {
        "{'source': 'kohya', 'path': $(PyPath $Dataset), 'repeat_prefix': 'first', 'artist_from_folder': 'name', 'txt_as': 'tags'}"
    } else {
        "{'source': 'folder', 'path': $(PyPath $Dataset)}"
    }
    $algoLines = if ($Algo -eq "lokr") {
        "use_config($(PyPath (Join-Path $cfgRoot 'algo\algo_lokr.py')))`nDIM = 10000`nALPHA = 1.0`nALGO_KWARGS = {'factor': $Factor, 'full_matrix': True}"
    } else {
        "use_config($(PyPath (Join-Path $cfgRoot 'algo\algo_lora.py')))`nDIM = $Dim`nALPHA = float($Dim)"
    }
    $precisionLines = if ($Precision -eq "fp16" -and $FixFile) {
        "PRECISION = 'fp16'`nFP16_FIX = 'none'`nWEIGHTS = $(PyPath $Weights)`nTRANSFORMER = $(PyPath (FullPath $FixCheckpoint))"
    } elseif ($Precision -eq "fp16") {
        "use_config($(PyPath (Join-Path $cfgRoot 'precision\precision_fp16.py')))`nWEIGHTS = $(PyPath $Weights)"
    } else {
        "use_config($(PyPath (Join-Path $cfgRoot 'precision\precision_bf16.py')))`nWEIGHTS = $(PyPath $Weights)"
    }
    $overrideLine = (@(
        "BASE_PRECISION = '$BasePrecision'", "FP8_RECIPE = '$Fp8Recipe'", "FP8_DGRAD = '$Fp8Dgrad'"
    ) + $manual) -join "`n"
    $kernelPairs = foreach ($k in $Kernel) {
        $op, $backend = $k -split "=", 2
        if (-not $backend) { throw "-Kernel takes op=backend, got $k" }
        "$(PyStr $op): $(PyStr $backend)"
    }
    $kernelLine = if ($kernelPairs) { "KERNELS = {" + ($kernelPairs -join ", ") + "}" } else { "" }
    $sampleLines = if ($SampleEvery -gt 0) {
        "SAMPLE_EVERY = $SampleEvery`nSAMPLE_PROMPTS = [$(PyStr $SamplePrompt)]`nSAMPLE_STEPS = 20`nSAMPLE_SIZE = (1024, 1024)"
    } else { "SAMPLE_EVERY = 0" }
    $warmup = [math]::Min($Steps - 1, [math]::Min(20, [math]::Max(1, [int]($Steps / 10))))

    $config = @"
"""Generated by scripts/examples/train.ps1 ($tag); drives scripts/train/moss.py."""

from kohakuengine import use_config

use_config($(PyPath (Join-Path $cfgRoot '_base\base_lora.py')))
$algoLines
$precisionLines

DATASETS = [$dataset]
$cacheLine
LATENT_CACHE = '$Cache'
TEXT_CACHE = '$Cache'
TEXT_ENCODE = {'memory': '$TextMemory', 'ahead': 8}
MEMORY_PROFILE = $(PyStr $Memory)
$overrideLine
$kernelLine
TARGET_AREA = $TargetArea
MAX_TOKENS = $MaxTokens
LR = $lrValue
OPTIMIZER = 'adamw'
STEPS = $Steps
SCHEDULE = {'mode': 'constant', 'warmup': $warmup, 'end': -1}
SAVE_EVERY = $SaveEvery
KEEP_LAST = 0
SEED = $Seed
TUNE = '$Tune'
LOG_EVERY = $LogEvery
LOGGER = '$Logger'
PROJECT = $(PyStr $Project)
$sampleLines
OUTPUT_DIR = $(PyPath $Out)
RUN_NAME = '$tag'
"@
    $cfgPath = Join-Path $runDir "config.py"
    [IO.File]::WriteAllText($cfgPath, $config, (New-Object Text.UTF8Encoding $false))

    Write-Host "== $Device $Algo $Precision$(if ($FixFile) {' (fix file)'}) -> $runDir"
    Write-Host "   config: $cfgPath"
    # validate every knob against the trainer's rules before loading any weights
    Push-Location $repo
    try {
        & python -m kohakumoss.cli.config.main check $cfgPath
        $checked = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    if ($checked -ne 0) { throw "config check failed for $cfgPath" }
    if ($DryRun) {
        Write-Host "   dry run: kogine run $script --config $cfgPath"
        return
    }
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & kogine run $script --config $cfgPath
    $code = $LASTEXITCODE
    Write-Host ("   wall {0:N1} s, exit {1}" -f $sw.Elapsed.TotalSeconds, $code)
    if ($code -ne 0) { throw "training run failed ($tag), exit $code" }
} finally {
    foreach ($n in $envNames) { [Environment]::SetEnvironmentVariable($n, $envSaved[$n]) }
}
