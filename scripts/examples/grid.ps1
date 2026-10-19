# Example comparison grid: the base model vs trained adapters (LoRA / LoKr exports or checkpoints).
# Columns = prompts, rows = base, then each adapter at each multiplier (-Layout prompt-rows swaps
# them). Same seed per prompt across rows.
#
#   .\scripts\examples\grid.ps1 -Adapter runs\examples\train\<run>\export\<run>-lycoris.safetensors -Prompt "a lighthouse at dusk"
#   .\scripts\examples\grid.ps1 -Adapter a.safetensors,b.safetensors -Multiplier 0.5,1.0 -Prompt "a cat","a dog"
#   .\scripts\examples\grid.ps1 -Run <run> -Prompt "a cat"                  # the train example run's lycoris export
#   .\scripts\examples\grid.ps1 -Run <run> -Checkpoints -Prompt "a cat"     # every saved checkpoint of the run
#   .\scripts\examples\grid.ps1 -Run <run> -Dataset <dir> -Artist <name>    # captions of that artist's folder
#
# Run it from the activated venv (kogine and python on PATH). The prompts come from
# scripts/examples/grid_prompts.py, run with a generated prompts_config.py. -Weights defaults to
# $env:KOHAKUMOSS_WEIGHTS, -Dataset to $env:KOHAKUMOSS_DATASET. Output:
# <Out>\<yyyyMMdd-HHmmss>-<device>-grid\ with grid.png, every single image (r<row>_c<col>.png),
# config.py and metadata.json. Environment variables set for the device are restored when the
# script exits. See scripts/examples/README.md.

[CmdletBinding()]
param(
    # Device kind: cuda (NVIDIA) | xpu (Intel Arc).
    [ValidateSet("cuda", "xpu")] [string] $Device = "cuda",
    # GPU index within that kind: cuda -> CUDA_VISIBLE_DEVICES (nvidia-smi order), xpu -> ZE_AFFINITY_MASK.
    [int] $Index = 0,
    # Weights: a ComfyUI models folder or an HF weights root; text encoder, VAE and tokenizer come from here.
    [string] $Weights = $env:KOHAKUMOSS_WEIGHTS,
    # Arch preset: base | sft.
    [ValidateSet("base", "sft")] [string] $Variant = "base",
    # DiT override ("" = the one in -Weights); e.g. an fp16-fix file for -Precision fp16.
    [string] $Transformer = "",
    # Adapter files; every adapter gets one grid line per multiplier.
    [string[]] $Adapter = @(),
    # A train example run name under -TrainOut: adds its export\<run>-lycoris.safetensors (or, with
    # -Checkpoints, every checkpoints\step_*\trainable.safetensors).
    [string] $Run = "",
    # With -Run: use every saved checkpoint of the run (step lines) instead of the final export.
    [switch] $Checkpoints,
    # Root of the train example runs.
    [string] $TrainOut = "runs\examples\train",
    # Multipliers; every adapter gets one grid line per multiplier.
    [double[]] $Multiplier = @(1.0),
    # How adapters are applied: bypass = beside each layer (works on fp8 / quantized bases) |
    # merge = folded into the weights (16-bit bases). Undone between lines.
    [ValidateSet("bypass", "merge")] [string] $AdapterMode = "bypass",
    # Prompt bodies; each artist tag of -Artist is prepended ("<artist>, <prompt>").
    [string[]] $Prompt = @(),
    # Kohya-style dataset (<repeats>_..._<artist>\ folders + .txt tags), read for captions when -Prompt is empty.
    [string] $Dataset = $env:KOHAKUMOSS_DATASET,
    # Artist tags, as the dataset's folder names without the numeric prefix; "-" = no artist tag.
    # Training captions are "<artist>, <tags>" with underscores as spaces, so the prompts are too.
    [string[]] $Artist = @(),
    # Captions per artist when -Prompt is empty: tag lists from random .txt files of its folder
    # (picked with -Seed, so the same prompts every run).
    [int] $Captions = 1,
    # Text appended to every prompt (e.g. "masterpiece, best quality"); "" = none.
    [string] $Suffix = "",
    # Negative prompt of every image.
    [string] $Negative = "blurry, distortion, lowres",
    # Image size HxW.
    [string] $Size = "1024x1024",
    # Sampler steps.
    [int] $Steps = 30,
    # Classifier-free guidance scale.
    [double] $Cfg = 3.0,
    # Seed of prompt 0; prompt i uses -Seed + i (the same for every line); also picks the captions.
    [int] $Seed = 0,
    # Compute dtype: bf16 | fp16 (fp16 needs -Transformer <fp16-fix file> or -Fp16Fix online).
    [ValidateSet("bf16", "fp16")] [string] $Precision = "bf16",
    # fp16 residual fix when no -Transformer is given: online | fp32_residual | none.
    [ValidateSet("online", "fp32_residual", "none")] [string] $Fp16Fix = "online",
    # DiT base: compute = 16-bit; int8 / fp8 = weight-only; fp8_compute / mxfp8 = fp8 GEMMs (sm89+).
    [ValidateSet("auto", "compute", "int8", "fp8", "fp8_compute", "mxfp8")] [string] $BasePrecision = "auto",
    # DiT fp8_compute scaling: row | tensor | block.
    [ValidateSet("row", "tensor", "block")] [string] $Fp8Recipe = "row",
    # DiT memory profile: auto = the fastest profile that fits the card's free memory | minimal (2 resident
    # blocks, the rest streamed; fits 8 GB) | low | balanced | any catalog name | "" = whole DiT on the GPU.
    [string] $Memory = "auto",
    # Text encoder: auto | resident | offload | ring (layers from host RAM, ~2 GB; "stream" is accepted too) | cpu.
    [ValidateSet("auto", "resident", "offload", "ring", "stream", "cpu")] [string] $TextMemory = "auto",
    # Text-encoder precision: compute | int8 | fp8 | fp8_compute | mxfp8.
    [ValidateSet("compute", "int8", "fp8", "fp8_compute", "mxfp8")] [string] $TextPrecision = "compute",
    # Free the text encoder after the one encode of every prompt.
    [switch] $FreeTextEncoder,
    # VAE placement: auto | resident | offload | ring (block rings; "stream" is accepted too) | cpu.
    [ValidateSet("auto", "resident", "offload", "ring", "stream", "cpu")] [string] $VaeMemory = "auto",
    # VAE dtype: auto | bf16 | fp16 | fp32.
    [ValidateSet("auto", "bf16", "fp16", "fp32")] [string] $VaeDtype = "auto",
    # Prompts per generate batch (one packed DiT forward per step for the whole batch).
    [int] $Batch = 4,
    # Kernel tuning: off = stored per-card decisions, else fixed defaults; nothing timed |
    # table = missing decisions also tuned on first use, then stored.
    [ValidateSet("off", "table")] [string] $Tune = "off",
    # Grid layout: prompt-cols (prompts head the columns, weights the rows) | prompt-rows.
    [ValidateSet("prompt-cols", "prompt-rows")] [string] $Layout = "prompt-cols",
    # Label font size in pixels; 0 = scaled to the image size.
    [int] $FontSize = 0,
    # Output root (relative paths are under the current directory).
    [string] $Out = "runs\examples\grid",
    # Write and check config.py and stop (no model load).
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$script = Join-Path $repo "scripts\sample\compare_grid.py"
$helper = Join-Path $PSScriptRoot "grid_prompts.py"

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
    Need "kogine"
    Need "python"
    if (-not $Weights) { throw "no weights: pass -Weights <dir> or set KOHAKUMOSS_WEIGHTS" }
    if (-not (Test-Path $Weights)) { throw "weights root not found: $Weights" }
    if ($Transformer -and -not (Test-Path $Transformer)) { throw "transformer not found: $Transformer" }
    $Weights = FullPath $Weights
    if ($Transformer) { $Transformer = FullPath $Transformer }
    $Out = FullPath $Out

    # grid lines: explicit adapters, then the run's export or its checkpoints
    $files = @()
    $labels = @()
    foreach ($a in $Adapter) { $files += FullPath $a; $labels += [IO.Path]::GetFileNameWithoutExtension($a) }
    if ($Run) {
        $runDir = Join-Path (FullPath $TrainOut) $Run
        if (-not (Test-Path $runDir)) { throw "run not found: $runDir" }
        if ($Checkpoints) {
            $ckpts = @(Get-ChildItem (Join-Path $runDir "checkpoints") -Directory -Filter "step_*" | Sort-Object Name)
            if (-not $ckpts) { throw "no checkpoints in $runDir (the run saved none: -SaveEvery 0)" }
            foreach ($c in $ckpts) {
                $f = Join-Path $c.FullName "trainable.safetensors"
                if (Test-Path $f) { $files += $f; $labels += ("step " + [int]($c.Name -replace "step_", "")) }
            }
        } else {
            $f = Join-Path $runDir "export\$Run-lycoris.safetensors"
            if (-not (Test-Path $f)) { throw "export not found: $f" }
            $files += $f
            $labels += ($Run -replace "^\d{8}-\d{6}-", "")
        }
    }
    if (-not $files) { throw "no adapters: pass -Adapter or -Run" }
    foreach ($f in $files) { if (-not (Test-Path $f)) { throw "adapter not found: $f" } }

    # one grid line per (adapter, multiplier); labels carry the multiplier when there are several
    $cols = @()
    $colLabels = @()
    for ($i = 0; $i -lt $files.Count; $i++) {
        foreach ($m in $Multiplier) {
            $cols += "($(PyPath $files[$i]), $m)"
            $colLabels += $(if ($Multiplier.Count -gt 1) { PyStr "$($labels[$i]) x$m" } else { PyStr $labels[$i] })
        }
    }

    $runId = Get-Date -Format "yyyyMMdd-HHmmss"
    $dir = Join-Path $Out "$runId-$Device-grid"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null

    # prompts: per artist, the -Prompt bodies or -Captions captions of its dataset folder
    $artists = ($Artist | ForEach-Object { PyStr $(if ($_ -eq "-") { "" } else { $_ }) }) -join ", "
    $bodies = ($Prompt | ForEach-Object { PyStr $_ }) -join ", "
    $promptsJson = Join-Path $dir "prompts.json"
    $promptsCfg = Join-Path $dir "prompts_config.py"
    $promptsText = @"
"""Generated by scripts/examples/grid.ps1 ($runId); drives scripts/examples/grid_prompts.py."""

DATASET = $(PyPath $(if ($Dataset) { FullPath $Dataset } else { "" }))
ARTISTS = [$artists]
PROMPTS = [$bodies]
CAPTIONS = $Captions
SEED = $Seed
SUFFIX = $(PyStr $Suffix)
OUT = $(PyPath $promptsJson)
"@
    [IO.File]::WriteAllText($promptsCfg, $promptsText, (New-Object Text.UTF8Encoding $false))
    & kogine run $helper --config $promptsCfg
    if ($LASTEXITCODE -ne 0) { throw "no prompts (grid_prompts.py exit $LASTEXITCODE)" }
    $rows = [IO.File]::ReadAllText($promptsJson).Trim()

    $fix = if ($Precision -eq "fp16" -and -not $Transformer) { $Fp16Fix } else { "none" }
    $fixKw = if ($fix -eq "online") { "{'c': 0.0625}" } else { "{}" }
    $text = "{'memory': '$TextMemory', 'precision': '$TextPrecision'"
    $text += $(if ($FreeTextEncoder) { ", 'free_after_use': True}" } else { "}" })
    $config = @"
"""Generated by scripts/examples/grid.ps1 ($runId); drives scripts/sample/compare_grid.py."""

WEIGHTS = $(PyPath $Weights)
VARIANT = '$Variant'
TRANSFORMER = $(PyPath $Transformer)
CHECKPOINTS = [$($cols -join ', ')]
LABELS = [$($colLabels -join ', ')]
ADAPTER_MODE = '$AdapterMode'
PRECISION = '$Precision'
FP16_FIX = '$fix'
FP16_FIX_KWARGS = $fixKw
TUNE = '$Tune'

DEVICE = '$Device'
TEXT_ENCODE = $text
VAE_ENCODE = {'memory': '$VaeMemory'}
VAE_DTYPE = '$VaeDtype'
MEMORY_PROFILE = $(PyStr $Memory)
BASE_PRECISION = '$BasePrecision'
FP8_RECIPE = '$Fp8Recipe'

PROMPTS = $rows
NEGATIVE = $(PyStr $Negative)
SIZES = [$(PyStr $Size)]
SEED_BASE = $Seed

STEPS = $Steps
CFG = $Cfg
BATCH = $Batch

OUT = $(PyPath $dir)
GRID_LAYOUT = '$Layout'
FONT_SIZE = $FontSize
"@
    $cfgPath = Join-Path $dir "config.py"
    [IO.File]::WriteAllText($cfgPath, $config, (New-Object Text.UTF8Encoding $false))
    Write-Host "== $Device grid: $($cols.Count + 1) weight lines -> $dir"
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
    if ($LASTEXITCODE -ne 0) { throw "grid run failed, exit $LASTEXITCODE" }
    Write-Host ("   wall {0:N1} s; grid at {1}" -f $sw.Elapsed.TotalSeconds, (Join-Path $dir "grid.png"))
} finally {
    foreach ($n in $envNames) { [Environment]::SetEnvironmentVariable($n, $envSaved[$n]) }
}
