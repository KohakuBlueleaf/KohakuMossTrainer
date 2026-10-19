# Example sampling run: images from the base weights (and adapters) through a generated config.
#
#   .\scripts\examples\sample.ps1 -Weights models\MOSS-Image-Base
#   .\scripts\examples\sample.ps1 -Prompt "a fox in snow" -Size 832x1216 -Steps 30 -Count 4
#   .\scripts\examples\sample.ps1 -Adapter runs\examples\train\<run>\export\<run>-lycoris.safetensors:0.8
#   .\scripts\examples\sample.ps1 -Precision fp16 -Transformer <fp16-fix .safetensors> -Compare
#   .\scripts\examples\sample.ps1 -Memory "" -BasePrecision fp8_compute      # whole DiT resident, fp8 GEMMs (sm89+)
#   .\scripts\examples\sample.ps1 -TuneOnly                                    # tune the sampling shapes, no weights loaded
#
# Run it from the activated venv (kogine, kohakumoss-tune and python on PATH). -Weights defaults
# to $env:KOHAKUMOSS_WEIGHTS. Each run writes <Out>\<yyyyMMdd-HHmmss>-<device>-<tag>\ with its
# config.py, the PNGs and metadata.json (generate time, peak VRAM, text-encoder stats).
# Environment variables set for the device are restored when the script exits. See
# scripts/examples/README.md.

[CmdletBinding()]
param(
    # Device kind: cuda (NVIDIA) | xpu (Intel Arc).
    [ValidateSet("cuda", "xpu")] [string] $Device = "cuda",
    # GPU index within that kind: cuda -> CUDA_VISIBLE_DEVICES (nvidia-smi order), xpu -> ZE_AFFINITY_MASK.
    [int] $Index = 0,
    # Weights: a ComfyUI models folder or an HF weights root; the text encoder, VAE and tokenizer are loaded from here.
    [string] $Weights = $env:KOHAKUMOSS_WEIGHTS,
    # Arch preset: base | sft (on a root with base\ / sft\ folders, WEIGHTS\<variant>\ is read).
    [ValidateSet("base", "sft")] [string] $Variant = "base",
    # DiT checkpoint that replaces the one in -Weights ("" = none), e.g. an fp16-fix file (it carries its fold).
    [string] $Transformer = "",
    # Prompt text.
    [string] $Prompt = "a lighthouse on a rocky coast at sunset",
    # Negative prompt ("" = none).
    [string] $Negative = "blurry, distortion",
    # Image size as HxW (height x width), e.g. 1024x1024, 832x1216.
    [string] $Size = "1024x1024",
    # Sampler steps.
    [int] $Steps = 50,
    # Classifier-free guidance scale.
    [double] $Cfg = 3.0,
    # Noise seed of the first image (same seed + prompt + size = same image across runs).
    [int] $Seed = 0,
    # Compute dtype: bf16 | fp16 (fp16 uses -Fp16Fix, or the fold of an fp16-fix -Transformer).
    [ValidateSet("bf16", "fp16")] [string] $Precision = "bf16",
    # fp16 residual fix when no -Transformer is given: online | fp32_residual | none.
    [ValidateSet("online", "fp32_residual", "none")] [string] $Fp16Fix = "online",
    # Text-encoder placement: auto = fastest that fits; resident = whole TE on the GPU; offload = moved in
    # per prompt batch; ring = layers streamed from host RAM (~2 GB on the GPU; "stream" is accepted too);
    # cpu = run on the CPU.
    [ValidateSet("auto", "resident", "offload", "ring", "stream", "cpu")] [string] $TextMemory = "auto",
    # Text-encoder precision: compute = bf16 / fp16; int8 / fp8 = quantized weight storage (16-bit math);
    # fp8_compute / mxfp8 = fp8 GEMMs (sm89+).
    [ValidateSet("compute", "int8", "fp8", "fp8_compute", "mxfp8")] [string] $TextPrecision = "compute",
    # Text-encoder fp8_compute scaling: row (per token / channel) | tensor | block (1x128 / 128x128).
    [ValidateSet("row", "tensor", "block")] [string] $TextFp8Recipe = "row",
    # Free the text encoder after it encoded the prompts.
    [switch] $FreeTextEncoder,
    # DiT base: compute = bf16 / fp16; int8 / fp8 = weight-only; fp8_compute / mxfp8 = fp8 GEMMs (sm89+).
    [ValidateSet("auto", "compute", "int8", "fp8", "fp8_compute", "mxfp8")] [string] $BasePrecision = "auto",
    # DiT fp8_compute scaling: row | tensor | block.
    [ValidateSet("row", "tensor", "block")] [string] $Fp8Recipe = "row",
    # DiT memory profile: auto = the fastest profile that fits the card's free memory | minimal (2 resident
    # blocks, the rest streamed; fits 8 GB) | low | balanced (16 resident) | any catalog name (see train.ps1)
    # | "" = whole DiT on the GPU (~10 GB in 16 bit).
    [string] $Memory = "auto",
    # VAE placement: auto = fastest that fits | resident | offload | ring (block rings; "stream" is accepted too) | cpu.
    [ValidateSet("auto", "resident", "offload", "ring", "stream", "cpu")] [string] $VaeMemory = "auto",
    # VAE dtype: auto (bf16 where supported, else fp16) | bf16 | fp16 | fp32.
    [ValidateSet("auto", "bf16", "fp16", "fp32")] [string] $VaeDtype = "auto",
    # Adapters to apply: one or more "path" or "path:multiplier" (LyCORIS / ComfyUI LoRA, LoKr, LoHa ... files).
    [string[]] $Adapter = @(),
    # How adapters are applied: bypass = run beside each layer (works on every base) | merge = folded
    # into the weights once (16-bit bases; a quantized or streamed base runs bypass with a warning).
    [ValidateSet("merge", "bypass")] [string] $AdapterMode = "bypass",
    # Kernel tuning during the run: off = stored per-card decisions, else fixed defaults; nothing timed
    # (default) | table = missing decisions also tuned on first use, then stored. -TuneOnly tunes ahead.
    [ValidateSet("off", "table")] [string] $Tune = "off",
    # Only tune: run kohakumoss-tune for this device / -Precision / -Size / -Batch and exit (no weights
    # loaded, no image); later runs bind the stored results.
    [switch] $TuneOnly,
    # Images to generate from -Prompt: seeds -Seed, -Seed + 1, ... (ignored when -PromptsFile is set).
    [int] $Count = 4,
    # Images per generate batch (one packed DiT forward per step for the whole batch; more VRAM).
    [int] $Batch = 4,
    # Optional prompts file: .txt (one prompt per line) or .json (list of strings, or objects
    # {"prompt", "negative", "size": [H, W], "seed"}); replaces -Prompt / -Count.
    [string] $PromptsFile = "",
    # With -Transformer: also render a bf16 reference from the stock transformer with the same prompts.
    [switch] $Compare,
    # Output root (relative paths are under the current directory).
    [string] $Out = "runs\examples\sample",
    # Write and check the generated config and print the command, without sampling.
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$script = Join-Path $repo "scripts\sample\sample.py"

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
function WriteText([string] $path, [string] $text) {
    [IO.File]::WriteAllText($path, $text, (New-Object Text.UTF8Encoding $false))
}
function Need([string] $tool) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "$tool not on PATH: activate the KohakuMoss venv first"
    }
}

# one config per rendering: <Out>\<runId>-<device>-<tag>\config.py, checked, then run
function Invoke-Sample([string] $precision, [string] $dit, [string] $tag) {
    $dir = Join-Path $Out "$runId-$Device-$tag"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $fix = if ($precision -eq "fp16" -and -not $dit) { $Fp16Fix } else { "none" }
    $fixKw = if ($fix -eq "online") { "{'c': 0.0625}" } else { "{}" }
    $text = "{'memory': '$TextMemory', 'precision': '$TextPrecision', 'fp8_recipe': '$TextFp8Recipe'"
    $text += $(if ($FreeTextEncoder) { ", 'free_after_use': True}" } else { "}" })
    $adapters = ($Adapter | ForEach-Object { PyStr ($_ -replace "\\", "/") }) -join ", "
    $config = @"
"""Generated by scripts/examples/sample.ps1 ($runId-$Device-$tag); drives scripts/sample/sample.py."""

WEIGHTS = $(PyPath $Weights)
VARIANT = '$Variant'
TRANSFORMER = $(PyPath $dit)
ADAPTERS = [$adapters]
ADAPTER_MODE = '$AdapterMode'
PRECISION = '$precision'
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

PROMPTS_FILE = $(PyPath $PromptsFile)
NEGATIVE = $(PyStr $Negative)
SIZES = [$(PyStr $Size)]
SEED_BASE = $Seed
STEPS = $Steps
CFG = $Cfg
BATCH = $Batch

OUT = $(PyPath $dir)
"@
    $cfgPath = Join-Path $dir "config.py"
    WriteText $cfgPath $config
    Write-Host "== $Device $tag -> $dir"
    # validate every knob against the sample script's rules before loading any weights
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
    if ($LASTEXITCODE -ne 0) { throw "sample run failed ($tag), exit $LASTEXITCODE" }
    Write-Host ("   wall {0:N1} s; see {1}" -f $sw.Elapsed.TotalSeconds, $dir)
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
        # DiT block ops + attention for this device, -Size, -Batch images (CFG packed); no weights loaded
        Need "kohakumoss-tune"
        $dtypes = if ($Compare) { @($Precision, "bf16") | Select-Object -Unique } else { @($Precision) }
        foreach ($dt in $dtypes) {
            Write-Host "== tune $Device $dt $Size"
            & kohakumoss-tune --device $Device --dtype $dt --sizes $Size --batch $Batch
            if ($LASTEXITCODE -ne 0) { throw "tuning failed ($dt), exit $LASTEXITCODE" }
        }
        return
    }

    Need "kogine"
    Need "python"
    if (-not $Weights) { throw "no weights: pass -Weights <dir> or set KOHAKUMOSS_WEIGHTS" }
    if (-not (Test-Path $Weights)) { throw "weights root not found: $Weights" }
    if ($Transformer -and -not (Test-Path $Transformer)) { throw "transformer not found: $Transformer" }
    if ($Compare -and -not $Transformer) { throw "-Compare renders the stock transformer next to -Transformer; pass -Transformer" }
    $Weights = FullPath $Weights
    if ($Transformer) { $Transformer = FullPath $Transformer }
    $Out = FullPath $Out

    # one run id per invocation, so repeated runs never overwrite
    $runId = Get-Date -Format "yyyyMMdd-HHmmss"

    # prompts: a given file, or -Count copies of -Prompt in a JSON file next to the outputs
    # (a file keeps prompts with commas / newlines intact; seeds follow SEED_BASE + index)
    if ($PromptsFile) {
        if (-not (Test-Path $PromptsFile)) { throw "prompts file not found: $PromptsFile" }
        $PromptsFile = FullPath $PromptsFile
    } else {
        New-Item -ItemType Directory -Force -Path $Out | Out-Null
        $PromptsFile = Join-Path $Out "$runId-prompts.json"
        WriteText $PromptsFile (ConvertTo-Json -InputObject @(@($Prompt) * [math]::Max($Count, 1)))
    }

    $tag = if ($Transformer) { "$Precision-transformer" } else { $Precision }
    Invoke-Sample $Precision $Transformer $tag
    # optional reference: bf16 with the stock transformer, same prompts / seeds
    if ($Compare) { Invoke-Sample "bf16" "" "bf16-stock" }
} finally {
    foreach ($n in $envNames) { [Environment]::SetEnvironmentVariable($n, $envSaved[$n]) }
}
