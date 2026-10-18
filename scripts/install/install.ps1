# Windows install: torch from the index for this GPU / driver, kohakumoss with its default extra, the doctor.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\install\install.ps1 (a clone installs editable).
$ErrorActionPreference = "Stop"

$TorchSpec = if ($env:KM_TORCH) { $env:KM_TORCH } else { "torch>=2.13,<2.15" }
$Python = "3.12"
$Manual = "install by hand: docs/guides/install.md#manual-install"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Gpus = (Get-CimInstance Win32_VideoController).Name -join "; "

# Runs one native command; a nonzero exit code stops the install.
function Invoke-Step([string]$What, [scriptblock]$Step) {
    & $Step
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit code $LASTEXITCODE)" }
}

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    # "CUDA Version: X.Y", or "CUDA UMD Version: X.Y" on newer drivers
    $m = [regex]::Match((nvidia-smi | Out-String), "CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)")
    if (-not $m.Success) { throw "could not read the CUDA version from nvidia-smi; $Manual" }
    $ver = [int]$m.Groups[1].Value * 100 + [int]$m.Groups[2].Value
    if ($ver -ge 1300) { $Tag = "cu130" }
    elseif ($ver -ge 1208) { $Tag = "cu128" }
    elseif ($ver -ge 1206) { $Tag = "cu126" }
    else { throw "driver CUDA $($m.Groups[1].Value).$($m.Groups[2].Value) < 12.6: update the NVIDIA driver" }
    $Extra = "[cuda]"
} elseif ($Gpus -match "Intel\(R\) Arc") {
    $Tag = "xpu"; $Extra = "[xpu]"
} else {
    Write-Output "no NVIDIA driver (nvidia-smi) or Intel Arc GPU found: CPU build"
    $Tag = "cpu"; $Extra = ""
}
$Index = "https://download.pytorch.org/whl/$Tag"
Write-Output "GPUs: $Gpus"
Write-Output "torch index: $Index"

if (Test-Path (Join-Path $Root "pyproject.toml")) { $Pkg = @("-e", "$Root$Extra") } else { $Pkg = @("kohakumoss$Extra") }

if (Get-Command uv -ErrorAction SilentlyContinue) {
    if (-not (Test-Path .venv)) { Invoke-Step "uv venv" { uv venv -p $Python } }
    Invoke-Step "torch install" { uv pip install $TorchSpec torchvision --index-url $Index }
    Invoke-Step "kohakumoss install" { uv pip install @Pkg $TorchSpec --extra-index-url $Index --index-strategy unsafe-best-match }
} else {
    if (-not (Test-Path .venv)) {
        # the py launcher of a python.org install; `python` alone may be the Microsoft Store alias
        if (Get-Command py -ErrorAction SilentlyContinue) { py "-$Python" -m venv .venv }
        else { python -m venv .venv }
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path .venv\Scripts\python.exe)) {
            throw "could not create .venv with Python $($Python): install Python $Python from python.org (with the py launcher) or uv"
        }
    }
    Invoke-Step "torch install" { .venv\Scripts\python.exe -m pip install $TorchSpec torchvision --index-url $Index }
    Invoke-Step "kohakumoss install" { .venv\Scripts\python.exe -m pip install @Pkg $TorchSpec --extra-index-url $Index }
}
.venv\Scripts\kohakumoss-doctor.exe
