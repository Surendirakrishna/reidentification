<#
.SYNOPSIS
    Creates a virtual environment and installs all dependencies (CUDA torch if an NVIDIA GPU is present).

.EXAMPLE
    .\setup.ps1                       # venv in .\.venv (or $env:EYE_REID_VENV)
    .\setup.ps1 -VenvPath C:\venvs\reid -Cpu
#>
param(
    [string]$VenvPath = $(if ($env:EYE_REID_VENV) { $env:EYE_REID_VENV } else { ".venv" }),
    [string]$Python = "",
    [switch]$Cpu,
    [string]$CudaIndex = "https://download.pytorch.org/whl/cu130"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not $Python) {
    # Prefer Python 3.12 (best wheel coverage), fall back to whatever `py`/`python` resolves to.
    $candidates = @("py -3.12", "py -3.11", "py -3.13", "python")
    foreach ($c in $candidates) {
        try { & cmd /c "$c -c ""import sys; print(sys.version)""" 2>$null | Out-Null; if ($LASTEXITCODE -eq 0) { $Python = $c; break } } catch {}
    }
}
if (-not $Python) { throw "No Python interpreter found. Install Python 3.10-3.13 first." }

Write-Host "Creating venv at $VenvPath using '$Python' ..."
& cmd /c "$Python -m venv ""$VenvPath"""
$pip = Join-Path $VenvPath "Scripts\python.exe"
& $pip -m pip install --upgrade pip

$hasGpu = $false
if (-not $Cpu) {
    try { & nvidia-smi | Out-Null; if ($LASTEXITCODE -eq 0) { $hasGpu = $true } } catch {}
}
if ($hasGpu) {
    Write-Host "NVIDIA GPU detected - installing CUDA build of torch/torchvision from $CudaIndex ..."
    & $pip -m pip install torch torchvision --index-url $CudaIndex
} else {
    Write-Host "Installing CPU build of torch/torchvision ..."
    & $pip -m pip install torch torchvision
}
Write-Host "Installing remaining requirements ..."
& $pip -m pip install -r requirements.txt

Write-Host ""
Write-Host "Done. Start the app with:  .\run.ps1   (or: $VenvPath\Scripts\Activate.ps1; streamlit run app.py)"
