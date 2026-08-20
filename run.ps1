<#
.SYNOPSIS
    Starts the Streamlit application using the project virtual environment.

.EXAMPLE
    .\run.ps1                 # http://localhost:8501
    .\run.ps1 -Port 8600
#>
param(
    [int]$Port = 8501,
    [string]$VenvPath = ""
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not $VenvPath) {
    foreach ($candidate in @($env:EYE_REID_VENV, ".venv", "$env:USERPROFILE\.venvs\eye-reid")) {
        if ($candidate -and (Test-Path (Join-Path $candidate "Scripts\python.exe"))) { $VenvPath = $candidate; break }
    }
}
if (-not $VenvPath) {
    throw "No virtual environment found. Run .\setup.ps1 first (or set EYE_REID_VENV)."
}

$python = Join-Path $VenvPath "Scripts\python.exe"
Write-Host "Using $python"
& $python -m streamlit run app.py --server.port $Port
