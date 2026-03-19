param(
    [int]$Port = 10001,
    [string]$BindHost = "127.0.0.1",
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

if ($Help) {
    Write-Host "Usage: .\start_windows.ps1 [-Port 10001] [-BindHost 127.0.0.1] [extra serve args...]"
    Write-Host ""
    Write-Host "Start the translator web console on native Windows using the local .venv-windows environment."
    exit 0
}

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$VenvDir = Join-Path $ScriptDir ".venv-windows"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv $VenvDir
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv $VenvDir
    }
    else {
        throw "Python launcher not found. Install Python 3 first."
    }
}

& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install -r (Join-Path $ScriptDir "requirements.txt")

& $PythonExe -m translator serve --host $BindHost --port $Port @ExtraArgs
