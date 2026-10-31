<#
.SYNOPSIS
    Keep the VoiceBrief web app running: start the services if needed, then `voicebrief serve`.

.DESCRIPTION
    This is the entry the "VoiceBrief Web" scheduled task calls at logon, so the site is
    at http://localhost:8000 whenever the laptop is on. It blocks for as long as the server
    runs; the task restarts it if it exits.

    It serves on localhost only. To reach it from a phone, set VB_API_TOKEN and pass
    -Address 0.0.0.0 (see the README).
#>
[CmdletBinding()]
param(
    [string]$Address = '127.0.0.1',
    [int]$Port = 8000,
    [int]$DockerWaitSeconds = 240
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv\Scripts\python.exe'

$logDir = Join-Path $env:LOCALAPPDATA 'VoiceBrief\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("serve-{0}.log" -f (Get-Date -Format 'yyyy-MM-dd'))

function Write-Log([string]$message) {
    $line = "{0:HH:mm:ss} {1}" -f (Get-Date), $message
    Add-Content -Path $log -Value $line -Encoding utf8
}

Get-ChildItem $logDir -Filter 'serve-*.log' -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
    Remove-Item -Force -ErrorAction SilentlyContinue

Write-Log '--- VoiceBrief web start ---'

if (-not (Test-Path $python)) {
    Write-Log "Python environment not found at $python"
    exit 1
}

# Someone (or an earlier start) already serves on this port: nothing to do.
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    Write-Log "Port $Port is already in use; leaving it alone"
    exit 0
}

# Same reasoning as run-daily.ps1: native stderr must not abort the script.
function Test-Docker { cmd.exe /c 'docker info >nul 2>&1'; return ($LASTEXITCODE -eq 0) }

if (-not (Test-Docker)) {
    $desktop = Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'
    if (Test-Path $desktop) {
        Write-Log 'Docker is not running; starting Docker Desktop'
        Start-Process $desktop
    }
    $deadline = (Get-Date).AddSeconds($DockerWaitSeconds)
    while (-not (Test-Docker) -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 5 }
    if (-not (Test-Docker)) {
        Write-Log "Docker did not become ready within $DockerWaitSeconds seconds"
        exit 1
    }
}

Push-Location $repo
try {
    Write-Log 'Ensuring services are up'
    cmd.exe /c ('docker compose -p voicebrief up -d --wait >> "{0}" 2>&1' -f $log)
    if ($LASTEXITCODE -ne 0) { Write-Log 'Services did not become healthy'; exit 1 }

    $env:PYTHONUTF8 = '1'
    if (-not $env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS = '4' }
    Write-Log "Serving on http://${Address}:$Port"
    cmd.exe /c ('""{0}" -m voicebrief.cli serve --host {1} --port {2} >> "{3}" 2>&1"' -f $python, $Address, $Port, $log)
    $code = $LASTEXITCODE
    Write-Log "voicebrief serve exited with code $code"
    exit $code
}
finally {
    Pop-Location
}
