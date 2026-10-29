<#
.SYNOPSIS
    Run today's VoiceBrief: start the services if needed, then `voicebrief daily`.

.DESCRIPTION
    This is the entry the scheduled task calls, but it is equally safe to run by hand.
    `voicebrief daily` is idempotent, so triggering this twice in one day, or after a
    missed schedule, never produces two briefs.

    Exit code mirrors `voicebrief daily`: 0 done or nothing to do, 2 a degraded brief
    (for example silent audio), 1 failure.
#>
[CmdletBinding()]
param(
    [switch]$Force,
    [int]$DockerWaitSeconds = 240
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv\Scripts\python.exe'

$logDir = Join-Path $env:LOCALAPPDATA 'VoiceBrief\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("daily-{0}.log" -f (Get-Date -Format 'yyyy-MM-dd'))

function Write-Log([string]$message) {
    $line = "{0:HH:mm:ss} {1}" -f (Get-Date), $message
    Add-Content -Path $log -Value $line -Encoding utf8
    Write-Host $line
}

# Keep a month of logs; older ones are noise.
Get-ChildItem $logDir -Filter 'daily-*.log' -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
    Remove-Item -Force -ErrorAction SilentlyContinue

Write-Log '--- VoiceBrief daily start ---'

if (-not (Test-Path $python)) {
    Write-Log "Python environment not found at $python. Run: python -m venv .venv ; pip install -e .[ml,llm,docs,dev]"
    exit 1
}

# Native commands run through cmd.exe with plain redirection. Under Windows PowerShell
# 5.1, stderr from a native program (Docker prints its progress there) is wrapped as an
# error record and, with ErrorActionPreference=Stop, aborts the script and fills the log
# with PowerShell noise instead of the program's own output.
function Invoke-Logged([string]$exe, [string[]]$arguments) {
    $quoted = ($arguments | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' '
    cmd.exe /c ('""{0}" {1} >> "{2}" 2>&1"' -f $exe, $quoted, $log)
    return $LASTEXITCODE
}

# 1. Docker. After a reboot or a login the engine may still be starting.
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
        Write-Log "Docker did not become ready within $DockerWaitSeconds seconds; giving up for now"
        exit 1
    }
}

# 2. The datastores. `--wait` returns only when every healthcheck passes.
Push-Location $repo
try {
    Write-Log 'Ensuring services are up'
    $up = Invoke-Logged 'docker' @('compose', '-p', 'voicebrief', 'up', '-d', '--wait')
    if ($up -ne 0) { Write-Log 'Services did not become healthy'; exit 1 }

    # 3. Today's brief.
    $env:PYTHONUTF8 = '1'
    if (-not $env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS = '4' }
    $arguments = @('-m', 'voicebrief.cli', 'daily', '--trigger', 'schedule')
    if ($Force) { $arguments += '--force' }

    Write-Log ("Running: voicebrief " + ($arguments[2..($arguments.Length - 1)] -join ' '))
    $code = Invoke-Logged $python $arguments
    Write-Log "voicebrief daily finished with exit code $code"
    exit $code
}
finally {
    Pop-Location
}
