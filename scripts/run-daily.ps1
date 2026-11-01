<#
.SYNOPSIS
    Run today's VoiceBrief: start the services if needed, then `voicebrief daily`.

.DESCRIPTION
    This is the entry the scheduled task calls, but it is equally safe to run by hand.
    `voicebrief daily` is idempotent, so triggering this twice in one day, or after a
    missed schedule, never produces two briefs.

    Exit code mirrors `voicebrief daily`: 0 done or nothing to do, 2 a degraded brief
    (for example silent audio), 1 failure.

    -NotBefore HH:mm makes an earlier start a no-op. The scheduled task passes it so a
    logon at 08:00 does not produce the brief ahead of the 09:00 slot; by hand, omit it.
#>
[CmdletBinding()]
param(
    [switch]$Force,
    [int]$DockerWaitSeconds = 600,
    [string]$NotBefore
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
Get-ChildItem $logDir -Filter 'daily-*' -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
    Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem $logDir -Filter 'done-*.flag' -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
    Remove-Item -Force -ErrorAction SilentlyContinue

if ($NotBefore -and -not $Force) {
    $earliest = [datetime]::ParseExact($NotBefore, 'HH:mm', $null)
    if ((Get-Date) -lt $earliest) {
        Write-Log "Before $NotBefore; leaving it to the scheduled run"
        exit 0
    }
}

# The scheduled task re-fires every 30 minutes after 09:00 so a failed attempt (Docker not
# up yet, network down) is retried. Once a run has succeeded, the rest of the day's firings
# end here, without waking Docker or the database.
$doneFlag = Join-Path $logDir ("done-{0}.flag" -f (Get-Date -Format 'yyyy-MM-dd'))
if ((Test-Path $doneFlag) -and -not $Force) { exit 0 }

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

# 1. Docker. After a reboot or a login the engine may still be starting. If it never
# comes up, the next scheduled attempt (every 30 minutes) tries again.
. (Join-Path $PSScriptRoot 'docker-ready.ps1')
if (-not (Start-DockerIfNeeded $DockerWaitSeconds { param($m) Write-Log $m })) {
    Write-Log 'Giving up for now; the next scheduled attempt will try again'
    exit 1
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
    # 0 is "brief made, or today's already exists"; anything else is retried later today.
    if ($code -eq 0) { New-Item -ItemType File -Force -Path $doneFlag | Out-Null }
    exit $code
}
finally {
    Pop-Location
}
