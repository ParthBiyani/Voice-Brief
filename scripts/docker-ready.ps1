<#
.SYNOPSIS
    Shared by run-daily.ps1 and run-serve.ps1: make sure Docker is up, starting Docker
    Desktop if need be. Dot-source it, then call Start-DockerIfNeeded.

.DESCRIPTION
    Docker Desktop can be installed per user (%LOCALAPPDATA%\Programs\DockerDesktop, what
    the installer does without admin rights) or machine-wide (C:\Program Files\Docker).
    The scripts used to look only in the second place, so after a per-user reinstall they
    never started Docker, and a laptop that rebooted overnight produced no brief.
#>

# Native stderr (Docker prints its progress there) must not abort a script that runs with
# ErrorActionPreference=Stop, hence cmd.exe and a plain exit code.
function Test-Docker {
    cmd.exe /c 'docker info >nul 2>&1'
    return ($LASTEXITCODE -eq 0)
}

function Find-DockerDesktop {
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\Docker Desktop.exe'),
        (Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe')
    )
    # The installer also records where it put itself.
    $run = (Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run' -ErrorAction SilentlyContinue).'Docker Desktop'
    if ($run) { $candidates += ($run -replace '^"|"$', '') }
    $appPath = (Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\Docker Desktop.exe' -ErrorAction SilentlyContinue).'(default)'
    if ($appPath) { $candidates += $appPath }
    return $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
}

# Returns $true once `docker info` answers, $false if it never does within the wait.
# $Report is a script block taking one string. It is not named $Log: PowerShell variable
# names are case-insensitive, so it would shadow the callers' $log (the log file path)
# inside the block they pass, and every log line would fail.
function Start-DockerIfNeeded([int]$WaitSeconds, [scriptblock]$Report) {
    if (Test-Docker) { return $true }

    $running = Get-Process -Name 'Docker Desktop' -ErrorAction SilentlyContinue
    if ($running) {
        & $Report 'Docker Desktop is starting; waiting for the engine'
    } else {
        $desktop = Find-DockerDesktop
        if (-not $desktop) {
            & $Report 'Docker Desktop was not found in any known location; install it or start Docker by hand'
            return $false
        }
        & $Report "Docker is not running; starting $desktop"
        Start-Process -FilePath $desktop
    }

    $deadline = (Get-Date).AddSeconds($WaitSeconds)
    while (-not (Test-Docker) -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 5 }
    if (Test-Docker) {
        & $Report 'Docker is ready'
        return $true
    }
    & $Report "Docker did not become ready within $WaitSeconds seconds"
    return $false
}
