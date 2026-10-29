<#
.SYNOPSIS
    Register (or remove) the scheduled task that makes VoiceBrief run every morning.

.DESCRIPTION
    Creates one task, "VoiceBrief Daily Brief", for the current user. It fires at 07:00
    each day and again shortly after every logon, and it is set to run as soon as
    possible after a missed start, so a laptop that was asleep or off at 07:00 still
    produces the brief when it next wakes. `voicebrief daily` is idempotent, so the extra
    triggers never produce a second brief.

    The task runs only while the user is logged on (no password is stored), which is also
    what Docker Desktop needs.

    Only this task is ever created, changed or removed.

.EXAMPLE
    .\scripts\install-schedule.ps1                 # install, 07:00
    .\scripts\install-schedule.ps1 -At 06:30       # a different time
    .\scripts\install-schedule.ps1 -Remove         # uninstall
#>
[CmdletBinding()]
param(
    [string]$At = '07:00',
    [switch]$Remove
)

$ErrorActionPreference = 'Stop'
$taskName = 'VoiceBrief Daily Brief'

if ($Remove) {
    if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        Write-Host "Removed '$taskName'."
    } else {
        Write-Host "'$taskName' is not installed."
    }
    return
}

$runner = Join-Path $PSScriptRoot 'run-daily.ps1'
if (-not (Test-Path $runner)) { throw "Cannot find $runner" }

$action = New-ScheduledTaskAction `
    -Execute 'powershell.exe' `
    -Argument ('-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "{0}"' -f $runner)

$daily = New-ScheduledTaskTrigger -Daily -At $At
$logon = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
# Let the desktop, network and Docker settle after a login before the first attempt.
$logon.Delay = 'PT3M'

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -RunOnlyIfNetworkAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $taskName `
    -Description 'Generates the daily VoiceBrief audio briefing. Safe to run repeatedly.' `
    -Action $action `
    -Trigger @($daily, $logon) `
    -Settings $settings `
    -Principal $principal `
    -Force | Out-Null

Write-Host "Installed '$taskName': daily at $At and 3 minutes after each logon."
Write-Host "Logs: $env:LOCALAPPDATA\VoiceBrief\logs"
Write-Host "Run it now with:  Start-ScheduledTask -TaskName '$taskName'"
