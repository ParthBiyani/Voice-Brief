<#
.SYNOPSIS
    Register (or remove) the scheduled task that makes VoiceBrief run every morning.

.DESCRIPTION
    Creates one task, "VoiceBrief Daily Brief", for the current user. It fires at 09:00
    each day and again shortly after every logon, and it is set to run as soon as
    possible after a missed start, so a laptop that was asleep or off at 09:00 still
    produces the brief when it next wakes. `voicebrief daily` is idempotent, so the extra
    triggers never produce a second brief, and the task passes -NotBefore so a logon
    earlier in the morning waits for the 09:00 run instead of jumping ahead of it.

    A second task, "VoiceBrief Web", starts the web app at logon (run-serve.ps1), which
    restarts the server if it exits; the task re-fires every 10 minutes in case the script
    itself was stopped. So http://localhost:8000 is there whenever the laptop is on.
    -NoWeb skips it.

    The tasks run only while the user is logged on (no password is stored), which is also
    what Docker Desktop needs.

    Only these two tasks are ever created, changed or removed.

.EXAMPLE
    .\scripts\install-schedule.ps1                 # install, 09:00
    .\scripts\install-schedule.ps1 -At 06:30       # a different time
    .\scripts\install-schedule.ps1 -NoWeb          # the brief only, no always-on web app
    .\scripts\install-schedule.ps1 -Remove         # uninstall both
#>
[CmdletBinding()]
param(
    [string]$At = '09:00',
    [switch]$NoWeb,
    [switch]$Remove
)

$ErrorActionPreference = 'Stop'
$taskName = 'VoiceBrief Daily Brief'
$webTaskName = 'VoiceBrief Web'

if ($Remove) {
    foreach ($name in @($taskName, $webTaskName)) {
        if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
            Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
            Unregister-ScheduledTask -TaskName $name -Confirm:$false
            Write-Host "Removed '$name'."
        } else {
            Write-Host "'$name' is not installed."
        }
    }
    return
}

$runner = Join-Path $PSScriptRoot 'run-daily.ps1'
if (-not (Test-Path $runner)) { throw "Cannot find $runner" }

# Through `conhost --headless`, not a "hidden" powershell.exe: where Windows Terminal is the
# default console, a hidden PowerShell started by Task Scheduler still gets a terminal that
# goes away about 25 seconds later, killing the run (exit 0xC000013A). Short runs survived;
# a real three-minute generation did not. A headless conhost has no window to lose.
function New-HeadlessAction([string]$script, [string]$extra = '') {
    New-ScheduledTaskAction `
        -Execute 'conhost.exe' `
        -Argument ('--headless powershell.exe -NoProfile -ExecutionPolicy Bypass -File "{0}" {1}' -f $script, $extra).Trim()
}

$action = New-HeadlessAction $runner "-NotBefore $At"

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

if ($NoWeb) {
    if (Get-ScheduledTask -TaskName $webTaskName -ErrorAction SilentlyContinue) {
        Stop-ScheduledTask -TaskName $webTaskName -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $webTaskName -Confirm:$false
        Write-Host "Removed '$webTaskName'."
    }
} else {
    $serveRunner = Join-Path $PSScriptRoot 'run-serve.ps1'
    $webAction = New-HeadlessAction $serveRunner
    $webLogon = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    $webLogon.Delay = 'PT1M'
    # Re-fire every 10 minutes for the rest of the session: if the server is already up the
    # script sees the port in use and exits at once; if anything stopped it, it comes back.
    $webLogon.Repetition = (New-ScheduledTaskTrigger -Once -At (Get-Date) `
        -RepetitionInterval (New-TimeSpan -Minutes 10)).Repetition
    # And each morning with the brief, in case it was stopped; a running server ignores it.
    $webDaily = New-ScheduledTaskTrigger -Daily -At $At
    # Long-running by design: no time limit. run-serve.ps1 restarts the server itself.
    $webSettings = New-ScheduledTaskSettingsSet `
        -StartWhenAvailable `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -MultipleInstances IgnoreNew `
        -ExecutionTimeLimit ([TimeSpan]::Zero) `
        -RestartCount 999 `
        -RestartInterval (New-TimeSpan -Minutes 1)

    Register-ScheduledTask `
        -TaskName $webTaskName `
        -Description 'Keeps the VoiceBrief web app running at http://localhost:8000.' `
        -Action $webAction `
        -Trigger @($webLogon, $webDaily) `
        -Settings $webSettings `
        -Principal $principal `
        -Force | Out-Null
    Write-Host "Installed '$webTaskName': http://localhost:8000 from 1 minute after each logon (and checked daily at $At)."
}

Write-Host "Logs: $env:LOCALAPPDATA\VoiceBrief\logs"
Write-Host "Run the brief now with:  Start-ScheduledTask -TaskName '$taskName'"
