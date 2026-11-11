<#
.SYNOPSIS
    First-time setup on Windows: environment, services, database, sources, web app.

.DESCRIPTION
    Safe to run again; every step checks before it acts. It does not need administrator
    rights and does not touch anything outside this folder, Docker's "voicebrief" project
    and your per-user VoiceBrief data folder. Finish by running `voicebrief doctor`.

.EXAMPLE
    .\scripts\setup.ps1
    .\scripts\setup.ps1 -SkipWeb          # no Node needed; use the CLI and Slack only
#>
[CmdletBinding()]
param([switch]$SkipWeb)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

function Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }
function Need($name, $hint) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) { throw "$name was not found. $hint" }
}

Step 'Checking the tools'
Need docker 'Install Docker Desktop and start it: https://www.docker.com/products/docker-desktop'
$py = (Get-Command py -ErrorAction SilentlyContinue)
if (-not $py -and -not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw 'Python 3.11 or newer was not found: https://www.python.org/downloads/'
}
if (-not $SkipWeb) { Need npm 'Install Node 20 or newer: https://nodejs.org' }
docker info *> $null
if ($LASTEXITCODE -ne 0) { throw 'Docker is installed but not running. Start Docker Desktop and run this again.' }

Step 'Python environment'
$venvPython = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    if ($py) { py -3 -m venv .venv } else { python -m venv .venv }
}
& $venvPython -m pip install --quiet --upgrade pip
& $venvPython -m pip install --quiet -e ".[ml,llm,docs,tts,dev]"

Step 'Settings'
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host 'Created .env. Add your ANTHROPIC_API_KEY to it (https://console.anthropic.com).' -ForegroundColor Yellow
} else {
    Write-Host '.env already exists; left as it is.'
}

Step 'Database, vector store and audio storage (Docker)'
docker compose -p voicebrief up -d --wait
if ($LASTEXITCODE -ne 0) { throw 'The services did not become healthy. Run: docker compose -p voicebrief logs' }
& $venvPython -m alembic upgrade head
& $venvPython -m voicebrief.cli sources sync

if (-not $SkipWeb) {
    Step 'Web app'
    Push-Location web
    npm ci
    npm run build
    Pop-Location
}

Step 'Done'
& $venvPython -m voicebrief.cli doctor
Write-Host "`nNext:  .\.venv\Scripts\python.exe -m voicebrief.cli daily --github YOUR_GITHUB_LOGIN"
Write-Host 'Every morning:  .\scripts\install-schedule.ps1'
