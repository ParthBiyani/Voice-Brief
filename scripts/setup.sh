#!/usr/bin/env bash
# First-time setup on Linux and macOS: environment, services, database, sources, web app.
# Safe to run again. Needs python3 (3.11+), docker, and node (20+) unless --skip-web.
set -euo pipefail

cd "$(dirname "$0")/.."
skip_web=0
[[ "${1:-}" == "--skip-web" ]] && skip_web=1

step() { printf '\n== %s\n' "$1"; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "$1 was not found. $2" >&2; exit 1; }; }

step "Checking the tools"
need docker "Install Docker: https://docs.docker.com/get-docker/"
need python3 "Install Python 3.11 or newer."
[[ $skip_web -eq 1 ]] || need npm "Install Node 20 or newer: https://nodejs.org"
docker info >/dev/null 2>&1 || { echo "Docker is installed but not running. Start it and run this again." >&2; exit 1; }

step "Python environment"
[[ -d .venv ]] || python3 -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -e ".[ml,llm,docs,tts,dev]"

step "Settings"
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env. Add your ANTHROPIC_API_KEY to it (https://console.anthropic.com)."
else
  echo ".env already exists; left as it is."
fi

step "Database, vector store and audio storage (Docker)"
docker compose -p voicebrief up -d --wait
.venv/bin/python -m alembic upgrade head
.venv/bin/python -m voicebrief.cli sources sync

if [[ $skip_web -eq 0 ]]; then
  step "Web app"
  (cd web && npm ci && npm run build)
fi

step "Done"
.venv/bin/python -m voicebrief.cli doctor || true
echo
echo "Next:   .venv/bin/python -m voicebrief.cli daily --github YOUR_GITHUB_LOGIN"
echo "Every morning: see 'Schedule it' in the README (cron or systemd timer)."
