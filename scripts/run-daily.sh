#!/usr/bin/env bash
# Today's brief on Linux and macOS: start the services if needed, then `voicebrief daily`.
# `daily` is idempotent, so running this from cron more than once a day is harmless.
#
#   crontab -e
#   0 9 * * *  /path/to/Voice-Brief/scripts/run-daily.sh >> ~/.local/share/voicebrief/daily.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p "$HOME/.local/share/voicebrief"
docker compose -p voicebrief up -d --wait
exec .venv/bin/python -m voicebrief.cli daily --trigger schedule
