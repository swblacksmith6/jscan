#!/usr/bin/env bash
# Run the job matcher for every enabled profile in profiles.yaml.
# Intended for cron (2am and 2pm daily). Logs each run to logs/cron.log.
#
# Cron entry (crontab -e):
#   0 2,14 * * * /Users/swbs/src/jscan/run_matches.sh >> /Users/swbs/src/jscan/logs/cron.log 2>&1

set -euo pipefail

# Project root = directory this script lives in (so cron's cwd doesn't matter).
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

VENV_PY="$PROJECT_DIR/.venv/bin/python"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') starting run ====="

"$VENV_PY" run_profiles.py --profiles profiles.yaml

echo "===== $(date '+%Y-%m-%d %H:%M:%S') finished run ====="
