#!/usr/bin/env bash
# Run the job-match agent and email the report via Gmail.
# Intended for cron (2am and 2pm daily). Logs each run to logs/cron.log.
#
# Cron entry (crontab -e):
#   0 2,14 * * * /Users/swbs/src/cc_jscan/run_matches.sh >> /Users/swbs/src/cc_jscan/logs/cron.log 2>&1

set -euo pipefail

# Project root = directory this script lives in (so cron's cwd doesn't matter).
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

VENV_PY="$PROJECT_DIR/.venv/bin/python"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') starting run ====="

# NOTE: --top 10 (matches to report). Your original had "--jobs 10", but --jobs
# is the jobs FILE path; the count flag is --top. Change if you meant otherwise.
"$VENV_PY" match_jobs.py \
  --agent \
  --from prosprc@gmail.com \
  --to reach.sriharsha@gmail.com \
  --max-age-hours 24 \
  --top 20

echo "===== $(date '+%Y-%m-%d %H:%M:%S') finished run for abhishek====="


"$VENV_PY" match_jobs.py \
  --agent \
  --from prosprc@gmail.com \
  --to reach.sriharsha@gmail.com \
  --max-age-hours 24 \
  --top 20
echo "===== $(date '+%Y-%m-%d %H:%M:%S') finished run  for harsha====="
