#!/usr/bin/env bash
# Regenerate RQ2 outputs and figures. Run from repo root: ./run_rq2.sh
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  echo "Creating venv and installing dependencies..."
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
  PY=".venv/bin/python"
fi

echo "=== validate_rq2.py ==="
"$PY" validate_rq2.py

echo "=== rq2_endogenous_coverage.py ==="
"$PY" rq2_endogenous_coverage.py

echo "=== rq2_exinterim_guardrails.py (slow, ~10-15 min) ==="
"$PY" rq2_exinterim_guardrails.py

echo "=== figures_rq2.py ==="
"$PY" figures_rq2.py

echo "Done. Figures are in ./figures/"
