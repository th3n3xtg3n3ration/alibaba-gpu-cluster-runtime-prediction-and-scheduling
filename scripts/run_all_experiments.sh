#!/usr/bin/env bash
# ============================================================
#  run_all_experiments.sh
#  ------------------------------------------------------------
#  Automated pipeline for thesis experiments and visualization.
#
#  Usage:
#    bash scripts/run_all_experiments.sh [all|workload|models|scheduler]
#
#  The MODE argument defaults to "all".
# ============================================================

set -euo pipefail

# Derive project root robustly from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change into the project root so all relative paths resolve correctly
cd "$PROJECT_ROOT"

# Configuration
MODE="${1:-all}"

echo "[Pipeline] Starting Thesis Experimental Pipeline (Mode: $MODE)"
echo "[Pipeline] Project root: $PROJECT_ROOT"

if [ "$MODE" == "test" ]; then
    echo "[Pipeline] Running Unit Tests..."
    # Always use the specific python version identified for this project
    python3 -m unittest discover tests
    exit 0
fi

# Step 1: Generate visualizations
python scripts/generate_all_figures.py --mode "$MODE"

echo "[Pipeline] Done. Results are in results/figures/"
