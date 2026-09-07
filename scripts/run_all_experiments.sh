#!/usr/bin/env bash
# ==============================================================================
#  Alibaba GPU Runtime Prediction & Scheduling Thesis Pipeline
# ==============================================================================
#  Automated execution and artifact export script.
#
#  Usage:
#    bash scripts/run_all_experiments.sh [all|export|test|run|force|tr]
#
#  Options:
#    all     : (Default) Runs unit tests and extracts all thesis figures & tables.
#    export  : Fast export (<1s) of figures & tables from existing notebook outputs.
#    test    : Runs the automated Python unit test suite (tests/).
#    run     : Runs notebooks if outputs are missing, then exports results.
#    force   : Forces complete re-execution of all research notebooks from scratch.
#    tr      : Exports results from Turkish notebooks (notebooks/tr/).
# ==============================================================================

set -euo pipefail

# ANSI Colors
BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
CYAN="\033[0;36m"
YELLOW="\033[1;33m"
RESET="\033[0m"

# Project Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Use this project's own venv, not whatever `python3` resolves to on $PATH
# (a pyenv/system shim can point at a completely different environment,
# with different library versions than the one that produced the published
# results -- reproducibility-6). Falls back to `python3` with a warning if
# the venv hasn't been created yet, rather than failing outright.
if [ -x "$PROJECT_ROOT/venv/bin/python" ]; then
    PYTHON="$PROJECT_ROOT/venv/bin/python"
else
    echo -e "${YELLOW}⚠ $PROJECT_ROOT/venv not found -- falling back to 'python3' on PATH.${RESET}"
    echo -e "${YELLOW}  Results may not match the published environment; see README.md.${RESET}"
    PYTHON="python3"
fi

MODE="${1:-all}"

echo -e "${BOLD}${BLUE}==============================================================================${RESET}"
echo -e "${BOLD}${BLUE}  ALIBABA GPU RUNTIME PREDICTION & SCHEDULING RESEARCH PIPELINE${RESET}"
echo -e "${BOLD}${BLUE}==============================================================================${RESET}"
echo -e "${CYAN}📁 Project Root :${RESET} $PROJECT_ROOT"
echo -e "${CYAN}⚙️  Pipeline Mode:${RESET} ${BOLD}$MODE${RESET}\n"

case "$MODE" in
    test)
        echo -e "${BOLD}${YELLOW}▶ [Step 1/1] Running Automated Unit Tests...${RESET}"
        "$PYTHON" -m unittest discover tests
        echo -e "\n${BOLD}${GREEN}✅ All unit tests passed successfully!${RESET}"
        ;;

    export)
        echo -e "${BOLD}${YELLOW}▶ [Step 1/1] Fast Exporting Figures & Tables from Notebooks...${RESET}"
        "$PYTHON" scripts/export_thesis_results.py
        echo -e "\n${BOLD}${GREEN}✅ Export complete. Artifacts saved in results/figures/thesis_export/${RESET}"
        ;;

    run|execute)
        echo -e "${BOLD}${YELLOW}▶ [Step 1/1] Auto-Executing Notebooks & Exporting Results...${RESET}"
        "$PYTHON" scripts/export_thesis_results.py --execute
        echo -e "\n${BOLD}${GREEN}✅ Execution & export complete!${RESET}"
        ;;

    force)
        echo -e "${BOLD}${YELLOW}▶ [Step 1/1] Force Re-Executing All Notebooks from Scratch...${RESET}"
        "$PYTHON" scripts/export_thesis_results.py --force-execute
        echo -e "\n${BOLD}${GREEN}✅ Complete pipeline re-run & export finished!${RESET}"
        ;;

    tr)
        echo -e "${BOLD}${YELLOW}▶ [Step 1/1] Exporting from Turkish Notebooks (notebooks/tr/)...${RESET}"
        "$PYTHON" scripts/export_thesis_results.py --lang tr
        echo -e "\n${BOLD}${GREEN}✅ Turkish notebook results exported successfully!${RESET}"
        ;;

    all)
        echo -e "${BOLD}${YELLOW}▶ [Step 1/2] Quality Assurance (Unit Tests)...${RESET}"
        "$PYTHON" -m unittest discover tests

        echo -e "\n${BOLD}${YELLOW}▶ [Step 2/2] Extracting Thesis Figures & Tables...${RESET}"
        "$PYTHON" scripts/export_thesis_results.py

        echo -e "\n${BOLD}${BLUE}==============================================================================${RESET}"
        echo -e "${BOLD}${GREEN}🎉 PIPELINE COMPLETED SUCCESSFULLY!${RESET}"
        echo -e "  📊 PNG Figures  : results/figures/thesis_export/png/"
        echo -e "  📋 HTML Tables  : results/figures/thesis_export/html/"
        echo -e "${BOLD}${BLUE}==============================================================================${RESET}"
        ;;

    -h|--help|help)
        echo -e "Available commands:"
        echo -e "  bash scripts/run_all_experiments.sh         # Runs unit tests + fast export"
        echo -e "  bash scripts/run_all_experiments.sh export  # Fast export (<1s)"
        echo -e "  bash scripts/run_all_experiments.sh test    # Run unit tests only"
        echo -e "  bash scripts/run_all_experiments.sh run     # Auto-run unexecuted notebooks"
        echo -e "  bash scripts/run_all_experiments.sh force   # Force re-run all notebooks"
        echo -e "  bash scripts/run_all_experiments.sh tr      # Export from Turkish notebooks"
        ;;

    *)
        echo -e "${YELLOW}Unknown mode: '$MODE'. Defaulting to fast export...${RESET}"
        "$PYTHON" scripts/export_thesis_results.py
        ;;
esac
