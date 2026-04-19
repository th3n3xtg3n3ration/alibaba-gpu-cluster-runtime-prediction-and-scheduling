"""
generate_all_figures.py

Standardized Entry Point for Thesis Visualization

This script generates all workload characterization, model performance, and
scheduler simulation figures for the thesis. It is designed to be modular and
parametric, delegating all plotting logic to :mod:`src.visualization`.

Usage::

    python scripts/generate_all_figures.py --mode all
    python scripts/generate_all_figures.py --mode workload
    python scripts/generate_all_figures.py --mode models
    python scripts/generate_all_figures.py --mode scheduler
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path before any src.* imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import visualization as vis
from src.data_loading import load_main_sample
from src.feature_engineering import (
    build_job_table_from_sample,
    prepare_features_for_model,
)
from src.simulation import (
    ClusterSimulator,
    FIFOScheduler,
    SJFPredScheduler,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task runners
# ---------------------------------------------------------------------------


def run_workload_tasks(output_dir: str) -> None:
    """Load, normalize, and plot workload characterization figures."""
    logger.info("Loading and normalizing workload data...")
    raw_df = load_main_sample()
    job_df = build_job_table_from_sample(raw_df)

    logger.info("Generating workload characterization figures...")
    vis.plot_workload_summary(job_df, output_dir)
    logger.info("Workload figures written to: %s", output_dir)


def run_model_tasks(output_dir: str) -> None:
    """
    Generate model performance figures using saved RF baseline.
    """
    logger.info("Generating model performance figures (Exp A: RF Numeric)...")
    
    # 1. Prepare data
    _, _, X_test, _, y_test, _, _ = prepare_features_for_model(
        dataset="main", feature_mode="numeric_only"
    )

    # 2. Load model
    model_path = PROJECT_ROOT / "results/models/rf_numeric.joblib"
    if not model_path.exists():
        logger.warning("Model not found at %s. Run Notebook 04 first.", model_path)
        return

    model = joblib.load(model_path)
    
    # 3. Predict & Plot
    y_pred = model.predict(X_test)
    vis.plot_regression_analysis(y_test, y_pred, "Random Forest (Numeric)", output_dir)
    logger.info("Model performance figures written to: %s", output_dir)


def run_scheduler_tasks(output_dir: str) -> None:
    """
    Generate scheduler comparison figures via mini-simulation.
    """
    logger.info("Generating scheduler comparison figures (Mini-Simulation: 1000 jobs)...")
    
    # 1. Prepare simulation workload
    job_df, _, X_test, _, y_test, _, _ = prepare_features_for_model(
        dataset="main", feature_mode="numeric_only"
    )
    
    # 2. Add predictions for SJF-Pred
    model_path = PROJECT_ROOT / "results/models/rf_numeric.joblib"
    if not model_path.exists():
        logger.warning("Model not found at %s. Run Notebook 04 first.", model_path)
        return
    
    model = joblib.load(model_path)
    sim_df = X_test.iloc[:1000].copy()
    sim_df["runtime"] = y_test[:1000]
    sim_df["predicted_runtime"] = model.predict(sim_df.drop(columns=["runtime"]))
    sim_df["job_id"] = range(len(sim_df))
    sim_df["submit_time"] = sim_df["arrival_sec"]

    # 3. Run simulations
    results = []
    
    logger.info("Running FIFO simulation...")
    fifo_res = ClusterSimulator(FIFOScheduler()).run(sim_df)
    fifo_res["policy"] = "FIFO"
    results.append(fifo_res)
    
    logger.info("Running SJF-Pred simulation...")
    sjf_res = ClusterSimulator(SJFPredScheduler()).run(sim_df)
    sjf_res["policy"] = "SJF-Pred (RF)"
    results.append(sjf_res)
    
    all_results = pd.concat(results)
    
    # 4. Plot
    vis.plot_scheduler_comparison(all_results, output_dir)
    logger.info("Scheduler comparison figures written to: %s", output_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Thesis Visualization Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "workload", "models", "scheduler"],
        default="all",
        help="Which figures to generate (default: all).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/figures",
        help="Output directory for figures (default: results/figures).",
    )
    args = parser.parse_args()

    out_dir = str(PROJECT_ROOT / args.out)

    if args.mode in ("all", "workload"):
        run_workload_tasks(out_dir)

    if args.mode in ("all", "models"):
        run_model_tasks(out_dir)

    if args.mode in ("all", "scheduler"):
        run_scheduler_tasks(out_dir)

    logger.info("Task complete. Figures (if any) are in: %s", out_dir)


if __name__ == "__main__":
    main()
