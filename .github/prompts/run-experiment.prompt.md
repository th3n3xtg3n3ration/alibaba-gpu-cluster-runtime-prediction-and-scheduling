---
description: "Run a specific ML experiment (exp_a through exp_f) with full checkpoint saving, model serialization, and figure generation."
agent: "agent"
argument-hint: "Experiment tag (exp_a, exp_b, ...) and model name (rf, xgb, lgbm, cnn, lstm, hybrid)"
tools: [read, edit, execute, todo]
---

Run the ML experiment specified in the user's message.

Read [.github/tasks/run-experiment.md](.github/tasks/run-experiment.md) for the
complete procedure.

## Steps

1. Parse the experiment tag and model name from the user's request
2. Verify pre-flight checks pass (tests, processed dataset present)
3. Load config from `configs/paths.yaml` and `configs/models.yaml`
4. Build the correct feature matrix for the experiment's feature mode:
   - `exp_a`: `mode="numeric"`
   - `exp_b`: `mode="one_hot"`
   - `exp_c/e`: `mode="sequential"` (numeric)
   - `exp_d/f`: `mode="sequential"` (categorical)
5. Run hyperparameter search using `src/tuning.py`
6. Evaluate with `src/models/evaluation.evaluate_regression()` — report all 5 metrics
7. Save checkpoint to `results/checkpoints/exp_{tag}_{model}.json`
8. Save model to `results/models/`
9. Generate and save 3 figures (scatter, residuals, error_cdf)
10. Run tests again to confirm nothing broke: `python -m unittest discover tests -v`

## Constraints

- Use chronological 80/20 split — never shuffle time-series data
- Always seed 42: numpy, random, sklearn's `random_state`
- Never overwrite `data/alibaba_cluster_trace/pai_job_no_estimate_100K.csv`
- If experiment is already complete (checkpoint exists with all 5 metrics), ask before re-running
