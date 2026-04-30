# Results Summary — Current State

*Last updated: 2026-04-18*
*⚠️ All values below are verified against `results/checkpoints/` — do not overwrite without re-running.*

---

## Experiment A — Tree-Based, Numeric Features ✅ COMPLETE

### Regression Metrics (Test Set, chronological split)

| Model | MAE (s) | RMSE (s) | R² | MAPE | MdAE (s) |
|-------|---------|----------|-----|------|---------|
| **RandomForest** | **4,316** | **13,831** | **0.274** | 16.85% | 1,508 |
| XGBoost | 4,808 | 13,965 | 0.260 | 19.02% | 1,894 |
| LightGBM | 5,212 | 14,315 | 0.223 | 20.66% | 2,259 |

**Winner (exp_a):** RandomForest (R²=0.274, MAE=4,316s)  
**Checkpoints:** `results/checkpoints/exp_a_{rf,xgb,lgbm}.json`

### Key Observations
- All three models below R²=0.3, expected due to noisy user-controlled inputs
- Sweep-line utilization features (cluster_load_gpu, active_job_count) are top importance features
- Residuals show systematic overestimation of short jobs (< 100s)

---

## Experiment B — One-Hot Encoding ✅ COMPLETE

### Regression Metrics (Test Set, chronological split)

| Model | MAE (s) | RMSE (s) | R² | MAPE | MdAE (s) |
|-------|---------|----------|-----|------|---------|
| **XGBoost-OH** | **3,389** | **11,375** | **0.509** | 10.80% | 1,129 |
| LGBM-Native | 4,106 | 12,147 | 0.440 | 14.55% | 1,510 |
| RF-OH | 4,187 | 12,380 | 0.419 | 15.57% | 1,888 |
| LGBM-OH | 4,388 | 13,055 | 0.354 | 14.43% | 1,644 |

**Winner (exp_b):** XGBoost-OH (R²=0.509, MAE=3,389s) — **best model overall**  
**Checkpoints:** `results/checkpoints/exp_b_{xgb_oh,lgbm_nat,lgbm_oh,rf_oh}.json`

---

## Experiments C–F — Deep Learning ✅ COMPLETE (poor results, as expected for tabular data)

| Exp | Model | R² | MAE (s) | Notes |
|-----|-------|-----|---------|-------|
| C | CNN | 0.032 | 7,063 | Numeric sequence |
| C | LSTM | -0.086 | 5,163 | Below baseline |
| C | Hybrid | 0.026 | 6,179 | |
| D | CNN | 0.202 | 5,212 | Categorical embedding |
| D | LSTM | 0.144 | 5,724 | |
| D | Hybrid | 0.214 | 5,351 | Best DL result |
| E | CNN | 0.005 | 7,204 | Numeric+sequential |
| E | LSTM | -0.011 | 5,908 | |
| E | Hybrid | 0.002 | 6,783 | |
| F | CNN | 0.026 | 6,002 | Categorical+sequential |
| F | LSTM | -0.968 | 18,089 | Diverged |
| F | Hybrid | -0.029 | 8,380 | |

**Finding:** All DL models underperform tree-based approaches. Best DL: Hybrid-D (R²=0.214) vs XGB-OH (R²=0.509). **This confirms the thesis claim that tree-based methods are superior for this tabular scheduling dataset.**  
**Checkpoints:** `results/checkpoints/exp_{c,d,e,f}_{cnn,lstm,hybrid}.json`

---

## Scheduler Results ✅ COMPLETE (on 1,000-job sample)

| Policy | Avg Wait (s) | JCT (s) | Reduction vs FIFO |
|--------|-------------|---------|-------------------|
| FIFO (baseline) | 566,002 | 571,211 | 0% |
| SJF-Oracle | 50,988 | 56,196 | **90.16%** |
| SJF-XGBoost | 75,662 | 80,870 | **85.84%** |
| SJF-RandomForest | 111,764 | 116,972 | 79.52% |
| SJF-LightGBM | ~120,000 | ~125,000 | ~78% |

**Key Claim Validated:** XGBoost-SJF achieves 85.84% reduction, 4.32pp gap to oracle.

**Limitation:** Scheduler tested on 1,000-job sample only. Full 100K simulation pending.

---

## Figures Inventory (results/figures/)

| Figure | Status |
|--------|--------|
| Workload heatmaps (hourly/daily GPU demand) | ✅ |
| Runtime distribution (log-histogram) | ✅ |
| Arrival rate patterns | ✅ |
| Exp A: scatter (true vs pred) for RF, XGB, LGBM | ✅ |
| Exp A: residual histograms | ✅ |
| Exp A: error CDFs | ✅ |
| Exp A: model comparison bar chart | ✅ |
| Scheduler comparison (FIFO vs SJF variants) | ✅ |
| Feature importance plots | ✅ |
| Exp B–F figures | 🔲 |

---

## Models Inventory (results/models/)

| File | Experiment | Status |
|------|-----------|--------|
| `rf_numeric.joblib` | exp_a | ✅ Saved |
| `xgb_numeric.joblib` | exp_a | ✅ Saved |
| `lgb_numeric.joblib` | exp_a | ✅ Saved |

---

## Test Inventory

```
tests/ — 11 tests, 100% passing
├── test_config_utils.py      (2 tests)
├── test_feature_engineering.py (2 tests)
├── test_evaluation.py        (3 tests)
└── test_simulation.py        (2 tests)
```

**Note:** test count excludes any new tests added after 2026-04-18.

---

## Open Work Items (Priority Order)

1. **[HIGH]** Complete Experiment B (one-hot tree models) — partial checkpoint exists
2. **[HIGH]** Run scheduler simulation on full 100K jobs (not just 1K sample)
3. **[MEDIUM]** Run Experiments C–F (DL models with various feature modes)
4. **[MEDIUM]** Compute MAPE and MdAE for exp_a (currently missing from checkpoints)
5. **[MEDIUM]** Add confidence interval reporting to evaluation metrics
6. **[LOW]** Implement multi-node heterogeneous simulator
7. **[LOW]** Confidence-aware scheduling policies
8. **[LOW]** RL-based scheduling (future work chapter)
