# Experiment Map — Detailed Reference

## Experiment Matrix

| Tag | Models | Feature Mode | Status | Checkpoint |
|-----|--------|--------------|--------|-----------|
| `exp_a` | RF, XGB, LGBM | Numeric only | ✅ Complete | `exp_a_{rf,xgb,lgbm}.json` |
| `exp_b` | RF-OH, XGB-OH, LGBM-OH, LGBM-Native | One-Hot / Native Categorical | ✅ Complete | `exp_b_{rf_oh,xgb_oh,lgbm_oh,lgbm_nat}.json` |
| `exp_c` | CNN, LSTM, Hybrid | Numeric sequence | ✅ Complete | `exp_c_{cnn,lstm,hybrid}.json` |
| `exp_d` | CNN, LSTM, Hybrid | Categorical embedding | ✅ Complete | `exp_d_{cnn,lstm,hybrid}.json` |
| `exp_e` | CNN, LSTM, Hybrid | Numeric + sequential | ✅ Complete | `exp_e_{cnn,lstm,hybrid}.json` |
| `exp_f` | CNN, LSTM, Hybrid | Categorical + sequential | ✅ Complete | `exp_f_{cnn,lstm,hybrid}.json` |

---

## Experiment A — Tree-Based, Numeric Features (COMPLETE)

**Purpose:** Baseline performance of gradient boosting + random forest with minimal
feature engineering (numeric columns only, no categorical encoding).

**Feature Set:**
- `num_gpu`, `num_cpu`, `num_inst` (resource requests)
- `hour_of_day`, `day_of_week` (cyclical-encoded temporal)
- `cluster_load_gpu`, `cluster_load_cpu`, `active_job_count` (sweep-line utilization)

**Results (from checkpoints):**

| Model | MAE (s) | RMSE (s) | R² |
|-------|---------|----------|-----|
| XGBoost | ~3,000 | ~11,174 | 0.53 |
| LightGBM | ~4,070 | ~12,036 | 0.45 |
| RandomForest | ~4,187 | ~12,380 | 0.42 |

**Winner:** XGBoost (R²=0.53)  
**Key Finding:** Sweep-line utilization features contribute significantly to R².

---

## Experiment B — Tree-Based, One-Hot Encoding (PARTIAL)

**Purpose:** Add categorical features (user, gpu_type) via one-hot encoding to see
if user identity and GPU type improve prediction.

**Feature Set (exp_a + categorical):**
- All exp_a features
- `user` → One-hot encoded (top-K users + "other")
- `gpu_type` → One-hot encoded

**LGBM-Native variant:** Uses LightGBM's native categorical handling instead of one-hot.

**Expected:** One-hot XGBoost and LGBM-Native should outperform exp_a equivalents.

---

## Experiments C–F — Deep Learning

### Exp C: CNN, LSTM, Hybrid — Numeric Sequence
**Purpose:** Test if sequential (time-ordered) presentation of numeric features
enables CNN/LSTM to capture temporal patterns invisible to tree models.

### Exp D: CNN, LSTM, Hybrid — Categorical Embedding
**Purpose:** Learned embeddings for user and gpu_type (vs one-hot); may capture
semantic similarity between users/GPU types.

### Exp E: CNN, LSTM, Hybrid — Numeric + Sequential
**Purpose:** Full numeric feature set with sequential input structure.

### Exp F: CNN, LSTM, Hybrid — Categorical + Sequential
**Purpose:** Richest feature set — both categorical embeddings and sequential
structure. Expected best DL performance.

---

## Scheduling Experiments (Notebook 05)

### Baseline: FIFO
Jobs executed strictly in arrival order. No prediction used.
- Average waiting time: ~566,002s
- Establishes lower bound for improvement.

### Reference: SJF-Oracle
SJF using true runtimes (impossible in practice — theoretical best).
- Average waiting time: ~50,988s
- Improvement over FIFO: 90.16%

### Proposed: SJF-Pred (XGBoost)
SJF using XGBoost predictions from exp_a.
- Average waiting time: ~75,662s
- Improvement over FIFO: 85.84%
- **Gap to oracle:** 4.32 percentage points (due to prediction errors)

### Alternative: SJF-Pred (RandomForest)
- Average waiting time: ~111,764s
- Improvement over FIFO: 79.52%
- Demonstrates that prediction quality directly impacts scheduling.

---

## Completeness Requirements

An experiment is **complete** when ALL of the following exist:
1. `results/checkpoints/exp_{tag}_{model}.json` with all 5 metrics
2. `results/models/{tag}_{model}.joblib` (tree models) or `{tag}_{model}.pt` (DL)
3. `results/figures/exp_{tag}_{model}_scatter.png`
4. `results/figures/exp_{tag}_{model}_error_cdf.png`
5. Notebook narrative updated (EN + TR)
