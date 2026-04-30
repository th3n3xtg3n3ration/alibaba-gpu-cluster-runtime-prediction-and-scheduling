# Project Overview — GPU Scheduling Thesis

## Identity

**Project:** Runtime Prediction and Scheduling Optimization in Large-Scale GPU Clusters  
**Researcher:** MSc → PhD candidate  
**Academic Context:** 7-chapter PhD thesis in Computer Science / Systems  
**Status:** Active research — Experiment A complete, B–F in progress

---

## Research Motivation

Modern MLaaS platforms (like Alibaba PAI, Google Borg, Microsoft Azure ML) rely on
user-submitted runtime estimates to schedule GPU jobs. These estimates are systematically
inaccurate. The thesis demonstrates that a learned predictor can replace human estimates
and reduce cluster waiting times by ~85.8% with no additional hardware.

---

## Key Claim

> SJF-Pred (Shortest Job First using ML-predicted runtimes) reduces average cluster
> waiting time by **85.84%** vs FIFO baseline (from 566,002s to 75,662s per job on
> average), approaching the theoretical oracle limit of 90.16% (SJF with true runtimes).

---

## Technical Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10 |
| ML (trees) | scikit-learn (RandomForest), XGBoost, LightGBM |
| ML (deep) | PyTorch (CNN, LSTM, CNN-LSTM Hybrid) |
| Data | pandas, numpy, pyarrow |
| Visualization | matplotlib, seaborn, plotly |
| Config | PyYAML |
| Simulation | Custom discrete-event engine (heapq + heapdict) |
| Environment | conda (environment.yaml) |
| Type checking | Pyright (pyrightconfig.json) |

---

## Repository Layout

```
.github/                 ← Agent customization (instructions, agents, skills, prompts)
AGENTS.md                ← Authoritative project guidelines for all agents
src/                     ← Production Python library
notebooks/en/            ← English Jupyter notebooks (6 phases)
notebooks/tr/            ← Turkish counterparts
configs/                 ← paths.yaml, models.yaml
data/                    ← Raw + processed datasets
results/                 ← figures, models, checkpoints
tests/                   ← 11 unit tests
docs/                    ← thesis_outline.md
scripts/                 ← Pipeline scripts
scratch/                 ← Temporary patch scripts (do not import)
```

---

## Core Pipeline

```
Raw CSV (100K jobs)
    │
    ▼ [00_data_preparation]
    Normalized DataFrame + Sweep-line utilization features
    │
    ▼ [03_feature_engineering]
    Feature matrix (numeric, one-hot, sequential variants)
    │
    ▼ [04_runtime_prediction_models]
    Trained predictors (RF, XGB, LGBM, CNN, LSTM, Hybrid)
    │
    ▼ [05_scheduler_evaluation]
    Discrete-event simulation → FIFO vs SJF-Oracle vs SJF-Pred
    │
    ▼
    Thesis results: ~85.8% wait-time reduction
```

---

## Reproducibility Commitments

1. All random seeds fixed at `42`
2. Chronological 80/20 train/test split (no temporal leakage)
3. Hyperparameter grids versioned in `configs/models.yaml`
4. Checkpoint JSONs record all metrics + parameters + timestamp
5. conda `environment.yaml` pins all dependency versions
6. Dynamic project root discovery (works on any machine)

---

## Related Work Areas

- Cluster scheduling: Borg, Mesos, YARN, Tiresias, Gandiva, Optimus
- ML for systems: Ernest, Cherrypick, Clipper, Reef
- Workload characterization: Alibaba 2017/2022 traces, Google cluster traces
- Runtime prediction: Corral, DL-based predictors, feature engineering surveys
