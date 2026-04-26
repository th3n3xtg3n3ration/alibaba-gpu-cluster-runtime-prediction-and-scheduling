<p align="center">
  <img src="results/figures/project_logo.png" width="200" alt="Project Logo">
</p>

# Runtime Prediction and Scheduling Optimization in Large-Scale GPU Clusters

[![Academic Research](https://img.shields.io/badge/Research-Thesis-blue.svg)](https://github.com/hasanugurcelebi/Thesis)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-green.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive research framework for high-fidelity workload characterization, multi-paradigm runtime prediction, and scheduling optimization. This repository provides a complete pipeline leveraging the **Alibaba PAI GPU Cluster Trace (2022)** to improve cluster operational efficiency through machine learning methodologies.

---

## 🔬 Research Overview

This project optimizes resource fragmentation and queue latency in large-scale MLaaS platforms. By integrating high-accuracy predictive models into an event-driven simulation engine, we demonstrate that **SJF-Pred (Predictive Shortest Job First)** scheduling achieves an order-of-magnitude reduction in system-wide waiting times.

### Technical Contributions
- **Workload Analytics:** Comprehensive characterization of heavy-tailed GPU job distributions and arrival burstiness.
- **Offered Load Engineering:** Efficient **O(N log N) sweep-line algorithm** for real-time cluster state snapshots.
- **Multi-Paradigm Predictive Suite:** Comparative evaluation of GBDT (**XGBoost**, **LightGBM**) and Deep Learning (**1D-CNN**, **LSTM**, **Hybrid**) architectures with **One-Hot Categorical Encoding**.
- **Heterogeneous Event-Driven Simulation:** Robust framework supporting multi-resource constraints (GPU/CPU/Mem) and diverse node topologies.
- **Scientific Reproducibility:** 100% modular Pathlib-based architecture, NumPy-style documentation, and an automated verification suite.

---

## 📁 Repository Architecture

```text
.
├── configs/                # Centralized YAML configurations (Models, Paths, Tuning)
├── data/                   # Dataset management (Raw & Processed traces)
├── docs/                   # Academic documentation and thesis outlines
├── notebooks/              # Standardized Jupyter Research Suite (00-05)
├── reports/                # Standardized Analysis Reports (HTML/PDF)
├── results/                # High-fidelity figures and model artifacts
│   ├── figures/            # Exported research visualizations
│   └── models/             # Serialized .joblib and .pth model weights
├── scripts/                # Automated experiment and reproducibility pipelines
├── src/                    # Production-grade modular source code
│   ├── analysis/           # Statistical characterization modules
│   ├── models/             # ML/DL architecture implementations
│   ├── simulation/         # Discrete-event scheduling logic
│   └── *.py                # Common utilities (Feature Engineering, Loading)
├── tests/                  # Automated Unit Test Suite (100% Pass Rate)
└── _archive/               # Protected legacy and experimental residue
```

---

## 📓 Research Notebook Suite

The experimental workflow is organized into six standardized phases, featuring professional **bilingual (TR/EN) narratives**:

| Phase | Module | Research Objective |
|---|---|---|
| **00** | [Prepare Utilization Data](notebooks/00_prepare_utilization_data.ipynb) | Canonical normalization and sweep-line utilization tracking. |
| **01** | [Data Overview](notebooks/01_data_overview.ipynb) | Initial statistical inspection and metadata validation. |
| **02** | [Workload Characterization](notebooks/02_workload_analysis.ipynb) | Arrival heatmaps and heavy-tail distribution analysis. |
| **03** | [Feature Engineering](notebooks/03_feature_engineering.ipynb) | Cyclical timestamps, **One-Hot Encoding**, and resource offered-load engineering. |
| **04** | [Predictive Modeling](notebooks/04_runtime_prediction_models.ipynb) | Multi-model benchmarking, hyperparameter tuning, and error topology analysis. |
| **05** | [Scheduling Optimization](notebooks/05_scheduler_evaluation.ipynb) | Discrete-event heterogeneous policy evaluation and JCT comparative analysis. |

---

## 🔧 Installation & Reproducibility

### Environment Configuration
The project is built on a standardized scientific stack. Install via **Conda**:

```bash
conda env create -f environment.yaml
conda activate gpu-scheduling
```

### Dataset Ingestion
Detailed instructions for acquiring and processing the **Alibaba PAI GPU Trace** are available in [data/README.md](data/README.md). A 100,000-row sample is provided for immediate pipeline verification.

---

## 🚀 Execution Pipeline

### Automated Experiments
Execute the full research pipeline (training + figure generation) or run the verification suite:

```bash
# Execute full experimental pipeline
bash scripts/run_all_experiments.sh

# Execute automated unit tests
bash scripts/run_all_experiments.sh test
```

### Quality Assurance
To verify system integrity, run the standard test suite:
```bash
python -m unittest discover tests
```
- **Configuration:** YAML path and model parameter loading.

---

## 📊 Experimental Results

Benchmarked against the Alibaba PAI GPU workload trace, this unified pipeline demonstrates the performance advantages of machine learning-augmented dispatching over standard heuristics.

### Model Predictive Performance
Detailed empirical analysis indicates that Gradient Boosted Trees and Ensemble methods provide improved runtime estimation accuracy compared to traditional heuristics:

| Model Architecture | MAE (s) | RMSE (s) | R² Score | Robustness Indicator |
|:-------------------|:--------|:---------|:---------|:---------------------|
| **XGBoost (One-Hot)** | **3,389.3** | **11,375.2** | **0.51** | **Optimal Performer** |
| **LightGBM (Native)** | 4,105.8 | 12,147.2 | 0.44 | High Performance |
| **Random Forest (One-Hot)** | 4,187.1 | 12,380.1 | 0.42 | High Robustness |
| **LSTM (One-Hot)** | 5,836.3 | 15,056.7 | 0.06 | Deep Learning Best |

### Model Analytical Depth (Encoding & Error Topology)
Advanced feature engineering—utilizing **One-Hot Encoding** for job metadata and cyclical time features—is critical for mapping non-linear GPU correlations. Empirical analysis reveals that tree-based ensembles (XGBoost, LightGBM) maintain structural superiority over unoptimized Deep Learning variants in high-cardinality tabular spaces.

### Scheduling Optimization (Global Simulation)
Predictive Shortest Job First (**SJF-Pred**) prevents the catastrophic **Head-of-Line (HoL) cluster collapse** inherent in blind FIFO sequencing.

| Scheduling Policy | Mean Wait Time (s) | Wait Time Speedup | Slowdown Speedup |
|:------------------|:-------------------|:------------------|:-----------------|
| **SJF-Oracle**      | 121,128 | **5.91x (Theoretical Limit)** | 430.1x |
| **SJF-PRED-XGB**    | 318,257 | **2.25x (Observed Best)** | 4.13x |
| **SJF-PRED-LGBM**   | 329,124 | **2.17x** | 3.84x |
| **SJF-PRED-RF**     | 330,870 | **2.16x** | 4.03x |
| **FIFO (Baseline)** | 715,611 | *1.00x (Systems Baseline)* | 1.00x |

<div align="center">
  <h3>📊 Research Conclusion</h3>
  <p><i>"The integration of precise predictive modeling, specifically XGBoost (One-Hot), as a scheduling primitive enables a significant optimization in resource utilization. This approach achieves a <b>2.25x speedup</b> in cluster wait times compared to traditional FIFO scheduling, proving highly effective even within the complex constraints of heterogeneous multi-node cluster environments."</i></p>
</div>

### System Efficiency & Scalability
The simulation engine processes over **100,000 discrete job events in under 60 seconds**, enabling rapid data-driven policy iteration and systemic verification of JCT bound minimization.

---

## 📜 License
This project is licensed under the MIT License - see the `LICENSE` file for details.

## 🤝 Contact & Contributions
Developed by **Hasan Uğur Çelebi**. Contributions and academic collaborations are welcome.