# Thesis Outline: Multi-Paradigm Runtime Prediction and Heterogeneous Scheduling Optimization

**Project Title:** Multi-Paradigm Runtime Prediction and Heterogeneous Scheduling Optimization in Large-Scale GPU Clusters  
**Repository:** `alibaba-gpu-runtime-prediction-and-scheduling`  
**Author:** Ugur Celebi  
**Year:** 2025  

---

## 0. Front Matter
- Title Page
- Approval Page
- Acknowledgements
- **Abstract (English/Turkish):** Bridging the gap between tabular workload characterization and multi-resource cluster efficiency.
- Table of Contents
- List of Figures / List of Tables

---

## 1. Introduction

### 1.1 Background and Context  
- Large-scale distributed deep learning workloads and the explosion of GPU demand.  
- Managed MLaaS Platforms: Alibaba PAI, AWS SageMaker, and Google Vertex.  
- Multi-tenant GPU clusters: The necessity of shared infrastructure.

### 1.2 Motivation: The Efficiency Gap  
- Inaccuracy of user-provided runtime estimates and its impact on SLAs.  
- Resource fragmentation and the hidden costs of "Black Box" job durations.  
- Initial observations from the Alibaba GPU trace: Heavy-tails and burstiness.

### 1.3 Problem Statement & Objectives  
- **Objective 1:** Predictive Modeling — Can we outperform heuristics using multi-paradigm (Tree vs. DL) models?  
- **Objective 2:** Utilization Engineering — How does cluster-wide offered load influence prediction accuracy?  
- **Objective 3:** Simulation — Building a heterogeneous, multi-node event-driven framework for policy evaluation.

### 1.4 Global Contributions  
- Implementation of a high-performance (O(N log N)) sweep-line utilization tracker.  
- Comparative analysis of Gradient Boosting (XGB/LGBM) vs. Sequence Models (CNN/LSTM).  
- Development of a robust Heterogeneous Simulator with multi-resource constraints (CPU/GPU/Mem).  
- A fully reproducible, standardized "Elite" research repository.

---

## 2. Related Work

### 2.1 Cluster Scheduling Theory  
- Classic algorithms: FIFO, SJF, and Priority-based dispatching.  
- Backfilling and its critical dependence on runtime limits.

### 2.2 The "MLaaS in the Wild" Milestone (NSDI ’22)  
- Detailed review of the original Alibaba PAI workload analysis.  
- Limitations of simple User/Group/GPU heuristics in heterogeneous environments.

### 2.3 ML-Based Systems Optimization  
- Tabular modeling: The continued dominance of XGBoost/LightGBM.  
- Neural approaches: Using 1D-CNNs and LSTMs for system telemetry.

---

## 3. Dataset and Workload Characterization

### 3.1 Overview of Alibaba GPU Trace  
- Scope and cleaning: Handling 100k+ job records.  
- Data Filtering: Validity checks for runtime (duration > 0) and resources.

### 3.2 Feature Engineering Pipeline  
- **Static Features:** Job/Instance counts, GPU/CPU/Memory requests.  
- **Temporal Features:** Cyclical encoding of hour and weekday.  
- ** utilization Features (The Sweep-Line):**  
  - Background CPU/GPU load at arrival.  
  - Concurrently active job counts.

### 3.3 Workload Analysis (Visual Characterization)  
- **3.3.1 Runtime Distributions:** Log-histogram and CDF analysis (Heavy-tail verification).  
- **3.3.2 Arrival Patterns:** Hourly/Daily arrival rates and heatmap visualization.  
- **3.3.3 Resource Footprint:** GPU demand histograms and demand-runtime correlations.

---

## 4. Multi-Paradigm Runtime Prediction Models

### 4.1 Tree-Based Intelligence  
- **Random Forest:** Baseline non-linear regression.  
- **XGBoost & LightGBM:** Optimized Gradient Boosting for heavy-tailed data.

### 4.2 Deep Learning Architectures  
- **1D-CNN:** Mixing tabular features as spatial channels.  
- **LSTM:** Capturing localized job sequences.  
- **CNN-LSTM Hybrid:** Spatial feature extraction with temporal sequencing.

### 4.3 Evaluation Metrics (Scientific Rigor)  
- Primary Metrics: MAE, RMSE, R².  
- Secondary Stability Metrics: MedAE, MAPE, SMAPE.  
- Over-/Under-prediction bias analysis.

### 4.4 Comprehensive Error Analysis  
- **4.4.1 Scatter Analysis:** True vs. Predicted (Linear and Log-Log scales).  
- **4.4.2 Residual Analysis:** Residual histograms and Residual-vs-True plots.  
- **4.4.3 Error CDF:** Absolute error distribution on log-scales.  
- **4.4.4 Runtime-Binned Performance:** Analyzing accuracy across job size intervals.

---

## 5. The Heterogeneous Simulation Framework

### 5.1 Single-Server Simplified Simulation  
- Baseline Policy comparison: FIFO vs. Oracle SJF.  
- Impact of ML-error propagation in "SJF-Pred".

### 5.2 Multi-Node Heterogeneous Simulator  
- **Event-Driven Engine:** Heap-based discrete event management.  
- **Heterogeneous Provisioning:** High-Perf (8-GPU), Mid-Range (2-GPU), and CPU nodes.  
- **Resource Constraints:** Simultaneous CPU, GPU, and Memory accounting.  
- **Placement Logic:** First-Fit vs. Best-Fit in fragmented clusters.

---

## 6. Experimental Results and Discussion

### 6.1 Prediction Accuracy: Trees vs. DL  
- The performance plateau: Why GBDTs remain competitive on tabular data.  
- The role of Utilization features in reducing "Cluster Noise" errors.

### 6.2 Scheduling Outcomes  
- **6.2.1 Waiting Time Reduction:** Order-of-magnitude gains via SJF-Pred.  
- **6.2.2 Percentile Slowdown:** P50/P90/P99 analysis (The "Long Job" problem).  
- **6.2.3 Simulation Scalability:** O(N) performance verification.

### 6.3 Discussion: ML for Systems  
- Reliability of ML-estimates in high-flux environments.  
- Comparison with NSDI '22 heuristic baselines.

---

## 7. Conclusions and Future Work

### 7.1 Summary of Contributions  
- The bridge between predictive modeling and verifiable systems simulation.

### 7.2 Research Limitations  
- Static trace constraints and lack of real-time preemption data.

### 7.3 Future Work  
- **RL-Sched:** Reinforcement Learning for dynamic placement.  
- **Confidence-Aware Scheduling:** Using prediction intervals to manage risks.

---

## Appendix
- **Appendix A:** Automated Unit Test Suite Results (11/11 Pass).  
- **Appendix B:** Hyperparameter Tuning Grids.  
- **Appendix C:** List of Features and Sweep-Line implementation.  
- **Appendix D:** Full Scalability Profiles.