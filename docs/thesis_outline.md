# Thesis Outline: Multi-Paradigm Runtime Prediction and Heterogeneous Scheduling Optimization

**Project Title:** Multi-Paradigm Runtime Prediction and Heterogeneous Scheduling Optimization in Large-Scale GPU Clusters  
**Repository:** `alibaba-gpu-runtime-prediction-and-scheduling`  
**Author:** Hasan Uğur Çelebi  
**Year:** 2026  
**University:** Yeditepe University  

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

### 1.1 Problem Definition and Context (General)
- How to utilize GPU and CPU resources most efficiently in large-scale environments?
- The critical importance of hardware efficiency in massive Cloud Computing platforms (e.g., Alibaba).
- The necessity of shared infrastructure in multi-tenant GPU clusters.

### 1.2 Causes of the Problem (Narrowing Down)
- The inherent chaos and unpredictability of cloud workloads.
- Fluctuations in Alibaba trace jobs based on days of the week and hours of the day.
- Bursty traffic patterns and how heavy-tailed (long-running) jobs lead to severe resource fragmentation and system bottlenecks.

### 1.3 Search for a Solution and Scheduling (Specific)
- How is scheduling currently performed in existing systems? (e.g., FIFO, simple heuristic rules).
- Can we develop smarter, AI-driven, and more efficient scheduling solutions?
- Is it possible to establish a successful scheduling mechanism by utilizing historical workload data? (Answer: Yes).

### 1.4 Motivation, Definition, and Scope
- Initiating the development of a "Predictive Scheduling" solution using the Alibaba dataset.
- Defining the scope through time-series analysis, data cleaning, and the implementation of a Multi-Node Event-Driven Simulator.

### 1.5 Research Questions
1. Can Machine Learning (ML) and Deep Learning (DL) algorithms be successfully utilized for job scheduling problems in cloud computing clusters?
2. Which artificial intelligence methods are more suitable for predicting high-variance and chaotic workload runtimes in real-world (Alibaba) data?
3. To what extent do heuristic approaches (e.g., Smallest Resource First) that only consider hardware demand fall short against machine learning-supported (SJF-Pred) policies?
4. What is the performance disparity between tree-based models (XGBoost/LightGBM) and time-series-focused deep learning models (LSTM) when handling categorical (tabular) data in this domain?

### 1.6 Aims and Objectives
1. To clean and preprocess a massive 100,000-row Alibaba dataset to extract contextual cluster utilization features.
2. To develop multi-paradigm (Tree vs. DL) predictive models capable of estimating job runtimes accurately.
3. To make the models' decision-making processes explainable (Explainable AI) using Feature Importance (MDA/MDI) methods.
4. To code and deploy an advanced Multi-Node Simulator to test the real-world impact of these predictions.
5. To empirically prove how much ML-assisted scheduling (SJF-XGBoost) reduces waiting times compared to simple rule-based (Heuristic/FIFO) methods.

---

## 2. Related Work

### 2.1 Cluster Scheduling Theory (Classical Approaches)
- Traditional algorithms: FIFO, SJF, and simple rule-based (Heuristic / SRF) mechanisms.
- Backfilling and its critical limitations.

### 2.2 Machine Learning (ML) and Deep Learning (DL) for Scheduling
- Extensive literature review of ML/DL applications in cluster scheduling.
- Tabular data modeling: The dominance of Gradient Boosting (XGBoost and LightGBM).
- Time Series and System Telemetry: The role and limitations of LSTM and 1D-CNNs in scheduling literature.
- MLaaS datasets (Alibaba PAI NSDI '22 paper and related research).

---

## 3. Dataset and Workload Characterization

### 3.1 Overview of Alibaba GPU Trace  
- Scope and cleaning: Handling 100k+ job records.  
- Data Filtering: Validity checks for runtime (duration > 0) and resources.

### 3.2 Feature Engineering Pipeline  
- **Static Features:** Job/Instance counts, GPU/CPU/Memory requests.  
- **Temporal Features:** Time properties (hour, day) are extracted as raw integers. Categorical metadata uses One-Hot Encoding.
- **Utilization Features (The Sweep-Line):**  
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
