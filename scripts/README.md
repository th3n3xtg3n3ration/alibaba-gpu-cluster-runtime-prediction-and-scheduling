# 📜 Automation & Pipeline Scripts (`scripts/`)

This directory contains the automated pipeline runners, quality assurance suites, and publication artifact export utilities for the research project.

---

## 📂 File Inventory

| File | Type | Description |
|---|---|---|
| **[`run_all_experiments.sh`](run_all_experiments.sh)** | Bash Shell | Master orchestration pipeline (Automated unit tests + Artifact export). |
| **[`export_thesis_results.py`](export_thesis_results.py)** | Python 3 | Core extraction engine decoding and exporting notebook outputs to high-res PNGs and HTML tables. |

---

## 🚀 1. Master Pipeline Runner (`run_all_experiments.sh`)

Provides a unified command-line interface for running system verification tests and generating thesis deliverables.

```bash
# 1. Default Pipeline (Runs unit tests + Fast artifact export) [RECOMMENDED]
bash scripts/run_all_experiments.sh

# 2. Fast Artifact Export (<1s, extracts existing outputs directly)
bash scripts/run_all_experiments.sh export

# 3. Unit Test Suite Only (11/11 tests pass with 100% success rate)
bash scripts/run_all_experiments.sh test

# 4. Auto-Execute Missing Notebooks & Export
bash scripts/run_all_experiments.sh run

# 5. Force Re-Execution of All Notebooks from Scratch
bash scripts/run_all_experiments.sh force

# 6. Extract from Turkish Notebooks (notebooks/tr/)
bash scripts/run_all_experiments.sh tr
```

---

## ⚙️ 2. Artifact Extraction Engine (`export_thesis_results.py`)

Parses the JSON structure of research notebooks (`01` through `05`), extracting embedded visualizations (Base64 $\rightarrow$ PNG) and Pandas summary tables (HTML) into `results/figures/thesis_export/`.

### Execution Modes:
1. **Fast Extraction (Default — <1 second):**
   ```bash
   python scripts/export_thesis_results.py
   ```
   Directly parses pre-computed outputs, exporting **26 PNG figures** and **17 HTML tables** instantaneously.

2. **Auto-Execution (`--execute` / `-e`):**
   ```bash
   python scripts/export_thesis_results.py --execute
   ```
   If an unexecuted or cleared notebook is detected, it automatically runs the notebook in the background via `nbconvert`, saves outputs back to disk, and extracts the results.

3. **Force Re-Execution (`--force-execute`):**
   ```bash
   python scripts/export_thesis_results.py --force-execute
   ```
   Forces end-to-end re-execution of all notebooks from scratch before extraction.

4. **Language Selection (`--lang tr`):**
   ```bash
   python scripts/export_thesis_results.py --lang tr
   ```
   Extracts results directly from the localized Turkish notebooks.

---

## 📊 Exported Artifact Directory Structure

All generated figures and tables are organized under `results/figures/thesis_export/`:

```
results/figures/thesis_export/
├── png/                            # 26 publication-grade figures (.png)
│   ├── NB01-Figure01.png ... NB01-Figure06.png
│   ├── NB02-Figure01.png ... NB02-Figure03.png
│   ├── NB03-Figure01.png ... NB03-Figure02.png
│   ├── NB04-Figure01.png ... NB04-Figure05.png
│   ├── NB05_32GPU-Figure01.png ... NB05_32GPU-Figure05.png
│   └── NB05_256GPU-Figure01.png ... NB05_256GPU-Figure05.png
│
└── html/                           # 17 formatted HTML benchmark tables (.html)
    ├── NB04_Table01.html ... NB04_Table09.html
    ├── NB05_32GPU_Table01.html ... NB05_32GPU_Table04.html
    └── NB05_256GPU_Table01.html ... NB05_256GPU_Table04.html
```

---

## 🔒 Scientific Integrity & Single Source of Truth

By extracting deliverables directly from Jupyter Notebook cell outputs rather than relying on detached plotting scripts, this pipeline guarantees **100% mathematical and visual consistency** across all thesis chapters, defense slides, and interactive notebooks.
