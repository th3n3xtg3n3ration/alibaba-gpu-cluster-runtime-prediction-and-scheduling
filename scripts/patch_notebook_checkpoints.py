#!/usr/bin/env python3
"""
Patches 04_runtime_prediction_models.ipynb to add checkpoint saves after each experiment.
This ensures that if the kernel crashes mid-run, completed results are preserved on disk.
"""
import json
import sys
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "en" / "04_runtime_prediction_models.ipynb"

def load_notebook(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_notebook(path, nb):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

def find_cell_by_source_prefix(cells, prefix):
    """Find the index of a cell whose joined source starts with prefix."""
    for i, cell in enumerate(cells):
        src = "".join(cell.get("source", []))
        if src.strip().startswith(prefix):
            return i
    return None

def make_code_cell(source_lines, cell_id=None):
    """Create a new code cell."""
    import uuid
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id or str(uuid.uuid4())[:8],
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }

def main():
    nb = load_notebook(NB_PATH)
    cells = nb["cells"]
    
    # 1. Patch import cell to include save_checkpoint, load_all_checkpoints
    import_idx = find_cell_by_source_prefix(cells, "# ── 1. Imports")
    if import_idx is not None:
        src = cells[import_idx]["source"]
        # Check if already patched
        joined = "".join(src)
        if "save_checkpoint" not in joined:
            # Find the line with "from src.tuning import ("
            for i, line in enumerate(src):
                if "from src.tuning import (" in line:
                    break
            # Find the closing ")" 
            for j in range(i, len(src)):
                if src[j].strip().startswith(")"):
                    # Insert before the ")"
                    src.insert(j, "    save_checkpoint,\n")
                    src.insert(j+1, "    load_all_checkpoints,\n")
                    break
            cells[import_idx]["source"] = src
            print(f"[OK] Patched import cell (index {import_idx})")
    
    # 2. Define checkpoint save snippets for each experiment
    checkpoints = [
        # After Exp A RF (cell starting with "# ── 4. Random Forest")
        ("# ── 4. Random Forest", "ckpt_a_rf", [
            "# ── Checkpoint: Save Exp A / RF results ──────────────────────────────────\n",
            "save_checkpoint('exp_a_rf', {'metrics': rf_final_metrics, 'best_params': rf_gs_best})\n",
        ]),
        # After Exp A XGB
        ("# ── 5. XGBoost", "ckpt_a_xgb", [
            "# ── Checkpoint: Save Exp A / XGB results ─────────────────────────────────\n",
            "save_checkpoint('exp_a_xgb', {'metrics': xgb_final_metrics, 'best_params': xgb_gs_best})\n",
        ]),
        # After Exp A LGB
        ("# ── 6. LightGBM", "ckpt_a_lgb", [
            "# ── Checkpoint: Save Exp A / LGB results ─────────────────────────────────\n",
            "save_checkpoint('exp_a_lgb', {'metrics': lgb_final_metrics, 'best_params': lgb_gs_best})\n",
        ]),
        # After Exp B RF OH
        ("# ── 9. RF (One-Hot)", "ckpt_b_rf", [
            "# ── Checkpoint: Save Exp B / RF+OH results ───────────────────────────────\n",
            "save_checkpoint('exp_b_rf_oh', {'metrics': rf_oh_final_metrics, 'best_params': rf_oh_gs_best})\n",
        ]),
        # After Exp B XGB OH
        ("# ── 10. XGBoost (One-Hot)", "ckpt_b_xgb", [
            "# ── Checkpoint: Save Exp B / XGB+OH results ──────────────────────────────\n",
            "save_checkpoint('exp_b_xgb_oh', {'metrics': xgb_oh_final_metrics, 'best_params': xgb_oh_gs_best})\n",
        ]),
        # After Exp B LGB OH
        ("# ── 11. LightGBM (One-Hot)", "ckpt_b_lgb_oh", [
            "# ── Checkpoint: Save Exp B / LGB+OH results ──────────────────────────────\n",
            "save_checkpoint('exp_b_lgb_oh', {'metrics': lgb_oh_final_metrics, 'best_params': lgb_oh_gs_best})\n",
        ]),
        # After Exp B LGB Native
        ("# ── 12. LightGBM (Native Categorical)", "ckpt_b_lgb_nat", [
            "# ── Checkpoint: Save Exp B / LGB+Native results ──────────────────────────\n",
            "save_checkpoint('exp_b_lgb_nat', {'metrics': lgb_nat_final_metrics, 'best_params': lgb_nat_gs_best})\n",
        ]),
        # After Exp C CNN
        ("# ── 15. CNN (Numeric-Only)", "ckpt_c_cnn", [
            "# ── Checkpoint: Save Exp C / CNN results ─────────────────────────────────\n",
            "save_checkpoint('exp_c_cnn', {'metrics': cnn_metrics_num, 'best_params': best_cnn_gs_params_num})\n",
        ]),
        # After Exp C LSTM
        ("# ── 16. LSTM (Numeric-Only)", "ckpt_c_lstm", [
            "# ── Checkpoint: Save Exp C / LSTM results ────────────────────────────────\n",
            "save_checkpoint('exp_c_lstm', {'metrics': lstm_metrics_num, 'best_params': best_lstm_gs_params_num})\n",
        ]),
        # After Exp C Hybrid
        ("# ── 17. CNN-LSTM Hybrid (Numeric-Only)", "ckpt_c_hybrid", [
            "# ── Checkpoint: Save Exp C / Hybrid results ──────────────────────────────\n",
            "save_checkpoint('exp_c_hybrid', {'metrics': hybrid_metrics_num, 'best_params': best_hybrid_gs_params_num})\n",
        ]),
        # After Exp D CNN
        ("# ── 19. CNN (One-Hot)", "ckpt_d_cnn", [
            "# ── Checkpoint: Save Exp D / CNN results ─────────────────────────────────\n",
            "save_checkpoint('exp_d_cnn', {'metrics': cnn_metrics_cat, 'best_params': best_cnn_gs_params_cat})\n",
        ]),
        # After Exp D LSTM
        ("# ── 20. LSTM (One-Hot)", "ckpt_d_lstm", [
            "# ── Checkpoint: Save Exp D / LSTM results ────────────────────────────────\n",
            "save_checkpoint('exp_d_lstm', {'metrics': lstm_metrics_cat, 'best_params': best_lstm_gs_params_cat})\n",
        ]),
        # After Exp D Hybrid
        ("# ── 21. CNN-LSTM Hybrid (One-Hot)", "ckpt_d_hybrid", [
            "# ── Checkpoint: Save Exp D / Hybrid results ──────────────────────────────\n",
            "save_checkpoint('exp_d_hybrid', {'metrics': hybrid_metrics_cat, 'best_params': best_hybrid_gs_params_cat})\n",
        ]),
        # After Exp E CNN
        ("# ── 23. CNN (Sequence Numeric)", "ckpt_e_cnn", [
            "# ── Checkpoint: Save Exp E / CNN results ─────────────────────────────────\n",
            "save_checkpoint('exp_e_cnn', {'metrics': cnn_metrics_num_seq, 'best_params': best_cnn_gs_params_num_seq})\n",
        ]),
        # After Exp E LSTM
        ("# ── 24. LSTM (Sequence Numeric)", "ckpt_e_lstm", [
            "# ── Checkpoint: Save Exp E / LSTM results ────────────────────────────────\n",
            "save_checkpoint('exp_e_lstm', {'metrics': lstm_metrics_num_seq, 'best_params': best_lstm_gs_params_num_seq})\n",
        ]),
        # After Exp E Hybrid
        ("# ── 25. CNN-LSTM Hybrid (Sequence Numeric)", "ckpt_e_hybrid", [
            "# ── Checkpoint: Save Exp E / Hybrid results ──────────────────────────────\n",
            "save_checkpoint('exp_e_hybrid', {'metrics': hybrid_metrics_num_seq, 'best_params': best_hybrid_gs_params_num_seq})\n",
        ]),
        # After Exp F CNN
        ("# ── 27. CNN (Sequence One-Hot)", "ckpt_f_cnn", [
            "# ── Checkpoint: Save Exp F / CNN results ─────────────────────────────────\n",
            "save_checkpoint('exp_f_cnn', {'metrics': cnn_metrics_cat_seq, 'best_params': best_cnn_gs_params_cat_seq})\n",
        ]),
        # After Exp F LSTM
        ("# ── 28. LSTM (Sequence One-Hot)", "ckpt_f_lstm", [
            "# ── Checkpoint: Save Exp F / LSTM results ────────────────────────────────\n",
            "save_checkpoint('exp_f_lstm', {'metrics': lstm_metrics_cat_seq, 'best_params': best_lstm_gs_params_cat_seq})\n",
        ]),
        # After Exp F Hybrid
        ("# ── 29. CNN-LSTM Hybrid (Sequence One-Hot)", "ckpt_f_hybrid", [
            "# ── Checkpoint: Save Exp F / Hybrid results ──────────────────────────────\n",
            "save_checkpoint('exp_f_hybrid', {'metrics': hybrid_metrics_cat_seq, 'best_params': best_hybrid_gs_params_cat_seq})\n",
        ]),
    ]
    
    # Insert checkpoint cells AFTER the matching experiment cells
    inserted = 0
    for prefix, cell_id, source_lines in checkpoints:
        idx = find_cell_by_source_prefix(cells, prefix)
        if idx is not None:
            # Check if checkpoint already exists right after
            if idx + 1 < len(cells):
                next_src = "".join(cells[idx + 1].get("source", []))
                if "save_checkpoint" in next_src and cell_id.replace("ckpt_", "exp_") in next_src:
                    continue  # Already patched
            
            new_cell = make_code_cell(source_lines, cell_id)
            cells.insert(idx + 1, new_cell)
            inserted += 1
            print(f"[OK] Inserted checkpoint after '{prefix[:40]}...' (cell_id={cell_id})")
    
    # 3. Patch the "Overall Results Table" cell to load from checkpoints if variables are missing
    results_idx = find_cell_by_source_prefix(cells, "# ── 30. Overall Results Table")
    if results_idx is not None:
        old_src = cells[results_idx]["source"]
        joined = "".join(old_src)
        if "load_all_checkpoints" not in joined:
            # Prepend checkpoint loading code
            loader_lines = [
                "# ── 30. Overall Results Table ─────────────────────────────────────────────────\n",
                "import pandas as pd\n",
                "from IPython.display import display\n",
                "# Load results from disk checkpoints (survives kernel crashes!)\n",
                "ckpts = load_all_checkpoints()\n",
                "def _m(name, key='metrics'):\n",
                "    \"\"\"Get metrics from memory variable or disk checkpoint.\"\"\"\n",
                "    return ckpts.get(name, {}).get(key, {})\n",
                "\n",
                "# Use in-memory variables if available, otherwise fall back to checkpoints\n",
                "rf_final_metrics     = rf_final_metrics     if 'rf_final_metrics'     in dir() else _m('exp_a_rf')\n",
                "xgb_final_metrics    = xgb_final_metrics    if 'xgb_final_metrics'    in dir() else _m('exp_a_xgb')\n",
                "lgb_final_metrics    = lgb_final_metrics    if 'lgb_final_metrics'    in dir() else _m('exp_a_lgb')\n",
                "rf_oh_final_metrics  = rf_oh_final_metrics  if 'rf_oh_final_metrics'  in dir() else _m('exp_b_rf_oh')\n",
                "xgb_oh_final_metrics = xgb_oh_final_metrics if 'xgb_oh_final_metrics' in dir() else _m('exp_b_xgb_oh')\n",
                "lgb_oh_final_metrics = lgb_oh_final_metrics if 'lgb_oh_final_metrics' in dir() else _m('exp_b_lgb_oh')\n",
                "lgb_nat_final_metrics= lgb_nat_final_metrics if 'lgb_nat_final_metrics' in dir() else _m('exp_b_lgb_nat')\n",
                "cnn_metrics_num      = cnn_metrics_num      if 'cnn_metrics_num'      in dir() else _m('exp_c_cnn')\n",
                "lstm_metrics_num     = lstm_metrics_num     if 'lstm_metrics_num'     in dir() else _m('exp_c_lstm')\n",
                "hybrid_metrics_num   = hybrid_metrics_num   if 'hybrid_metrics_num'   in dir() else _m('exp_c_hybrid')\n",
                "cnn_metrics_cat      = cnn_metrics_cat      if 'cnn_metrics_cat'      in dir() else _m('exp_d_cnn')\n",
                "lstm_metrics_cat     = lstm_metrics_cat     if 'lstm_metrics_cat'     in dir() else _m('exp_d_lstm')\n",
                "hybrid_metrics_cat   = hybrid_metrics_cat   if 'hybrid_metrics_cat'   in dir() else _m('exp_d_hybrid')\n",
                "cnn_metrics_num_seq    = cnn_metrics_num_seq    if 'cnn_metrics_num_seq'    in dir() else _m('exp_e_cnn')\n",
                "lstm_metrics_num_seq   = lstm_metrics_num_seq   if 'lstm_metrics_num_seq'   in dir() else _m('exp_e_lstm')\n",
                "hybrid_metrics_num_seq = hybrid_metrics_num_seq if 'hybrid_metrics_num_seq' in dir() else _m('exp_e_hybrid')\n",
                "cnn_metrics_cat_seq    = cnn_metrics_cat_seq    if 'cnn_metrics_cat_seq'    in dir() else _m('exp_f_cnn')\n",
                "lstm_metrics_cat_seq   = lstm_metrics_cat_seq   if 'lstm_metrics_cat_seq'   in dir() else _m('exp_f_lstm')\n",
                "hybrid_metrics_cat_seq = hybrid_metrics_cat_seq if 'hybrid_metrics_cat_seq' in dir() else _m('exp_f_hybrid')\n",
                "\n",
            ]
            # Replace the first 3 lines (old header + imports) with the new loader
            cells[results_idx]["source"] = loader_lines + old_src[3:]  # Keep everything after 'from IPython...'
            print(f"[OK] Patched 'Overall Results Table' cell to load from checkpoints")
    
    # Save
    save_notebook(NB_PATH, nb)
    print(f"\n{'='*60}")
    print(f"  DONE — {inserted} checkpoint cells inserted")
    print(f"  Final summary cell patched to auto-load from disk")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
