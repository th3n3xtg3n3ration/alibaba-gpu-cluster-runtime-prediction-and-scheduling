#!/usr/bin/env python3
"""Generate the MAE/Spearman vs JCT gain figures for all 18 predictors (32-GPU and 256-GPU)."""

from __future__ import annotations

import gc
import html
import logging
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import load_paths_config
from src.feature_engineering import prepare_features_for_model
from src.models import dl_runtime_predictor  # noqa: F401

LOGGER = logging.getLogger(__name__)
SEED = 42
SEQ_LEN = 10
BATCH_SIZE = 8192

POLICY_TO_PRED_COL = {
    "SJF-RF (Numeric)": "pred_rf_num",
    "SJF-LGBM (Numeric)": "pred_lgb_num",
    "SJF-XGBoost (Numeric)": "pred_xgb_num",
    "SJF-RF (Categorical)": "pred_rf_cat",
    "SJF-LGBM (Categorical)": "pred_lgb_cat",
    "SJF-XGBoost (Categorical)": "pred_xgb_cat",
    "SJF-CNN (Numeric)": "pred_cnn",
    "SJF-LSTM (Numeric)": "pred_lstm",
    "SJF-CNN-LSTM (Numeric)": "pred_cnn_lstm",
    "SJF-CNN (Categorical)": "pred_cnn_cat",
    "SJF-LSTM (Categorical)": "pred_lstm_cat",
    "SJF-CNN-LSTM (Categorical)": "pred_cnn_lstm_cat",
    "SJF-CNN (Numeric Sequence)": "pred_cnn_num_seq",
    "SJF-LSTM (Numeric Sequence)": "pred_lstm_num_seq",
    "SJF-CNN-LSTM (Numeric Sequence)": "pred_hybrid_num_seq",
    "SJF-CNN (Categorical Sequence)": "pred_cnn_cat_seq",
    "SJF-LSTM (Categorical Sequence)": "pred_lstm_cat_seq",
    "SJF-CNN-LSTM (Categorical Sequence)": "pred_hybrid_cat_seq",
}

PLOT_LABELS = {
    "SJF-RF (Numeric)": "RF-Num",
    "SJF-LGBM (Numeric)": "LGBM-Num",
    "SJF-XGBoost (Numeric)": "XGB-Num",
    "SJF-RF (Categorical)": "RF-Cat",
    "SJF-LGBM (Categorical)": "LGBM-Cat",
    "SJF-XGBoost (Categorical)": "XGB-Cat",
    "SJF-CNN (Numeric)": "CNN-Num",
    "SJF-LSTM (Numeric)": "LSTM-Num",
    "SJF-CNN-LSTM (Numeric)": "CNNLSTM-Num",
    "SJF-CNN (Categorical)": "CNN-Cat",
    "SJF-LSTM (Categorical)": "LSTM-Cat",
    "SJF-CNN-LSTM (Categorical)": "CNNLSTM-Cat",
    "SJF-CNN (Numeric Sequence)": "CNN-Num-Seq",
    "SJF-LSTM (Numeric Sequence)": "LSTM-Num-Seq",
    "SJF-CNN-LSTM (Numeric Sequence)": "CNNLSTM-Num-Seq",
    "SJF-CNN (Categorical Sequence)": "CNN-Cat-Seq",
    "SJF-LSTM (Categorical Sequence)": "LSTM-Cat-Seq",
    "SJF-CNN-LSTM (Categorical Sequence)": "CNNLSTM-Cat-Seq",
}


def _load_torch_model(path: Path) -> Any:
    """Load a serialized PyTorch model onto CPU and freeze gradients."""
    model = torch.load(path, map_location="cpu", weights_only=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _predict_in_batches(
    model: Any,
    x_tensor_2d: torch.Tensor,
    *,
    is_sequence: bool,
    seq_len: int = SEQ_LEN,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Run batched inference for static or sequence DL models."""
    predictions: list[np.ndarray] = []
    n_samples = x_tensor_2d.shape[0]

    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            if is_sequence:
                batch_sequences = []
                for row_idx in range(start_idx, end_idx):
                    seq_start = max(0, row_idx - seq_len + 1)
                    sequence = x_tensor_2d[seq_start : row_idx + 1]
                    pad_len = seq_len - sequence.shape[0]
                    if pad_len > 0:
                        padding = sequence[0].unsqueeze(0).repeat(pad_len, 1)
                        sequence = torch.cat([padding, sequence], dim=0)
                    batch_sequences.append(sequence.unsqueeze(0))
                batch_x = torch.cat(batch_sequences, dim=0)
            else:
                batch_x = x_tensor_2d[start_idx:end_idx].unsqueeze(1)

            batch_pred = model(batch_x).cpu().numpy().reshape(-1)
            predictions.append(batch_pred)

            gc.collect()

    return np.concatenate(predictions)


def _load_export_table(path: Path) -> pd.DataFrame:
    """Load one exported HTML evaluation table."""
    if not path.exists():
        raise FileNotFoundError(f"Required exported table not found at {path}")

    html_text = path.read_text(encoding="utf-8")
    row_blocks = re.findall(r"<tr.*?>(.*?)</tr>", html_text, flags=re.DOTALL | re.IGNORECASE)
    if not row_blocks:
        raise ValueError(f"No HTML rows found in {path}")

    rows: list[list[str]] = []
    for row_block in row_blocks:
        cells = re.findall(r"<t[hd].*?>(.*?)</t[hd]>", row_block, flags=re.DOTALL | re.IGNORECASE)
        cleaned_cells = [
            re.sub(r"<.*?>", "", html.unescape(cell)).strip()
            for cell in cells
        ]
        if cleaned_cells:
            rows.append(cleaned_cells)

    if len(rows) < 2:
        raise ValueError(f"Could not parse table rows from {path}")

    header = rows[0][1:]
    data_rows = [row[1:] for row in rows[1:] if len(row) == len(rows[0])]
    return pd.DataFrame(data_rows, columns=header)


def _load_prediction_inputs() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Rebuild the test split and regenerate predictions for the 18 trained predictors."""
    paths = load_paths_config()
    model_dir = PROJECT_ROOT / paths["results"]["models_dir"]

    LOGGER.info("Loading test splits")
    (_, _, x_test_num, _, y_test, _, _) = prepare_features_for_model(
        dataset="main",
        time_unit="s",
        test_size=0.20,
        random_state=SEED,
        feature_mode="numeric_only",
    )
    (_, _, x_test_cat_oh, _, _, _, _) = prepare_features_for_model(
        dataset="main",
        time_unit="s",
        test_size=0.20,
        random_state=SEED,
        feature_mode="with_categorical_onehot",
    )
    (_, _, x_test_cat_base, _, _, _, cat_cols_base) = prepare_features_for_model(
        dataset="main",
        time_unit="s",
        test_size=0.20,
        random_state=SEED,
        feature_mode="with_categorical",
    )
    for col in cat_cols_base:
        x_test_cat_base[col] = x_test_cat_base[col].astype("category")

    LOGGER.info("Loading tree models and scalers")
    rf_num = joblib.load(model_dir / "rf_numeric.joblib")
    xgb_num = joblib.load(model_dir / "xgb_numeric.joblib")
    lgb_num = joblib.load(model_dir / "lgbm_numeric.joblib")
    rf_cat = joblib.load(model_dir / "rf_categorical.joblib")
    xgb_cat = joblib.load(model_dir / "xgb_categorical.joblib")
    lgb_cat = joblib.load(model_dir / "lgbm_categorical_native.joblib")
    lstm_scaler_x = joblib.load(model_dir / "lstm_scaler_x.joblib")
    lstm_scaler_y = joblib.load(model_dir / "lstm_scaler_y.joblib")
    lstm_scaler_x_cat = joblib.load(model_dir / "lstm_scaler_x_cat.joblib")
    lstm_scaler_y_cat = joblib.load(model_dir / "lstm_scaler_y_cat.joblib")

    LOGGER.info("Generating tree predictions")
    predictions: dict[str, np.ndarray] = {
        "pred_rf_num": rf_num.predict(x_test_num),
        "pred_xgb_num": xgb_num.predict(x_test_num),
        "pred_lgb_num": lgb_num.predict(x_test_num),
        "pred_rf_cat": rf_cat.predict(x_test_cat_oh),
        "pred_xgb_cat": xgb_cat.predict(x_test_cat_oh),
        "pred_lgb_cat": lgb_cat.predict(x_test_cat_base),
    }

    LOGGER.info("Preparing DL tensors")
    torch.set_num_threads(1)
    x_test_num_scaled = lstm_scaler_x.transform(x_test_num)
    x_test_cat_scaled = lstm_scaler_x_cat.transform(x_test_cat_oh)
    x_test_num_t = torch.tensor(x_test_num_scaled, dtype=torch.float32)
    x_test_cat_t = torch.tensor(x_test_cat_scaled, dtype=torch.float32)

    LOGGER.info("Loading DL models")
    cnn_static_num = _load_torch_model(model_dir / "cnn_numeric.pth")
    lstm_static_num = _load_torch_model(model_dir / "lstm_numeric.pth")
    hybrid_static_num = _load_torch_model(model_dir / "cnn_lstm_numeric.pth")
    cnn_static_cat = _load_torch_model(model_dir / "cnn_categorical_pt.pth")
    lstm_static_cat = _load_torch_model(model_dir / "lstm_categorical_pt.pth")
    hybrid_static_cat = _load_torch_model(model_dir / "cnn_lstm_categorical_pt.pth")
    cnn_num_seq = _load_torch_model(model_dir / "cnn_numeric_seq.pth")
    lstm_num_seq = _load_torch_model(model_dir / "lstm_numeric_seq.pth")
    hybrid_num_seq = _load_torch_model(model_dir / "cnn_lstm_numeric_seq.pth")
    cnn_cat_seq = _load_torch_model(model_dir / "cnn_categorical_seq.pth")
    lstm_cat_seq = _load_torch_model(model_dir / "lstm_categorical_seq.pth")
    hybrid_cat_seq = _load_torch_model(model_dir / "cnn_lstm_categorical_seq.pth")

    LOGGER.info("Generating DL predictions")
    predictions.update(
        {
            "pred_cnn": lstm_scaler_y.inverse_transform(
                _predict_in_batches(cnn_static_num, x_test_num_t, is_sequence=False).reshape(-1, 1)
            ).reshape(-1),
            "pred_lstm": lstm_scaler_y.inverse_transform(
                _predict_in_batches(lstm_static_num, x_test_num_t, is_sequence=False).reshape(-1, 1)
            ).reshape(-1),
            "pred_cnn_lstm": lstm_scaler_y.inverse_transform(
                _predict_in_batches(hybrid_static_num, x_test_num_t, is_sequence=False).reshape(-1, 1)
            ).reshape(-1),
            "pred_cnn_cat": lstm_scaler_y_cat.inverse_transform(
                _predict_in_batches(cnn_static_cat, x_test_cat_t, is_sequence=False).reshape(-1, 1)
            ).reshape(-1),
            "pred_lstm_cat": lstm_scaler_y_cat.inverse_transform(
                _predict_in_batches(lstm_static_cat, x_test_cat_t, is_sequence=False).reshape(-1, 1)
            ).reshape(-1),
            "pred_cnn_lstm_cat": lstm_scaler_y_cat.inverse_transform(
                _predict_in_batches(hybrid_static_cat, x_test_cat_t, is_sequence=False).reshape(-1, 1)
            ).reshape(-1),
            "pred_cnn_num_seq": lstm_scaler_y.inverse_transform(
                _predict_in_batches(cnn_num_seq, x_test_num_t, is_sequence=True).reshape(-1, 1)
            ).reshape(-1),
            "pred_lstm_num_seq": lstm_scaler_y.inverse_transform(
                _predict_in_batches(lstm_num_seq, x_test_num_t, is_sequence=True).reshape(-1, 1)
            ).reshape(-1),
            "pred_hybrid_num_seq": lstm_scaler_y.inverse_transform(
                _predict_in_batches(hybrid_num_seq, x_test_num_t, is_sequence=True).reshape(-1, 1)
            ).reshape(-1),
            "pred_cnn_cat_seq": lstm_scaler_y_cat.inverse_transform(
                _predict_in_batches(cnn_cat_seq, x_test_cat_t, is_sequence=True).reshape(-1, 1)
            ).reshape(-1),
            "pred_lstm_cat_seq": lstm_scaler_y_cat.inverse_transform(
                _predict_in_batches(lstm_cat_seq, x_test_cat_t, is_sequence=True).reshape(-1, 1)
            ).reshape(-1),
            "pred_hybrid_cat_seq": lstm_scaler_y_cat.inverse_transform(
                _predict_in_batches(hybrid_cat_seq, x_test_cat_t, is_sequence=True).reshape(-1, 1)
            ).reshape(-1),
        }
    )

    return y_test, predictions


def _to_float(value: Any) -> float:
    """Convert a numeric-looking table cell to float."""
    cleaned = str(value).replace("%", "").replace(",", "").strip()
    return float(cleaned)


def _build_analysis_frame() -> pd.DataFrame:
    """Create the joined metric table for all 18 predictors."""
    paths = load_paths_config()
    html_dir = PROJECT_ROOT / paths["results"]["figures_dir"] / "thesis_export" / "html"

    table_32 = _load_export_table(html_dir / "NB05_32GPU_Table02.html")
    table_256 = _load_export_table(html_dir / "NB05_256GPU_Table02.html")
    y_true, predictions = _load_prediction_inputs()

    gain_32_map = table_32.set_index("Policy / Architecture")["JCT Improvement %"].to_dict()
    gain_256_map = table_256.set_index("Policy / Architecture")["JCT Improvement %"].to_dict()

    rows: list[dict[str, Any]] = []
    for policy_name, pred_col in POLICY_TO_PRED_COL.items():
        y_pred = predictions[pred_col]
        rho, _ = spearmanr(y_true, y_pred)
        rows.append(
            {
                "policy": policy_name,
                "model": PLOT_LABELS[policy_name],
                "spearman_rho": float(rho),
                "mae": float(np.mean(np.abs(y_true - y_pred))),
                "jct_gain_32": _to_float(gain_32_map[policy_name]),
                "jct_gain_256": _to_float(gain_256_map[policy_name]),
            }
        )

    analysis_df = pd.DataFrame(rows).sort_values("jct_gain_256", ascending=False).reset_index(drop=True)
    return analysis_df


def _draw_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    x_col: str,
    xlabel: str,
    title: str,
    y_col: str = "jct_gain_256",
    ylabel: str = "256-GPU JCT gain over FIFO (%)",
) -> None:
    """Draw one scatter panel with fitted line and Pearson correlation text."""
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue="policy",
        palette="tab20",
        s=95,
        legend=False,
        ax=ax,
    )

    slope, intercept = np.polyfit(df[x_col], df[y_col], deg=1)
    x_values = np.linspace(df[x_col].min(), df[x_col].max(), 200)
    y_values = slope * x_values + intercept
    corr_value, _ = pearsonr(df[x_col], df[y_col])

    ax.plot(x_values, y_values, color="black", linestyle="--", linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)

    x_span = df[x_col].max() - df[x_col].min()
    y_span = df[y_col].max() - df[y_col].min()
    for row in df.itertuples(index=False):
        ax.annotate(
            row.model,
            (getattr(row, x_col), getattr(row, y_col)),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.text(
        df[x_col].min() + 0.02 * x_span,
        df[y_col].max() - 0.10 * y_span,
        f"Pearson r = {corr_value:.3f}",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "0.7", "boxstyle": "round,pad=0.3"},
    )


def _generate_figure(
    analysis_df: pd.DataFrame,
    *,
    y_col: str,
    ylabel: str,
    gpu_label: str,
    output_path: Path,
) -> None:
    """Generate and save a two-panel MAE/Spearman vs JCT gain figure."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    _draw_panel(
        axes[0],
        analysis_df,
        x_col="mae",
        xlabel="Test-set MAE (seconds)",
        title=f"Point Accuracy vs {gpu_label} JCT Gain",
        y_col=y_col,
        ylabel=ylabel,
    )
    _draw_panel(
        axes[1],
        analysis_df,
        x_col="spearman_rho",
        xlabel="Test-set Spearman $\\rho$",
        title=f"Ranking Quality vs {gpu_label} JCT Gain",
        y_col=y_col,
        ylabel=ylabel,
    )

    fig.suptitle(
        f"Predictor Quality vs Scheduling Gain Across 18 Runtime Models ({gpu_label})",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved figure to %s", output_path)


def main() -> None:
    """Generate and save both 256-GPU and 32-GPU rank-correlation figures."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    sns.set_theme(style="whitegrid")

    paths = load_paths_config()
    export_png_dir = PROJECT_ROOT / paths["results"]["figures_dir"] / "thesis_export" / "png"
    latex_fig_dir = PROJECT_ROOT / "thesis" / "latex" / "figures"
    export_png_dir.mkdir(parents=True, exist_ok=True)
    latex_fig_dir.mkdir(parents=True, exist_ok=True)

    analysis_df = _build_analysis_frame()

    for out_dir in [export_png_dir, latex_fig_dir]:
        _generate_figure(
            analysis_df,
            y_col="jct_gain_256",
            ylabel="256-GPU JCT gain over FIFO (%)",
            gpu_label="256-GPU",
            output_path=out_dir / "mae_spearman_vs_jct_gain_256gpu.png",
        )
        _generate_figure(
            analysis_df,
            y_col="jct_gain_32",
            ylabel="32-GPU JCT gain over FIFO (%)",
            gpu_label="32-GPU",
            output_path=out_dir / "mae_spearman_vs_jct_gain_32gpu.png",
        )

    LOGGER.info("\n%s", analysis_df[["model", "spearman_rho", "mae", "jct_gain_32", "jct_gain_256"]].to_string(index=False))


if __name__ == "__main__":
    main()
