"""
save_missing_models.py
======================
Checkpoint JSON'larından best_params okuyarak eksik modelleri yeniden
eğitir ve results/models/ altına kaydeder.

Sadece eksik .joblib / .pth dosyaları için çalışır; mevcut olanları
atlar.

Kullanım
--------
    python scripts/save_missing_models.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

# ── Proje kökünü sys.path'e ekle ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import prepare_features_for_model  # noqa: E402
from src.tuning import (  # noqa: E402
    finalize_dl_model,
    finalize_ml_model,
    load_checkpoint,
    prepare_dl_datasets,
)

MODEL_DIR = PROJECT_ROOT / "results" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı fonksiyonlar
# ─────────────────────────────────────────────────────────────────────────────

def _should_save(path: Path) -> bool:
    if path.exists():
        print(f"  [SKIP]  {path.name} zaten mevcut.")
        return False
    return True


def _load_ckpt(tag: str) -> dict | None:
    ckpt = load_checkpoint(tag)
    if ckpt is None:
        print(f"  [WARN]  Checkpoint bulunamadı: {tag} — atlanıyor.")
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# DL modeli kaydetme yardımcısı
# ─────────────────────────────────────────────────────────────────────────────

def save_dl_exp(train_dataset, val_dataset, test_dataset, y_test_raw,
                scaler_y, input_features, specs: list, label: str) -> None:
    """specs = [(tag, model_name, filename), ...]"""
    print(f"\n=== {label} ===")
    for tag, model_name, filename in specs:
        dest = MODEL_DIR / filename
        if not _should_save(dest):
            continue
        ckpt = _load_ckpt(tag)
        if ckpt is None:
            continue
        print(f"  Eğitiliyor: {tag} ({model_name}) …")
        model, metrics = finalize_dl_model(
            model_name=model_name,
            best_params=ckpt["best_params"],
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            input_features=input_features,
            scaler_y=scaler_y,
            y_test_raw=y_test_raw,
            test_dataset=test_dataset,
            final_epochs=50,
            patience=10,
        )
        torch.save(model, dest)
        print(f"  [SAVED] {dest}  MAE={metrics['mae']:.1f}  R²={metrics['r2']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Exp A — Numeric-only (RF, XGB, LGBM)
# ─────────────────────────────────────────────────────────────────────────────

def save_exp_a(X_train, y_train, X_test, y_test) -> None:
    print("\n=== Exp A: Numeric-only ===")

    specs = [
        ("exp_a_rf",   "rf",   "rf_numeric.joblib"),
        ("exp_a_xgb",  "xgb",  "xgb_numeric.joblib"),
        ("exp_a_lgbm", "lgbm", "lgbm_numeric.joblib"),
    ]

    for tag, model_name, filename in specs:
        dest = MODEL_DIR / filename
        if not _should_save(dest):
            continue
        ckpt = _load_ckpt(tag)
        if ckpt is None:
            continue
        print(f"  Eğitiliyor: {tag} …")
        model, _ = finalize_ml_model(
            model_name, ckpt["best_params"],
            X_train, y_train, X_test, y_test,
            random_state=42, verbose=False,
        )
        joblib.dump(model, dest)
        print(f"  [SAVED] {dest}")


# ─────────────────────────────────────────────────────────────────────────────
# Exp B — One-Hot & Native Categorical (RF, XGB, LGBM_OH, LGBM_NAT)
# ─────────────────────────────────────────────────────────────────────────────

def save_exp_b_oh(X_train_oh, y_train_oh, X_test_oh, y_test_oh) -> None:
    print("\n=== Exp B (One-Hot): RF, XGB, LGBM ===")

    specs = [
        ("exp_b_rf_oh",   "rf",   "rf_categorical.joblib",   None),
        ("exp_b_xgb_oh",  "xgb",  "xgb_categorical.joblib",  None),
        ("exp_b_lgbm_oh", "lgbm", "lgbm_categorical.joblib", None),
    ]

    for tag, model_name, filename, cat_feat in specs:
        dest = MODEL_DIR / filename
        if not _should_save(dest):
            continue
        ckpt = _load_ckpt(tag)
        if ckpt is None:
            continue
        print(f"  Eğitiliyor: {tag} …")
        model, _ = finalize_ml_model(
            model_name, ckpt["best_params"],
            X_train_oh, y_train_oh, X_test_oh, y_test_oh,
            random_state=42, verbose=False,
        )
        joblib.dump(model, dest)
        print(f"  [SAVED] {dest}")


def save_exp_b_nat(X_train_nat, y_train_nat, X_test_nat, y_test_nat,
                   cat_cols_nat) -> None:
    print("\n=== Exp B (Native): LGBM_NAT ===")

    dest = MODEL_DIR / "lgbm_categorical_native.joblib"
    if not _should_save(dest):
        return
    ckpt = _load_ckpt("exp_b_lgbm_nat")
    if ckpt is None:
        return
    print("  Eğitiliyor: exp_b_lgbm_nat …")
    model, _ = finalize_ml_model(
        "lgbm", ckpt["best_params"],
        X_train_nat, y_train_nat, X_test_nat, y_test_nat,
        random_state=42, verbose=False,
        categorical_feature=list(cat_cols_nat) if cat_cols_nat is not None else None,
    )
    joblib.dump(model, dest)
    print(f"  [SAVED] {dest}")


# ─────────────────────────────────────────────────────────────────────────────
# Ana akış
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("save_missing_models.py — Eksik modelleri yeniden eğit & kaydet")
    print("=" * 60)

    # ── Exp A: Numeric-only veriyi yükle ─────────────────────────────────────
    print("\n[Veri] Numeric-only özellik seti yükleniyor …")
    (_, X_train, X_test, y_train, y_test, _, _) = prepare_features_for_model(
        dataset="main", time_unit="s", test_size=0.20,
        random_state=42, feature_mode="numeric_only",
    )
    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")

    save_exp_a(X_train, y_train, X_test, y_test)

    # ── Exp B: One-Hot veriyi yükle ───────────────────────────────────────────
    print("\n[Veri] One-Hot özellik seti yükleniyor …")
    (_, X_train_oh, X_test_oh, y_train_oh, y_test_oh, _, _) = prepare_features_for_model(
        dataset="main", time_unit="s", test_size=0.20,
        random_state=42, feature_mode="with_categorical_onehot",
    )
    print(f"  X_train_oh: {X_train_oh.shape}  X_test_oh: {X_test_oh.shape}")

    save_exp_b_oh(X_train_oh, y_train_oh, X_test_oh, y_test_oh)

    # ── Exp B: Native Categorical veriyi yükle ────────────────────────────────
    print("\n[Veri] Native categorical özellik seti yükleniyor …")
    (_, X_train_nat, X_test_nat, y_train_nat, y_test_nat, _, cat_cols_nat) = prepare_features_for_model(
        dataset="main", time_unit="s", test_size=0.20,
        random_state=42, feature_mode="with_categorical_native",
    )
    print(f"  X_train_nat: {X_train_nat.shape}  X_test_nat: {X_test_nat.shape}")

    save_exp_b_nat(X_train_nat, y_train_nat, X_test_nat, y_test_nat, cat_cols_nat)

    # ── Exp C (numeric, seq_len=1) ────────────────────────────────────────────
    print("\n[Veri] Exp C/E: Numeric-only özellik seti yükleniyor …")
    (_, X_train_num, X_test_num, y_train_num, y_test_num, _, _) = prepare_features_for_model(
        dataset="main", time_unit="s", test_size=0.20,
        random_state=42, feature_mode="numeric_only",
    )
    # seq_len=1 için (exp_c)
    (train_c, val_c, test_c, y_raw_c,
     _, scaler_y_c, n_feat_c) = prepare_dl_datasets(
        X_train_num, X_test_num, y_train_num, y_test_num, seq_len=1
    )
    save_dl_exp(
        train_c, val_c, test_c, y_raw_c, scaler_y_c, n_feat_c,
        [
            ("exp_c_cnn",    "CNN",    "cnn_numeric.pth"),
            ("exp_c_lstm",   "LSTM",   "lstm_numeric.pth"),
            ("exp_c_hybrid", "Hybrid", "cnn_lstm_numeric.pth"),
        ],
        label="Exp C (numeric, seq_len=1)",
    )

    # seq_len=10 için (exp_e)
    (_, X_train_num_noshuf, X_test_num_noshuf,
     y_train_num_noshuf, y_test_num_noshuf, _, _) = prepare_features_for_model(
        dataset="main", time_unit="s", test_size=0.20,
        random_state=42, feature_mode="numeric_only", shuffle=False,
    )
    (train_e, val_e, test_e, y_raw_e,
     _, scaler_y_e, n_feat_e) = prepare_dl_datasets(
        X_train_num_noshuf, X_test_num_noshuf,
        y_train_num_noshuf, y_test_num_noshuf, seq_len=10
    )
    save_dl_exp(
        train_e, val_e, test_e, y_raw_e, scaler_y_e, n_feat_e,
        [
            ("exp_e_cnn",    "CNN",    "cnn_numeric_seq.pth"),
            ("exp_e_lstm",   "LSTM",   "lstm_numeric_seq.pth"),
            ("exp_e_hybrid", "Hybrid", "cnn_lstm_numeric_seq.pth"),
        ],
        label="Exp E (numeric seq_len=10)",
    )

    # ── Exp D (one_hot, seq_len=1) & Exp F (one_hot, seq_len=10) ─────────────
    print("\n[Veri] Exp D/F: One-Hot özellik seti yükleniyor …")
    (_, X_train_oh, X_test_oh, y_train_oh, y_test_oh, _, _) = prepare_features_for_model(
        dataset="main", time_unit="s", test_size=0.20,
        random_state=42, feature_mode="with_categorical_onehot",
    )
    (train_d, val_d, test_d, y_raw_d,
     _, scaler_y_d, n_feat_d) = prepare_dl_datasets(
        X_train_oh, X_test_oh, y_train_oh, y_test_oh, seq_len=1
    )
    save_dl_exp(
        train_d, val_d, test_d, y_raw_d, scaler_y_d, n_feat_d,
        [
            ("exp_d_cnn",    "CNN",    "cnn_categorical_pt.pth"),
            ("exp_d_lstm",   "LSTM",   "lstm_categorical_pt.pth"),
            ("exp_d_hybrid", "Hybrid", "cnn_lstm_categorical_pt.pth"),
        ],
        label="Exp D (one_hot, seq_len=1)",
    )

    (_, X_train_oh_noshuf, X_test_oh_noshuf,
     y_train_oh_noshuf, y_test_oh_noshuf, _, _) = prepare_features_for_model(
        dataset="main", time_unit="s", test_size=0.20,
        random_state=42, feature_mode="with_categorical_onehot", shuffle=False,
    )
    (train_f, val_f, test_f, y_raw_f,
     _, scaler_y_f, n_feat_f) = prepare_dl_datasets(
        X_train_oh_noshuf, X_test_oh_noshuf,
        y_train_oh_noshuf, y_test_oh_noshuf, seq_len=10
    )
    save_dl_exp(
        train_f, val_f, test_f, y_raw_f, scaler_y_f, n_feat_f,
        [
            ("exp_f_cnn",    "CNN",    "cnn_categorical_seq.pth"),
            ("exp_f_lstm",   "LSTM",   "lstm_categorical_seq.pth"),
            ("exp_f_hybrid", "Hybrid", "cnn_lstm_categorical_seq.pth"),
        ],
        label="Exp F (one_hot seq_len=10)",
    )

    print("\n[TAMAMLANDI] results/models/ güncel durumu:")
    for f in sorted(MODEL_DIR.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
