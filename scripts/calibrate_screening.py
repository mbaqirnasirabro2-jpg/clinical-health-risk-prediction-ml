from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ablation_and_threshold import TARGET_COL, load_or_build_dataset


def build_base_model(scale_pos_weight: float = 1.0):
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=15,
            subsample_for_bin=2000,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.03,
            max_depth=4,
            random_state=42,
        )


def summarize_probs(y: np.ndarray, probs: np.ndarray) -> dict:
    clipped = np.clip(probs, 1e-6, 1 - 1e-6)
    best_f1 = None
    best_scr = None

    for th in np.linspace(0.005, 0.60, 600):
        preds = (probs >= th).astype(int)
        f1 = f1_score(y, preds, zero_division=0)
        prec = precision_score(y, preds, zero_division=0)
        rec = recall_score(y, preds, zero_division=0)
        acc = accuracy_score(y, preds)

        if best_f1 is None or f1 > best_f1["f1"]:
            best_f1 = {
                "th": round(float(th), 4),
                "f1": round(f1, 4),
                "prec": round(prec, 4),
                "rec": round(rec, 4),
                "acc": round(acc, 4),
            }

        if rec >= 0.85:
            if best_scr is None or f1 > best_scr["f1"]:
                best_scr = {
                    "th": round(float(th), 4),
                    "f1": round(f1, 4),
                    "prec": round(prec, 4),
                    "rec": round(rec, 4),
                    "acc": round(acc, 4),
                }

    return {
        "roc_auc": round(float(roc_auc_score(y, probs)), 4),
        "pr_auc": round(float(average_precision_score(y, probs)), 4),
        "log_loss": round(float(log_loss(y, clipped)), 4),
        "brier": round(float(brier_score_loss(y, clipped)), 4),
        "best_f1": best_f1,
        "screening_rec_0.85": best_scr,
    }


def run_calibration_experiment(df: pd.DataFrame, y: np.ndarray):
    X_num = df.select_dtypes(include=[np.number]).copy()
    feature_cols = [c for c in X_num.columns if c != TARGET_COL]
    X_feat = X_num[feature_cols].fillna(X_num[feature_cols].median())

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_weighted = np.zeros(len(y), dtype=float)
    oof_calibrated = np.zeros(len(y), dtype=float)

    for train_idx, val_idx in skf.split(X_feat, y):
        X_tr, y_tr = X_feat.iloc[train_idx], y[train_idx]
        X_va = X_feat.iloc[val_idx]

        # Inner split taake calibration out-of-fold / held-out validation par fit ho
        inner_skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        tr_sub_idx, cal_idx = next(inner_skf.split(X_tr, y_tr))
        X_sub, y_sub = X_tr.iloc[tr_sub_idx], y_tr[tr_sub_idx]
        X_cal, y_cal = X_tr.iloc[cal_idx], y_tr[cal_idx]

        pos_sub = max(int((y_sub == 1).sum()), 1)
        neg_sub = max(int((y_sub == 0).sum()), 1)
        spw_sub = neg_sub / pos_sub

        pos_full = max(int((y_tr == 1).sum()), 1)
        neg_full = max(int((y_tr == 0).sum()), 1)
        spw_full = neg_full / pos_full

        # 1) Full-fold weighted model
        model_full = build_base_model(scale_pos_weight=spw_full)
        model_full.fit(X_tr, y_tr)
        oof_weighted[val_idx] = model_full.predict_proba(X_va)[:, 1]

        # 2) Calibrated weighted model (fitted on X_sub, calibrated on X_cal)
        model_sub = build_base_model(scale_pos_weight=spw_sub)
        model_sub.fit(X_sub, y_sub)
        calibrator = CalibratedClassifierCV(
            estimator=FrozenEstimator(model_sub),
            method="sigmoid",
            cv=None,
        )
        calibrator.fit(X_cal, y_cal)
        oof_calibrated[val_idx] = calibrator.predict_proba(X_va)[:, 1]

    return (
        len(feature_cols),
        summarize_probs(y, oof_weighted),
        summarize_probs(y, oof_calibrated),
    )


df = load_or_build_dataset()
y = df[TARGET_COL].astype(int).values
n_feat, res_weighted, res_calibrated = run_calibration_experiment(df, y)

print(f"Features count: {n_feat}")
print("=== A. 14 FEATURES + SCALE_POS_WEIGHT (Uncalibrated) ===")
print(res_weighted)
print("\n=== B. 14 FEATURES + SCALE_POS_WEIGHT + SIGMOID CALIBRATION ===")
print(res_calibrated)