import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

TARGET_COL = "high_risk_flag"
ENGINEERED_FEATURES = [
    "pulse_pressure",
    "map_pressure",
    "metabolic_index",
    "cardio_strain_index",
    "age_x_hba1c",
]


def load_or_build_dataset(csv_path: str = "data/clinical_patient_telemetry.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "pulse_pressure" not in df.columns and {"systolic_bp", "diastolic_bp"}.issubset(df.columns):
        df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    if "map_pressure" not in df.columns and {"diastolic_bp", "pulse_pressure"}.issubset(df.columns):
        df["map_pressure"] = df["diastolic_bp"] + df["pulse_pressure"] / 3.0
    if "metabolic_index" not in df.columns and {"bmi", "fasting_glucose"}.issubset(df.columns):
        df["metabolic_index"] = (df["bmi"] * df["fasting_glucose"]) / 100.0
    if "cardio_strain_index" not in df.columns and {"systolic_bp", "resting_hr"}.issubset(df.columns):
        df["cardio_strain_index"] = (df["systolic_bp"] * df["resting_hr"]) / 1000.0
    if "age_x_hba1c" not in df.columns and {"age", "hba1c"}.issubset(df.columns):
        df["age_x_hba1c"] = df["age"] * df["hba1c"]
    return df


def evaluate_oof_blend(X: pd.DataFrame, y: np.ndarray, random_state: int = 42) -> tuple[dict, np.ndarray]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    oof_probs = np.zeros(len(y), dtype=float)

    for train_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        lgb = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            subsample_for_bin=50000,
            random_state=random_state,
            verbose=-1,
        )
        lgb.fit(X_tr, y_tr)
        p_lgb = lgb.predict_proba(X_va)[:, 1]

        xgb = XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=2,
        )
        xgb.fit(X_tr, y_tr)
        p_xgb = xgb.predict_proba(X_va)[:, 1]

        oof_probs[val_idx] = 0.5 * p_lgb + 0.5 * p_xgb

    pred_050 = (oof_probs >= 0.50).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y, oof_probs),
        "logloss": log_loss(y, oof_probs),
        "f1_at_050": f1_score(y, pred_050),
        "acc_at_050": accuracy_score(y, pred_050),
    }
    return metrics, oof_probs


def run_feature_ablation(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = {TARGET_COL, "patient_id"}
    feature_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    y = df[TARGET_COL].astype(int).values

    base_metrics, base_oof = evaluate_oof_blend(df[feature_cols], y)
    rows = [
        {
            "experiment": "ALL_FEATURES (Baseline)",
            "dropped_feature": "None",
            "n_features": len(feature_cols),
            **base_metrics,
            "delta_roc_auc": 0.0,
            "delta_logloss": 0.0,
        }
    ]

    for feat in ENGINEERED_FEATURES:
        if feat in feature_cols:
            sub_cols = [c for c in feature_cols if c != feat]
            m, _ = evaluate_oof_blend(df[sub_cols], y)
            rows.append(
                {
                    "experiment": f"DROP_{feat}",
                    "dropped_feature": feat,
                    "n_features": len(sub_cols),
                    **m,
                    "delta_roc_auc": m["roc_auc"] - base_metrics["roc_auc"],
                    "delta_logloss": m["logloss"] - base_metrics["logloss"],
                }
            )

    return pd.DataFrame(rows), base_oof


def tune_thresholds(y: np.ndarray, oof_probs: np.ndarray, min_recall: float = 0.85) -> dict:
    thresholds = np.linspace(0.10, 0.90, 161)
    best_f1 = {"threshold": 0.50, "f1": -1.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0}
    best_recall_constrained = None

    for th in thresholds:
        preds = (oof_probs >= th).astype(int)
        f1 = f1_score(y, preds, zero_division=0)
        prec = precision_score(y, preds, zero_division=0)
        rec = recall_score(y, preds, zero_division=0)
        acc = accuracy_score(y, preds)

        if f1 > best_f1["f1"]:
            best_f1 = {"threshold": float(th), "f1": float(f1), "precision": float(prec), "recall": float(rec), "accuracy": float(acc)}

        if rec >= min_recall:
            if best_recall_constrained is None or f1 > best_recall_constrained["f1"]:
                best_recall_constrained = {
                    "threshold": float(th),
                    "f1": float(f1),
                    "precision": float(prec),
                    "recall": float(rec),
                    "accuracy": float(acc),
                }

    return {"best_f1": best_f1, "high_recall_target": best_recall_constrained}


if __name__ == "__main__":
    df = load_or_build_dataset()
    ablation_df, oof_probs = run_feature_ablation(df)
    print("\n=== FEATURE ABLATION RESULTS ===")
    print(ablation_df.to_string(index=False))

    th_results = tune_thresholds(df[TARGET_COL].astype(int).values, oof_probs, min_recall=0.85)
    print("\n=== THRESHOLD TUNING RESULTS ===")
    print("Best F1 Threshold:", th_results["best_f1"])
    print("Screening (Recall >= 0.85) Threshold:", th_results["high_recall_target"])