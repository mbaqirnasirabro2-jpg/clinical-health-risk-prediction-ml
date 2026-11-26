from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
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


def eval_oof(X: pd.DataFrame, y: np.ndarray, use_pos_weight: bool = False):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y), dtype=float)

    X_num = X.select_dtypes(include=[np.number]).copy()
    feature_cols = [c for c in X_num.columns if c != TARGET_COL]
    X_feat = X_num[feature_cols].fillna(X_num[feature_cols].median())

    has_lgb = False
    has_xgb = False
    try:
        import lightgbm as lgb
        has_lgb = True
    except ImportError:
        pass

    try:
        import xgboost as xgb
        has_xgb = True
    except ImportError:
        pass

    for train_idx, val_idx in skf.split(X_feat, y):
        X_tr, y_tr = X_feat.iloc[train_idx], y[train_idx]
        X_va = X_feat.iloc[val_idx]

        pos_count = max(int((y_tr == 1).sum()), 1)
        neg_count = max(int((y_tr == 0).sum()), 1)
        spw = (neg_count / pos_count) if use_pos_weight else 1.0

        fold_preds = []

        if has_lgb:
            clf_lgb = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.03,
                num_leaves=15,
                subsample_for_bin=2000,
                scale_pos_weight=spw,
                random_state=42,
                verbose=-1,
            )
            clf_lgb.fit(X_tr, y_tr)
            fold_preds.append(clf_lgb.predict_proba(X_va)[:, 1])

        if has_xgb:
            clf_xgb = xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=spw,
                eval_metric="logloss",
                random_state=42,
            )
            clf_xgb.fit(X_tr, y_tr)
            fold_preds.append(clf_xgb.predict_proba(X_va)[:, 1])

        if not fold_preds:
            from sklearn.ensemble import HistGradientBoostingClassifier
            clf_hgb = HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.03,
                max_depth=4,
                random_state=42,
            )
            clf_hgb.fit(X_tr, y_tr)
            fold_preds.append(clf_hgb.predict_proba(X_va)[:, 1])

        oof_probs[val_idx] = np.mean(fold_preds, axis=0)

    roc = roc_auc_score(y, oof_probs)
    pr_auc = average_precision_score(y, oof_probs)
    ll = log_loss(y, np.clip(oof_probs, 1e-6, 1 - 1e-6))

    best_f1 = None
    best_scr = None
    for th in np.linspace(0.005, 0.60, 600):
        preds = (oof_probs >= th).astype(int)
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
        "n_features": len(feature_cols),
        "roc_auc": round(float(roc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "log_loss": round(float(ll), 4),
        "best_f1": best_f1,
        "screening_rec_0.85": best_scr,
    }


df = load_or_build_dataset()
y = df[TARGET_COL].astype(int).values

drop_noisy = ["metabolic_index", "map_pressure", "age_x_hba1c"]
df_selected = df.drop(columns=[c for c in drop_noisy if c in df.columns])

res_base = eval_oof(df, y, use_pos_weight=False)
res_sel_unweighted = eval_oof(df_selected, y, use_pos_weight=False)
res_sel_weighted = eval_oof(df_selected, y, use_pos_weight=True)

print("=== 1. BASELINE (All Features, Unweighted) ===")
print(res_base)
print("\n=== 2. FEATURE-SELECTED (Drop Noisy, Unweighted) ===")
print(res_sel_unweighted)
print("\n=== 3. FEATURE-SELECTED + SCALE_POS_WEIGHT ===")
print(res_sel_weighted)