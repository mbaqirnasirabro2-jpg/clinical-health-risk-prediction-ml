import os, warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

os.makedirs("data", exist_ok=True)
os.makedirs("visuals", exist_ok=True)

print("==========================================================")
print("   CLINICAL HEALTH RISK PREDICTION & SHAP PIPELINE        ")
print("==========================================================\n")

# 1. Generate Synthetic Clinical Patient Telemetry Dataset
rng = np.random.default_rng(42)
n_patients = 3500

age = rng.integers(25, 85, size=n_patients)
bmi = np.round(np.clip(rng.normal(27.5, 4.8, size=n_patients), 16.5, 46.0), 1)
systolic_bp = np.round(np.clip(102 + 0.42 * age + 0.85 * (bmi - 25) + rng.normal(0, 11.0, size=n_patients), 85, 210), 1)
diastolic_bp = np.round(np.clip(0.58 * systolic_bp + rng.normal(4.0, 6.5, size=n_patients), 50, 125), 1)
fasting_glucose = np.round(np.clip(82 + 0.35 * (age - 30) + 1.4 * np.maximum(0, bmi - 25) + rng.normal(0, 18.0, size=n_patients), 65, 310), 1)
hba1c = np.round(np.clip(4.6 + 0.032 * (fasting_glucose - 85) + rng.normal(0, 0.45, size=n_patients), 4.2, 13.5), 2)
cholesterol_total = np.round(np.clip(155 + 0.65 * age + 1.2 * (bmi - 24) + rng.normal(0, 26.0, size=n_patients), 115, 340), 1)
resting_hr = np.round(np.clip(68 + 0.25 * np.maximum(0, bmi - 25) + rng.normal(0, 8.5, size=n_patients), 48, 120), 0)

# Clinical risk logit (cardio-metabolic risk indicator)
risk_logit = (
    -7.8
    + 0.045 * age
    + 0.075 * (bmi - 25)
    + 0.028 * (systolic_bp - 120)
    + 0.018 * (fasting_glucose - 100)
    + 0.35 * (hba1c - 5.5)
    + 0.012 * (cholesterol_total - 190)
    + 0.015 * (resting_hr - 70)
    + rng.normal(0, 0.55, size=n_patients)
)
risk_prob_true = 1.0 / (1.0 + np.exp(-risk_logit))
high_risk_flag = (rng.random(n_patients) < risk_prob_true).astype(int)

df = pd.DataFrame({
    "patient_id": np.arange(10001, 10001 + n_patients),
    "age": age,
    "bmi": bmi,
    "systolic_bp": systolic_bp,
    "diastolic_bp": diastolic_bp,
    "fasting_glucose": fasting_glucose,
    "hba1c": hba1c,
    "cholesterol_total": cholesterol_total,
    "resting_hr": resting_hr,
    "high_risk_flag": high_risk_flag,
})

# 2. Clinical Domain Feature Engineering
df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
df["map_pressure"] = df["diastolic_bp"] + (df["pulse_pressure"] / 3.0)
df["metabolic_index"] = (df["bmi"] * df["fasting_glucose"]) / 100.0
df["cardio_strain_index"] = (df["systolic_bp"] * df["resting_hr"]) / 1000.0
df["age_x_hba1c"] = df["age"] * df["hba1c"]

df.to_csv("data/clinical_patient_telemetry.csv", index=False)
print(f"[INFO] Saved dataset with {df.shape[0]} rows and {df.shape[1]} columns.")

feature_cols = [c for c in df.columns if c not in ["patient_id", "high_risk_flag"]]
X = df[feature_cols]
y = df["high_risk_flag"]

# 3. 5-Fold Stratified CV Ensemble (LightGBM + XGBoost)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_lgb = np.zeros(len(df))
oof_xgb = np.zeros(len(df))
oof_shap_list = []
oof_X_list = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

    m_lgb = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=42 + fold, verbose=-1
    )
    m_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(40, verbose=False)])
    oof_lgb[va_idx] = m_lgb.predict_proba(X_va)[:, 1]

    m_xgb = XGBClassifier(
        n_estimators=500, learning_rate=0.03, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        early_stopping_rounds=40, random_state=42 + fold, n_jobs=-1
    )
    m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[va_idx] = m_xgb.predict_proba(X_va)[:, 1]

    sample_va = X_va.sample(n=min(250, len(X_va)), random_state=42)
    sv = shap.TreeExplainer(m_lgb)(sample_va)
    if len(sv.shape) == 3:
        sv = sv[:, :, 1]
    oof_shap_list.append(sv)
    oof_X_list.append(sample_va)

oof_blend = 0.5 * oof_lgb + 0.5 * oof_xgb
auc = roc_auc_score(y, oof_blend)
ll = log_loss(y, oof_blend)
f1 = f1_score(y, (oof_blend >= 0.5).astype(int))
acc = accuracy_score(y, (oof_blend >= 0.5).astype(int))
print(f"[METRICS] 5-Fold OOF -> ROC-AUC: {auc:.4f} | LogLoss: {ll:.4f} | F1: {f1:.4f} | Acc: {acc:.4f}")

pd.DataFrame({
    "patient_id": df["patient_id"],
    "actual_high_risk": y,
    "predicted_risk_prob": np.round(oof_blend, 4)
}).to_csv("data/health_risk_oof_predictions.csv", index=False)

# 4. Visual 1: Predicted Risk Probability Distribution by Class
plt.figure(figsize=(10, 5))
plt.hist(oof_blend[y == 0], bins=30, alpha=0.7, label="Lower Risk (Class 0)")
plt.hist(oof_blend[y == 1], bins=30, alpha=0.7, label="Elevated Risk (Class 1)")
plt.title("1. Out-of-Fold Predicted Clinical Risk Probability Distribution", fontsize=11, fontweight="bold")
plt.xlabel("Predicted Risk Probability")
plt.ylabel("Patient Count")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("visuals/risk_probability_distribution.png", dpi=300)
plt.close()

# 5. Visual 2: Risk vs. Age & Metabolic Index
plt.figure(figsize=(10, 5))
plt.scatter(df["age"], oof_blend, c=y, cmap="coolwarm", alpha=0.45, s=14)
plt.colorbar(label="Actual Risk Flag (0/1)")
plt.title("2. Predicted Risk Score Across Patient Age", fontsize=11, fontweight="bold")
plt.xlabel("Patient Age (Years)")
plt.ylabel("Ensemble Predicted Risk Probability")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("visuals/risk_vs_age.png", dpi=300)
plt.close()

# 6. Visual 3: SHAP Global Feature Importance
X_shap_all = pd.concat(oof_X_list, axis=0).reset_index(drop=True)
sv_all = np.vstack([s.values for s in oof_shap_list])
base_all = np.concatenate([
    np.repeat(s.base_values, len(s.values)) if np.ndim(s.base_values) == 0 else s.base_values
    for s in oof_shap_list
])
shap_exp = shap.Explanation(values=sv_all, base_values=base_all, data=X_shap_all.values, feature_names=X_shap_all.columns.tolist())

plt.figure(figsize=(10, 6))
shap.plots.bar(shap_exp, show=False)
plt.title("3. SHAP Global Feature Importance for Clinical Risk", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("visuals/shap_health_drivers.png", dpi=300)
plt.close()

print("[SUCCESS] Generated clinical telemetry data, OOF predictions, and 3 visuals in visuals/!")

python - << 'EOF'
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

oof = pd.read_csv("data/health_risk_oof_predictions.csv")
print("Columns:", list(oof.columns))

y_true = oof["high_risk_flag"].values
prob_col = [c for c in oof.columns if c != "high_risk_flag" and "id" not in c.lower()][0]
y_prob = oof[prob_col].values
y_pred_050 = (y_prob >= 0.50).astype(int)

print(f"ROC-AUC : {roc_auc_score(y_true, y_prob):.4f}")
print(f"LogLoss : {log_loss(y_true, y_prob):.4f}")
print(f"F1 (@0.5): {f1_score(y_true, y_pred_050):.4f}")
print(f"Accuracy: {accuracy_score(y_true, y_pred_050):.4f}")
EOF

