# Experiments Log

## Overview
This page tracks model iterations, 5-fold cross-validation metrics, feature ablation results, and out-of-fold (OOF) threshold tuning findings for **Clinical Health Risk Prediction**.

---

## 1. Baseline 5-Fold CV Metrics (`LightGBM + XGBoost` Blend)
* **5-Fold ROC-AUC:** `0.7861`
* **OOF LogLoss:** `0.0954`
* **OOF F1 (@0.50):** `0.0278`
* **OOF Accuracy (@0.50):** `0.9800`

---

## 2. Feature Ablation Study (`src/ablation_and_threshold.py`)
| Experiment | Dropped Feature | ROC-AUC | LogLoss | Delta ROC-AUC | Delta LogLoss |
| :--- | :--- | :---: | :---: | :---: | :---: |
| ALL_FEATURES (Baseline) | None | 0.7861 | 0.0954 | +0.0000 | +0.0000 |
| DROP_pulse_pressure | pulse_pressure | 0.7791 | 0.0961 | -0.0069 | +0.0008 |
| DROP_map_pressure | map_pressure | 0.7887 | 0.0951 | +0.0026 | -0.0003 |
| DROP_metabolic_index | metabolic_index | 0.7911 | 0.0955 | +0.0050 | +0.0002 |
| DROP_cardio_strain_index | cardio_strain_index | 0.7829 | 0.0966 | -0.0032 | +0.0012 |
| DROP_age_x_hba1c | age_x_hba1c | 0.7876 | 0.0952 | +0.0015 | -0.0002 |

---

## 3. Threshold Tuning & Class Imbalance Findings
* **Dataset Positives (`high_risk_flag = 1`):** `68` out of `3500` (`1.94%`)
* **Best F1 Threshold (`0.10–0.90` grid):** `0.10` → `F1 = 0.1705`, `Precision = 0.1803`, `Recall = 0.1618`, `Accuracy = 0.9694`
* **Screening Recall (`>= 0.85`) Threshold:** `None`
