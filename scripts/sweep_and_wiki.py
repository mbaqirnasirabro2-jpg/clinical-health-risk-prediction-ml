from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ablation_and_threshold import TARGET_COL, load_or_build_dataset, run_feature_ablation

df = load_or_build_dataset()
y = df[TARGET_COL].astype(int).values
_, oof_probs = run_feature_ablation(df)

thresholds = np.linspace(0.005, 0.50, 500)
best_f1 = None
best_screening = None

for th in thresholds:
    preds = (oof_probs >= th).astype(int)
    f1 = f1_score(y, preds, zero_division=0)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    acc = accuracy_score(y, preds)

    if best_f1 is None or f1 > best_f1["f1"]:
        best_f1 = {
            "threshold": round(float(th), 4),
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "accuracy": round(acc, 4),
        }

    if rec >= 0.85:
        if best_screening is None or f1 > best_screening["f1"]:
            best_screening = {
                "threshold": round(float(th), 4),
                "f1": round(f1, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "accuracy": round(acc, 4),
            }

print("Best F1 (0.005-0.50):", best_f1)
print("Screening Recall >= 0.85:", best_screening)

wiki_dir = Path("/workspaces/clinical-health-risk-prediction-ml.wiki")
wiki_file = wiki_dir / "Experiments.md"

if wiki_file.exists():
    addition = (
        "\n## 4. Low-Threshold Sweep (`0.005–0.50`)\n"
        f"* **Best F1 Threshold:** `{best_f1}`\n"
        f"* **Screening Threshold (`Recall >= 0.85`):** `{best_screening}`\n"
    )
    with open(wiki_file, "a") as f:
        f.write(addition)
    print("Appended results to Experiments.md")
else:
    print("Wiki folder/Experiments.md abhi nahi mila.")
