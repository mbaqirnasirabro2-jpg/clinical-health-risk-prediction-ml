from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")

models = [
    "Baseline (14f)",
    "Feat-Sel (11f)",
    "11f + SPW",
    "14f + SPW",
    "14f + SPW + SigCal",
]

roc_auc = [0.8092, 0.8043, 0.7943, 0.7902, 0.7543]
pr_auc = [0.1247, 0.1069, 0.1203, 0.1094, 0.0779]

# 1. ROC-AUC vs PR-AUC Comparison
x = np.arange(len(models))
width = 0.36

fig, ax = plt.subplots(figsize=(9.5, 5))
b1 = ax.bar(x - width / 2, roc_auc, width, label="ROC-AUC", color="#1f77b4")
b2 = ax.bar(x + width / 2, pr_auc, width, label="PR-AUC (Avg Precision)", color="#ff7f0e")

ax.set_title("Model Ranking Performance Across Feature & Imbalance Setups", fontsize=12, fontweight="bold")
ax.set_ylabel("Score")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha="right")
ax.set_ylim(0, 0.92)
ax.legend()

for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "1_roc_vs_pr_auc.png", dpi=300)
plt.close()

# 2. LogLoss & Brier Score Comparison (Calibration Focus)
cal_models = ["Baseline (14f)", "14f + SPW", "14f + SPW + SigCal"]
log_loss_vals = [0.0851, 0.1077, 0.0869]
brier_vals = [np.nan, 0.0255, 0.0186]

x2 = np.arange(len(cal_models))
fig, ax = plt.subplots(figsize=(8, 4.8))
w2 = 0.35
bars_ll = ax.bar(x2 - w2 / 2, log_loss_vals, w2, label="LogLoss (lower is better)", color="#2ca02c")
bars_br = ax.bar(x2 + w2 / 2, np.nan_to_num(brier_vals, nan=0.0), w2, label="Brier Score (lower is better)", color="#9467bd")

ax.set_title("Probability Reliability: LogLoss & Brier Score", fontsize=12, fontweight="bold")
ax.set_ylabel("Error")
ax.set_xticks(x2)
ax.set_xticklabels(cal_models)
ax.set_ylim(0, 0.13)
ax.legend()

for bar in bars_ll:
    h = bar.get_height()
    ax.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

for idx, bar in enumerate(bars_br):
    if not np.isnan(brier_vals[idx]):
        h = bar.get_height()
        ax.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "2_calibration_metrics.png", dpi=300)
plt.close()

# 3. Best F1 vs Threshold Across Experiments
f1_models = ["Baseline (14f)", "Feat-Sel (11f)", "11f + SPW", "14f + SPW", "14f + SPW + SigCal"]
best_f1 = [0.2625, 0.2346, 0.2235, 0.2069, 0.1667]
best_th = [0.1063, 0.0884, 0.5374, 0.2215, 0.0775]

fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
x3 = np.arange(len(f1_models))
bars_f1 = ax1.bar(x3, best_f1, width=0.45, color="#17becf", label="Best F1 Score")
ax1.set_ylabel("Best F1 Score", color="#0e7c86")
ax1.set_ylim(0, 0.32)
ax1.set_xticks(x3)
ax1.set_xticklabels(f1_models, rotation=15, ha="right")

for bar in bars_f1:
    h = bar.get_height()
    ax1.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                 xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

ax2 = ax1.twinx()
ax2.plot(x3, best_th, color="#d62728", marker="o", linewidth=2, label="Optimal Threshold")
ax2.set_ylabel("Threshold", color="#d62728")
ax2.set_ylim(0, 0.65)

for i, th in enumerate(best_th):
    ax2.annotate(f"th={th:.4f}", xy=(x3[i], th), xytext=(0, 6),
                 textcoords="offset points", ha="center", fontsize=8, color="#d62728")

ax1.set_title("Best F1 Score & Corresponding Threshold", fontsize=12, fontweight="bold")
fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
plt.tight_layout()
plt.savefig(OUT_DIR / "3_best_f1_and_thresholds.png", dpi=300)
plt.close()

# 4. High-Recall Screening Regime (Recall >= 0.85)
scr_models = ["11f + SPW\n(th=0.0328)", "14f + SPW + SigCal\n(th=0.0100)"]
rec_vals = [0.8676, 0.8824]
prec_vals = [0.0357, 0.0297]
acc_vals = [0.5420, 0.4374]

x4 = np.arange(len(scr_models))
w4 = 0.25
fig, ax = plt.subplots(figsize=(8, 4.8))
b_r = ax.bar(x4 - w4, rec_vals, w4, label="Recall (>= 0.85 target)", color="#2ca02c")
b_p = ax.bar(x4, prec_vals, w4, label="Precision", color="#ff7f0e")
b_a = ax.bar(x4 + w4, acc_vals, w4, label="Accuracy", color="#1f77b4")

ax.axhline(0.85, color="red", linestyle="--", linewidth=1.2, label="0.85 Recall Target")
ax.set_title("High-Recall Screening Operating Points", fontsize=12, fontweight="bold")
ax.set_ylabel("Metric Value")
ax.set_xticks(x4)
ax.set_xticklabels(scr_models)
ax.set_ylim(0, 1.02)
ax.legend()

for grp in (b_r, b_p, b_a):
    for bar in grp:
        h = bar.get_height()
        ax.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "4_high_recall_screening.png", dpi=300)
plt.close()

print("Saved 4 graphs to:", OUT_DIR.resolve())