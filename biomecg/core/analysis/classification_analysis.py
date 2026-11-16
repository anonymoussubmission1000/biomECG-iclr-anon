# ================================================================================
# Analyse the result dataframe to compute metrics, thresholds, ROC, EER, etc.
# ================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

from biomecg.core.utils.tools import *


# ================================================================================
class ClassificationAnalyser:
    # ------------------------------------------------------------------------------
    def __init__(self, config: dict, loader_name: str):
        self.config = config
        base_path = get_base_path()
        inference_repo = (
            base_path
            + f"/results/{self.config['network']}/inference/{self.config['name']}/{loader_name}/"
        )
        self.path_results = inference_repo + f"results_{loader_name}.csv"
        self.path_scores = inference_repo + f"results_{loader_name}_scores.csv"
        self.path_confusion = inference_repo + f"results_{loader_name}_confusion.png"
        self.path_roc = inference_repo + f"results_{loader_name}_roc.png"
    # ------------------------------------------------------------------------------

    def compute_metrics_at_threshold(self, y_true, y_score, threshold):
        y_pred = (y_score >= threshold).astype(int)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return acc, prec, rec, f1

    def analyse_model(self):
        df = pd.read_csv(self.path_results)

        y_true = df["Label"].to_numpy()
        y_score = df["Probability"].to_numpy()

        # ROC
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auc_roc = roc_auc_score(y_true, y_score)

        # EER (FPR ~= FNR)
        eer_list = np.abs((1 - tpr) - fpr)
        idx_eer = int(np.nanargmin(eer_list))
        eer_fpr = fpr[idx_eer]
        eer_tpr = tpr[idx_eer]
        eer_threshold = thresholds[idx_eer]
        eer_value = (eer_fpr + (1 - eer_tpr)) / 2

        plt.figure(figsize=(6, 5))
        plt.plot(
            fpr, tpr, linewidth=2.2,
            label=f"AUC = {auc_roc:.3f}"
        )
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5)

        # EER point
        plt.scatter(eer_fpr, eer_tpr, s=70, zorder=5)
        plt.text(
            eer_fpr,
            eer_tpr,
            f"EER = {eer_value:.3f}\nt = {eer_threshold:.3f}",
            fontsize=10,
            ha="left",
            va="bottom",
        )

        # Threshold deciles
        valid_thr = thresholds[np.isfinite(thresholds)]
        thr_deciles = np.quantile(valid_thr, np.linspace(0.1, 0.9, 9))
        for thr in thr_deciles:
            idx = np.argmin(np.abs(thresholds - thr))
            plt.scatter(fpr[idx], tpr[idx], s=30)
            plt.text(
                fpr[idx],
                tpr[idx],
                f"{thr:.2f}",
                fontsize=8,
                ha="left",
                va="bottom",
            )

        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curve", fontsize=14)
        plt.grid(True, linestyle=":", linewidth=0.8, alpha=0.8)
        plt.legend(loc="lower right", fontsize=12, frameon=True, framealpha=0.9)
        plt.tight_layout()
        plt.savefig(self.path_roc, dpi=400)
        plt.close()

        # Fixed threshold
        threshold = 0.5
        acc, prec, rec, f1 = self.compute_metrics_at_threshold(
            y_true, y_score, threshold
        )

        results = {
            "ROC_AUC": auc_roc,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "EER": eer_value,
            "EER_Threshold": eer_threshold,
        }

        pd.DataFrame(list(results.items())).to_csv(
            self.path_scores, header=False, index=False
        )

        # Confusion matrix at threshold 0.5
        y_pred = (y_score >= threshold).astype(int)
        c_matrix = confusion_matrix(y_true, y_pred).astype(np.float32)

        plt.figure(figsize=(5, 4))
        sns.heatmap(
            c_matrix,
            annot=True,
            fmt=".0f",
            cmap="Blues",
            xticklabels=["0", "1"],
            yticklabels=["0", "1"],
            cbar=False,
        )
        plt.title("Confusion Matrix (t = 0.5)")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        plt.savefig(self.path_confusion, dpi=300)
        plt.close()

        print("\n# --- Metrics saved to:", self.path_scores)
        print("# --- ROC curve saved to:", self.path_roc)
        print("# --- Confusion matrix saved to:", self.path_confusion, "\n")
