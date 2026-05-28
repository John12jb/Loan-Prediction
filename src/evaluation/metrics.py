"""
src/evaluation/metrics.py
--------------------------
Classification evaluation:
  - Accuracy, Precision, Recall, F1, ROC-AUC
  - Confusion Matrix (with visualisation)
  - Stakeholder-oriented interpretation
  - Deepchecks model validation (optional)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)


# ── Style setup ───────────────────────────────────────────────────────
import matplotlib as mpl
mpl.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f8f9fa",
    "axes.edgecolor":    "#dee2e6",
    "axes.linewidth":    0.8,
    "axes.grid":         True,
    "grid.color":        "#e9ecef",
    "grid.linewidth":    0.6,
    "xtick.color":       "#495057",
    "ytick.color":       "#495057",
    "text.color":        "#212529",
    "font.family":       "DejaVu Sans",
    "font.size":         10,
})

_C = {
    "blue":   "#2563eb",
    "green":  "#16a34a",
    "red":    "#dc2626",
    "orange": "#ea580c",
    "purple": "#7c3aed",
    "gray":   "#6b7280",
    "light":  "#f1f5f9",
}


def _style_ax(ax, title: str = "") -> None:
    ax.set_facecolor("#f8f9fa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#dee2e6")
        spine.set_linewidth(0.8)
    ax.tick_params(colors="#495057", labelsize=9)
    ax.xaxis.label.set_color("#212529")
    ax.yaxis.label.set_color("#212529")
    if title:
        ax.set_title(title, color="#212529", fontsize=12,
                     fontweight="semibold", pad=12)


# ── Core evaluation ───────────────────────────────────────────────────

def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    model_name: str = "Model",
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Print and return a full classification scorecard.

    Parameters
    ----------
    y_true     : True binary labels.
    y_pred     : Predicted binary labels.
    y_prob     : Predicted probabilities for the positive class.
    model_name : Display name for printout.
    threshold  : Decision threshold.
    """
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)

    width = 60
    print("\n" + "─" * width)
    print(f"  {model_name.upper()}  ·  Evaluation Report")
    print("─" * width)
    print(f"  {'Accuracy':<20} {metrics['accuracy']:.4f}")
    print(f"  {'Precision':<20} {metrics['precision']:.4f}")
    print(f"  {'Recall':<20} {metrics['recall']:.4f}   ← Key for loan risk")
    print(f"  {'F1 Score':<20} {metrics['f1']:.4f}")
    if "roc_auc" in metrics:
        print(f"  {'ROC-AUC':<20} {metrics['roc_auc']:.4f}")
    print("─" * width)
    print(classification_report(y_true, y_pred, target_names=["No Default", "Default"]))

    # Stakeholder impact note
    _print_stakeholder_notes(metrics)

    return metrics


def _print_stakeholder_notes(metrics: Dict[str, float]) -> None:
    """Print plain-language impact of the metrics for each stakeholder."""
    print("  STAKEHOLDER IMPACT NOTES")
    print("  " + "─" * 40)

    recall = metrics["recall"]
    precision = metrics["precision"]

    # Loan officer
    missed = round((1 - recall) * 100, 1)
    false_flags = round((1 - precision) * 100, 1)
    print(f"  Loan Officers  : {missed}% of defaulters are being missed "
          f"(not flagged as risky).")
    print(f"                   {false_flags}% of flagged loans are actually safe "
          f"(false alarms).")

    # Regulator
    print(f"  Regulators     : Recall of {recall:.0%} means the model is "
          + ("meeting" if recall >= 0.75 else "falling short of")
          + " standard risk-capture thresholds.")

    # Data scientist
    f1 = metrics["f1"]
    print(f"  Data Scientists: F1 = {f1:.4f} — "
          + ("good balance of precision and recall."
             if f1 >= 0.70 else
             "consider threshold tuning or further class-balance work."))
    print()


# ── Confusion Matrix ──────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_dir: str = "outputs/plots",
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fig.patch.set_facecolor("white")

    sns.heatmap(
        cm, annot=True, fmt="d",
        cmap="Blues",
        linewidths=1.5, linecolor="white",
        xticklabels=["No Default", "Default"],
        yticklabels=["No Default", "Default"],
        ax=ax,
        annot_kws={"size": 16, "weight": "bold"},
    )
    _style_ax(ax, f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted Label", labelpad=8, fontsize=10)
    ax.set_ylabel("Actual Label", labelpad=8, fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Annotate with TP/TN/FP/FN labels
    labels = ["TN", "FP", "FN", "TP"]
    for idx, label in enumerate(labels):
        r, c = divmod(idx, 2)
        ax.text(c + 0.85, r + 0.15, label,
                ha="right", va="top",
                color="#6b7280", fontsize=8)

    plt.tight_layout()
    _save(fig, save_dir, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    save_dir: str = "outputs/plots",
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fig.patch.set_facecolor("white")
    _style_ax(ax, f"ROC Curve — {model_name}  (AUC = {auc:.3f})")

    ax.plot(fpr, tpr, color=_C["blue"], lw=2.5, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color=_C["gray"], lw=1.5, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.08, color=_C["blue"])
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    _save(fig, save_dir, f"roc_curve_{model_name.lower().replace(' ', '_')}.png")


# ── Deepchecks integration ────────────────────────────────────────────

def run_deepchecks(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names,
    save_dir: str = "outputs/plots",
) -> None:
    """
    Run Deepchecks model validation suite.
    Identifies data drift, label drift, and model performance issues.
    """
    try:
        import pandas as pd
        from deepchecks.tabular import Dataset
        from deepchecks.tabular.suites import model_evaluation

        train_ds = Dataset(
            pd.DataFrame(X_train, columns=feature_names),
            label=pd.Series(y_train),
            cat_features=[],
        )
        test_ds = Dataset(
            pd.DataFrame(X_test, columns=feature_names),
            label=pd.Series(y_test),
            cat_features=[],
        )

        suite = model_evaluation()
        result = suite.run(model, train_dataset=train_ds, test_dataset=test_ds)

        out_path = Path(save_dir) / "deepchecks_report.html"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.save_as_html(str(out_path))
        print(f"[Deepchecks] Report saved → {out_path}")

    except ImportError:
        print("[Deepchecks] Not installed. Run: pip install deepchecks")
    except Exception as e:
        print(f"[Deepchecks] Skipped — {e}")


# ── Utility ───────────────────────────────────────────────────────────

def _save(fig: plt.Figure, directory, filename: str) -> None:
    plt.tight_layout()
    if directory:
        path = Path(directory) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
