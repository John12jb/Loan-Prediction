"""
src/models/tuning.py
---------------------
Shared hyperparameter tuning utilities used by both models.

Provides:
  - plot_hyperopt_trials  : Visualise Hyperopt trial losses over time
  - plot_grid_search_heatmap : Heatmap of GridSearch CV scores
  - compare_models        : Side-by-side metric comparison bar chart
  - threshold_analysis    : F1 / Precision / Recall vs decision threshold
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8f9fa",
    "axes.edgecolor":   "#dee2e6",
    "axes.grid":        True,
    "grid.color":       "#e9ecef",
    "grid.linewidth":   0.6,
    "text.color":       "#212529",
    "font.family":      "DejaVu Sans",
    "font.size":        10,
})
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

from sklearn.metrics import f1_score, precision_score, recall_score

_BLUE   = "#2563eb"
_GREEN  = "#16a34a"
_RED    = "#dc2626"
_ORANGE = "#ea580c"
_PURPLE = "#7c3aed"
_GRAY   = "#6b7280"


# ── Hyperopt trial losses ─────────────────────────────────────────────

def plot_hyperopt_trials(
    trials,
    model_name: str = "XGBoost",
    save_dir: str = "outputs/plots",
) -> None:
    """
    Plot the Hyperopt trial losses over successive evaluations.
    A descending trend confirms the search is converging.
    """
    losses = [t["result"]["loss"] for t in trials.trials]
    best_so_far = pd.Series(losses).cummin().tolist()

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f9fa")

    ax.scatter(range(len(losses)), [-l for l in losses],
               color=_BLUE, alpha=0.4, s=22, label="Trial F1", zorder=3)
    ax.plot(range(len(best_so_far)), [-l for l in best_so_far],
            color=_GREEN, lw=2.5, label="Best so far", zorder=4)

    ax.set_xlabel("Trial", fontsize=10)
    ax.set_ylabel("F1 Score", fontsize=10)
    ax.set_title(f"Hyperopt Search — {model_name}",
                 fontsize=12, fontweight="semibold", pad=12)
    ax.tick_params(labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#dee2e6")
    ax.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, save_dir, f"hyperopt_trials_{model_name.lower().replace(' ', '_')}.png")


# ── GridSearch heatmap ────────────────────────────────────────────────

def plot_grid_search_heatmap(
    grid_search_cv,
    param_x: str,
    param_y: str,
    model_name: str = "Random Forest",
    save_dir: str = "outputs/plots",
) -> None:
    """
    Heatmap of mean CV scores across two hyperparameter axes.

    Parameters
    ----------
    grid_search_cv : Fitted GridSearchCV object.
    param_x        : Parameter name for the x-axis.
    param_y        : Parameter name for the y-axis.
    """
    results = pd.DataFrame(grid_search_cv.cv_results_)
    pivot = results.pivot_table(
        values="mean_test_score",
        index=f"param_{param_y}",
        columns=f"param_{param_x}",
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f9fa")

    sns_import = _try_import_seaborn()
    if sns_import:
        sns_import.heatmap(
            pivot, annot=True, fmt=".3f",
            cmap="Blues", linewidths=1.5,
            linecolor="white", ax=ax,
            annot_kws={"size": 9},
        )
    else:
        im = ax.imshow(pivot.values, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticklabels(pivot.index)
        plt.colorbar(im, ax=ax)

    ax.set_title(f"GridSearch CV F1 — {model_name}",
                 fontsize=12, fontweight="semibold", pad=12)
    ax.set_xlabel(param_x, fontsize=10)
    ax.set_ylabel(param_y, fontsize=10)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    _save(fig, save_dir, f"grid_search_heatmap_{model_name.lower().replace(' ', '_')}.png")


# ── Model comparison bar chart ────────────────────────────────────────

def compare_models(
    results: Dict[str, Dict[str, float]],
    save_dir: str = "outputs/plots",
) -> None:
    """
    Side-by-side bar chart comparing metrics across models.

    Parameters
    ----------
    results : { "XGBoost": {"accuracy": 0.88, "f1": 0.79, ...}, ... }
    """
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    models  = list(results.keys())
    n_metrics = len(metrics)
    n_models  = len(models)
    x = np.arange(n_metrics)
    width = 0.8 / n_models
    colors = [_BLUE, _ORANGE, _RED, _PURPLE, "#f59e0b"]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f9fa")

    for i, (model_name, scores) in enumerate(results.items()):
        vals = [scores.get(m, 0) for m in metrics]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9,
                      label=model_name, color=colors[i % len(colors)],
                      edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=9)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title("Model Comparison", fontsize=12, fontweight="semibold", pad=12)
    ax.tick_params(labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#dee2e6")
    ax.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, save_dir, "model_comparison.png")


# ── Threshold analysis ────────────────────────────────────────────────

def threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    save_dir: str = "outputs/plots",
) -> float:
    """
    Plot F1, Precision, and Recall across decision thresholds.
    Returns the threshold that maximises F1.

    Useful for tuning the operating point of the classifier without
    retraining — important for lending where recall often takes priority.
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    f1s, precs, recs = [], [], []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
        precs.append(precision_score(y_true, y_pred, zero_division=0))
        recs.append(recall_score(y_true, y_pred, zero_division=0))

    best_t = thresholds[np.argmax(f1s)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f9fa")

    ax.plot(thresholds, f1s,   color=_BLUE,   lw=2.5, label="F1")
    ax.plot(thresholds, precs, color=_GREEN,  lw=2.5, label="Precision")
    ax.plot(thresholds, recs,  color=_RED,    lw=2.5, label="Recall")
    ax.axvline(best_t, color=_GRAY, lw=1.5, linestyle="--",
               label=f"Best F1 threshold = {best_t:.2f}")

    ax.set_xlabel("Decision Threshold", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(f"Threshold Analysis — {model_name}",
                 fontsize=12, fontweight="semibold", pad=12)
    ax.tick_params(labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#dee2e6")
    ax.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, save_dir, f"threshold_analysis_{model_name.lower().replace(' ', '_')}.png")
    print(f"[Tuning] Optimal threshold for {model_name}: {best_t:.2f}  "
          f"(F1={max(f1s):.4f})")
    return float(best_t)


# ── Utilities ─────────────────────────────────────────────────────────

def _save(fig: plt.Figure, directory, filename: str) -> None:
    plt.tight_layout()
    if directory:
        path = Path(directory) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def _try_import_seaborn():
    try:
        import seaborn
        return seaborn
    except ImportError:
        return None
