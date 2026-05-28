"""
src/explainability/shap_explainer.py
--------------------------------------
SHAP (SHapley Additive exPlanations) for:
  1. Global feature importance (summary plot)
  2. Feature dependence plots
  3. Local / single-prediction explanation (waterfall + force plot)
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
from pathlib import Path
from typing import List, Optional, Any


class SHAPExplainer:
    """
    Unified SHAP explainer that works with both tree-based models
    (XGBoost, Random Forest) using TreeExplainer for speed.
    """

    def __init__(
        self,
        model: Any,
        X_background: np.ndarray,
        feature_names: List[str],
        background_samples: int = 100,
    ):
        import shap
        self.shap = shap
        self.feature_names = feature_names

        print("[SHAP] Initialising TreeExplainer...")
        # TreeExplainer is exact and fast for tree-based models
        background = shap.sample(X_background, min(background_samples, len(X_background)))
        self.explainer = shap.TreeExplainer(model, data=background)

    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """Compute SHAP values for a dataset. Returns array of shape (n, features)."""
        print(f"[SHAP] Computing SHAP values for {X.shape[0]} samples...")
        shap_values = self.explainer.shap_values(X)

        # RandomForest returns a list [class0_shap, class1_shap]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        return shap_values

    # ── 1. Global summary plot ────────────────────────────────────────

    def plot_summary(
        self,
        X: np.ndarray,
        shap_values: np.ndarray,
        max_display: int = 15,
        save_dir: str = "outputs/plots",
    ) -> None:
        """
        Beeswarm summary plot — shows direction and magnitude of each
        feature's impact across all predictions.
        """
        print("[SHAP] Generating summary (beeswarm) plot...")
        fig, ax = plt.subplots(figsize=(9, 6))
        fig.patch.set_facecolor("white")

        self.shap.summary_plot(
            shap_values, X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
            plot_size=None,
        )

        plt.gcf().set_facecolor("white")
        for ax_ in plt.gcf().axes:
            ax_.set_facecolor("#f8f9fa")
            ax_.tick_params(labelsize=9)
        plt.gcf().axes[0].set_xlabel(
            "SHAP Value  (impact on model output)", fontsize=10)
        plt.gcf().axes[0].set_title(
            "Global Feature Importance — SHAP Summary",
            fontsize=12, fontweight="semibold", pad=12)

        _save(plt.gcf(), save_dir, "shap_summary_plot.png")

    def plot_bar_importance(
        self,
        shap_values: np.ndarray,
        save_dir: str = "outputs/plots",
        max_display: int = 15,
    ) -> None:
        """Mean |SHAP| bar chart — easy to communicate to non-technical stakeholders."""
        mean_abs = np.abs(shap_values).mean(axis=0)
        indices = np.argsort(mean_abs)[-max_display:]

        features = [self.feature_names[i] for i in indices]
        values   = mean_abs[indices]

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#f8f9fa")

        colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(indices)))
        bars = ax.barh(features, values, color=colors, edgecolor="white", height=0.6)
        ax.set_xlabel("Mean |SHAP Value|", fontsize=10)
        ax.set_title("Feature Importance (Mean |SHAP|)",
                     fontsize=12, fontweight="semibold", pad=12)
        ax.tick_params(labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#dee2e6")

        for bar, val in zip(bars, values):
            ax.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8, color="#374151")

        plt.tight_layout()
        _save(fig, save_dir, "shap_bar_importance.png")

    # ── 2. Dependence plots ───────────────────────────────────────────

    def plot_dependence(
        self,
        X: np.ndarray,
        shap_values: np.ndarray,
        feature: str,
        interaction_feature: str = "auto",
        save_dir: str = "outputs/plots",
    ) -> None:
        """
        SHAP dependence plot for a single feature.
        Shows how that feature's SHAP value changes across its range,
        coloured by an interaction feature.
        """
        if feature not in self.feature_names:
            print(f"[SHAP] Feature '{feature}' not found. Skipping.")
            return

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#f8f9fa")

        self.shap.dependence_plot(
            feature, shap_values, X,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False, ax=ax,
        )

        ax.set_title(f"SHAP Dependence — {feature}",
                     fontsize=12, fontweight="semibold", pad=12)
        ax.tick_params(labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#dee2e6")
        fig.patch.set_facecolor("white")

        safe_name = feature.replace(" ", "_").replace("/", "_")
        _save(fig, save_dir, f"shap_dependence_{safe_name}.png")

    # ── 3. Local / single-prediction explanation ──────────────────────

    def plot_waterfall(
        self,
        X: np.ndarray,
        sample_index: int = 0,
        save_dir: str = "outputs/plots",
    ) -> None:
        """
        Waterfall plot for a single prediction — shows how each feature
        pushes the model output from the base value towards the prediction.
        """
        print(f"[SHAP] Generating waterfall plot for sample #{sample_index}...")

        explanation = self.explainer(X[sample_index : sample_index + 1])

        # For models that return list (RandomForest), take class-1 explanation
        if hasattr(explanation, "__len__") and isinstance(explanation.values, list):
            import shap
            exp = shap.Explanation(
                values=explanation.values[0][1],
                base_values=explanation.base_values[0][1],
                data=explanation.data[0],
                feature_names=self.feature_names,
            )
        else:
            exp = explanation[0]
            exp.feature_names = self.feature_names

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#f8f9fa")

        self.shap.plots.waterfall(exp, show=False, max_display=12)

        plt.gcf().set_facecolor("white")
        for ax_ in plt.gcf().axes:
            ax_.set_facecolor("#f8f9fa")
        plt.title(f"SHAP Waterfall — Sample #{sample_index} Explanation",
                  fontsize=12, fontweight="semibold")

        _save(plt.gcf(), save_dir, f"shap_waterfall_sample_{sample_index}.png")
        plt.close("all")

    def explain_prediction_text(
        self,
        X: np.ndarray,
        shap_values: np.ndarray,
        sample_index: int = 0,
        top_n: int = 5,
    ) -> None:
        """Print a plain-English explanation of a single prediction."""
        sv = shap_values[sample_index]
        feature_values = X[sample_index]

        sorted_idx = np.argsort(np.abs(sv))[::-1][:top_n]

        print(f"\n  Explanation for sample #{sample_index}:")
        print("  " + "─" * 50)
        for i in sorted_idx:
            direction = "↑ increased" if sv[i] > 0 else "↓ decreased"
            fname = self.feature_names[i]
            fval = feature_values[i]
            impact = abs(sv[i])
            print(f"  {fname:<28} = {fval:>10.3f}  →  {direction} default risk by {impact:.4f}")
        print()


# ── Utility ───────────────────────────────────────────────────────────

def _save(fig: plt.Figure, directory, filename: str) -> None:
    plt.tight_layout()
    if directory:
        path = Path(directory) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
