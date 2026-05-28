"""
main.py
--------
Entry point for the Loan Default Prediction + Explainable AI pipeline.

Run:
    python main.py

Steps executed:
    1.  Load raw data
    2.  EDA summary
    3.  Preprocessing (impute → transform → encode → split → SMOTE)
    4.  Train XGBoost with Hyperopt tuning
    5.  Train Random Forest with GridSearchCV tuning
    6.  Evaluate both models (metrics + confusion matrix + ROC curve)
    7.  Deepchecks model validation (optional)
    8.  SHAP global + local explanations
    9.  Anchor rule explanations
    10. Counterfactual explanations
    11. Save models to outputs/models/
"""

import sys
import yaml
import numpy as np
from pathlib import Path

# ── Project imports ───────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data_preparation.loader import load_csv, quick_eda
from src.data_preparation.preprocessor import LoanPreprocessor
from src.models.xgboost_model import XGBoostModel
from src.models.random_forest_model import RandomForestModel
from src.evaluation.metrics import (
    evaluate,
    plot_confusion_matrix,
    plot_roc_curve,
    run_deepchecks,
)
from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.anchors_explainer import AnchorsExplainer
from src.explainability.counterfactuals import CounterfactualExplainer


# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def banner(text: str) -> None:
    width = 65
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


# ─────────────────────────────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()

    # ── 0. Ensure output directories exist ────────────────────────────
    import os
    os.makedirs(cfg["paths"]["plots_dir"],  exist_ok=True)
    os.makedirs(cfg["paths"]["models_dir"], exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────
    banner("STEP 1 · Loading Data")
    data_path = ROOT / cfg["paths"]["raw_data"]
    if not data_path.exists():
        print(f"  Data not found at {data_path}")
        print("  Generating synthetic data now...\n")
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "data/generate_sample_data.py")], check=True)

    df = load_csv(data_path)
    quick_eda(df)

    # ── 2. Preprocessing ──────────────────────────────────────────────
    banner("STEP 2 · Preprocessing")
    data_cfg = cfg["data"]
    preprocessor = LoanPreprocessor(
        target_col=data_cfg["target_column"],
        drop_cols=data_cfg["drop_columns"],
        categorical_cols=data_cfg["categorical_columns"],
        numeric_cols=data_cfg["numeric_columns"],
        skew_threshold=data_cfg["skew_threshold"],
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
        smote_strategy=data_cfg["smote_strategy"],
    )
    X_train, X_test, y_train, y_test, feature_names = preprocessor.fit_transform(df)

    # ── 3. XGBoost ────────────────────────────────────────────────────
    banner("STEP 3 · XGBoost — Hyperopt Tuning + Training")
    xgb = XGBoostModel(cfg["xgboost"])
    xgb.tune(
        X_train, y_train,
        max_evals=cfg["xgboost"]["hyperopt"]["max_evals"]
    )
    xgb.fit(X_train, y_train, use_tuned=True)
    xgb.save(cfg["paths"]["models_dir"])

    # ── 4. Random Forest ──────────────────────────────────────────────
    banner("STEP 4 · Random Forest — GridSearchCV Tuning + Training")
    rf = RandomForestModel(cfg["random_forest"])
    rf.tune(X_train, y_train)
    rf.fit(X_train, y_train, use_tuned=True)
    rf.save(cfg["paths"]["models_dir"])

    # ── 5. Evaluation ─────────────────────────────────────────────────
    banner("STEP 5 · Model Evaluation")
    plots_dir = cfg["paths"]["plots_dir"]
    eval_cfg = cfg["evaluation"]

    for name, model in [("XGBoost", xgb), ("Random Forest", rf)]:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        evaluate(y_true=y_test, y_pred=y_pred, y_prob=y_prob,
                 model_name=name, threshold=eval_cfg["threshold"])
        plot_confusion_matrix(y_test, y_pred, model_name=name, save_dir=plots_dir)
        plot_roc_curve(y_test, y_prob, model_name=name, save_dir=plots_dir)

    # ── 6. Deepchecks ─────────────────────────────────────────────────
    banner("STEP 6 · Deepchecks Model Validation")
    run_deepchecks(
        xgb.model, X_train, X_test, y_train, y_test,
        feature_names=feature_names, save_dir=plots_dir
    )

    # ── 7. SHAP Explanations ──────────────────────────────────────────
    banner("STEP 7 · SHAP Explainability")
    shap_cfg = cfg["explainability"]["shap"]

    # Use XGBoost as primary model for SHAP (TreeExplainer is exact for it)
    shap_exp = SHAPExplainer(
        model=xgb.model,
        X_background=X_train,
        feature_names=feature_names,
        background_samples=shap_cfg["background_samples"],
    )
    shap_values = shap_exp.compute_shap_values(X_test)

    # Global: summary and bar plots
    shap_exp.plot_summary(X_test, shap_values,
                          max_display=shap_cfg["max_display"], save_dir=plots_dir)
    shap_exp.plot_bar_importance(shap_values, save_dir=plots_dir,
                                 max_display=shap_cfg["max_display"])

    # Dependence plots for key features
    for feat in shap_cfg["dependence_features"]:
        shap_exp.plot_dependence(X_test, shap_values, feature=feat, save_dir=plots_dir)

    # Local: waterfall for a single prediction
    sample_idx = cfg["explainability"]["anchors"]["sample_index"]
    shap_exp.plot_waterfall(X_test, sample_index=sample_idx, save_dir=plots_dir)
    shap_exp.explain_prediction_text(X_test, shap_values, sample_index=sample_idx)

    # ── 8. Anchor Explanations ────────────────────────────────────────
    banner("STEP 8 · Anchor Explanations (IF-THEN Rules)")
    anchor_cfg = cfg["explainability"]["anchors"]
    anchor_exp = AnchorsExplainer(
        model=xgb.model,
        feature_names=feature_names,
        threshold=anchor_cfg["threshold"],
    )
    anchor_exp.fit(X_train)
    anchor_exp.explain(X_test, sample_index=anchor_cfg["sample_index"],
                       threshold=anchor_cfg["threshold"])

    # ── 9. Counterfactual Explanations ────────────────────────────────
    banner("STEP 9 · Counterfactual Explanations")
    cf_cfg = cfg["explainability"]["counterfactuals"]
    cf_exp = CounterfactualExplainer(
        model=xgb.model,
        feature_names=feature_names,
        X_train=X_train,
        y_train=y_train,
        num_cfs=cf_cfg["num_cfs"],
        desired_class=cf_cfg["desired_class"],
    )
    cf_exp.explain(X_test, sample_index=cf_cfg["sample_index"])

    # ── Done ──────────────────────────────────────────────────────────
    banner("PIPELINE COMPLETE")
    print(f"  Models  → {cfg['paths']['models_dir']}")
    print(f"  Plots   → {cfg['paths']['plots_dir']}")
    print()


if __name__ == "__main__":
    main()
