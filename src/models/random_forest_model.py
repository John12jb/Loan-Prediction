"""
src/models/random_forest_model.py
-----------------------------------
Random Forest classifier with GridSearchCV tuning.
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class RandomForestModel:
    """
    Wraps RandomForestClassifier with:
      - Sensible default parameters
      - GridSearchCV exhaustive tuning
      - Save / load helpers
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model: RandomForestClassifier = None
        self.best_params: Dict[str, Any] = {}

    def build(self, params: Dict[str, Any] = None) -> RandomForestClassifier:
        p = params or self.config
        self.model = RandomForestClassifier(
            n_estimators=int(p.get("n_estimators", 200)),
            max_depth=p.get("max_depth", None),
            min_samples_split=int(p.get("min_samples_split", 2)),
            random_state=self.config.get("random_state", 42),
            n_jobs=-1,
            class_weight="balanced",
        )
        return self.model

    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Dict[str, Any]:
        """
        GridSearchCV over n_estimators, max_depth, and min_samples_split.
        Optimises for F1 score to handle class imbalance well.
        """
        gs_cfg = self.config.get("grid_search", {})
        param_grid = gs_cfg.get("param_grid", {})

        # YAML None → Python None
        if "max_depth" in param_grid:
            param_grid["max_depth"] = [
                None if v is None else v
                for v in param_grid["max_depth"]
            ]

        base = RandomForestClassifier(
            random_state=self.config.get("random_state", 42),
            n_jobs=-1,
            class_weight="balanced",
        )

        print(f"[RandomForest] Running GridSearchCV "
              f"(cv={gs_cfg.get('cv_folds', 5)}, "
              f"scoring={gs_cfg.get('scoring', 'f1')})...")

        gs = GridSearchCV(
            estimator=base,
            param_grid=param_grid,
            cv=gs_cfg.get("cv_folds", 5),
            scoring=gs_cfg.get("scoring", "f1"),
            n_jobs=gs_cfg.get("n_jobs", -1),
            verbose=0,
            refit=True,
        )
        gs.fit(X_train, y_train)

        self.best_params = gs.best_params_
        self.model = gs.best_estimator_
        print(f"[RandomForest] Best params : {self.best_params}")
        print(f"[RandomForest] Best CV F1  : {gs.best_score_:.4f}")
        return self.best_params

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        use_tuned: bool = True,
    ) -> "RandomForestModel":
        if use_tuned and self.model is not None:
            # Already fitted by GridSearchCV.refit=True — nothing to do
            print("[RandomForest] Using GridSearchCV best estimator (already fitted).")
        else:
            params = self.best_params if (use_tuned and self.best_params) else {}
            self.build(params)
            self.model.fit(X_train, y_train)
            print("[RandomForest] Training complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, directory: str) -> Path:
        path = Path(directory) / "random_forest_model.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"[RandomForest] Model saved → {path}")
        return path

    @classmethod
    def load(cls, filepath: str) -> RandomForestClassifier:
        model = joblib.load(filepath)
        print(f"[RandomForest] Model loaded from {filepath}")
        return model
