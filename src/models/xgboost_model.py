"""
src/models/xgboost_model.py
----------------------------
XGBoost classifier with Hyperopt Bayesian hyperparameter tuning.
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple

from xgboost import XGBClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score


class XGBoostModel:
    """
    Wraps XGBClassifier with:
      - Sensible default parameters
      - Hyperopt Bayesian search
      - Save / load helpers
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model: XGBClassifier = None
        self.best_params: Dict[str, Any] = {}

    def build(self, params: Dict[str, Any] = None) -> XGBClassifier:
        """Instantiate an XGBClassifier with given or default params."""
        p = params or self.config
        self.model = XGBClassifier(
            n_estimators=int(p.get("n_estimators", 300)),
            max_depth=int(p.get("max_depth", 6)),
            learning_rate=float(p.get("learning_rate", 0.05)),
            subsample=float(p.get("subsample", 0.8)),
            colsample_bytree=float(p.get("colsample_bytree", 0.8)),
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=self.config.get("random_state", 42),
            n_jobs=-1,
        )
        return self.model

    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        max_evals: int = 50,
    ) -> Dict[str, Any]:
        """
        Run Hyperopt Bayesian search over the hyperparameter space.
        Returns the best parameter dictionary.
        """
        hp_cfg = self.config.get("hyperopt", {}).get("space", {})

        space = {
            "max_depth":        hp.quniform(
                "max_depth",
                hp_cfg.get("max_depth", [3, 10])[0],
                hp_cfg.get("max_depth", [3, 10])[1], 1
            ),
            "learning_rate":    hp.loguniform(
                "learning_rate",
                np.log(hp_cfg.get("learning_rate", [0.01, 0.3])[0]),
                np.log(hp_cfg.get("learning_rate", [0.01, 0.3])[1]),
            ),
            "n_estimators":     hp.quniform(
                "n_estimators",
                hp_cfg.get("n_estimators", [100, 500])[0],
                hp_cfg.get("n_estimators", [100, 500])[1], 50
            ),
            "subsample":        hp.uniform(
                "subsample",
                hp_cfg.get("subsample", [0.6, 1.0])[0],
                hp_cfg.get("subsample", [0.6, 1.0])[1],
            ),
            "colsample_bytree": hp.uniform(
                "colsample_bytree",
                hp_cfg.get("colsample_bytree", [0.6, 1.0])[0],
                hp_cfg.get("colsample_bytree", [0.6, 1.0])[1],
            ),
        }

        def objective(params):
            clf = XGBClassifier(
                max_depth=int(params["max_depth"]),
                learning_rate=params["learning_rate"],
                n_estimators=int(params["n_estimators"]),
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
            score = cross_val_score(
                clf, X_train, y_train,
                cv=3, scoring="f1", n_jobs=-1
            ).mean()
            return {"loss": -score, "status": STATUS_OK}

        print(f"[XGBoost] Running Hyperopt ({max_evals} evaluations)...")
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            verbose=False,
        )

        self.best_params = {
            "max_depth":        int(best["max_depth"]),
            "learning_rate":    best["learning_rate"],
            "n_estimators":     int(best["n_estimators"]),
            "subsample":        best["subsample"],
            "colsample_bytree": best["colsample_bytree"],
        }
        print(f"[XGBoost] Best params: {self.best_params}")
        return self.best_params

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        use_tuned: bool = True,
    ) -> "XGBoostModel":
        """Train the model. Uses tuned params if available and use_tuned=True."""
        params = self.best_params if (use_tuned and self.best_params) else {}
        self.build(params)
        self.model.fit(X_train, y_train)
        print("[XGBoost] Training complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, directory: str) -> Path:
        path = Path(directory) / "xgboost_model.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"[XGBoost] Model saved → {path}")
        return path

    @classmethod
    def load(cls, filepath: str) -> XGBClassifier:
        model = joblib.load(filepath)
        print(f"[XGBoost] Model loaded from {filepath}")
        return model
