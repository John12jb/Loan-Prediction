"""
src/models/experiment_tracker.py
----------------------------------
Optional Neptune experiment tracking wrapper.

Logs:
  - Hyperparameters
  - Metrics (accuracy, precision, recall, f1, roc_auc)
  - Model artifacts (saved .pkl files)
  - Plots (confusion matrix, ROC curve, SHAP summary)

Usage:
  Set neptune.enabled = true in config/config.yaml
  Set neptune.api_token to your Neptune API key
  Set neptune.project to your workspace/project name

If Neptune is not configured, all calls silently no-op so the
rest of the pipeline is unaffected.
"""

from typing import Dict, Any, Optional
from pathlib import Path


class ExperimentTracker:
    """
    Thin wrapper around Neptune that gracefully degrades to a no-op
    logger when Neptune is disabled or not installed.
    """

    def __init__(self, config: Dict[str, Any]):
        self._run = None
        self._enabled = config.get("enabled", False)

        if self._enabled:
            self._init_neptune(config)

    def _init_neptune(self, config: Dict[str, Any]) -> None:
        try:
            import neptune
            api_token = config.get("api_token", "")
            project   = config.get("project", "")

            if not api_token or not project:
                print("[Neptune] api_token or project not set in config — tracking disabled.")
                self._enabled = False
                return

            self._run = neptune.init_run(
                project=project,
                api_token=api_token,
            )
            print(f"[Neptune] Run initialised → {project}")

        except ImportError:
            print("[Neptune] neptune-client not installed — tracking disabled.")
            self._enabled = False
        except Exception as e:
            print(f"[Neptune] Could not initialise run: {e} — tracking disabled.")
            self._enabled = False

    # ── Logging methods ───────────────────────────────────────────────

    def log_params(self, params: Dict[str, Any], prefix: str = "") -> None:
        """Log a dictionary of hyperparameters."""
        if not self._enabled or self._run is None:
            return
        for k, v in params.items():
            key = f"{prefix}/{k}" if prefix else k
            self._run[key] = v

    def log_metrics(self, metrics: Dict[str, float], prefix: str = "") -> None:
        """Log scalar evaluation metrics."""
        if not self._enabled or self._run is None:
            return
        for k, v in metrics.items():
            key = f"{prefix}/{k}" if prefix else k
            self._run[key] = v
        print(f"[Neptune] Metrics logged: {list(metrics.keys())}")

    def log_artifact(self, filepath: str, destination: Optional[str] = None) -> None:
        """Upload a file (model, plot, report) to Neptune."""
        if not self._enabled or self._run is None:
            return
        path = Path(filepath)
        if not path.exists():
            print(f"[Neptune] Artifact not found: {filepath}")
            return
        dest = destination or f"artifacts/{path.name}"
        self._run[dest].upload(str(path))
        print(f"[Neptune] Artifact uploaded → {dest}")

    def log_text(self, key: str, value: str) -> None:
        if not self._enabled or self._run is None:
            return
        self._run[key] = value

    def stop(self) -> None:
        """Close the Neptune run cleanly."""
        if self._run is not None:
            self._run.stop()
            print("[Neptune] Run closed.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()
