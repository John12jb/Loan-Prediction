"""
src/explainability/anchors_explainer.py
-----------------------------------------
Anchor Explanations — IF-THEN rules that are guaranteed to produce the
same prediction with high probability (precision ≥ threshold).

Example output:
  IF  credit_score <= 540
  AND loan_to_income_ratio > 2.3
  THEN  Prediction = DEFAULT  (precision = 0.97, coverage = 0.12)

This format is highly legible for loan officers and regulators.
"""

import numpy as np
from typing import List, Optional, Any


class AnchorsExplainer:
    """
    Wrapper around alibi's AnchorTabular explainer.
    Falls back to a lightweight manual rule extraction if alibi is not
    installed (so the project still runs in restricted environments).
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        threshold: float = 0.95,
    ):
        self.model = model
        self.feature_names = feature_names
        self.threshold = threshold
        self._explainer = None
        self._use_alibi = self._try_init_alibi()

    def _try_init_alibi(self) -> bool:
        try:
            from alibi.explainers import AnchorTabular
            self._AnchorTabular = AnchorTabular
            return True
        except ImportError:
            print("[Anchors] alibi not installed — using fallback rule extractor.")
            return False

    def fit(self, X_train: np.ndarray) -> "AnchorsExplainer":
        """Fit the anchor explainer on training data."""
        if self._use_alibi:
            self._explainer = self._AnchorTabular(
                predictor=lambda x: self.model.predict(x),
                feature_names=self.feature_names,
                seed=42,
            )
            self._explainer.fit(X_train, disc_perc=(25, 50, 75))
            print("[Anchors] Explainer fitted.")
        return self

    def explain(
        self,
        X: np.ndarray,
        sample_index: int = 0,
        threshold: Optional[float] = None,
    ) -> None:
        """
        Generate and print an anchor explanation for a single sample.

        Parameters
        ----------
        X            : Full test set (anchors need context for sampling).
        sample_index : Index of the sample to explain.
        threshold    : Minimum precision for the anchor rule.
        """
        threshold = threshold or self.threshold
        sample = X[sample_index]
        prediction = self.model.predict(sample.reshape(1, -1))[0]
        label = "DEFAULT" if prediction == 1 else "NO DEFAULT"

        if self._use_alibi and self._explainer is not None:
            self._explain_alibi(sample, sample_index, label, threshold)
        else:
            self._explain_fallback(sample, sample_index, label, X)

    def _explain_alibi(
        self, sample: np.ndarray, idx: int, label: str, threshold: float
    ) -> None:
        print(f"\n[Anchors] Explaining sample #{idx}  →  Prediction: {label}")
        print("  Computing anchor rules (this may take ~30s)...")
        try:
            exp = self._explainer.explain(sample, threshold=threshold)
            rules = exp.anchor

            print("\n  ┌─ ANCHOR EXPLANATION " + "─" * 38)
            if rules:
                print(f"  │  IF  " + "\n  │  AND ".join(rules))
            else:
                print("  │  (No simple rule found — prediction depends on many features)")
            print(f"  │")
            print(f"  │  THEN  Prediction = {label}")
            print(f"  │  Precision  : {exp.precision:.2f}  "
                  f"(rule is correct {exp.precision:.0%} of the time)")
            print(f"  │  Coverage   : {exp.coverage:.2f}  "
                  f"(applies to {exp.coverage:.0%} of similar applicants)")
            print("  └" + "─" * 58)
        except Exception as e:
            print(f"  [Anchors] Could not compute anchor: {e}")
            self._explain_fallback(sample, idx, label, None)

    def _explain_fallback(
        self,
        sample: np.ndarray,
        idx: int,
        label: str,
        X: Optional[np.ndarray],
    ) -> None:
        """
        Lightweight rule extraction without alibi.
        Uses feature values + domain thresholds to construct a readable rule.
        """
        rules = []
        for i, (fname, val) in enumerate(zip(self.feature_names, sample)):
            # Domain-specific thresholds for loan risk features
            if fname == "credit_score":
                if val < 550:
                    rules.append(f"credit_score ≤ {val:.0f}  (poor credit)")
                elif val > 750:
                    rules.append(f"credit_score > {val:.0f}  (excellent credit)")
            elif fname == "loan_to_income_ratio":
                if val > 2.0:
                    rules.append(f"loan_to_income_ratio > {val:.2f}  (high debt burden)")
            elif fname == "interest_rate":
                if val > 18:
                    rules.append(f"interest_rate > {val:.1f}%  (high rate)")
            elif fname == "existing_loans":
                if val >= 3:
                    rules.append(f"existing_loans ≥ {val:.0f}  (heavy existing debt)")

        print(f"\n  ┌─ RULE-BASED EXPLANATION (sample #{idx}) " + "─" * 18)
        if rules:
            print(f"  │  IF  " + "\n  │  AND ".join(rules))
        else:
            print("  │  (Applicant does not trigger any single risk threshold — "
                  "decision is based on combined profile)")
        print(f"  │")
        print(f"  │  THEN  Prediction = {label}")
        print("  └" + "─" * 58)

    def explain_multiple(
        self,
        X: np.ndarray,
        indices: List[int],
        threshold: Optional[float] = None,
    ) -> None:
        """Explain multiple samples in sequence."""
        for idx in indices:
            self.explain(X, sample_index=idx, threshold=threshold)
