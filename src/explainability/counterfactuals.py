"""
src/explainability/counterfactuals.py
---------------------------------------
Counterfactual explanations — "What is the minimum change to this
applicant's profile that would flip the decision from DEFAULT to
NO DEFAULT?"

Uses DiCE (Diverse Counterfactual Explanations) when available,
with a gradient-free fallback for restricted environments.

Example output:
  Original  : credit_score=480, loan_amount=800,000, interest_rate=22.5
  Flip to   : credit_score=582, loan_amount=720,000, interest_rate=18.0
  Change    : Improve credit score by 102 points, reduce loan by ₹80,000
"""

import numpy as np
import pandas as pd
from typing import List, Any, Optional, Dict


class CounterfactualExplainer:
    """
    Generates diverse counterfactual explanations using DiCE or a
    built-in genetic search fallback.
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        X_train: np.ndarray,
        y_train: np.ndarray,
        num_cfs: int = 5,
        desired_class: int = 0,
    ):
        self.model = model
        self.feature_names = feature_names
        self.X_train = X_train
        self.y_train = y_train
        self.num_cfs = num_cfs
        self.desired_class = desired_class
        self._use_dice = self._try_init_dice()

    def _try_init_dice(self) -> bool:
        try:
            import dice_ml
            self._dice_ml = dice_ml
            return True
        except ImportError:
            print("[Counterfactuals] dice-ml not installed — using genetic fallback.")
            return False

    def explain(
        self,
        X: np.ndarray,
        sample_index: int = 0,
    ) -> None:
        """
        Generate counterfactual explanations for a single sample.

        Parameters
        ----------
        X            : Test set (numpy array).
        sample_index : Index of the sample to explain.
        """
        sample = X[sample_index]
        original_pred = self.model.predict(sample.reshape(1, -1))[0]

        print(f"\n[Counterfactuals] Explaining sample #{sample_index}")
        print(f"  Original prediction : {'DEFAULT' if original_pred == 1 else 'NO DEFAULT'}")
        print(f"  Target prediction   : {'DEFAULT' if self.desired_class == 1 else 'NO DEFAULT'}")

        self._print_original_profile(sample)

        if original_pred == self.desired_class:
            print("  ✓  Prediction already matches desired class — no changes needed.\n")
            return

        if self._use_dice:
            self._explain_dice(sample, sample_index)
        else:
            self._explain_genetic(sample, sample_index, X)

    def _explain_dice(self, sample: np.ndarray, sample_index: int) -> None:
        try:
            df_train = pd.DataFrame(self.X_train, columns=self.feature_names)
            df_train["Defaulter"] = self.y_train

            d = self._dice_ml.Data(
                dataframe=df_train,
                continuous_features=self.feature_names,
                outcome_name="Defaulter",
            )
            m = self._dice_ml.Model(model=self.model, backend="sklearn")

            exp = self._dice_ml.Dice(d, m, method="random")
            query = pd.DataFrame([sample], columns=self.feature_names)
            result = exp.generate_counterfactuals(
                query,
                total_CFs=self.num_cfs,
                desired_class=self.desired_class,
            )
            cfs = result.cf_examples_list[0].final_cfs_df

            print(f"\n  ┌─ TOP {self.num_cfs} COUNTERFACTUALS " + "─" * 34)
            for i, (_, row) in enumerate(cfs.iterrows(), 1):
                changes = self._diff_sample(sample, row[self.feature_names].values)
                print(f"  │  Counterfactual #{i}:")
                for feat, orig, new in changes:
                    delta = new - orig
                    sign = "+" if delta > 0 else ""
                    print(f"  │    {feat:<28} {orig:>10.2f} → {new:>10.2f}  ({sign}{delta:.2f})")
                print(f"  │")
            print("  └" + "─" * 58)

        except Exception as e:
            print(f"  [DiCE] Error: {e} — switching to fallback.")
            self._explain_genetic(None, sample_index, None, sample=sample)

    def _explain_genetic(
        self,
        sample: Optional[np.ndarray],
        sample_index: int,
        X: Optional[np.ndarray],
        sample_override: Optional[np.ndarray] = None,
    ) -> None:
        """
        Simple perturbation-based counterfactual search.
        Perturbs features from the training distribution until the
        model flips to the desired class.
        """
        s = sample_override if sample_override is not None else sample
        best_cfs = []
        np.random.seed(42)

        for trial in range(2000):
            candidate = s.copy().astype(float)
            n_perturb = np.random.randint(1, 4)
            feat_idx = np.random.choice(len(self.feature_names), n_perturb, replace=False)

            for fi in feat_idx:
                col_vals = self.X_train[:, fi]
                std = col_vals.std()
                candidate[fi] += np.random.uniform(-std, std)

            pred = self.model.predict(candidate.reshape(1, -1))[0]
            if pred == self.desired_class:
                dist = np.linalg.norm(candidate - s)
                best_cfs.append((dist, candidate.copy()))

            if len(best_cfs) >= self.num_cfs:
                break

        if not best_cfs:
            print("  [Counterfactuals] Could not find a valid counterfactual within budget.")
            return

        best_cfs.sort(key=lambda x: x[0])
        print(f"\n  ┌─ COUNTERFACTUAL EXPLANATIONS " + "─" * 27)
        for i, (dist, cf) in enumerate(best_cfs[:self.num_cfs], 1):
            changes = self._diff_sample(s, cf)
            print(f"  │  Option #{i}  (distance={dist:.3f}):")
            for feat, orig, new in changes:
                delta = new - orig
                sign = "+" if delta > 0 else ""
                print(f"  │    {feat:<28} {orig:>10.2f} → {new:>10.2f}  ({sign}{delta:.2f})")
            print(f"  │")
        print("  └" + "─" * 58)
        print("  Interpretation: The smallest realistic changes to the applicant's")
        print("  profile that would flip the loan decision to NO DEFAULT.\n")

    def _print_original_profile(self, sample: np.ndarray) -> None:
        print("\n  Original Applicant Profile:")
        for fname, val in zip(self.feature_names, sample):
            print(f"    {fname:<28} {val:>12.3f}")

    def _diff_sample(
        self,
        original: np.ndarray,
        modified: np.ndarray,
        tol: float = 1e-4,
    ) -> List[tuple]:
        changes = []
        for i, (o, m) in enumerate(zip(original, modified)):
            if abs(o - m) > tol:
                changes.append((self.feature_names[i], o, m))
        return changes
