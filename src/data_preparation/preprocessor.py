"""
src/data_preparation/preprocessor.py
--------------------------------------
Full preprocessing pipeline:
  1. Drop irrelevant columns
  2. Handle missing values
  3. Address skewness via log-transform
  4. Check and handle multicollinearity
  5. Encode categorical features
  6. Train / test split
  7. Handle class imbalance with SMOTE
"""

import numpy as np
import pandas as pd
import warnings
from typing import Tuple, List, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")


class LoanPreprocessor:
    """
    Stateful preprocessor — fit on training data, transform on test data.
    This prevents data leakage from test set into encoders or imputers.
    """

    def __init__(
        self,
        target_col: str = "Defaulter",
        drop_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
        skew_threshold: float = 1.0,
        test_size: float = 0.2,
        random_state: int = 42,
        smote_strategy: float = 0.5,
    ):
        self.target_col = target_col
        self.drop_cols = drop_cols or ["user_id"]
        self.categorical_cols = categorical_cols or ["loan_category", "employment_type"]
        self.numeric_cols = numeric_cols or ["loan_amount", "annual_income", "loan_to_income_ratio"]
        self.skew_threshold = skew_threshold
        self.test_size = test_size
        self.random_state = random_state
        self.smote_strategy = smote_strategy

        # Fitted state
        self._encoders: dict = {}
        self._numeric_medians: dict = {}
        self._categorical_modes: dict = {}
        self._log_transformed_cols: List[str] = []
        self._feature_names: List[str] = []

    # ────────────────────────────────────────────────────────────────
    #  Public API
    # ────────────────────────────────────────────────────────────────

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Full pipeline on raw data.
        Returns X_train, X_test, y_train, y_test, feature_names.
        """
        df = df.copy()

        df = self._drop_columns(df)
        df = self._impute_missing(df, fit=True)
        df = self._transform_skewed(df, fit=True)
        self._check_multicollinearity(df)
        df = self._encode_categoricals(df, fit=True)

        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        # Safety net — auto-drop any remaining non-numeric columns
        # (e.g. ID columns not listed in drop_cols, unexpected strings).
        # Prevents "could not convert string to float" errors in SMOTE.
        leftover = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if leftover:
            print(f"[Preprocessor] Auto-dropping non-numeric columns: {leftover}")
            X = X.drop(columns=leftover)

        self._feature_names = list(X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        X_train, y_train = self._apply_smote(X_train, y_train)

        print(f"[Preprocessor] Train: {X_train.shape}  |  Test: {X_test.shape}")
        print(f"[Preprocessor] Train class balance after SMOTE: "
              f"{np.bincount(y_train.astype(int))}")

        return (
            X_train.values if hasattr(X_train, "values") else X_train,
            X_test.values if hasattr(X_test, "values") else X_test,
            y_train.values if hasattr(y_train, "values") else y_train,
            y_test.values if hasattr(y_test, "values") else y_test,
            self._feature_names,
        )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data (after fit_transform has been called)."""
        df = df.copy()
        df = self._drop_columns(df, drop_target=False)
        df = self._impute_missing(df, fit=False)
        df = self._transform_skewed(df, fit=False)
        df = self._encode_categoricals(df, fit=False)
        return df[self._feature_names].values

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    # ────────────────────────────────────────────────────────────────
    #  Private helpers
    # ────────────────────────────────────────────────────────────────

    def _drop_columns(self, df: pd.DataFrame, drop_target: bool = True) -> pd.DataFrame:
        cols_to_drop = [c for c in self.drop_cols if c in df.columns]
        if drop_target and self.target_col in df.columns:
            pass  # keep target — we separate X/y later
        df = df.drop(columns=cols_to_drop, errors="ignore")
        return df

    def _impute_missing(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        # Numeric → median imputation
        for col in df.select_dtypes(include=[np.number]).columns:
            if col == self.target_col:
                continue
            if fit:
                self._numeric_medians[col] = df[col].median()
            df[col] = df[col].fillna(self._numeric_medians.get(col, 0))

        # Categorical → mode imputation
        for col in df.select_dtypes(include=["object", "category"]).columns:
            if fit:
                self._categorical_modes[col] = df[col].mode()[0]
            df[col] = df[col].fillna(self._categorical_modes.get(col, "Unknown"))

        return df

    def _transform_skewed(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Log-transform columns with high skewness to reduce it."""
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric = [c for c in numeric if c != self.target_col]

        if fit:
            self._log_transformed_cols = []
            for col in self.numeric_cols:
                if col in df.columns:
                    skew = df[col].skew()
                    if abs(skew) > self.skew_threshold:
                        self._log_transformed_cols.append(col)
                        print(f"[Preprocessor] Log-transform '{col}'  (skew={skew:.2f})")

        for col in self._log_transformed_cols:
            if col in df.columns:
                df[col] = np.log1p(df[col].clip(lower=0))

        return df

    def _check_multicollinearity(self, df: pd.DataFrame) -> None:
        """Print a warning for highly correlated numeric feature pairs."""
        numerics = df.select_dtypes(include=[np.number]).drop(
            columns=[self.target_col], errors="ignore"
        )
        corr = numerics.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high = [(c, r, upper.loc[r, c]) for r in upper.index for c in upper.columns
                if not pd.isna(upper.loc[r, c]) and upper.loc[r, c] > 0.85]
        if high:
            print("[Preprocessor] ⚠  High multicollinearity detected:")
            for c1, c2, v in high:
                print(f"   {c1}  ↔  {c2}  (r={v:.2f})")
        else:
            print("[Preprocessor] ✓  No high multicollinearity detected.")

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        for col in self.categorical_cols:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self._encoders[col] = le
            else:
                le = self._encoders.get(col)
                if le:
                    # Handle unseen labels gracefully
                    known = set(le.classes_)
                    df[col] = df[col].astype(str).apply(
                        lambda x: x if x in known else le.classes_[0]
                    )
                    df[col] = le.transform(df[col])
        return df

    def _apply_smote(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE to oversample the minority class."""
        sm = SMOTE(
            sampling_strategy=self.smote_strategy,
            random_state=self.random_state,
            k_neighbors=5,
        )
        X_res, y_res = sm.fit_resample(X, y)
        return X_res, y_res
