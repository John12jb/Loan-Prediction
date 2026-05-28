"""
src/data_preparation/feature_engineering.py
---------------------------------------------
Derives additional predictive features from the raw loan dataset.

New features created:
  - emi_estimate         : Estimated monthly EMI (loan_amount / tenure_months)
  - emi_to_income_ratio  : EMI as a fraction of monthly income
  - credit_risk_band     : Ordinal credit risk category from credit_score
  - high_interest_flag   : Binary flag for interest_rate > 18%
  - income_per_year      : Normalised annual income bucket
  - debt_service_ratio   : (existing_loans * 5000 + emi_estimate) / monthly_income

These features are optional — they are derived before preprocessing and
appended to the DataFrame. Toggle them in config.yaml if needed.
"""

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps.
    Input  : Raw DataFrame (post-load, pre-preprocessing).
    Output : DataFrame with additional derived columns.
    """
    df = df.copy()

    # ── EMI estimate (flat division — approximate) ─────────────────
    df["emi_estimate"] = df["loan_amount"] / df["tenure_months"].replace(0, np.nan)

    # ── EMI-to-monthly-income ratio ────────────────────────────────
    monthly_income = df["annual_income"] / 12
    df["emi_to_income_ratio"] = (
        df["emi_estimate"] / monthly_income.replace(0, np.nan)
    ).round(4)

    # ── Debt service ratio ─────────────────────────────────────────
    # Assumes ₹5,000/month per existing loan as a rough estimate
    existing_debt_monthly = df["existing_loans"] * 5000
    df["debt_service_ratio"] = (
        (existing_debt_monthly + df["emi_estimate"]) / monthly_income.replace(0, np.nan)
    ).round(4)

    # ── Credit risk band (ordinal) ─────────────────────────────────
    # Ranges based on standard credit bureau categorisation
    bins   = [0,   549,  649,  749,  849,  901]
    labels = [4,     3,    2,    1,    0     ]   # 0 = best, 4 = worst
    df["credit_risk_band"] = pd.cut(
        df["credit_score"].fillna(650),
        bins=bins,
        labels=labels,
        right=True,
    ).astype(float)

    # ── High interest flag ─────────────────────────────────────────
    df["high_interest_flag"] = (df["interest_rate"] > 18).astype(int)

    # ── Income bucket (log-scaled) ─────────────────────────────────
    df["log_annual_income"] = np.log1p(df["annual_income"].clip(lower=0))

    # ── Drop intermediates that would leak info ────────────────────
    df = df.drop(columns=["emi_estimate"], errors="ignore")

    _report(df)
    return df


def _report(df: pd.DataFrame) -> None:
    new_cols = [
        "emi_to_income_ratio",
        "debt_service_ratio",
        "credit_risk_band",
        "high_interest_flag",
        "log_annual_income",
    ]
    print("[FeatureEngineering] New features added:")
    for col in new_cols:
        if col in df.columns:
            print(f"  {col:<28}  "
                  f"mean={df[col].mean():.3f}  "
                  f"null={df[col].isnull().sum()}")
