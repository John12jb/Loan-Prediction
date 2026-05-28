"""
generate_sample_data.py
-----------------------
Generates a realistic synthetic loan dataset for the Loan Default
Prediction project. Saves to data/loan_data.csv.

Run:  python data/generate_sample_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_SAMPLES = 5000

np.random.seed(SEED)


def generate_loan_data(n: int = N_SAMPLES) -> pd.DataFrame:
    user_ids = [f"USR_{str(i).zfill(5)}" for i in range(1, n + 1)]

    loan_categories = np.random.choice(
        ["Personal", "Home", "Auto", "Business", "Education"],
        size=n, p=[0.35, 0.25, 0.20, 0.12, 0.08]
    )

    annual_income = np.random.lognormal(mean=11.0, sigma=0.6, size=n)
    annual_income = np.clip(annual_income, 150_000, 5_000_000)

    loan_amount = annual_income * np.random.uniform(0.5, 4.0, size=n)
    loan_amount = np.clip(loan_amount, 10_000, 10_000_000).round(-3)

    interest_rate = np.random.normal(loc=12.5, scale=4.0, size=n)
    interest_rate = np.clip(interest_rate, 5.0, 28.0).round(2)

    tenure_months = np.random.choice([12, 24, 36, 48, 60, 84, 120], size=n)

    employment_type = np.random.choice(
        ["Salaried", "Self-Employed", "Unemployed"],
        size=n, p=[0.60, 0.32, 0.08]
    )

    credit_score = np.random.normal(loc=650, scale=90, size=n).astype(int)
    credit_score = np.clip(credit_score, 300, 900)

    existing_loans = np.random.poisson(lam=1.2, size=n)
    existing_loans = np.clip(existing_loans, 0, 6)

    loan_to_income_ratio = (loan_amount / annual_income).round(4)

    # ── Target variable: Defaulter ──
    # Default probability is driven by realistic risk factors
    log_odds = (
        -3.0
        + 0.8  * (loan_to_income_ratio > 2.5).astype(float)
        + 0.6  * (interest_rate > 18).astype(float)
        + 1.2  * (credit_score < 550).astype(float)
        + 0.5  * (credit_score < 650).astype(float)
        + 0.4  * (existing_loans >= 3).astype(float)
        + 0.9  * (employment_type == "Unemployed").astype(float)
        + 0.3  * (employment_type == "Self-Employed").astype(float)
        + 0.2  * np.random.normal(0, 1, n)   # noise
    )
    prob_default = 1 / (1 + np.exp(-log_odds))
    defaulter = (np.random.rand(n) < prob_default).astype(int)

    df = pd.DataFrame({
        "user_id":              user_ids,
        "loan_category":        loan_categories,
        "loan_amount":          loan_amount.round(2),
        "interest_rate":        interest_rate,
        "tenure_months":        tenure_months,
        "employment_type":      employment_type,
        "annual_income":        annual_income.round(2),
        "credit_score":         credit_score,
        "existing_loans":       existing_loans,
        "loan_to_income_ratio": loan_to_income_ratio,
        "Defaulter":            defaulter,
    })

    # Introduce ~3% missing values realistically
    for col in ["credit_score", "annual_income", "employment_type"]:
        mask = np.random.rand(n) < 0.03
        df.loc[mask, col] = np.nan

    return df


if __name__ == "__main__":
    out_path = Path(__file__).parent / "loan_data.csv"
    df = generate_loan_data()
    df.to_csv(out_path, index=False)

    print(f"Dataset saved → {out_path}")
    print(f"Shape         : {df.shape}")
    print(f"Default rate  : {df['Defaulter'].mean():.2%}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
