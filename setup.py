from setuptools import setup, find_packages

setup(
    name="loan_default_xai",
    version="1.0.0",
    description="Loan Default Prediction with Explainable AI (XGBoost + Random Forest + SHAP + Anchors + Counterfactuals)",
    author="John",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.1",
        "numpy>=1.26",
        "scikit-learn>=1.3",
        "xgboost>=2.0",
        "imbalanced-learn>=0.11",
        "hyperopt>=0.2.7",
        "shap>=0.44",
        "matplotlib>=3.8",
        "seaborn>=0.13",
        "pyyaml>=6.0",
        "joblib>=1.3",
        "tqdm>=4.66",
    ],
    extras_require={
        "full": [
            "alibi>=0.9",
            "dice-ml>=0.11",
            "deepchecks>=0.18",
        ],
        "tracking": [
            "neptune-client>=1.8",
        ],
    },
)
