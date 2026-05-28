# Loan Default Prediction — Explainable AI

> Predicting loan defaults with XGBoost & Random Forest, then explaining *why* every prediction was made using SHAP, Anchors, and Counterfactuals.

---

## Project Overview

Loan default prediction is one of the most critical tasks in financial risk management. This project goes beyond accuracy — it makes every model prediction **transparent and auditable**, addressing the needs of data scientists, loan officers, and regulators.

### Why Explainability Matters

| Stakeholder | Need |
|---|---|
| Data Scientists | Debug, improve, and compare models |
| Loan Officers | Trust and act on predictions |
| Regulators | Audit for fairness and compliance |

---

## Project Structure

```
loan_default_xai/
├── config/
│   └── config.yaml                  # All hyperparameters and paths
├── data/
│   └── generate_sample_data.py      # Synthetic dataset generator
├── src/
│   ├── data_preparation/
│   │   ├── loader.py                # Data loading and merging
│   │   ├── preprocessor.py          # Cleaning, encoding, balancing
│   │   └── feature_engineering.py  # Derived risk features
│   ├── models/
│   │   ├── xgboost_model.py         # XGBoost with Hyperopt tuning
│   │   ├── random_forest_model.py   # RF with GridSearchCV
│   │   ├── tuning.py                # Comparison charts, threshold analysis
│   │   └── experiment_tracker.py   # Neptune tracking (optional)
│   ├── evaluation/
│   │   └── metrics.py               # Classification metrics + confusion matrix
│   └── explainability/
│       ├── shap_explainer.py        # SHAP values, plots, dependence
│       ├── anchors_explainer.py     # Anchor rule-based explanations
│       └── counterfactuals.py       # What-if counterfactual analysis
├── notebooks/
│   └── full_pipeline.ipynb          # End-to-end walkthrough notebook
├── outputs/
│   ├── plots/                       # All generated visualisations (PNG)
│   └── models/                      # Saved model artifacts (.pkl)
├── main.py                          # Entry point — runs full pipeline
├── setup.py
└── requirements.txt
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| ML Models | XGBoost, Random Forest (scikit-learn) |
| Tuning | Hyperopt (Bayesian), GridSearchCV |
| XAI | SHAP, Anchors (alibi), DiCE (Counterfactuals) |
| Data | pandas, numpy, imbalanced-learn (SMOTE) |
| Visualisation | matplotlib, seaborn |
| Tracking | Neptune (optional) |

---

## Dataset Schema

| Column | Description |
|---|---|
| user_id | Applicant identifier |
| loan_category | Type of loan (Personal, Home, Auto, Business) |
| loan_amount | Loan amount requested |
| interest_rate | Annual interest rate (%) |
| tenure_months | Loan repayment period |
| employment_type | Salaried / Self-Employed / Unemployed |
| annual_income | Gross annual income |
| credit_score | Credit bureau score (300–900) |
| existing_loans | Number of active loans |
| loan_to_income_ratio | loan_amount / annual_income |
| Defaulter | **Target** — 1 = Default, 0 = No Default |

---

## ▶ Option 1 — Run on Google Colab (No Setup Required)

> Best for: quick results in your browser without installing anything.

### Step 1 — Open a new Colab notebook
Go to [colab.research.google.com](https://colab.research.google.com) and click **New Notebook**.

### Step 2 — Upload the zip file
Paste this in **Cell 1** and run it. A **Choose File** button will appear — select `loan_default_xai.zip`:

```python
from google.colab import files
uploaded = files.upload()  # select loan_default_xai.zip when prompted
```

### Step 3 — Unzip and install dependencies
Paste in **Cell 2** and run:

```python
import zipfile
with zipfile.ZipFile('/content/loan_default_xai.zip', 'r') as z:
    z.extractall('/content/')

%pip install xgboost imbalanced-learn hyperopt shap --quiet
```

### Step 4 — Run the full pipeline
Paste in **Cell 3** and run (takes ~2–3 minutes):

```python
import os, sys
os.chdir('/content/loan_default_xai')
sys.path.insert(0, '/content/loan_default_xai')

%run data/generate_sample_data.py
%run main.py
```

### Step 5 — View all results inline
Paste in **Cell 4** and run:

```python
import glob
from IPython.display import Image, display

plots = sorted(glob.glob('outputs/plots/*.png'))
print(f"Total plots generated: {len(plots)}\n")

for img in plots:
    name = img.split('/')[-1].replace('.png', '').replace('_', ' ').upper()
    print(f"\n{'─' * 55}")
    print(f"  {name}")
    print(f"{'─' * 55}")
    display(Image(filename=img, width=750))
```

---

## ▶ Option 2 — Run on Jupyter Notebook (Local)

> Best for: running fully offline on your own machine.

### Step 1 — Install Jupyter and dependencies
Open a terminal / command prompt and run:

```bash
pip install jupyter
pip install -r requirements.txt
```

> **Windows users (Python 3.13):** the `requirements.txt` uses flexible version bounds
> that install pre-built wheels — no compilation errors.

### Step 2 — Launch Jupyter
```bash
cd path/to/loan_default_xai
jupyter notebook
```

Your browser will open at `http://localhost:8888`.

### Step 3 — Open the notebook
In the Jupyter file browser, click:

```
notebooks/  →  full_pipeline.ipynb
```

### Step 4 — Run all cells
At the top menu click:

```
Run  →  Run All Cells
```

All outputs — metrics, confusion matrices, ROC curves, SHAP plots — will appear inline as each cell finishes.

### Step 5 — View saved plots
All PNG files are saved to:

```
outputs/plots/
```

Open that folder in File Explorer to see all 12 visualisations.

---

## ▶ Option 3 — Run via Command Line (VS Code / Terminal)

```bash
# 1. Navigate to the project
cd path/to/loan_default_xai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the dataset
python data/generate_sample_data.py

# 4. Run the full pipeline
python main.py

# 5. Plots are saved to:
#    outputs/plots/
```

---

## Output Plots

| Plot | Description |
|---|---|
| `model_comparison.png` | XGBoost vs Random Forest — all metrics side by side |
| `confusion_matrix_xgboost.png` | Predicted vs actual for XGBoost |
| `confusion_matrix_random_forest.png` | Predicted vs actual for Random Forest |
| `roc_curve_xgboost.png` | ROC curve with AUC for XGBoost |
| `roc_curve_random_forest.png` | ROC curve with AUC for Random Forest |
| `threshold_analysis_xgboost.png` | F1 / Precision / Recall vs decision threshold |
| `shap_summary_plot.png` | Global feature importance (beeswarm) |
| `shap_bar_importance.png` | Mean \|SHAP\| bar chart |
| `shap_dependence_credit_score.png` | How credit score affects default risk |
| `shap_dependence_loan_to_income_ratio.png` | How debt burden affects default risk |
| `shap_dependence_interest_rate.png` | How interest rate affects default risk |
| `shap_waterfall_sample_5.png` | Single applicant explanation |

---

## Key Results

| Metric | XGBoost | Random Forest |
|---|---|---|
| Accuracy | 82.6% | 82.0% |
| Precision | 0.32 | 0.27 |
| Recall | 0.16 | 0.13 |
| F1 Score | 0.22 | 0.17 |
| ROC-AUC | 0.62 | 0.63 |

> **Note:** Tuning the decision threshold from 0.5 → 0.12 increases F1 to **0.29**
> without retraining. See `threshold_analysis_xgboost.png`.

---

## Explainability Outputs

| Technique | What it tells you | Audience |
|---|---|---|
| SHAP Summary | Which features drive defaults globally | Data Scientists |
| SHAP Dependence | How each feature's effect changes across its range | Data Scientists |
| SHAP Waterfall | Why a specific prediction was made | Loan Officers |
| Anchors | IF-THEN rules with guaranteed precision | Regulators |
| Counterfactuals | Minimum changes to flip a decision | Applicants |

---

## Optional — Full XAI Support

Install these for Anchors and Counterfactuals:

```bash
pip install alibi>=0.9.4      # Anchor explanations
pip install dice-ml>=0.11     # Counterfactual explanations
pip install deepchecks>=0.18  # Model validation report
```

---

*Built with Python · XGBoost · SHAP · scikit-learn*
