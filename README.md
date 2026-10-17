# MLOps Lab 4 & 5: Advanced MLflow & Data Versioning with DVC

**Student:** Chirag Jindani  
**Roll No:** 24AI020  
**Course:** Machine Learning Operations (MLOps)  

---

## 📌 Project Overview

This repository demonstrates an end-to-end reproducible MLOps workflow integrating:
1. **DVC (Data Version Control)**: Versioning large datasets, configuring local remote storage, and ensuring reliable data rollbacks.
2. **MLflow Tracking & Model Registry**: Tracking hyperparameters, evaluating regression metrics ($MAE, RMSE, R^2$), logging model signatures and visual artifacts, and enforcing automated lifecycle promotion (`@candidate` / `Staging`, `@production` / `Production`, `@archived` / `Archived`).
3. **Git Integration**: Full auditability linking Git commit SHAs, DVC dataset content hashes, and MLflow run tags.

---

## 🗂 Repository Structure

```
mlops-lab-4-5/
├── .dvc/                   # DVC configuration & internal metadata
├── .dvcignore              # DVC ignore rules
├── .gitignore              # Git ignore configuration
├── configs/
│   └── config.yaml         # Central pipeline configuration
├── data/
│   ├── .gitignore          # Ignores large raw dataset from Git
│   └── dataset.csv.dvc     # DVC pointer tracking dataset.csv
├── dvc.yaml                # DVC multi-stage pipeline definition
├── dvc.lock                # DVC pipeline lockfile ensuring reproducibility
├── models/                 # Model persistence directory
├── reports/
│   └── figures/            # Diagnostic plots (residuals, actual vs predicted, feature importance)
│       ├── v1_ridge/
│       ├── v2_random_forest/
│       └── v3_gradient_boosting/
├── requirements.txt        # Exact environment dependencies
├── src/
│   ├── evaluate.py         # Evaluation metrics & diagnostic plotting routines
│   ├── prepare_data.py     # Data download & version perturbation generator
│   └── train.py            # MLflow experiment tracking & model registry lifecycle
└── README.md
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/CHIRAG-JINDANI/mlops-lab-4-5.git
cd mlops-lab-4-5

# Create and activate virtual environment (Python 3.12 - 3.14 recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. DVC Data Setup & Remote Verification

```bash
# Pull the dataset from the configured DVC remote
dvc pull

# Verify dataset row count
wc -l data/dataset.csv
```

### 3. Run Pipeline via DVC

```bash
# Reproduce entire pipeline (skips unchanged steps automatically)
dvc repro

# Visualize pipeline dependency graph
dvc dag
```

### 4. Launch MLflow UI

```bash
# Start MLflow UI pointing to SQLite backend
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Then open your browser at `http://127.0.0.1:5000` to inspect runs, artifacts, signatures, and Model Registry versions.

### 5. Retrieve & Evaluate the Production Model

```bash
python src/evaluate.py --model-uri "models:/HousePriceModel@production"
```

---

## 📊 Summary of Model Candidates

| Registered Version | Candidate Name | Model Architecture | Test RMSE | Test MAE | Test $R^2$ | Lifecycle State | Promotion Decision |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Version 1** | `v1_ridge` | Ridge Regression ($L_2$) | 0.7530 | 0.5315 | 0.5715 | `Archived` (`@archived`) | Fails RMSE threshold ($\le 0.65$); archived baseline. |
| **Version 2** | `v2_random_forest` | Random Forest (100 trees, depth 10) | 0.5428 | 0.3639 | 0.7774 | `Staging` (`@candidate`) | Passes threshold; candidate for validation. |
| **Version 3** | `v3_gradient_boosting` | Gradient Boosting (200 trees, lr 0.08) | **0.4783** | **0.3175** | **0.8272** | **`Production` (`@production`)** | **Best performance across all metrics; promoted.** |

---

## 🔁 Dataset Versioning Summary

| Dataset Version | Git Commit | Row Count | Column Count | DVC MD5 Hash | Verification |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Dataset v1** | `6767005` | 20,640 | 9 | `ca5b757f0729c68d74d05bf9d9ca8486` | Restored via `dvc checkout` and verified |
| **Dataset v2** | `666c6cd` | 20,740 | 9 | `5968e75c88d31ad1a1083b00d57fba4e` | Added 100 housing survey samples |
